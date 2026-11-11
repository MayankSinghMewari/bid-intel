"""Tracing layer: no-op without keys; with a (fake) Langfuse client, a run becomes one trace of typed
observations with stable names, LLM calls become generations with token usage, and orchestrator decisions
become events."""
from contextlib import contextmanager
from pathlib import Path

from bid_intel import llm, observability
from bid_intel.agents import Orchestrator
from bid_intel.ingest import ingest_folder
from bid_intel.search import BidSearch

BID2 = Path(__file__).resolve().parent.parent / "data" / "bid_folders" / "Bid2"


class FakeObs:
    def __init__(self, store, kw):
        self.trace_id, self.id = "trace-1", f"obs-{len(store['obs'])}"
        store["obs"].append({"as_type": kw.get("as_type", "span"), **kw})
        self.record = store["obs"][-1]

    def update(self, **kw):
        self.record.update(kw)


class FakeClient:
    def __init__(self):
        self.store = {"obs": [], "events": []}

    @contextmanager
    def _cm(self, kw):
        yield FakeObs(self.store, kw)

    def start_as_current_observation(self, **kw):
        return self._cm(kw)

    def create_event(self, **kw):
        self.store["events"].append(kw)

    def get_trace_url(self, trace_id):
        return f"https://langfuse.example/trace/{trace_id}"

    def shutdown(self):
        pass


def test_everything_is_a_no_op_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(observability, "_client", None)
    assert not observability.enabled() and observability.client() is None
    with observability.trace("x") as root, observability.span("y") as sp, \
            observability.generation("g", "m", "in") as gen:
        assert root is None and sp is None and gen is None
    assert observability.check()["enabled"] is False


def test_spans_outside_a_trace_are_no_ops(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(observability, "client", lambda: fake)
    with observability.span("search-evidence", "retriever") as sp:
        assert sp is None
    assert fake.store["obs"] == []


def test_run_becomes_one_trace_of_typed_observations(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(observability, "client", lambda: fake)
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "_call", lambda *a, **k: {
        "text": "Summary.", "model": "fake-model", "parsed": None,
        "input_tokens": 120, "output_tokens": 30, "thinking_tokens": 50})
    orch = Orchestrator(BID2, engine=BidSearch(ingest_folder(BID2)[0], dense=False), use_llm=True)
    orch.run()
    obs = fake.store["obs"]
    root = obs[0]
    # root: stable verb-first name, run-specific values in metadata, reviewer-friendly input/output
    assert root["name"] == "extract-bid" and root["as_type"] == "agent"
    assert root["metadata"]["bid_id"] == "Bid2" and "validation" in root["output"]
    assert orch.trace_url == "https://langfuse.example/trace/trace-1"
    types = {}
    for o in obs[1:]:
        types.setdefault(o["name"], set()).add(o["as_type"])
    # most specific observation type per step
    assert types["ingest-documents"] == {"tool"}
    assert types["extract-identity-fields"] == {"agent"} and types["extract-product-fields"] == {"agent"}
    assert types["search-evidence"] == {"retriever"} and types["apply-field-rules"] == {"tool"}
    assert types["validate-field"] == {"evaluator"} and types["reconcile-addenda"] == {"chain"}
    assert types["llm-extract-dates-fields"] == {"agent"} and types["gather-evidence"] == {"retriever"}
    assert types["generate-text"] == {"generation"} and types["summarize-bid"] == {"chain"}
    assert not any(" " in n for n in types)  # stable names, no run-specific values
    # every observation has an input or an output
    assert all(o.get("input") is not None or o.get("output") is not None for o in obs)
    # generations: model, role-labelled messages, token usage including thinking tokens
    gens = [o for o in obs if o["as_type"] == "generation"]
    assert all(g["model"] in (llm.GEMINI_MODEL, llm.MODEL) for g in gens)
    assert all(g["input"][0]["role"] == "system" and g["input"][1]["role"] == "user" for g in gens)
    assert {"input": 120, "output": 30, "reasoning": 50} in [g.get("usage_details") for g in gens]
    # failed validations are flagged; orchestrator decisions are events, agent work is not duplicated
    assert any(o.get("level") == "WARNING" for o in obs if o["name"] == "validate-field")
    events = fake.store["events"]
    assert events and all(e["name"].startswith("orchestrator.") for e in events)
    assert any(e["name"] == "orchestrator.retry" and e["level"] == "WARNING" for e in events)
