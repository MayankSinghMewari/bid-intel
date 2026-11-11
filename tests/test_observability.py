"""Tracing layer: no-op without keys; with a (fake) Langfuse client, runs become traces, LLM calls become
generations with token usage, and every local trace step is mirrored as an event of the same trace."""
from contextlib import contextmanager
from pathlib import Path

from bid_intel import llm, observability
from bid_intel.bid_agents import Orchestrator
from bid_intel.docs import ingest_folder
from bid_intel.search import BidSearch

BID2 = Path(__file__).resolve().parent.parent / "data" / "bid_folders" / "Bid2"


class FakeObs:
    def __init__(self, store, kind, kw):
        self.store, self.trace_id, self.id = store, "trace-1", f"obs-{len(store['obs'])}"
        store["obs"].append({"kind": kind, **kw})
        self.record = store["obs"][-1]

    def update(self, **kw):
        self.record.update(kw)


class FakeClient:
    def __init__(self):
        self.store = {"obs": [], "events": []}

    @contextmanager
    def _cm(self, kw):
        yield FakeObs(self.store, kw.get("as_type", "span"), kw)

    def start_as_current_observation(self, **kw):
        return self._cm(kw)

    def create_event(self, **kw):
        self.store["events"].append(kw)

    def get_trace_url(self, trace_id):
        return f"https://langfuse.example/trace/{trace_id}"

    def flush(self):
        pass


def test_everything_is_a_no_op_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(observability, "_client", None)
    assert not observability.enabled() and observability.client() is None
    with observability.trace("x") as root, observability.generation("g", "m", "in") as gen:
        assert root is None and gen is None
    assert observability.langgraph_callbacks() == [] and observability.check()["enabled"] is False


def test_run_becomes_a_trace_with_events_and_generations(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(observability, "client", lambda: fake)
    monkeypatch.setattr(observability, "langgraph_callbacks", lambda: [])
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "_call", lambda *a, **k: {
        "text": "Summary.", "model": "fake-model", "parsed": None,
        "input_tokens": 120, "output_tokens": 30, "thinking_tokens": 50})
    orch = Orchestrator(BID2, engine=BidSearch(ingest_folder(BID2)[0], dense=False), use_llm=True)
    orch.run()

    root = fake.store["obs"][0]
    assert root["name"] == "extract Bid2" and root["kind"] == "agent" and "validation" in root["output"]
    assert orch.trace_url == "https://langfuse.example/trace/trace-1"
    # every local trace step is an event of this trace
    names = [e["name"] for e in fake.store["events"]]
    assert len(names) == len(orch.trace()) and "orchestrator.plan" in names and "validator.validate" in names
    assert all(e["trace_context"]["trace_id"] == "trace-1" for e in fake.store["events"])
    assert any(e["level"] == "WARNING" for e in fake.store["events"] if e["name"] == "orchestrator.retry")
    # LLM calls are generations with model and token usage (incl. thinking tokens)
    gens = [o for o in fake.store["obs"] if o["kind"] == "generation"]
    assert gens and all(g["model"] in (llm.GEMINI_MODEL, llm.MODEL) for g in gens)
    assert {"input": 120, "output": 30, "reasoning": 50} in [g.get("usage_details") for g in gens]
