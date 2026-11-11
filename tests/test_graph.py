"""LangGraph orchestration: end-to-end on a real bid folder, offline (no dense index, LLM off or faked)."""
import re
from pathlib import Path

from bid_intel import llm
from bid_intel.bid_agents import Orchestrator
from bid_intel.docs import ingest_folder
from bid_intel.graph import mermaid
from bid_intel.llm_extract import FieldValue, GroupExtraction
from bid_intel.search import BidSearch

BID2 = Path(__file__).resolve().parent.parent / "data" / "bid_folders" / "Bid2"


def _engine():
    return BidSearch(ingest_folder(BID2)[0], dense=False)


def _steps(orch, agent):
    return [s for s in orch.trace() if s["agent"] == agent]


def test_graph_runs_rules_path_with_feedback_loop():
    orch = Orchestrator(BID2, engine=_engine(), use_llm=False)
    result = orch.run()
    assert result["fields"]["Bid Number"]["value"] == "BPM044557"
    assert result["fields"]["Delivery Date"]["value"].startswith("Delivery within 45 days")
    assert result["validation"]["failed"] == 0 and len(result["fields"]) == 20
    actions = [s["action"] for s in _steps(orch, "orchestrator")]
    assert actions[0] == "plan" and "retry" in actions and actions[-1] == "finish"
    assert {"next": "reconcile", "reason": "LLM not available"} in [s["output"] for s in _steps(orch, "orchestrator")]
    # four extraction specialists ran (one per field group)
    assert {s["agent"] for s in orch.trace() if s["agent"].startswith("extractor:")} == \
        {"extractor:identity", "extractor:dates", "extractor:commercial", "extractor:product"}


def test_graph_routes_through_llm_specialists_and_merges(monkeypatch):
    calls = []

    def fake_complete(system, user, max_tokens=2048, schema=None, timeout_s=30):
        calls.append(user)
        if schema is None:  # bid summary
            return {"text": "A short summary.", "model": "fake", "parsed": None,
                    "input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0}
        fields = []
        if "- MFG for Registration:" in user:  # the LLM finds one field and cites the passage holding the quote
            passages = re.findall(r"^\[(\d+)\] \(file:.*?\)\n(.*?)(?=\n\n\[\d+\] \(file:|\n\nFIELDS TO EXTRACT)",
                                  user, re.S | re.M)
            n = next(int(i) for i, text in passages if "authorized reseller for Dell" in " ".join(text.split()))
            fields = [FieldValue(name="MFG for Registration", value="Dell", citations=[n],
                                 quote="authorized reseller for Dell")]
        return {"text": "", "model": "fake", "parsed": GroupExtraction(fields=fields),
                "input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0}

    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "_complete", fake_complete)
    orch = Orchestrator(BID2, engine=_engine(), use_llm=True)
    result = orch.run()
    agents = {s["agent"] for s in orch.trace()}
    assert {"llm_extractor:identity", "llm_extractor:dates", "llm_extractor:commercial",
            "llm_extractor:product"} <= agents
    assert any(s["action"] == "llm_merge" for s in _steps(orch, "orchestrator"))
    assert result["fields"]["Bid Summary"]["value"] == "A short summary."
    mfg = result["fields"]["MFG for Registration"]
    assert mfg["value"] == "Dell" and mfg["notes"].startswith("LLM extraction (")
    assert "agrees with rule-based value" in mfg["notes"] and mfg["confidence"] == 0.95
    # rule values survive when the LLM returns nothing for a field
    assert result["fields"]["Delivery Date"]["value"].startswith("Delivery within 45 days")


def test_mermaid_diagram_is_generated_from_the_graph():
    diagram = mermaid()
    for node in ("ingest", "plan", "rule_extract", "validate", "llm_extract", "llm_merge", "reconcile",
                 "summarize", "finalize"):
        assert node in diagram
