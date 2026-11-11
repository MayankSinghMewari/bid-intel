"""LangGraph orchestration of the extraction agents.

    START -> ingest -> plan --(Send per group, parallel)--> rule_extract --> validate
    validate --(rejected fields, attempts left: Send per group)--> rule_extract      [feedback loop]
    validate --(LLM available: Send per group, parallel)--> llm_extract --> llm_merge --> reconcile
    validate --(no LLM)--> reconcile
    reconcile -> summarize -> finalize -> END

The graph state is the run's working memory (plan, draft fields, attempt counts, retry list, LLM results).
Parallel branches return partial updates that reducers merge. The agents themselves (RetrievalAgent,
ExtractionAgent, ValidatorAgent, ReconcileAgent, LLM specialists) are unchanged; the Orchestrator object
supplies them, the search engine and the trace logger to the nodes.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import observability as obs
from .fields import FIELDS

if TYPE_CHECKING:
    from .agents import Orchestrator

from .agents import MAX_RETRIES  # noqa: E402  (single source of truth)
SPECS = {s.name: s for s in FIELDS}


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class GraphState(TypedDict, total=False):
    bid_id: str
    plan: list[dict]                               # [{"group": ..., "fields": [...]}]
    drafts: Annotated[dict[str, Any], merge_dicts]   # field name -> FieldResult
    attempts: Annotated[dict[str, int], merge_dicts]
    checked: Annotated[dict[str, int], merge_dicts]   # field -> attempt number already validated
    retry: list[str]                               # fields the validator sent back this round
    llm_results: Annotated[dict[str, dict], merge_dicts]  # group -> LLM specialist results
    route: str                                     # last routing decision, for the trace


class FieldTask(TypedDict):
    group: str
    fields: list[str]
    attempt: int


def build_graph(o: "Orchestrator"):
    from . import llm_extract as llm_specialists  # not "llm_extract": that name is the graph node below
    from .agents import (NOT_FOUND, ExtractionAgent, FieldResult, ReconcileAgent, RetrievalAgent,
                             ValidatorAgent)

    retriever = RetrievalAgent(o.engine, o.tracer) if o.engine else None
    extractor, validator = ExtractionAgent(o.tracer), ValidatorAgent(o.tracer)

    def ingest(state: GraphState) -> dict:
        with obs.span("ingest-documents", "tool", input={"folder": str(o.folder)}) as sp:
            o.chunks = o._ingest()
            o.facts = [(c, c.facts) for c in o.chunks if c.facts]
            obs.set_output(sp, {"chunks": len(o.chunks), "files": [
                {k: f.get(k) for k in ("file", "doc_type", "pages", "chunks", "status")} for f in o.ingestion_report]})
        return {}

    def plan(state: GraphState) -> dict:
        t0 = time.perf_counter()
        groups: dict[str, list[str]] = {}
        for spec in FIELDS:
            if spec.group != "summary":
                groups.setdefault(spec.group, []).append(spec.name)
        p = [{"group": g, "fields": f} for g, f in groups.items()]
        o.tracer.log("orchestrator", "plan", {"goal": "extract bid"}, {"plan": p}, t0)
        return {"plan": p}

    def fan_out_rules(state: GraphState) -> list[Send]:
        return [Send("rule_extract", FieldTask(group=p["group"], fields=p["fields"], attempt=1))
                for p in state["plan"]]

    def rule_extract(task: FieldTask) -> dict:
        """One extraction specialist (field group): retrieve, then apply the field rules."""
        with obs.span(f"extract-{task['group']}-fields", "agent",
                      input={"fields": task["fields"]}, metadata={"attempt": task["attempt"]}) as sp:
            out = _rule_extract(task)
            obs.set_output(sp, {n: r.value for n, r in out["drafts"].items()})
        return out

    def _rule_extract(task: FieldTask) -> dict:
        nonlocal retriever
        retriever = retriever or RetrievalAgent(o.engine, o.tracer)
        drafts = {}
        for name in task["fields"]:
            spec = SPECS[name]
            try:
                hits = retriever.run(o.bid_id, spec, task["attempt"])
                drafts[name] = extractor.run(spec, hits, o.facts)
            except Exception as exc:  # one failing field must not crash the run
                drafts[name] = FieldResult(value=None, notes=f"error: {type(exc).__name__}: {str(exc)[:150]}")
            drafts[name].attempts = task["attempt"]
        return {"drafts": drafts, "attempts": {n: task["attempt"] for n in task["fields"]}}

    def validate(state: GraphState) -> dict:
        """Validate the drafts produced this round; rejected fields with attempts left are retried."""
        with obs.span("validate-fields", "span") as sp:
            out = _validate(state)
            obs.set_output(sp, {"rejected_for_retry": out["retry"], "checked": list(out["checked"])})
        return out

    def _validate(state: GraphState) -> dict:
        retry, updates = [], {}
        checked = state.get("checked") or {}
        this_round = [n for n in state["drafts"] if n in SPECS and checked.get(n) != state["attempts"][n]]
        for name in this_round:
            r = validator.run(SPECS[name], state["drafts"][name])
            if r.validated:
                continue
            if state["attempts"][name] <= MAX_RETRIES:
                retry.append(name)
            elif r.value is None:  # retries exhausted: report as absent, with confidence that it is absent
                r.confidence, r.notes = 0.8, NOT_FOUND
            updates[name] = r
        return {"retry": retry, "drafts": updates, "checked": {n: state["attempts"][n] for n in this_round}}

    def after_validate(state: GraphState):
        retry = state.get("retry") or []
        if retry:
            by_group: dict[str, list[str]] = {}
            for name in retry:
                by_group.setdefault(SPECS[name].group, []).append(name)
            attempt = max(state["attempts"][n] for n in retry) + 1
            o.tracer.log("orchestrator", "retry", {"fields": retry, "next_attempt": attempt},
                         {"reason": "validator rejected; widening retrieval"}, time.perf_counter())
            return [Send("rule_extract", FieldTask(group=g, fields=f, attempt=attempt)) for g, f in by_group.items()]
        o.state.fields = dict(state["drafts"])  # rule results become the record the LLM refines
        if o.llm_on():
            o.tracer.log("orchestrator", "route", {}, {"next": "llm_extract"}, time.perf_counter())
            return [Send("llm_extract", FieldTask(group=p["group"], fields=p["fields"], attempt=1))
                    for p in state["plan"]]
        o.tracer.log("orchestrator", "route", {}, {"next": "reconcile", "reason": "LLM not available"},
                     time.perf_counter())
        return "reconcile"

    def llm_extract(task: FieldTask) -> dict:
        """One LLM specialist (field group): structured output over retrieved evidence, own feedback loop."""
        with obs.span(f"llm-extract-{task['group']}-fields", "agent", input={"fields": task["fields"]}) as sp:
            out = _llm_extract(task)
            res = out["llm_results"][task["group"]]
            obs.set_output(sp, {n: {"value": v[0].value if v[0] else None, "accepted": v[2], "reason": v[3]}
                                for n, v in res.items()})
        return out

    def _llm_extract(task: FieldTask) -> dict:
        specs = [SPECS[n] for n in task["fields"]]
        seeds = [src.chunk_id for s in specs for src in o.state.fields[s.name].sources]
        try:
            res = llm_specialists.extract_group(o.engine, o.bid_id, task["group"], specs, o.tracer.log, seeds)
        except Exception as exc:  # a failed group keeps its rule values
            o.tracer.log("orchestrator", "llm_group_error", {"group": task["group"]}, {"error": str(exc)[:200]},
                         time.perf_counter())
            res = {}
        return {"llm_results": {task["group"]: res}}

    def llm_merge(state: GraphState) -> dict:
        t0 = time.perf_counter()
        with obs.span("merge-llm-results", "span") as sp:
            o.merge_llm(state.get("llm_results") or {})
            obs.set_output(sp, {n: {"value": r.value, "notes": r.notes} for n, r in o.state.fields.items()})
        accepted = sum(1 for r in o.state.fields.values() if r.notes.startswith("LLM extraction"))
        o.tracer.log("orchestrator", "llm_merge", {"groups": list(state.get("llm_results") or {})},
                     {"llm_fields_accepted": accepted}, t0)
        return {"drafts": dict(o.state.fields)}

    def reconcile(state: GraphState) -> dict:
        o.state.fields = dict(state["drafts"])
        with obs.span("reconcile-addenda", "chain", input={"due_date_before": o.state.fields.get("Due Date") and
                                                           o.state.fields["Due Date"].value}) as sp:
            ReconcileAgent(o.tracer).run(o.chunks, o.state)
            obs.set_output(sp, {"addendum_changes": o.state.addendum_changes,
                                "due_date_after": o.state.fields["Due Date"].value})
        return {"drafts": dict(o.state.fields)}

    def summarize(state: GraphState) -> dict:
        o.state.fields = dict(state["drafts"])
        with obs.span("summarize-bid", "chain") as sp:
            summary = o._summary()
            obs.set_output(sp, {"summary": summary.value, "notes": summary.notes})
        return {"drafts": {"Bid Summary": summary}}

    def finalize(state: GraphState) -> dict:
        o.state.fields = {s.name: state["drafts"][s.name] for s in FIELDS}
        return {}

    g = StateGraph(GraphState)
    for name, fn in [("ingest", ingest), ("plan", plan), ("rule_extract", rule_extract), ("validate", validate),
                     ("llm_extract", llm_extract), ("llm_merge", llm_merge), ("reconcile", reconcile),
                     ("summarize", summarize), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "plan")
    g.add_conditional_edges("plan", fan_out_rules, ["rule_extract"])
    g.add_edge("rule_extract", "validate")
    g.add_conditional_edges("validate", after_validate, ["rule_extract", "llm_extract", "reconcile"])
    g.add_edge("llm_extract", "llm_merge")
    g.add_edge("llm_merge", "reconcile")
    g.add_edge("reconcile", "summarize")
    g.add_edge("summarize", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def mermaid() -> str:
    """Mermaid diagram of the compiled graph (generated from the code, so it cannot drift)."""
    class _Stub:  # build_graph only needs attributes at run time, not to draw the structure
        engine = tracer = None
    return build_graph(_Stub()).get_graph().draw_mermaid()  # type: ignore[arg-type]
