"""Multi-agent extraction over a whole bid folder.

Agents (own orchestration; each step is logged to the trace with input, output and latency):
  Orchestrator    plans field groups, runs them in parallel, re-routes rejected fields, assembles output
  IngestionAgent  parses the folder (docs.py) and builds the search index (search.py)
  RetrievalAgent  expands the query, applies metadata filters, calls hybrid search
  ExtractionAgent one specialist per field group; applies the field rule to retrieved evidence only
  ReconcileAgent  compares addendums with the base documents and applies the latest change
  ValidatorAgent  checks citation support, format and consistency; rejected fields are retried
Shared state lives in BidState (pydantic models), which is also the structured message format.
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .docs import DocChunk, ingest_folder
from .fields import DATE, DATETIME, FIELDS, FieldSpec, _sentences
from .search import BidSearch, SearchHit

MAX_RETRIES = 2  # validator -> extractor feedback rounds per field (used by graph.py)
NOT_FOUND = "Not found in documents"


class Source(BaseModel):
    file: str
    page: int
    doc_type: str
    chunk_id: str
    evidence: str = Field(description="passage text the value was read from (truncated)")


class FieldResult(BaseModel):
    value: str | None
    sources: list[Source] = []
    confidence: float = 0.0
    notes: str = ""
    attempts: int = 1
    validated: bool = False


class TraceStep(BaseModel):
    step: int
    agent: str
    action: str
    input: dict
    output: dict
    latency_ms: float


class BidState(BaseModel):
    bid_id: str
    plan: list[dict] = []
    evidence: dict[str, list[dict]] = {}
    fields: dict[str, FieldResult] = {}
    addendum_changes: list[dict] = []
    validation: dict = {}
    trace: list[TraceStep] = []


class Tracer:
    """Local structured trace (logs/<bid>_trace.json), mirrored to Langfuse when tracing is enabled."""

    def __init__(self, state: BidState):
        self.state, self.lock = state, threading.Lock()
        self.ctx: dict | None = None  # Langfuse trace context of the current run (set by Orchestrator.run)

    def log(self, agent: str, action: str, inp: dict, out: dict, t0: float) -> None:
        latency = round((time.perf_counter() - t0) * 1000, 1)
        with self.lock:
            step = len(self.state.trace) + 1
            self.state.trace.append(TraceStep(step=step, agent=agent, action=action, input=inp, output=out,
                                              latency_ms=latency))
        # agent work is traced as typed observations; orchestrator decisions and errors become events
        if self.ctx is not None and (agent == "orchestrator" or "error" in action):
            from . import observability
            warn = "error" in action or action == "retry"
            observability.event(f"{agent}.{action}", input=inp, output=out,
                                metadata={"step": step, "latency_ms": latency},
                                level="WARNING" if warn else "DEFAULT")


def _source(h: SearchHit, value: str | None = None, full: bool = False) -> Source:
    """Citation with the evidence window centred on the extracted value, so the cited text shows it."""
    c = h.chunk
    flat = re.sub(r"\s+", " ", c.text)
    pos = flat.lower().find(re.sub(r"\s+", " ", value.split(";")[0]).lower()[:40]) if value else -1
    start = max(0, pos - 150) if pos >= 0 else 0
    return Source(file=c.file_name, page=c.page, doc_type=c.doc_type, chunk_id=c.chunk_id,
                  evidence=flat[:1500] if full else flat[start:start + 450])


class RetrievalAgent:
    def __init__(self, engine: BidSearch, tracer: Tracer):
        self.engine, self.tracer = engine, tracer

    def run(self, bid_id: str, spec: FieldSpec, attempt: int) -> list[SearchHit]:
        from . import observability
        t0 = time.perf_counter()
        # attempt 1 uses the field's preferred document types; retries widen the filter and the depth
        doc_types = spec.doc_types if (attempt == 1 or spec.keep_filter) else None
        k = 8 * attempt
        with observability.span("search-evidence", "retriever",
                                input={"query": spec.query, "bid_id": bid_id, "doc_types": doc_types, "top_k": k},
                                metadata={"field": spec.name, "attempt": attempt}) as obs:
            hits = self.engine.search(spec.query, top_k=k, bid_id=bid_id, doc_types=doc_types)
            observability.set_output(obs, [f"{h.chunk.file_name} p.{h.chunk.page} (score {h.score})" for h in hits])
        self.tracer.log("retriever", "search", {"field": spec.name, "query": spec.query, "doc_types": doc_types,
                                                "top_k": k, "attempt": attempt},
                        {"hits": [{"chunk_id": h.chunk.chunk_id, "file": h.chunk.file_name, "page": h.chunk.page,
                                   "score": h.score} for h in hits]}, t0)
        return hits


class ExtractionAgent:
    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def run(self, spec: FieldSpec, hits: list[SearchHit], facts: list[tuple[DocChunk, dict]]) -> FieldResult:
        from . import observability
        with observability.span("apply-field-rules", "tool", input={"field": spec.name, "passages": len(hits)}) as obs:
            result = self._run(spec, hits, facts)
            observability.set_output(obs, {"value": result.value, "notes": result.notes,
                                           "sources": [f"{s.file} p.{s.page}" for s in result.sources]})
        return result

    def _run(self, spec: FieldSpec, hits: list[SearchHit], facts: list[tuple[DocChunk, dict]]) -> FieldResult:
        t0 = time.perf_counter()
        result = None
        # 1) structured facts from the bid portal page are the most reliable source
        for chunk, f in facts:
            for key in spec.fact_keys:
                if f.get(key):
                    val = re.sub(r"\s*\*\*.*?\*\*\s*", " ", f[key]).strip() if key == "title" else f[key]
                    src = Source(file=chunk.file_name, page=chunk.page, doc_type=chunk.doc_type,
                                 chunk_id=chunk.chunk_id, evidence=f"{key}: {f[key]}"[:400])
                    result = FieldResult(value=val[:500], sources=[src], confidence=0.9,
                                         notes=f"from bid page field '{key}'")
                    break
            if result:
                break
        # 2) otherwise apply the field rule to retrieved passages, best rank first
        if result is None:
            values, sources = [], []
            for h in hits:
                v = spec.rule(h.chunk.text)
                if v:
                    values.append(v)
                    sources.append(_source(h, v, full=spec.derived))
                    if not spec.multi:
                        break
            if values:
                merged = values[0] if not spec.multi else "; ".join(dict.fromkeys(
                    p.strip() for v in values for p in v.split(";") if p.strip()))
                rank = next(h.rank for h in hits if h.chunk.chunk_id == sources[0].chunk_id)
                result = FieldResult(value=merged, sources=sources[:5],
                                     confidence=round(0.55 + 0.35 / rank, 2),
                                     notes=f"rule match in retrieved passage (rank {rank})")
        if result is None:
            result = FieldResult(value=None, confidence=0.0, notes=NOT_FOUND)
        self.tracer.log(f"extractor:{spec.group}", "extract", {"field": spec.name, "evidence": len(hits)},
                        {"value": result.value, "sources": [s.chunk_id for s in result.sources]}, t0)
        return result


class ReconcileAgent:
    """Find what each addendum changes and apply the latest change to the affected fields."""
    CHANGE = re.compile(r"\b(extend|new due date|revis|replac|amend|chang|delet|added|is hereby|clarif)\w*", re.I)

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def run(self, chunks: list[DocChunk], state: BidState) -> None:
        t0 = time.perf_counter()
        adds = sorted({(c.addendum_number or 0, c.file_name) for c in chunks if c.doc_type == "addendum"})
        changes = []
        for num, fname in adds:
            for c in (c for c in chunks if c.file_name == fname):
                for s in _sentences(c.text):
                    if self.CHANGE.search(s) and len(s) < 400:
                        changes.append({"addendum": num, "file": fname, "page": c.page, "change": s})
        # dedupe and keep each addendum's first changes (Q&A addenda can contain dozens of clarifications)
        seen, kept = set(), []
        for ch in changes:
            key = (ch["addendum"], ch["change"][:80])
            if key not in seen and sum(1 for k in kept if k["addendum"] == ch["addendum"]) < 8:
                seen.add(key)
                kept.append(ch)
        state.addendum_changes = kept

        # due date: the latest addendum that states a new date overrides the base documents
        for ch in sorted(kept, key=lambda c: -c["addendum"]):
            m = re.search(rf"(?:new due date|due date[^.]{{0,40}}(?:extended|changed|revised)|extended to)"
                          rf"[^.]{{0,40}}?({DATETIME})", ch["change"], re.I)
            if m:
                old = state.fields.get("Due Date")
                src = Source(file=ch["file"], page=ch["page"], doc_type="addendum",
                             chunk_id=f"{state.bid_id}:addendum{ch['addendum']}", evidence=ch["change"][:400])
                new_date = re.search(rf"{DATE}", m.group(1)).group(0)
                if old and old.value and new_date.lower() in old.value.lower():
                    # the extractor already applied this addendum: confirm instead of overwriting
                    if not any(s.file == ch["file"] for s in old.sources):
                        old.sources.insert(0, src)
                    old.notes += f"; confirmed by reconciler: Addendum {ch['addendum']} sets this date"
                    old.confidence = max(old.confidence, 0.95)
                    break
                note = f"Updated by Addendum {ch['addendum']}"
                if old and old.value:
                    note += f" (previously stated: {old.value})"
                state.fields["Due Date"] = FieldResult(
                    value=m.group(1), sources=[src] + (old.sources if old else []), confidence=0.93, notes=note,
                    validated=True)
                break
        self.tracer.log("reconciler", "reconcile", {"addenda": [f for _, f in adds]},
                        {"changes": len(kept), "due_date": state.fields.get("Due Date").value
                         if state.fields.get("Due Date") else None}, t0)


class ValidatorAgent:
    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def check(self, spec: FieldSpec, r: FieldResult) -> tuple[bool, str]:
        if r.value is None:
            return False, "no value"
        if not r.sources:
            return False, "value has no citation"
        if spec.kind == "date" and not re.search(r"\d{4}|\d{1,2}/\d{1,2}/\d{2}", r.value):
            return False, f"not a date: {r.value!r}"
        # evidence support: the value (or its first token group) must appear in a cited passage
        joined = " ".join(re.sub(r"\s+", " ", s.evidence).lower() for s in r.sources)
        if spec.derived:
            # every number and component value must be present in the cited evidence
            parts = [p.split(":", 1)[-1].strip().lower()[:25] for p in r.value.split(";")]
            nums = re.findall(r"\d+", r.value) if spec.name == "Term of Bid" else []
            missing = [p for p in parts + nums if p and p not in joined
                       and not (spec.name == "Term of Bid" and not p.isdigit())]
            return (False, f"components not in evidence: {missing[:3]}") if missing else (True, "ok")
        probe = re.sub(r"\s+", " ", r.value.split(";")[0]).lower()[:40]
        if spec.group != "summary" and probe and probe not in joined and not r.notes.startswith("Updated"):
            return False, "value not found verbatim in cited evidence"
        return True, "ok"

    def run(self, spec: FieldSpec, r: FieldResult) -> FieldResult:
        from . import observability
        t0 = time.perf_counter()
        with observability.span("validate-field", "evaluator", input={"field": spec.name, "value": r.value,
                                                                     "cited": [s.evidence[:200] for s in r.sources]}) as obs:
            ok, reason = self.check(spec, r)
            observability.set_output(obs, {"passed": ok, "reason": reason}, level=None if ok else "WARNING")
        r.validated = ok
        if not ok and r.value is not None:
            r.confidence = round(r.confidence * 0.6, 2)
            r.notes = (r.notes + f"; validator: {reason}").strip("; ")
        self.tracer.log("validator", "validate", {"field": spec.name, "value": r.value},
                        {"passed": ok, "reason": reason}, t0)
        return r


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", s.lower()))


def _agree(rule_val: str, llm_val: str) -> bool:
    """Rule and LLM values agree when most of the rule value's tokens appear in the LLM value."""
    rt = _tokens(rule_val)
    return bool(rt) and len(rt & _tokens(llm_val)) / len(rt) >= 0.5


def _quote_source(h: SearchHit, quote: str | None) -> Source:
    c = h.chunk
    flat = re.sub(r"\s+", " ", c.text)
    pos = flat.lower().find(re.sub(r"\s+", " ", quote or "").lower()[:60]) if quote else -1
    start = max(0, pos - 120) if pos >= 0 else 0
    return Source(file=c.file_name, page=c.page, doc_type=c.doc_type, chunk_id=c.chunk_id,
                  evidence=flat[start:start + 450])


class Orchestrator:
    def __init__(self, folder: Path, bid_id: str | None = None, engine: BidSearch | None = None,
                 use_llm: bool | None = None):
        self.folder = folder
        self.use_llm = use_llm  # None: use the LLM when a key is configured
        self.trace_url: str | None = None  # Langfuse link for the last run, when tracing is enabled
        self.bid_id = bid_id or folder.name
        self.state = BidState(bid_id=self.bid_id)
        self.tracer = Tracer(self.state)
        self.engine = engine

    def _ingest(self) -> list[DocChunk]:
        t0 = time.perf_counter()
        chunks, report = ingest_folder(self.folder, self.bid_id)
        if self.engine is None:
            self.engine = BidSearch(chunks)
        elif not any(c.bid_id == self.bid_id for c in self.engine.chunks):
            self.engine.add(chunks)  # incremental: only the new bid is indexed
        self.tracer.log("ingestion", "parse+index", {"folder": str(self.folder)},
                        {"files": report, "chunks": len(chunks)}, t0)
        self.ingestion_report = report
        return chunks

    def _summary(self) -> FieldResult:
        f = {k: v.value for k, v in self.state.fields.items()}
        from . import llm
        llm_on = llm.enabled() if self.use_llm is None else (self.use_llm and llm.enabled())
        if llm_on:
            t0 = time.perf_counter()
            facts = "\n".join(f"{k}: {v}" for k, v in f.items() if v)
            try:
                out = llm._complete(
                    "Write a 3 to 6 sentence summary of a public-sector bid using only the extracted facts given. "
                    "Do not add facts that are not listed. Plain prose, no lists.", facts, max_tokens=600)
                srcs = [s for k in ("Title", "Due Date", "Product") for s in
                        (self.state.fields[k].sources if self.state.fields.get(k) else [])][:3]
                self.tracer.log("qa_report", "summary", {"facts": len(facts)},
                                {"model": out["model"], "input_tokens": out["input_tokens"],
                                 "output_tokens": out["output_tokens"]}, t0)
                if out["text"]:
                    return FieldResult(value=out["text"], sources=srcs, confidence=0.75, validated=True,
                                       notes=f"generated by {out['model']} from the validated extracted fields")
            except Exception as exc:  # LLM failure must not stop the pipeline: use the template summary
                self.tracer.log("qa_report", "summary_error", {}, {"error": str(exc)[:200]}, t0)
        parts = []
        if f.get("Title"):
            parts.append(f"{f.get('company_name') or 'The issuing agency'} has issued bid "
                         f"{f.get('Bid Number') or ''} for {f['Title']}.".replace("  ", " "))
        if f.get("Product"):
            parts.append(f"Scope: {f['Product'][:240].rstrip('.')}.")
        if f.get("Due Date"):
            parts.append(f"Responses are due {f['Due Date']}" +
                         (f" via {f['Bid Submission Type'][:120].rstrip('.')}." if f.get("Bid Submission Type") else "."))
        if f.get("Term of Bid"):
            parts.append(f"Contract term: {f['Term of Bid'][:160].rstrip('.')}.")
        if f.get("Any Additional Documentation Required"):
            parts.append(f"Required documents include {f['Any Additional Documentation Required'][:200]}.")
        if self.state.addendum_changes:
            parts.append(f"{len({c['addendum'] for c in self.state.addendum_changes})} addendum(s) modify the "
                         f"original documents.")
        srcs = [s for k in ("Title", "Product", "Due Date") for s in (self.state.fields.get(k).sources
                                                                      if self.state.fields.get(k) else [])][:3]
        return FieldResult(value=" ".join(parts[:6]) or None, sources=srcs, confidence=0.7 if parts else 0.0,
                           notes="template summary from the extracted fields (LLM " +
                                 ("call failed, see trace)" if llm_on else "not configured)"),
                           validated=bool(parts))

    def llm_on(self) -> bool:
        from . import llm
        return llm.enabled() if self.use_llm is None else bool(self.use_llm and llm.enabled())

    def merge_llm(self, results_by_group: dict[str, dict]) -> None:
        """Merge LLM specialist results into self.state.fields: a validated LLM value replaces the rule
        value (higher confidence when both agree); otherwise the rule value stays."""
        from . import llm
        model = llm.GEMINI_MODEL if llm.provider() == "gemini" else llm.MODEL
        for g, results in results_by_group.items():
            for name, (fv, passages, ok, reason, attempts) in results.items():
                rule = self.state.fields.get(name)
                rule_val = rule.value if rule is not None and rule.validated else None
                if fv is not None and ok and fv.value is not None:
                    agree = rule_val is not None and _agree(rule_val, fv.value)
                    notes = [f"LLM extraction ({model}); quote verified in cited passage"]
                    if fv.note:
                        notes.append(fv.note)
                    if agree:
                        notes.append("agrees with rule-based value")
                    elif rule_val:
                        notes.append(f"rule-based value was: {rule_val[:120]}")
                    if attempts > 1:
                        notes.append("accepted after validator feedback")
                    srcs = [_quote_source(passages[n - 1], fv.quote) for n in dict.fromkeys(fv.citations)][:5]
                    self.state.fields[name] = FieldResult(value=fv.value, sources=srcs,
                                                          confidence=0.95 if agree else 0.85, notes="; ".join(notes),
                                                          validated=True, attempts=attempts)
                elif rule_val is not None:
                    rule.notes += "; LLM: " + ("not stated in its evidence" if fv is not None and fv.value is None
                                               else f"rejected ({reason})")
                elif fv is not None and fv.value is None and ok:
                    self.state.fields[name] = FieldResult(value=None, confidence=0.85,
                                                          notes=NOT_FOUND + " (rules and LLM agree)")

    def run(self) -> dict:
        """Run the LangGraph workflow (graph.py) and return the bid record."""
        from . import observability
        from .graph import GraphState, build_graph
        t_start = time.perf_counter()
        with observability.trace("extract-bid", input={"bid_id": self.bid_id, "folder": str(self.folder)},
                                 metadata={"bid_id": self.bid_id, "use_llm": self.llm_on()},
                                 tags=["extraction"]) as root:
            self.tracer.ctx = observability.trace_context()
            final: GraphState = build_graph(self).invoke(
                {"bid_id": self.bid_id}, config={"max_concurrency": 4, "recursion_limit": 60})
            self.state.plan = final.get("plan", [])
            result = self._result(t_start)
            observability.set_output(root, {"validation": result["validation"],
                                            "fields": {k: v["value"] for k, v in result["fields"].items()}})
            self.trace_url = observability.trace_url(root)
        self.tracer.ctx = None
        return result

    def _result(self, t_start: float) -> dict:
        ordered = {s.name: self.state.fields[s.name] for s in FIELDS}
        passed = sum(1 for r in ordered.values() if r.value is not None and r.validated)
        failed = sum(1 for r in ordered.values() if r.value is not None and not r.validated)
        nf = sum(1 for r in ordered.values() if r.value is None)
        self.state.validation = {"passed": passed, "failed": failed, "not_found": nf}
        self.tracer.log("orchestrator", "finish", {}, self.state.validation, t_start)
        return {
            "bid_id": self.bid_id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fields": {k: v.model_dump(exclude={"validated", "attempts"}) | {"validated": v.validated}
                       for k, v in ordered.items()},
            "addendum_changes": self.state.addendum_changes,
            "validation": self.state.validation,
            "ingestion": self.ingestion_report,
        }

    def trace(self) -> list[dict]:
        return [t.model_dump() for t in self.state.trace]
