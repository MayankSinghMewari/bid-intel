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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .docs import DocChunk, ingest_folder
from .fields import DATETIME, FIELDS, FieldSpec, _sentences
from .search import BidSearch, SearchHit

MAX_RETRIES = 2
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
    def __init__(self, state: BidState):
        self.state, self.lock = state, threading.Lock()

    def log(self, agent: str, action: str, inp: dict, out: dict, t0: float) -> None:
        with self.lock:
            self.state.trace.append(TraceStep(step=len(self.state.trace) + 1, agent=agent, action=action,
                                              input=inp, output=out,
                                              latency_ms=round((time.perf_counter() - t0) * 1000, 1)))


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
        t0 = time.perf_counter()
        # attempt 1 uses the field's preferred document types; retries widen the filter and the depth
        doc_types = spec.doc_types if (attempt == 1 or spec.keep_filter) else None
        k = 8 * attempt
        hits = self.engine.search(spec.query, top_k=k, bid_id=bid_id, doc_types=doc_types)
        self.tracer.log("retriever", "search", {"field": spec.name, "query": spec.query, "doc_types": doc_types,
                                                "top_k": k, "attempt": attempt},
                        {"hits": [{"chunk_id": h.chunk.chunk_id, "file": h.chunk.file_name, "page": h.chunk.page,
                                   "score": h.score} for h in hits]}, t0)
        return hits


class ExtractionAgent:
    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def run(self, spec: FieldSpec, hits: list[SearchHit], facts: list[tuple[DocChunk, dict]]) -> FieldResult:
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
                note = f"Updated by Addendum {ch['addendum']}"
                if old and old.value:
                    note += f" (previously stated: {old.value})"
                src = Source(file=ch["file"], page=ch["page"], doc_type="addendum",
                             chunk_id=f"{state.bid_id}:addendum{ch['addendum']}", evidence=ch["change"][:400])
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
        t0 = time.perf_counter()
        ok, reason = self.check(spec, r)
        r.validated = ok
        if not ok and r.value is not None:
            r.confidence = round(r.confidence * 0.6, 2)
            r.notes = (r.notes + f"; validator: {reason}").strip("; ")
        self.tracer.log("validator", "validate", {"field": spec.name, "value": r.value},
                        {"passed": ok, "reason": reason}, t0)
        return r


class Orchestrator:
    def __init__(self, folder: Path, bid_id: str | None = None, engine: BidSearch | None = None):
        self.folder = folder
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

    def _run_field(self, spec: FieldSpec, facts) -> FieldResult:
        retriever, extractor, validator = (RetrievalAgent(self.engine, self.tracer),
                                           ExtractionAgent(self.tracer), ValidatorAgent(self.tracer))
        result = None
        for attempt in range(1, MAX_RETRIES + 2):
            hits = retriever.run(self.bid_id, spec, attempt)
            result = validator.run(spec, extractor.run(spec, hits, facts))
            result.attempts = attempt
            if result.validated:
                break
            # feedback loop: rejected or missing field goes back for wider retrieval
            self.tracer.log("orchestrator", "retry", {"field": spec.name, "attempt": attempt},
                            {"reason": result.notes}, time.perf_counter())
        if not result.validated and result.value is None:
            result.confidence = 0.8 if result.attempts > MAX_RETRIES else 0.5  # confidence it is truly absent
            result.notes = NOT_FOUND
        return result

    def _summary(self) -> FieldResult:
        f = {k: v.value for k, v in self.state.fields.items()}
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
                           notes="generated from the extracted fields (template summary; no LLM key configured)",
                           validated=bool(parts))

    def run(self) -> dict:
        t_start = time.perf_counter()
        chunks = self._ingest()
        facts = [(c, c.facts) for c in chunks if c.facts]
        groups: dict[str, list[FieldSpec]] = {}
        for spec in FIELDS:
            if spec.group != "summary":
                groups.setdefault(spec.group, []).append(spec)
        self.state.plan = [{"group": g, "fields": [s.name for s in specs]} for g, specs in groups.items()]
        self.tracer.log("orchestrator", "plan", {"goal": "extract bid"}, {"plan": self.state.plan}, t_start)

        # field groups run concurrently (one extraction specialist per group)
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = {s.name: pool.submit(self._run_field, s, facts) for specs in groups.values() for s in specs}
        for name, fut in futures.items():
            try:
                self.state.fields[name] = fut.result()
            except Exception as exc:  # one failing field must not crash the pipeline
                self.state.fields[name] = FieldResult(value=None, notes=f"error: {type(exc).__name__}: {exc}")

        ReconcileAgent(self.tracer).run(chunks, self.state)
        self.state.fields["Bid Summary"] = self._summary()

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
