"""Multi-agent extraction. Agents share no state; they communicate only by posting messages
to a shared Bus. The orchestrator sequences them: Retriever -> Extractor -> Verifier, per field."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .ingest import Chunk
from .retrieval import Hit, SearchEngine
from .schema import FieldSpec


@dataclass
class Message:
    step: int
    sender: str
    receiver: str
    action: str
    summary: str
    data: dict
    ts: float


class Bus:
    def __init__(self):
        self.log: list[Message] = []

    def send(self, sender: str, receiver: str, action: str, summary: str, data: dict | None = None) -> Message:
        msg = Message(len(self.log) + 1, sender, receiver, action, summary, data or {}, round(time.time(), 3))
        self.log.append(msg)
        return msg


class RetrieverAgent:
    name = "retriever"

    def __init__(self, engine: SearchEngine, bus: Bus):
        self.engine, self.bus = engine, bus

    def retrieve(self, query: str, k: int) -> list[Hit]:
        hits = self.engine.search(query, k)
        self.bus.send(self.name, "extractor", "retrieve", f"{len(hits)} hits",
                      {"query": query,
                       "hits": [{"chunk_id": h.chunk.chunk_id, "score": round(h.score, 3), "rank": h.rank}
                                for h in hits]})
        return hits


def match_field(spec: FieldSpec, text: str):
    """Apply a field rule to chunk text. Returns (value, source, span) or None.
    source is 'answer' when the match is in the District's answer, 'question' when it is only in
    the vendor's question (a weaker signal)."""
    m = spec.regex.search(text)
    if not m:
        return None
    value = spec.static if spec.static else m.group("value")
    marker = text.find("Answer:")
    source = "question" if (marker != -1 and m.start() < marker) else "answer"
    span = text[max(0, m.start() - 80): m.end() + 80]
    return value, source, span


@dataclass
class Candidate:
    field_key: str
    chunk_id: str
    value: str
    source: str
    span: str
    norm: float
    page: int
    qnums: list[int]


class ExtractorAgent:
    name = "extractor"

    def __init__(self, retriever: RetrieverAgent, bus: Bus, top_k: int = 5):
        self.retriever, self.bus, self.top_k = retriever, bus, top_k

    def extract(self, spec: FieldSpec) -> Candidate | None:
        for hit in self.retriever.retrieve(spec.query, self.top_k):
            found = match_field(spec, hit.chunk.text)
            if not found:
                continue
            value, source, span = found
            cand = Candidate(spec.key, hit.chunk.chunk_id, value, source, span,
                             round(hit.norm, 3), hit.chunk.page, hit.chunk.qnums)
            self.bus.send(self.name, "verifier", "candidate",
                          f"{spec.key} = {value!r} ({source}) from {hit.chunk.chunk_id}", asdict(cand))
            return cand
        self.bus.send(self.name, "orchestrator", "not_found", f"no rule match for {spec.key}",
                      {"field": spec.key})
        return None


class VerifierAgent:
    """Re-checks each candidate against the stored chunk. Three checks:
    1. The cited span appears verbatim in the cited chunk.
    2. The cited question numbers exist in the document.
    3. The field rule, re-run on the cited chunk, gives the same value."""
    name = "verifier"

    def __init__(self, chunks: list[Chunk], valid_qnums: set[int], bus: Bus):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.valid_qnums = valid_qnums
        self.bus = bus

    def verify(self, spec: FieldSpec, cand: Candidate) -> dict[str, bool]:
        chunk = self.chunks.get(cand.chunk_id)
        checks = {
            "span_in_chunk": bool(chunk) and cand.span in chunk.text,
            "qnums_exist": bool(cand.qnums) and all(q in self.valid_qnums for q in cand.qnums),
            "rule_rederives_value": False,
        }
        if chunk:
            again = match_field(spec, chunk.text)
            checks["rule_rederives_value"] = bool(again) and again[0] == cand.value
        self.bus.send(self.name, "orchestrator", "verify",
                      f"{spec.key}: {'pass' if all(checks.values()) else 'fail'}",
                      {"chunk_id": cand.chunk_id, "checks": checks})
        return checks


def score_confidence(cand: Candidate, verified: bool) -> float:
    """0.4 x retrieval strength + 0.4 x source location + 0.2 x verification.
    Source location: 1.0 if the value comes from the District's answer, 0.5 if only from the question."""
    source_weight = 1.0 if cand.source == "answer" else 0.5
    return round(0.4 * cand.norm + 0.4 * source_weight + 0.2 * (1.0 if verified else 0.0), 2)


class Orchestrator:
    def __init__(self, engine: SearchEngine, items_qnums: set[int], specs: list[FieldSpec], top_k: int = 5):
        self.engine, self.specs = engine, specs
        self.bus = Bus()
        self.retriever = RetrieverAgent(engine, self.bus)
        self.extractor = ExtractorAgent(self.retriever, self.bus, top_k)
        self.verifier = VerifierAgent(engine.chunks, items_qnums, self.bus)

    def run(self, bid_id: str, source_document: str, generated_at: str) -> dict:
        fields: dict[str, dict] = {}
        for spec in self.specs:
            cand = self.extractor.extract(spec)
            if cand is None:
                fields[spec.key] = {"label": spec.label, "value": None, "status": "not_found",
                                    "confidence": 0.0, "verified": False, "citation": None}
                continue
            checks = self.verifier.verify(spec, cand)
            verified = all(checks.values())
            fields[spec.key] = {
                "label": spec.label,
                "value": cand.value,
                "status": "found",
                "confidence": score_confidence(cand, verified),
                "verified": verified,
                "citation": {
                    "text": "Addendum No. 1, " + ", ".join(f"Q{q}" for q in cand.qnums) + f", p. {cand.page}",
                    "chunk_id": cand.chunk_id,
                    "qnums": cand.qnums,
                    "page": cand.page,
                    "evidence_source": cand.source,
                    "evidence": cand.span,
                },
            }
        found = [f for f in fields.values() if f["status"] == "found"]
        self.bus.send("orchestrator", "writer", "complete", f"{len(found)}/{len(fields)} fields found",
                      {"bid_id": bid_id})
        return {
            "bid_id": bid_id,
            "schema_version": "1.0",
            "generated_at": generated_at,
            "source_document": source_document,
            "engine": self.engine.describe(),
            "fields": fields,
            "summary": {
                "fields_total": len(fields),
                "fields_found": len(found),
                "mean_confidence": round(sum(f["confidence"] for f in found) / len(found), 2) if found else 0.0,
            },
        }

    def trace(self) -> list[dict]:
        return [asdict(m) for m in self.bus.log]
