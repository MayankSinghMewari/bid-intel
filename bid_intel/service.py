"""Service layer shared by the CLI and the API."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .agents import Orchestrator
from .ingest import Chunk, build_chunks, load_items
from .retrieval import SearchEngine
from .schema import FIELDS

ROOT = Path(__file__).resolve().parent.parent
BIDS_DIR = ROOT / "data" / "bids"
DEFAULT_CHUNKING = "qa"


def bid_dir(bid_id: str) -> Path:
    d = BIDS_DIR / bid_id
    if not (d / "items.json").exists():
        raise FileNotFoundError(f"bid {bid_id} is not ingested; run: python -m bid_intel ingest")
    return d


def make_engine(bid_id: str, chunking: str = DEFAULT_CHUNKING, k1: float = 1.2, b: float = 0.75) -> SearchEngine:
    _, items = load_items(bid_dir(bid_id))
    return SearchEngine(build_chunks(items, chunking), k1=k1, b=b)


def ask(bid_id: str, question: str, k: int = 3, chunking: str = DEFAULT_CHUNKING) -> dict:
    engine = make_engine(bid_id, chunking)
    hits = engine.search(question, k)
    if not hits:
        return {"question": question, "answer": None, "confidence": 0.0, "citations": [],
                "method": "extractive BM25"}
    top = hits[0].chunk
    passage = top.text.split(" Answer: ", 1)[1] if " Answer: " in top.text else top.text
    total = sum(h.score for h in hits)
    return {
        "question": question,
        "answer": passage,
        "confidence": round(hits[0].score / total, 2),  # top hit's share of the top-k score mass
        "method": "extractive BM25 (no LLM generation in this version)",
        "citations": [
            {"rank": h.rank, "score": round(h.score, 3), "chunk_id": h.chunk.chunk_id,
             "qnums": h.chunk.qnums, "page": h.chunk.page,
             "text": "Addendum No. 1, " + ", ".join(f"Q{q}" for q in h.chunk.qnums) + f", p. {h.chunk.page}",
             "passage": h.chunk.text}
            for h in hits
        ],
    }


def extract(bid_id: str, k1: float = 1.2, b: float = 0.75) -> tuple[dict, list[dict]]:
    """Run the multi-agent extraction. Writes extraction.json and returns (result, trace)."""
    d = bid_dir(bid_id)
    meta, items = load_items(d)
    engine = SearchEngine(build_chunks(items, DEFAULT_CHUNKING), k1=k1, b=b)
    orch = Orchestrator(engine, {i.qnum for i in items}, FIELDS)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = orch.run(bid_id, meta["source"], now)
    (d / "extraction.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, orch.trace()
