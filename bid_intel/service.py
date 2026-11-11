"""Service layer shared by the CLI and the API."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import embeddings, llm
from .agents import Orchestrator
from .ingest import Chunk, build_chunks, load_items
from .retrieval import SearchEngine
from .schema import FIELDS

ROOT = Path(__file__).resolve().parent.parent
BIDS_DIR = ROOT / "data" / "bids"
DEFAULT_CHUNKING = "qa"
# hybrid retrieval when the embedding model is installed; BM25 otherwise
DEFAULT_MODE = "hybrid" if embeddings.available() else "bm25"


def bid_dir(bid_id: str) -> Path:
    d = BIDS_DIR / bid_id
    if not (d / "items.json").exists():
        raise FileNotFoundError(f"bid {bid_id} is not ingested; run: python -m bid_intel ingest")
    return d


def make_engine(bid_id: str, chunking: str = DEFAULT_CHUNKING, k1: float = 1.2, b: float = 0.75,
                mode: str = "bm25") -> SearchEngine:
    _, items = load_items(bid_dir(bid_id))
    return SearchEngine(build_chunks(items, chunking), k1=k1, b=b, mode=mode)


def ask(bid_id: str, question: str, k: int = 3, chunking: str = DEFAULT_CHUNKING,
        mode: str | None = None, use_llm: bool = True) -> dict:
    mode = mode or DEFAULT_MODE
    engine = make_engine(bid_id, chunking, mode=mode)
    t0 = time.perf_counter()
    hits = engine.search(question, k)
    retrieval_s = round(time.perf_counter() - t0, 3)
    if not hits:
        return {"question": question, "answer": None, "confidence": 0.0, "citations": [],
                "method": f"{mode} retrieval, no match"}
    top = hits[0].chunk
    passage = top.text.split(" Answer: ", 1)[1] if " Answer: " in top.text else top.text
    total = sum(h.score for h in hits)
    generated = llm.generate_answer(question, hits) if use_llm else None
    if generated and "answer" in generated:
        answer, method = generated["answer"], f"{mode} retrieval + LLM answer ({generated['model']})"
    else:
        answer, method = passage, f"{mode} retrieval + extractive answer (top passage)"
    return {
        "question": question,
        "answer": answer,
        "confidence": round(hits[0].score / total, 2),  # top hit's share of the top-k score mass
        "method": method,
        "retrieval": engine.describe(),
        "timing": {"retrieval_s": retrieval_s, **({k_: v for k_, v in generated.items()
                                                    if k_ in ("latency_s", "input_tokens", "output_tokens")}
                                                 if generated and "answer" in generated else {})},
        **({"llm_fallback_reason": generated["error"]} if generated and "error" in generated else {}),
        "extractive_passage": passage,
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
