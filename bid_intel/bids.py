"""Service layer for bid folders: index, search, extract, ask. Shared by the CLI and the API."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import llm
from .bid_agents import Orchestrator
from .docs import ingest_folder
from .search import BidSearch

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ROOT / "data" / "bid_folders"
OUT = ROOT / "outputs"

_engine: BidSearch | None = None


def engine() -> BidSearch:
    """One shared index; each bid folder is added once (incremental indexing)."""
    global _engine
    if _engine is None:
        _engine = BidSearch()
    return _engine


def index_folder(folder: Path, bid_id: str | None = None) -> dict:
    bid_id = bid_id or folder.name
    eng = engine()
    if any(c.bid_id == bid_id for c in eng.chunks):
        return {"bid_id": bid_id, "status": "already indexed", "chunks": sum(c.bid_id == bid_id for c in eng.chunks)}
    chunks, report = ingest_folder(folder, bid_id)
    eng.add(chunks)
    return {"bid_id": bid_id, "status": "indexed", "chunks": len(chunks), "files": report}


def ensure_indexed(bid_id: str | None = None) -> None:
    for f in sorted(FOLDERS.iterdir()):
        if f.is_dir() and (bid_id is None or f.name == bid_id):
            index_folder(f)


def search(q: str, bid_id: str | None = None, top_k: int = 5, doc_types: list[str] | None = None,
           mode: str = "hybrid", rerank: bool = False) -> list[dict]:
    ensure_indexed(bid_id)
    hits = engine().search(q, top_k=top_k, bid_id=bid_id, doc_types=doc_types, mode=mode, rerank=rerank)
    return [{"rank": h.rank, "score": h.score, "bid_id": h.chunk.bid_id, **h.citation()} for h in hits]


def extract(folder: Path, bid_id: str | None = None) -> tuple[dict, list[dict]]:
    ensure_indexed()
    orch = Orchestrator(folder, bid_id, engine=engine())
    result = orch.run()
    OUT.mkdir(exist_ok=True)
    (OUT / f"{orch.bid_id}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, orch.trace()


def _bids_in(question: str) -> list[str] | None:
    known = [f.name for f in FOLDERS.iterdir() if f.is_dir()]
    named = [b for b in known if re.search(rf"\b{re.escape(b)}\b", question, re.I)]
    return named or None


def ask(question: str, bid_id: str | None = None, top_k: int = 5) -> dict:
    ensure_indexed()
    bids = [bid_id] if bid_id else (_bids_in(question) or [None])
    hits = []
    for b in bids:  # "compare both bids" style questions retrieve per bid so each is represented
        hits += search(question, bid_id=b, top_k=top_k if len(bids) == 1 else 3)
    generated = llm.answer_from_citations(question, hits)
    if generated and "answer" in generated:
        answer, method = generated["answer"], f"hybrid search + LLM ({generated['model']})"
    else:
        best = hits[0]["text"] if hits else None
        answer = _best_sentences(question, hits) if hits else "Not found in documents"
        method = "hybrid search + extractive answer (no LLM key configured)"
    return {"question": question, "answer": answer, "method": method,
            **({"llm_fallback_reason": generated["error"]} if generated and "error" in generated else {}),
            "citations": [{"n": i + 1, "bid_id": h["bid_id"], "file": h["file"], "page": h["page"],
                           "passage": h["text"][:500]} for i, h in enumerate(hits)]}


def _best_sentences(question: str, hits: list[dict], n: int = 2) -> str:
    from .bm25 import tokenize
    from .fields import _sentences
    q = set(tokenize(question))
    scored = []
    for i, h in enumerate(hits[:3]):
        for s in _sentences(h["text"]):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                scored.append((overlap - 0.1 * i, f"{s[:300]} [{i + 1}]"))
    scored.sort(key=lambda t: -t[0])
    return " ".join(s for _, s in scored[:n]) or hits[0]["text"][:300] + " [1]"
