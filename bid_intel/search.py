"""Search engine over ingested bid folders: hybrid BM25 + dense retrieval with reciprocal rank fusion,
metadata filters, query expansion, and an optional cross-encoder re-ranker. Every hit carries its
file, page and chunk text for citation."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from . import embeddings
from .bm25 import BM25, tokenize
from .docs import DocChunk

RRF_K = 60
RERANK_MODEL = os.environ.get("BID_INTEL_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# Query understanding: RFP vocabulary varies by agency, so each concept expands to its synonyms
EXPANSIONS = {
    "deadline": ["due date", "submission deadline", "closing date", "proposal due"],
    "due": ["deadline", "closing date", "submission"],
    "bond": ["bid bond", "bid security", "surety", "guarantee"],
    "payment": ["net 30", "invoice", "invoicing", "payment terms"],
    "term": ["contract term", "initial term", "renewal", "duration"],
    "meeting": ["pre-bid", "pre-proposal", "prebid conference"],
    "warranty": ["extended warranty", "manufacturer warranty", "years"],
    "affidavit": ["affidavit", "certification", "form", "attachment"],
    "contact": ["procurement officer", "point of contact", "email", "phone"],
    "delivery": ["ship", "shipping", "lead time", "days of award"],
    "installation": ["imaging", "deployment", "setup", "asset tagging"],
    "cooperative": ["interlocal", "master contract", "piggyback", "purchasing cooperative"],
    "manufacturer": ["authorized reseller", "OEM", "letter of authorization"],
    "specs": ["processor", "memory", "RAM", "storage", "display"],
}


def expand_query(q: str) -> str:
    extra = []
    low = q.lower()
    for key, syns in EXPANSIONS.items():
        if re.search(rf"\b{key}", low):
            extra += [s for s in syns if s.lower() not in low]
    return q + (" " + " ".join(extra) if extra else "")


@dataclass
class SearchHit:
    chunk: DocChunk
    score: float
    rank: int

    def citation(self) -> dict:
        return {"file": self.chunk.file_name, "page": self.chunk.page, "doc_type": self.chunk.doc_type,
                "chunk_id": self.chunk.chunk_id, "text": self.chunk.text}


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=RERANK_MODEL)


class BidSearch:
    """Index over the chunks of one or more bids. add() indexes a new bid without touching others."""

    def __init__(self, chunks: list[DocChunk] | None = None, dense: bool | None = None):
        self.chunks: list[DocChunk] = []
        self.use_dense = embeddings.available() if dense is None else dense
        self._bm25 = None
        self._dense = None
        if chunks:
            self.add(chunks)

    def add(self, chunks: list[DocChunk]) -> None:
        self.chunks += chunks
        # BM25 statistics are global, so the (cheap, pure-Python) lexical index is rebuilt;
        # the expensive dense index only embeds the new chunks
        self._bm25 = BM25([tokenize(c.text) for c in self.chunks])
        if self.use_dense:
            if self._dense is None:
                self._dense = embeddings.DenseIndex([c.text for c in chunks])
            else:
                self._dense.add([c.text for c in chunks])

    def search(self, query: str, top_k: int = 5, bid_id: str | None = None, doc_types: list[str] | None = None,
               addendum_number: int | None = None, mode: str = "hybrid", expand: bool = True,
               rerank: bool = False) -> list[SearchHit]:
        allowed = [i for i, c in enumerate(self.chunks)
                   if (bid_id is None or c.bid_id == bid_id)
                   and (not doc_types or c.doc_type in doc_types)
                   and (addendum_number is None or c.addendum_number == addendum_number)]
        if not allowed:
            return []
        q = expand_query(query) if expand else query
        fused = {i: 0.0 for i in allowed}
        if mode in ("bm25", "hybrid"):
            bm = self._bm25.scores(tokenize(q))
            order = sorted((i for i in allowed if bm[i] > 0), key=lambda i: -bm[i])
            for r, i in enumerate(order, 1):
                fused[i] += 1.0 / (RRF_K + r) if mode == "hybrid" else bm[i]
        if mode in ("dense", "hybrid") and self._dense is not None:
            ds = self._dense.scores(query)  # dense uses the raw query; expansion helps lexical search most
            order = sorted(allowed, key=lambda i: -ds[i])
            for r, i in enumerate(order, 1):
                fused[i] += 1.0 / (RRF_K + r) if mode == "hybrid" else max(ds[i], 0.0)
        ranked = [i for i in sorted(fused, key=lambda i: -fused[i]) if fused[i] > 0]
        pool = ranked[: max(top_k * 4, 20)] if rerank else ranked[:top_k]
        if rerank and pool:
            try:
                scores = list(_reranker().rerank(query, [self.chunks[i].text for i in pool]))
                pool = [i for _, i in sorted(zip(scores, pool), key=lambda t: -t[0])]
                fused = dict(zip(pool, sorted(scores, reverse=True)))
            except Exception:  # re-ranker model unavailable: keep the fused order
                pass
        return [SearchHit(self.chunks[i], round(float(fused[i]), 4), r) for r, i in enumerate(pool[:top_k], 1)]
