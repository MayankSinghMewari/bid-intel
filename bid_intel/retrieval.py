"""RAG search engine: BM25, dense, or hybrid retrieval over chunks, plus Recall@k / MRR evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .bm25 import BM25, tokenize
from .ingest import Chunk

MODES = ("bm25", "dense", "hybrid")
RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass
class Hit:
    chunk: Chunk
    score: float
    norm: float  # score / top score for this query, in (0, 1]
    rank: int


class SearchEngine:
    """mode="bm25": lexical BM25 (default for extraction; pure Python, deterministic).
    mode="dense": cosine similarity of local embeddings (see embeddings.py).
    mode="hybrid": reciprocal rank fusion of the BM25 and dense rankings."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75, mode: str = "bm25"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.chunks = chunks
        self.k1, self.b, self.mode = k1, b, mode
        self.index = BM25([tokenize(c.text) for c in chunks], k1=k1, b=b)
        self.dense = None
        if mode in ("dense", "hybrid"):
            from .embeddings import DenseIndex
            self.dense = DenseIndex([c.text for c in chunks])

    def describe(self) -> dict:
        from .embeddings import MODEL_NAME
        return {
            "retrieval": {"bm25": "BM25 (pure Python)", "dense": f"dense ({MODEL_NAME})",
                          "hybrid": f"hybrid RRF of BM25 + dense ({MODEL_NAME})"}[self.mode],
            "k1": self.k1,
            "b": self.b,
            "chunking": self.chunks[0].strategy if self.chunks else None,
            "chunks": len(self.chunks),
        }

    def _bm25_scores(self, query: str) -> list[float]:
        return self.index.scores(tokenize(query))

    def _scores(self, query: str) -> list[float]:
        if self.mode == "bm25":
            return self._bm25_scores(query)
        dense = self.dense.scores(query)
        if self.mode == "dense":
            return [max(s, 0.0) for s in dense]
        # hybrid: RRF over both rankings; chunks with no lexical match still get their dense share
        bm = self._bm25_scores(query)
        fused = [0.0] * len(self.chunks)
        for scores, require_positive in ((bm, True), (dense, False)):
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            for r, i in enumerate(order, start=1):
                if require_positive and scores[i] <= 0:
                    break
                fused[i] += 1.0 / (RRF_K + r)
        return fused

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._scores(query)
        top = max(scores) if scores else 0.0
        if top <= 0:
            return []
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            Hit(self.chunks[i], scores[i], scores[i] / top, rank)
            for rank, i in enumerate((i for i in order if scores[i] > 0), start=1)
        ]


def evaluate(engine: SearchEngine, questions: list[dict], ks=(1, 3, 5)) -> dict:
    """Recall@k = mean fraction of gold question numbers covered by the top k chunks.
    MRR = mean reciprocal rank of the first chunk containing any gold question."""
    recall = {k: [] for k in ks}
    reciprocal = []
    for q in questions:
        gold = set(q["gold"])
        hits = engine.search(q["question"], k=max(ks))
        for k in ks:
            covered: set[int] = set()
            for h in hits[:k]:
                covered |= set(h.chunk.qnums) & gold
            recall[k].append(len(covered) / len(gold))
        first = next((h.rank for h in hits if set(h.chunk.qnums) & gold), None)
        reciprocal.append(1.0 / first if first else 0.0)
    result = {f"Recall@{k}": round(mean(recall[k]), 4) for k in ks}
    result["MRR"] = round(mean(reciprocal), 4)
    result["questions"] = len(questions)
    return result
