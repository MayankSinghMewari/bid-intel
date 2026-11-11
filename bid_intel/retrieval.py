"""RAG search engine: BM25 retrieval over chunks plus Recall@k / MRR evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .bm25 import BM25, tokenize
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    norm: float  # score / top score for this query, in (0, 1]
    rank: int


class SearchEngine:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.index = BM25([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def describe(self) -> dict:
        return {
            "retrieval": "BM25 (pure Python)",
            "k1": self.k1,
            "b": self.b,
            "chunking": self.chunks[0].strategy if self.chunks else None,
            "chunks": len(self.chunks),
        }

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.index.scores(tokenize(query))
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
