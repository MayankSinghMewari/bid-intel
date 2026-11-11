"""Sentence embeddings with fastembed (ONNX, CPU, no API key).

Model: BAAI/bge-small-en-v1.5 (384 dimensions, about 70 MB, downloaded once on first use).
Vectors are L2-normalised, so cosine similarity is a dot product. Vectors for the bid folders are
stored in the Chroma vector store (vectorstore.py); DenseIndex is the small in-memory index used by
the earlier single-addendum prototype (retrieval.py).
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

MODEL_NAME = os.environ.get("BID_INTEL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DIM = 384
# bge models are trained with this instruction prefix on the query side only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_lock = threading.Lock()  # agents search from several threads; one ONNX session at a time


def available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=2)
def _model(name: str = MODEL_NAME):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=name)


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    with _lock:
        # small batches: one 256-text batch pads every text to the longest and is slow and memory-hungry
        return _normalise(np.array(list(_model().embed(list(texts), batch_size=16)), dtype=np.float32))


@lru_cache(maxsize=512)
def _embed_query_cached(query: str) -> tuple[float, ...]:
    with _lock:
        v = _normalise(np.array(list(_model().embed([QUERY_PREFIX + query])), dtype=np.float32))[0]
    return tuple(v.tolist())


def embed_query(query: str) -> np.ndarray:
    return np.array(_embed_query_cached(query), dtype=np.float32)


class DenseIndex:
    """In-memory index for the prototype's 39-chunk addendum; the bid folders use Chroma."""

    def __init__(self, texts: list[str]):
        self.matrix = embed_passages(texts)

    def add(self, texts: list[str]) -> None:
        if texts:
            self.matrix = np.vstack([self.matrix, embed_passages(texts)])

    def scores(self, query: str) -> list[float]:
        return (self.matrix @ embed_query(query)).tolist()
