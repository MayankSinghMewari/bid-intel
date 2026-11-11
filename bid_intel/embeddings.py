"""Dense retrieval: local sentence embeddings with fastembed (ONNX, no GPU, no API key).

Model: BAAI/bge-small-en-v1.5 (384 dimensions, about 70 MB, downloaded once on first use).
Vectors are L2-normalised, so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_NAME = os.environ.get("BID_INTEL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models are trained with this instruction prefix on the query side only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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


CACHE_DIR = Path(os.environ.get("BID_INTEL_CACHE", Path(__file__).resolve().parent.parent / ".cache"))


def _cache_path() -> Path:
    return CACHE_DIR / f"emb_{hashlib.sha1(MODEL_NAME.encode()).hexdigest()[:8]}.npz"


@lru_cache(maxsize=1)
def _disk_cache() -> dict[str, np.ndarray]:
    p = _cache_path()
    if p.exists():
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    return {}


def _embed_passages(texts: tuple[str, ...]) -> np.ndarray:
    """Embed passages, reusing vectors cached on disk (keyed by a hash of the text), so a chunk is
    embedded once across runs and adding a bid only embeds that bid's chunks."""
    cache = _disk_cache()
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    if missing:
        vecs = _normalise(np.array(list(_model().embed([t for _, t in missing])), dtype=np.float32))
        for (k, _), v in zip(missing, vecs):
            cache[k] = v
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(_cache_path(), **cache)
    return np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 384), dtype=np.float32)


class DenseIndex:
    def __init__(self, texts: list[str]):
        self.matrix = _embed_passages(tuple(texts)) if texts else np.zeros((0, 384), dtype=np.float32)

    def add(self, texts: list[str]) -> None:
        """Incremental indexing: embed only the new passages and append them."""
        if texts:
            self.matrix = np.vstack([self.matrix, _embed_passages(tuple(texts))])

    def scores(self, query: str) -> list[float]:
        q = _normalise(np.array(list(_model().embed([QUERY_PREFIX + query])), dtype=np.float32))[0]
        return (self.matrix @ q).tolist()
