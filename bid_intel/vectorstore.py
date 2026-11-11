"""Vector store: Chroma collection holding chunk embeddings plus their metadata.

- Embeddings come from our own model (embeddings.py), so Chroma never downloads a default model.
- Metadata (bid_id, file_name, doc_type, page, addendum_number) is stored with each vector, so
  metadata filters run inside the vector search ("only Bid2 specs").
- Incremental indexing: upsert() compares chunk ids and a hash of the chunk text with what the
  collection already holds, embeds only new or changed chunks, and deletes chunks of a re-indexed
  bid that no longer exist.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import embeddings
from .ingest import DocChunk

DEFAULT_PATH = Path(os.environ.get("BID_INTEL_CHROMA", Path(__file__).resolve().parent.parent / ".cache" / "chroma"))
COLLECTION = "bid_chunks"
MAX_DENSE_CANDIDATES = 200  # RRF contributions below rank 200 are negligible


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _metadata(c: DocChunk) -> dict:
    md = {"bid_id": c.bid_id, "file_name": c.file_name, "doc_type": c.doc_type, "page": c.page,
          "text_hash": text_hash(c.text)}
    if c.addendum_number is not None:  # Chroma metadata values cannot be None
        md["addendum_number"] = c.addendum_number
    return md


def where_filter(bid_id: str | None = None, doc_types: list[str] | None = None,
                 addendum_number: int | None = None) -> dict | None:
    conds = []
    if bid_id is not None:
        conds.append({"bid_id": bid_id})
    if doc_types:
        conds.append({"doc_type": {"$in": list(doc_types)}})
    if addendum_number is not None:
        conds.append({"addendum_number": addendum_number})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class ChromaStore:
    def __init__(self, path: Path | None = DEFAULT_PATH, collection: str = COLLECTION):
        import chromadb
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False)
        if path is None:  # in-memory store, used by tests
            self.client = chromadb.EphemeralClient(settings=settings)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path), settings=settings)
        self.collection = self.client.get_or_create_collection(
            collection, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
            metadata={"embedding_model": embeddings.MODEL_NAME})

    def upsert(self, chunks: list[DocChunk]) -> dict:
        """Index chunks; returns counts of embedded, unchanged and deleted chunks."""
        if not chunks:
            return {"embedded": 0, "unchanged": 0, "deleted": 0}
        ids = [c.chunk_id for c in chunks]
        existing = self.collection.get(ids=ids, include=["metadatas"])
        known = {i: (m or {}).get("text_hash") for i, m in zip(existing["ids"], existing["metadatas"])}
        todo = [c for c in chunks if known.get(c.chunk_id) != text_hash(c.text)]
        if todo:
            vectors = embeddings.embed_passages([c.text for c in todo])
            self.collection.upsert(ids=[c.chunk_id for c in todo], embeddings=vectors.tolist(),
                                   documents=[c.text for c in todo], metadatas=[_metadata(c) for c in todo])
        # remove chunks of these bids that disappeared (file deleted or re-chunked)
        deleted = 0
        for bid in {c.bid_id for c in chunks}:
            stored = set(self.collection.get(where={"bid_id": bid}, include=[])["ids"])
            stale = list(stored - set(ids))
            if stale:
                self.collection.delete(ids=stale)
                deleted += len(stale)
        return {"embedded": len(todo), "unchanged": len(chunks) - len(todo), "deleted": deleted}

    def query(self, query: str, n: int, where: dict | None = None) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine similarity) pairs, best first, restricted by the metadata filter."""
        n = max(1, min(n, MAX_DENSE_CANDIDATES))
        res = self.collection.query(query_embeddings=[embeddings.embed_query(query).tolist()],
                                    n_results=n, where=where, include=["distances"])
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def count(self) -> int:
        return self.collection.count()
