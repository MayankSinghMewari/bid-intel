"""Chroma vector store: incremental upsert, change detection, stale-chunk deletion, metadata filters.
Uses an in-memory Chroma client and deterministic fake embeddings, so it runs offline and fast."""
import hashlib

import numpy as np
import pytest

pytest.importorskip("chromadb")

from bid_intel import embeddings, vectorstore  # noqa: E402
from bid_intel.ingest import DocChunk  # noqa: E402
from bid_intel.search import BidSearch  # noqa: E402


def _fake_vec(text: str) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=embeddings.DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    calls = {"passages": 0}

    def embed_passages(texts):
        calls["passages"] += len(texts)
        return np.stack([_fake_vec(t) for t in texts]) if texts else np.zeros((0, embeddings.DIM), np.float32)

    monkeypatch.setattr(embeddings, "embed_passages", embed_passages)
    monkeypatch.setattr(embeddings, "embed_query", _fake_vec)  # query == passage text -> exact match
    return calls


def _chunks(bid="B1"):
    return [DocChunk(f"{bid}:rfp:p1:0", bid, "rfp.pdf", "rfp", 1, f"{bid} proposals are due July 1"),
            DocChunk(f"{bid}:spec:p2:0", bid, "specs.pdf", "specs", 2, f"{bid} processor Intel i5 memory 16GB"),
            DocChunk(f"{bid}:add:p1:0", bid, "Addendum 1.pdf", "addendum", 1, f"{bid} new due date July 9", 1)]


def test_incremental_upsert_embeds_only_new_or_changed(fake_embeddings):
    store = vectorstore.ChromaStore(path=None, collection="test_incremental")
    assert store.upsert(_chunks("B1"))["embedded"] == 3
    assert store.upsert(_chunks("B1")) == {"embedded": 0, "unchanged": 3, "deleted": 0}
    changed = _chunks("B1")
    changed[0].text = "B1 proposals are due July 2"
    assert store.upsert(changed)["embedded"] == 1
    assert store.upsert(_chunks("B2"))["embedded"] == 3  # a new bid does not touch B1
    assert store.count() == 6
    assert fake_embeddings["passages"] == 7


def test_stale_chunks_of_a_reindexed_bid_are_deleted():
    store = vectorstore.ChromaStore(path=None, collection="test_stale")
    store.upsert(_chunks("B1"))
    assert store.upsert(_chunks("B1")[:2])["deleted"] == 1
    assert store.count() == 2


def test_metadata_filters_run_inside_the_vector_store():
    store = vectorstore.ChromaStore(path=None, collection="test_filters")
    store.upsert(_chunks("B1") + _chunks("B2"))
    q = "B2 processor Intel i5 memory 16GB"
    assert store.query(q, 5, vectorstore.where_filter("B2", ["specs"]))[0][0] == "B2:spec:p2:0"
    ids = [i for i, _ in store.query(q, 10, vectorstore.where_filter(bid_id="B1"))]
    assert ids and all(i.startswith("B1:") for i in ids)
    addenda = sorted(i for i, _ in store.query(q, 10, vectorstore.where_filter(addendum_number=1)))
    assert addenda == ["B1:add:p1:0", "B2:add:p1:0"]


def test_hybrid_search_uses_the_store():
    store = vectorstore.ChromaStore(path=None, collection="test_hybrid")
    eng = BidSearch(_chunks("B1") + _chunks("B2"), dense=True, store=store)
    hits = eng.search("B2 processor Intel i5 memory 16GB", top_k=3, bid_id="B2", mode="dense")
    assert hits[0].chunk.chunk_id == "B2:spec:p2:0"
    assert {h.chunk.bid_id for h in eng.search("due date", bid_id="B1")} == {"B1"}
