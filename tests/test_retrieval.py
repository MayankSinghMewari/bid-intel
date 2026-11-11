from bid_intel.bm25 import BM25, tokenize
from bid_intel.ingest import Chunk
from bid_intel.retrieval import SearchEngine, evaluate


def _chunks():
    return [
        Chunk("a", "qa", "Monitor needs USB 3.1 minimum port", [1], 1),
        Chunk("b", "qa", "Warranty for Chromebooks is one year", [7], 2),
        Chunk("c", "qa", "Form 1295 goes with the response", [31], 4),
    ]


def test_tokenizer_keeps_negations_and_versions():
    toks = tokenize("Is a PDF version NOT required? USB 3.1")
    assert "not" in toks and "3.1" in toks and "is" not in toks


def test_bm25_ranks_the_relevant_chunk_first():
    engine = SearchEngine(_chunks())
    hits = engine.search("USB port minimum version for the monitor", k=3)
    assert hits[0].chunk.chunk_id == "a" and hits[0].rank == 1


def test_search_returns_nothing_for_unrelated_query():
    assert SearchEngine(_chunks()).search("zebra spaceship", k=3) == []


def test_recall_and_mrr_on_a_known_case():
    engine = SearchEngine(_chunks())
    questions = [{"question": "warranty for Chromebooks", "gold": [7]},
                 {"question": "Form 1295 response", "gold": [31]}]
    m = evaluate(engine, questions, ks=(1, 3))
    assert m["Recall@1"] == 1.0 and m["MRR"] == 1.0


def test_bm25_scores_are_zero_for_absent_terms():
    idx = BM25([["alpha", "beta"], ["gamma"]])
    assert idx.scores(["delta"]) == [0.0, 0.0]
