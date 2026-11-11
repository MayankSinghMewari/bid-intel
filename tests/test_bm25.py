"""BM25 (the lexical half of hybrid search)."""
from bid_intel.bm25 import BM25, tokenize


def test_tokenizer_keeps_negations_versions_and_part_numbers():
    toks = tokenize("Is a PDF version NOT required? USB 3.1 and part WD22TB4")
    assert "not" in toks and "3.1" in toks and "wd22tb4" in toks and "is" not in toks


def test_bm25_ranks_the_matching_document_first():
    idx = BM25([tokenize("Monitor needs USB 3.1 minimum port"), tokenize("Warranty for Chromebooks is one year")])
    scores = idx.scores(tokenize("USB port for the monitor"))
    assert scores[0] > scores[1] == 0


def test_bm25_scores_are_zero_for_absent_terms():
    assert BM25([["alpha", "beta"], ["gamma"]]).scores(["delta"]) == [0.0, 0.0]
