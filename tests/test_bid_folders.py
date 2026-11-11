"""Tests for generic bid-folder ingestion, search filters and field rules (no network needed)."""
from pathlib import Path

from bid_intel.docs import DocChunk, doc_type_of, ingest_folder
from bid_intel.fields import FIELDS
from bid_intel.search import BidSearch, expand_query

ROOT = Path(__file__).resolve().parent.parent
RULES = {f.name: f.rule for f in FIELDS}


def test_doc_type_detection():
    assert doc_type_of("Addendum 2 RFP JA-207652.pdf") == ("addendum", 2)
    assert doc_type_of("Bid Information _ BidNet Direct.html") == ("bid_page", None)
    assert doc_type_of("Mercury_Affidavit.pdf") == ("affidavit", None)
    assert doc_type_of("Dell_Laptop_Specs.pdf") == ("specs", None)
    assert doc_type_of("PORFP_-_Dell_Laptop_Final.pdf") == ("rfp", None)


def test_html_bid_page_facts_are_parsed():
    chunks, report = ingest_folder(ROOT / "data" / "bid_folders" / "Bid2")
    facts = next(c.facts for c in chunks if c.facts)
    assert facts["solicitation number"] == "BPM044557"
    assert "Treasurer" in facts["issuing organization"]
    assert all(r["status"] == "ok" for r in report)
    assert all(c.bid_id == "Bid2" and c.page >= 1 for c in chunks)


def _engine():
    return BidSearch([
        DocChunk("a", "B1", "rfp.pdf", "rfp", 3, "Proposals are due July 1, 2024 at 2:00 PM CST."),
        DocChunk("b", "B1", "Addendum 1.pdf", "addendum", 1, "The new due date will be July 9, 2024 at 2:00 PM CST.", 1),
        DocChunk("c", "B2", "specs.pdf", "specs", 1, "Processor Intel Core i5. Memory 16GB. Model Latitude 5550."),
    ], dense=False)


def test_metadata_filters_restrict_results():
    eng = _engine()
    assert {h.chunk.bid_id for h in eng.search("due date", bid_id="B1")} == {"B1"}
    assert [h.chunk.chunk_id for h in eng.search("due date", doc_types=["addendum"])] == ["b"]
    assert eng.search("due date", bid_id="NOPE") == []


def test_query_expansion_adds_synonyms():
    assert "closing date" in expand_query("what is the deadline")


def test_incremental_add_keeps_existing_chunks():
    eng = _engine()
    eng.add([DocChunk("d", "B3", "x.pdf", "rfp", 1, "Bid bond of 5% is required.")])
    assert len(eng.chunks) == 4 and eng.search("bid bond", bid_id="B3")[0].chunk.chunk_id == "d"


def test_field_rules():
    assert RULES["Bid Bond Requirement"]("A bid bond of 5% of the total bid amount must be submitted.")
    assert RULES["Bid Bond Requirement"]("A bid bond is not required for this solicitation.") == "Not required"
    assert RULES["Delivery Date"]("9. Delivery within 45 days of Award.").lower().startswith("delivery within 45 days")
    assert "Mercury Affidavit" in RULES["Any Additional Documentation Required"]("Provide a Mercury Affidavit and a W-9.")
    assert RULES["Term of Bid"]("Initial Term: 3 Renewal 1: 1 Renewal 2: 1").startswith("Initial term 3 year")
