from pathlib import Path

from bid_intel import service
from bid_intel.schema import FIELDS

BID = "JA-207652"
ADDENDUM = Path(__file__).resolve().parent.parent / "data" / "bids" / BID / "addendum_1.pdf"


def _ensure_ingested():
    if not (service.BIDS_DIR / BID / "items.json").exists():
        from bid_intel.ingest import ingest
        ingest(ADDENDUM, BID, service.BIDS_DIR / BID)


def test_extraction_values_match_the_addendum():
    _ensure_ingested()
    result, trace = service.extract(BID)
    f = result["fields"]
    assert f["usb_monitor_minimum"]["value"] == "3.1"
    assert f["chromebook_warranty_minimum"]["value"] == "one-year"
    assert f["windows_warranty_minimum"]["value"] == "three years"
    assert f["warranty_backing"]["value"] == "OEM"
    assert f["etch_and_asset_tag"]["value"] == "Both are required"
    assert f["form_1295_timing"]["value"] == "RFP response"
    assert trace and trace[0]["sender"] == "retriever"


def test_every_found_field_has_a_citation_and_confidence():
    _ensure_ingested()
    result, _ = service.extract(BID)
    for key, field in result["fields"].items():
        if field["status"] == "found":
            assert field["citation"]["qnums"], key
            assert 0.0 <= field["confidence"] <= 1.0, key


def test_charger_value_is_flagged_as_question_sourced():
    _ensure_ingested()
    result, _ = service.extract(BID)
    charger = result["fields"]["chromebook_charger_watts"]
    assert charger["value"] == "65"
    assert charger["citation"]["evidence_source"] == "question"


def test_ask_returns_cited_passage():
    _ensure_ingested()
    res = service.ask(BID, "Is accidental damage protection required?")
    assert res["citations"] and res["citations"][0]["qnums"] == [9]
