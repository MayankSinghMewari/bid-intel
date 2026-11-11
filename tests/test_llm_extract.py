"""LLM extraction: quote-grounded validation and the validator -> extractor feedback loop.
The LLM is replaced by a scripted fake, so these tests are offline and deterministic."""
from bid_intel import llm, llm_extract
from bid_intel.ingest import DocChunk
from bid_intel.fields import FIELDS
from bid_intel.llm_extract import FieldValue, GroupExtraction, validate
from bid_intel.search import BidSearch, SearchHit

SPEC = {f.name: f for f in FIELDS}
PASSAGES = [SearchHit(DocChunk("c1", "B1", "rfp.pdf", "rfp", 1, "Proposals are due June 10, 2024 at 2:00 PM EDT."), 1, 1),
            SearchHit(DocChunk("c2", "B1", "rfp.pdf", "rfp", 2, "9. Delivery within 45 days of Award."), 1, 2)]


def fv(name, value, cites, quote):
    return FieldValue(name=name, value=value, citations=cites, quote=quote)


def test_verbatim_quote_is_accepted_even_with_different_whitespace_and_case():
    ok, _ = validate(fv("Delivery Date", "45 days after award", [2], "delivery  within 45 days of award"),
                     SPEC["Delivery Date"], PASSAGES)
    assert ok


def test_hallucinated_quote_is_rejected():
    ok, reason = validate(fv("Delivery Date", "30 days", [2], "Delivery within 30 days of Award"),
                          SPEC["Delivery Date"], PASSAGES)
    assert not ok and "not found verbatim" in reason


def test_quote_must_come_from_a_cited_passage():
    ok, _ = validate(fv("Delivery Date", "45 days", [1], "Delivery within 45 days of Award"),
                     SPEC["Delivery Date"], PASSAGES)
    assert not ok


def test_bad_citation_and_missing_citation_are_rejected():
    assert not validate(fv("Delivery Date", "45 days", [7], "x"), SPEC["Delivery Date"], PASSAGES)[0]
    assert not validate(fv("Delivery Date", "45 days", [], None), SPEC["Delivery Date"], PASSAGES)[0]


def test_null_value_means_not_stated_and_is_valid():
    assert validate(fv("Bid Bond Requirement", None, [], None), SPEC["Bid Bond Requirement"], PASSAGES)[0]


def test_date_fields_must_look_like_dates():
    ok, reason = validate(fv("Due Date", "soon", [1], "Proposals are due"), SPEC["Due Date"], PASSAGES)
    assert not ok and "not a date" in reason


def test_feedback_loop_resends_only_rejected_fields_with_the_reason(monkeypatch):
    engine = BidSearch([h.chunk for h in PASSAGES], dense=False)
    prompts = []
    replies = [
        GroupExtraction(fields=[fv("Due Date", "June 10, 2024 at 2:00 PM EDT", [1], "due June 10, 2024 at 2:00 PM EDT"),
                                fv("Delivery Date", "30 days", [2], "Delivery within 30 days")]),     # hallucinated
        GroupExtraction(fields=[fv("Delivery Date", "45 days of award", [2], "Delivery within 45 days of Award")]),
    ]

    def fake_complete(system, user, max_tokens=2048, schema=None, timeout_s=30):
        prompts.append(user)
        return {"parsed": replies[len(prompts) - 1], "model": "fake", "text": "",
                "input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0}

    monkeypatch.setattr(llm, "_complete", fake_complete)
    log = lambda *a, **k: None  # noqa: E731
    res = llm_extract.extract_group(engine, "B1", "dates", [SPEC["Due Date"], SPEC["Delivery Date"]], log)
    assert len(prompts) == 2
    assert "REJECTED" in prompts[1] and "Delivery Date" in prompts[1] and "not found verbatim" in prompts[1]
    assert "- Due Date:" not in prompts[1].split("FIELDS TO EXTRACT")[1]  # accepted field is not re-asked
    value, _, ok, _, attempts = res["Delivery Date"]
    assert ok and attempts == 2 and value.value == "45 days of award"
    assert res["Due Date"][2] and res["Due Date"][4] == 1
