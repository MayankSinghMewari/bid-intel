"""REST API: endpoints answer with citations; folders are addressed by name only (no arbitrary paths)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from bid_intel import api, service  # noqa: E402


@pytest.fixture(autouse=True)
def keyword_only(monkeypatch):
    monkeypatch.setenv("BID_INTEL_DENSE", "off")
    monkeypatch.setattr(service, "_engine", None)


client = TestClient(api.app)


def test_health_and_bid_list():
    assert client.get("/health").json() == {"status": "ok"}
    assert {"bid_id": "Bid2", "extracted": True} in client.get("/bids").json()


def test_search_returns_cited_passages_filtered_by_bid():
    hits = client.get("/search", params={"q": "Mercury affidavit", "bid_id": "Bid2", "top_k": 3}).json()
    assert hits and all(h["bid_id"] == "Bid2" and h["file"] and h["page"] >= 1 for h in hits)
    assert any(h["file"] == "Mercury_Affidavit.pdf" for h in hits)


def test_ask_answers_with_citations():
    r = client.post("/ask", json={"question": "What is the delivery requirement for Bid2?"}).json()
    assert "45 days" in r["answer"] and r["citations"]


def test_stored_extraction_and_unknown_bid():
    assert client.get("/extraction/Bid2").json()["fields"]["Bid Number"]["value"] == "BPM044557"
    assert client.get("/extraction/NOPE").status_code == 404


def test_folders_cannot_escape_the_bid_directory():
    assert client.post("/index", json={"bid_id": "../../etc"}).status_code == 400
    assert client.post("/index", json={"bid_id": "C:"}).status_code == 400
    assert client.get("/search", params={"q": "bond", "bid_id": "../Bid1"}).status_code == 400
