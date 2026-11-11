import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from bid_intel.api import app  # noqa: E402


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_ask_endpoint_returns_citations():
    client = TestClient(app)
    r = client.post("/bids/JA-207652/ask", json={"question": "When will the RFP be awarded?"})
    assert r.status_code == 200
    assert r.json()["citations"][0]["qnums"] == [3]


def test_unknown_bid_is_404():
    r = TestClient(app).post("/bids/NOPE/ask", json={"question": "anything here"})
    assert r.status_code == 404
