"""Streamlit UI smoke test (headless AppTest): every tab renders; search and chat return cited results.
Runs offline: keyword-only index, LLM and tracing off (conftest)."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def keyword_only(monkeypatch):
    monkeypatch.setenv("BID_INTEL_DENSE", "off")
    from bid_intel import service
    monkeypatch.setattr(service, "_engine", None)


def test_app_renders_all_tabs_and_the_extraction_record():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs][:5] == [":material/table_view: Extraction", ":material/search: Search",
                                             ":material/forum: Ask", ":material/upload_file: Add bid",
                                             ":material/analytics: Evaluation"]
    at.selectbox(key="extract_bid").set_value("Bid2").run()
    assert not at.exception
    table = at.dataframe[0].value
    assert len(table) == 19 and "BPM044557" in table["Value"].tolist()


def test_search_returns_cited_passages():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input[0].set_value("Mercury affidavit").run()
    assert not at.exception
    results = " ".join(m.value for m in at.markdown)
    assert "Mercury_Affidavit.pdf" in results and "page" in results


def test_chat_answers_with_citations():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("When is the pre-proposal meeting for Bid1?").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assistant = at.chat_message[1]
    assert "June" in assistant.markdown[0].value or "06/10/2024" in assistant.markdown[0].value
