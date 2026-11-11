"""Streamlit UI smoke tests (headless AppTest): every tab renders; key facts, search, chat, examples and demo
mode work. Runs offline: keyword-only index, LLM and tracing off (conftest)."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TABS = [":material/fact_check: Key facts", ":material/search: Search documents", ":material/forum: Ask a question",
        ":material/upload_file: Add a bid", ":material/insights: How well it works"]


@pytest.fixture(autouse=True)
def keyword_only(monkeypatch):
    monkeypatch.setenv("BID_INTEL_DENSE", "off")
    from bid_intel import service
    monkeypatch.setattr(service, "_engine", None)


def _text(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_app_renders_all_tabs_and_the_key_facts():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs][:5] == TABS
    assert not any("Starting up" in i.value for i in at.info)  # start-up notice removed once ready
    at.selectbox(key="extract_bid").set_value("Bid2").run()
    assert not at.exception
    tables = [d.value for d in at.dataframe if "Fact" in d.value.columns]
    assert sum(len(t) for t in tables) == 19  # all extracted fields, split into four plain-language groups
    values = [v for t in tables for v in t["Value"].tolist()]
    assert "BPM044557" in values
    assert "Mercury Affidavit; Contract Affidavit" in values  # duplicates merged, readable casing
    assert "contact_info" not in _text(at.markdown) and "Contact person" in [r for t in tables for r in t["Fact"]]


def test_search_returns_cited_passages_with_readable_names():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input(key="search_q").set_value("Mercury affidavit").run()
    assert not at.exception
    results = _text(at.markdown)
    assert "Mercury Affidavit" in results and "page" in results


def test_chat_answers_with_citations():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("When is the pre-proposal meeting for Bid1?").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    answer = _text(at.chat_message[1].markdown)
    assert "June" in answer or "06/10/2024" in answer


def test_example_question_pill_asks_it():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.session_state["pending_question"] = "Which affidavits are required for the Dell laptop bid?"
    at.run()
    assert not at.exception
    assert "Mercury Affidavit" in _text(at.chat_message[1].markdown)


def test_public_demo_mode_is_read_only_and_never_uses_the_llm(monkeypatch):
    monkeypatch.setenv("BID_INTEL_DEMO", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "would-be-spent-if-demo-mode-failed")
    monkeypatch.delenv("BID_INTEL_LLM", raising=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert not any("Add a bid" in label for label in labels) and len(labels) == 4
    assert not any(b.label == "Run extraction" for b in at.button)
    assert any("Public read-only demo" in i.value for i in at.info)
    assert os.environ["BID_INTEL_LLM"] == "off"
    assert any("switched off on purpose" in i.value for i in at.sidebar.info)
