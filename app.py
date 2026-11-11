"""RFP Intelligence: web UI (Streamlit). Run with:  streamlit run app.py

Tabs: Extraction (20 fields with citations), Search (hybrid search with filters), Ask (cited chat over the
bids), Add bid (upload a new folder, index and extract it), Evaluation (retrieval and extraction results).
The UI calls the same service layer as the CLI and the REST API (bid_intel/bids.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from bid_intel import bids, llm, observability

ROOT = Path(__file__).resolve().parent
DOC_TYPES = ["bid_page", "rfp", "addendum", "specs", "affidavit"]

st.set_page_config(page_title="RFP Intelligence", page_icon=":material/description:", layout="wide")


@st.cache_resource(show_spinner="Indexing bid folders (first run embeds every chunk; later runs reuse the index)...")
def warm_index() -> int:
    bids.ensure_indexed()
    return len(bids.engine().chunks)


def bid_ids() -> list[str]:
    return sorted(f.name for f in bids.FOLDERS.iterdir() if f.is_dir())


def load_record(bid: str) -> dict | None:
    path = bids.OUT / f"{bid}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("RFP Intelligence")
    st.caption("Hybrid RAG search and multi-agent extraction over public bid documents.")
    chunks = warm_index()
    st.metric("Indexed chunks", chunks, help="BM25 in memory + Chroma vector store (bge-small)")
    provider = llm.provider()
    if provider and not llm.breaker_reason():
        model = llm.GEMINI_MODEL if provider == "gemini" else llm.MODEL
        st.success(f"LLM: {model}", icon=":material/smart_toy:")
    elif llm.breaker_reason():
        st.warning(f"LLM paused: {llm.breaker_reason()}. Answers fall back to cited passages.",
                   icon=":material/hourglass:")
    else:
        st.info("No LLM key: rule-based extraction and extractive answers.", icon=":material/info:")
    if observability.enabled():
        st.success("Tracing: Langfuse", icon=":material/monitoring:")
    else:
        st.caption("Tracing: local JSON only (no Langfuse keys)")

tab_extract, tab_search, tab_ask, tab_add, tab_eval = st.tabs(
    [":material/table_view: Extraction", ":material/search: Search", ":material/forum: Ask",
     ":material/upload_file: Add bid", ":material/analytics: Evaluation"])

# ---------------------------------------------------------------- extraction
with tab_extract:
    ids = bid_ids()
    col_pick, col_run = st.columns([3, 1], vertical_alignment="bottom")
    bid = col_pick.selectbox("Bid folder", ids, key="extract_bid")
    if col_run.button("Run extraction", icon=":material/play_arrow:", width="stretch",
                      help="Runs the LangGraph agents on this folder and overwrites outputs/<bid>.json"):
        with st.status(f"Extracting {bid}...", expanded=False) as status:
            result, trace = bids.extract(bids.FOLDERS / bid)
            status.update(label=f"Done: {result['validation']}", state="complete")
        if bids.LAST_TRACE_URL:
            st.link_button("Open Langfuse trace", bids.LAST_TRACE_URL, icon=":material/open_in_new:")
    record = load_record(bid) if bid else None
    if record is None:
        st.info("No extraction yet for this bid. Click **Run extraction**.")
    else:
        v = record["validation"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Validated", v["passed"])
        m2.metric("Failed validation", v["failed"])
        m3.metric("Not found in documents", v["not_found"])
        m4.metric("Addendum changes", len(record.get("addendum_changes", [])))
        summary = record["fields"].get("Bid Summary", {})
        if summary.get("value"):
            st.markdown(f"**Summary.** {summary['value']}")
            st.caption(summary.get("notes", ""))

        rows = []
        for name, f in record["fields"].items():
            if name == "Bid Summary":
                continue
            src = f["sources"][0] if f["sources"] else None
            rows.append({"Field": name, "Value": f["value"] if f["value"] is not None else "Not found",
                         "Confidence": f["confidence"], "Validated": f["validated"],
                         "Source": f"{src['file']} · p.{src['page']}" if src else "",
                         "Method": "LLM" if f["notes"].startswith("LLM extraction") else
                                   ("addendum" if "Addendum" in f["notes"] else "rules")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={
                         "Value": st.column_config.TextColumn(width="large"),
                         "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                         "Validated": st.column_config.CheckboxColumn(),
                     })

        st.subheader("Evidence for a field")
        field = st.selectbox("Field", [r["Field"] for r in rows], key="evidence_field")
        f = record["fields"][field]
        st.markdown(f"**Value:** {f['value'] if f['value'] is not None else '_Not found in documents_'}")
        st.caption(f"Confidence {f['confidence']} · {'validated' if f['validated'] else 'not validated'} · {f['notes']}")
        for s in f["sources"]:
            with st.expander(f"{s['file']} · page {s['page']} · {s['doc_type']}"):
                st.text(s["evidence"])

        if record.get("addendum_changes"):
            with st.expander(f"Addendum changes ({len(record['addendum_changes'])})"):
                st.dataframe(pd.DataFrame(record["addendum_changes"]), hide_index=True, width="stretch")
        st.download_button("Download JSON", json.dumps(record, indent=2, ensure_ascii=False),
                           file_name=f"{bid}.json", mime="application/json", icon=":material/download:")

# ---------------------------------------------------------------- search
with tab_search:
    q = st.text_input("Query", placeholder="e.g. bid bond, delivery within days of award, Latitude 5550")
    c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
    s_bid = c1.selectbox("Bid", ["All bids", *bid_ids()], key="search_bid")
    s_types = c2.multiselect("Document types", DOC_TYPES, key="search_types")
    s_mode = c3.segmented_control("Mode", ["hybrid", "bm25", "dense"], default="hybrid", key="search_mode")
    s_k = c4.number_input("Top k", 1, 20, 5, key="search_k")
    s_rerank = st.toggle("Cross-encoder re-ranking", help="Slower (first use downloads the model); "
                                                          "lowered MRR on the evaluation set")
    if q:
        hits = bids.search(q, None if s_bid == "All bids" else s_bid, int(s_k), s_types or None,
                           s_mode or "hybrid", s_rerank)
        if not hits:
            st.warning("No matching passages.")
        for h in hits:
            with st.container(border=True):
                st.markdown(f"**#{h['rank']}** · `{h['bid_id']}` · {h['file']} · page {h['page']} · "
                            f"*{h['doc_type']}* · score {h['score']}")
                st.text(" ".join(h["text"].split())[:900])

# ---------------------------------------------------------------- ask (chat)
with tab_ask:
    st.caption("Answers come only from retrieved passages and cite them as [n]. Name a bid (\"Bid2\", "
               "\"the Dell laptop bid\") to restrict the search; \"both bids\" compares them.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                with st.expander(f"Citations ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(f"**[{c['n']}]** `{c['bid_id']}` · {c['file']} · page {c['page']}")
                        st.text(" ".join(c["passage"].split())[:500])
                st.caption(msg["method"])
                if msg.get("trace_url"):
                    st.markdown(f"[Langfuse trace]({msg['trace_url']})")
    if prompt := st.chat_input("Ask about the bids, e.g. Which affidavits are required for the Dell laptop bid?"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.spinner("Searching and answering..."):
            r = bids.ask(prompt)
        method = r["method"] + (f" (LLM fallback: {r['llm_fallback_reason'][:80]})"
                                if r.get("llm_fallback_reason") else "")
        st.session_state.chat.append({"role": "assistant", "content": r["answer"], "citations": r["citations"],
                                      "method": method, "trace_url": r.get("trace_url")})
        st.rerun()
    if st.session_state.chat and st.button("Clear chat", icon=":material/delete:"):
        st.session_state.chat = []
        st.rerun()

# ---------------------------------------------------------------- add bid
with tab_add:
    st.markdown("Upload the files of **one** bid (HTML bid page, RFP, addendums, specs, affidavits). "
                "They are saved as a new bid folder, indexed incrementally, and extracted by the agents.")
    new_id = st.text_input("Bid folder name", placeholder="e.g. Bid3")
    files = st.file_uploader("Bid documents", type=["pdf", "html", "htm"], accept_multiple_files=True)
    if st.button("Index and extract", icon=":material/rocket_launch:", disabled=not (new_id and files)):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", new_id):
            st.error("Use letters, digits, '-' or '_' only (max 40 characters).")
        elif (bids.FOLDERS / new_id).exists():
            st.error(f"A bid folder named {new_id} already exists.")
        else:
            folder = bids.FOLDERS / new_id
            folder.mkdir(parents=True)
            for f in files:
                (folder / Path(f.name).name).write_bytes(f.getvalue())  # basename only: no path traversal
            with st.status(f"Processing {new_id}...", expanded=True) as status:
                st.write(bids.index_folder(folder)["status"])
                result, _ = bids.extract(folder)
                status.update(label=f"{new_id}: {result['validation']}", state="complete")
            st.success(f"Done. Open the **Extraction** tab and pick {new_id}.")

# ---------------------------------------------------------------- evaluation
with tab_eval:
    st.subheader("Retrieval (Recall@k, MRR)")
    rpath = ROOT / "eval" / "bid_results.json"
    if rpath.exists():
        st.dataframe(pd.DataFrame(json.loads(rpath.read_text(encoding="utf-8"))["configs"]), hide_index=True,
                     width="stretch")
        st.caption("20 questions over Bid1 and Bid2, gold file and page. `python -m bid_intel eval-bids`")
    st.subheader("Extraction accuracy")
    epath = ROOT / "eval" / "extraction_results.json"
    if epath.exists():
        res = json.loads(epath.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame([{"Mode": m, "Bid": b, "Correct": r["correct"], "Scored": r["scored"],
                                    "Accuracy": r["accuracy"], "LLM-sourced fields": r.get("llm_fields", 0)}
                                   for m, per in res.items() for b, r in per.items()]),
                     hide_index=True, width="stretch")
        st.caption("Hand-checked labels in eval/extraction_gold.json. `python -m bid_intel eval-extraction`")
