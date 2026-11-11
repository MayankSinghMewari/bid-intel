"""Bid Document Analyzer: web UI (Streamlit). Run with:  streamlit run app.py

Tabs: Key facts (the 20 extracted fields with their evidence), Search documents (hybrid search with filters),
Ask a question (cited answers), Add a bid (upload a new folder; not in the public demo), How well it works
(retrieval and extraction evaluation). The UI calls the same service layer as the CLI and the REST API.

BID_INTEL_DEMO=1 turns it into a safe public demo: the LLM and tracing are forced off (strangers cannot spend
the owner's quota) and write actions (upload, re-run extraction) are hidden. Demo mode is also forced on when
the app runs on Streamlit Community Cloud.
"""
from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st


def _demo_mode() -> bool:
    """Read-only public demo: BID_INTEL_DEMO in the environment or in Streamlit secrets (Community Cloud),
    and always when running on Streamlit Community Cloud (apps live under /mount/src), so a public
    deployment is safe even if the setting is forgotten."""
    def truthy(v) -> bool:
        return str(v).strip().lower() in ("1", "true", "yes", "on")
    if truthy(os.environ.get("BID_INTEL_DEMO", "")):
        return True
    try:
        if truthy(st.secrets.get("BID_INTEL_DEMO", "")):
            return True
    except Exception:  # no secrets file configured
        pass
    return Path(__file__).resolve().as_posix().startswith("/mount/src/")


DEMO = _demo_mode()
if DEMO:  # set before bid_intel is imported, so no LLM or tracing client is ever created
    os.environ["BID_INTEL_LLM"] = "off"
    os.environ["BID_INTEL_TRACING"] = "off"

# Render something immediately: importing the search stack (Chroma, ONNX, LangGraph) and building the index
# take a while after a restart, and a blank page looks broken.
st.set_page_config(page_title="Bid Document Analyzer", page_icon=":material/description:", layout="wide")
_booting = st.empty()
_booting.info("**Starting up...** loading the search engine. After a restart the first visit takes 1-3 minutes "
              "while the embedding model loads and the bid documents are indexed.", icon=":material/hourglass_top:")

from bid_intel import llm, observability, service  # noqa: E402
from bid_intel.agents import dedupe_parts  # noqa: E402

REPO_URL = "https://github.com/MayankSinghMewari/bid-intel"
ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ plain-language labels
LABELS = {
    "Bid Number": "Bid number", "Title": "Title", "company_name": "Issued by", "contact_info": "Contact person",
    "Due Date": "Submission deadline", "Bid Submission Type": "How to submit", "Pre Bid Meeting": "Pre-bid meeting",
    "Delivery Date": "Delivery", "Term of Bid": "Contract length", "Bid Bond Requirement": "Bid bond (deposit)",
    "Payment Terms": "Payment terms", "Any Additional Documentation Required": "Documents to submit",
    "MFG for Registration": "Manufacturer authorization", "Contract or Cooperative to use": "Contract / cooperative",
    "Installation": "Installation and setup", "Model_no": "Model numbers", "Part_no": "Part numbers",
    "Product": "Products and quantities", "Product Specification": "Technical specifications",
    "Bid Summary": "Summary",
}
GROUPS = [
    ("Who and what", ["Bid Number", "Title", "company_name", "contact_info"]),
    ("Dates and how to submit", ["Due Date", "Bid Submission Type", "Pre Bid Meeting", "Delivery Date"]),
    ("Contract, money and paperwork", ["Term of Bid", "Bid Bond Requirement", "Payment Terms",
                                       "Any Additional Documentation Required", "Contract or Cooperative to use"]),
    ("Products", ["Product", "Product Specification", "Model_no", "Part_no", "MFG for Registration",
                  "Installation"]),
]
DOC_TYPES = {"bid_page": "Bid portal page", "rfp": "Main RFP document", "addendum": "Addendum (change notice)",
             "specs": "Specification sheet", "affidavit": "Affidavit / form"}
SEARCH_METHODS = {"hybrid": "Smart: keywords + meaning", "bm25": "Exact keywords", "dense": "Meaning only"}
METHOD_NAMES = {"bm25": "Exact keywords", "bm25 + query expansion": "Keywords + synonyms",
                "dense (bge-small)": "Meaning only", "hybrid RRF": "Smart (keywords + meaning)",
                "hybrid RRF + cross-encoder rerank": "Smart + re-ranker"}


def doc_label(file: str | None, doc_type: str | None = None) -> str:
    """Readable document name: the portal page gets a name; long file names are shortened."""
    if not file:
        return ""
    if doc_type == "bid_page" or file.lower().endswith((".html", ".htm")):
        return "Bid portal page (BidNet)"
    stem = re.sub(r"\s+", " ", Path(file).stem.replace("_", " ")).strip(" -")
    return stem if len(stem) <= 52 else stem[:49].rstrip() + "..."


def clean(value: str | None) -> str | None:
    if value is None:
        return None
    if ";" in value:
        return "; ".join(dedupe_parts(value.split(";")))
    return value


def highlight(text: str, terms: list[str]) -> str:
    """HTML-escaped passage with the search terms marked (works in light and dark themes)."""
    out = html.escape(" ".join(text.split()))
    for t in sorted({t for t in terms if len(t) > 2}, key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(html.escape(t))})", r"<mark>\1</mark>", out)
    return out


PASSAGE_CSS = ("<style>.passage{line-height:1.6;font-size:0.95rem;}"
               ".passage mark{background:rgba(57,135,229,.25);color:inherit;padding:0 .15em;border-radius:3px;}"
               "</style>")


def theme_type() -> str:
    t = st.context.theme
    return (t.get("type") if t else None) or "light"


def theme_toggle() -> None:
    """Dark-mode switch for this visitor only: sets Streamlit's own stored theme in the visitor's browser and
    reloads, so every widget uses the configured light/dark palette (.streamlit/config.toml)."""
    current_dark = theme_type() == "dark"
    want_dark = st.toggle("Dark mode", value=current_dark, key="dark_mode", help="Switch between light and dark.")
    if want_dark != current_dark:
        choice = "Dark" if want_dark else "Light"
        st.html("<script>(function(){const k='stActiveTheme-'+window.location.pathname+'-v2';"
                f"window.localStorage.setItem(k, JSON.stringify('{choice}'));window.location.reload();}})();</script>",
                unsafe_allow_javascript=True)


@st.cache_resource(show_spinner="Building the search index (first visit after a restart only)...")
def warm_index() -> int:
    service.ensure_indexed()
    return len(service.engine().chunks)


def bid_ids() -> list[str]:
    return sorted(f.name for f in service.FOLDERS.iterdir() if f.is_dir())


def load_record(bid: str) -> dict | None:
    path = service.OUT / f"{bid}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def bid_label(bid: str) -> str:
    r = load_record(bid)
    if not r:
        return bid
    org = r["fields"].get("company_name", {}).get("value") or ""
    title = r["fields"].get("Title", {}).get("value") or ""
    return f"{bid} · {org} · {title}" if org else f"{bid} · {title}"


def status_of(f: dict) -> str:
    if f["value"] is None:
        return "Not in the documents"
    return "Verified" if f["validated"] else "Needs review"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### :material/description: Bid Document Analyzer")
    st.caption("Reads public tender documents, finds answers with their sources, and extracts the key facts "
               "of every bid.")
    theme_toggle()
    st.divider()
    chunks = warm_index()
    st.markdown("**Status**")
    st.markdown(f":material/check_circle: Search ready · {chunks} passages indexed")
    provider = llm.provider()
    if DEMO:
        st.info("Public demo: AI writing is switched off on purpose. Answers come from the verified facts and "
                "the documents themselves.", icon=":material/info:")
    elif provider and not llm.breaker_reason():
        model = llm.GEMINI_MODEL if provider == "gemini" else llm.MODEL
        st.markdown(f":material/smart_toy: AI answers on · {model}")
    elif llm.breaker_reason():
        st.warning(f"AI paused: {llm.breaker_reason()}. Answers fall back to the documents.",
                   icon=":material/hourglass:")
    else:
        st.markdown(":material/do_not_disturb_on: AI answers off (no API key)")
    if DEMO:
        st.caption("Tracing: off in the public demo (Langfuse is used in the full version)")
    elif observability.enabled():
        st.markdown(":material/monitoring: Tracing on · Langfuse")
    else:
        st.caption("Tracing: local JSON only (no Langfuse keys)")
    st.divider()
    st.markdown("**How it works**")
    st.caption("1. Every document is split into short passages and indexed for keyword and meaning search.\n\n"
               "2. AI agents look up each of 20 facts, check every answer against its source, and apply "
               "addendum changes.\n\n"
               "3. Every answer shows the document and page it came from.")
    st.link_button("Source code on GitHub", REPO_URL, icon=":material/code:", width="stretch")

_booting.empty()  # ready: remove the start-up notice

# ------------------------------------------------------------------ header
st.title("Bid Document Analyzer")
st.markdown("Explore real public tenders: see the **key facts** the AI agents extracted, **search** every "
            "document, and **ask questions**. Every answer shows the document and page it came from.")
if DEMO:
    st.info(f"**Public read-only demo.** AI writing and uploads are off here, so answers come from the verified "
            f"facts and the documents. The full version with AI answers is on [GitHub]({REPO_URL}).",
            icon=":material/visibility:")

TAB_LABELS = [":material/fact_check: Key facts", ":material/search: Search documents",
              ":material/forum: Ask a question", ":material/upload_file: Add a bid",
              ":material/insights: How well it works"]
if DEMO:
    TAB_LABELS.remove(":material/upload_file: Add a bid")
tabs = dict(zip(TAB_LABELS, st.tabs(TAB_LABELS)))
tab_facts = tabs[":material/fact_check: Key facts"]
tab_search = tabs[":material/search: Search documents"]
tab_ask = tabs[":material/forum: Ask a question"]
tab_eval = tabs[":material/insights: How well it works"]
tab_add = tabs.get(":material/upload_file: Add a bid")

# ------------------------------------------------------------------ key facts
with tab_facts:
    ids = bid_ids()
    bid = st.selectbox("Choose a bid", ids, format_func=bid_label, key="extract_bid")
    if not DEMO:
        with st.expander("Re-run the AI agents on this bid"):
            st.caption("Runs the full workflow again and overwrites the saved results for this bid.")
            if st.button("Run extraction", icon=":material/play_arrow:"):
                with st.status(f"Extracting {bid}...", expanded=False) as status:
                    result, _ = service.extract(service.FOLDERS / bid)
                    status.update(label=f"Done: {result['validation']}", state="complete")
                if service.LAST_TRACE_URL:
                    st.link_button("Open the Langfuse trace", service.LAST_TRACE_URL, icon=":material/open_in_new:")
    record = load_record(bid) if bid else None
    if record is None:
        st.info("No results for this bid yet. Run the extraction first.", icon=":material/info:")
    else:
        fields = record["fields"]
        v = record["validation"]
        changes = record.get("addendum_changes", [])
        b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
        with b1:
            st.badge(f"{v['passed']} facts verified", icon=":material/check_circle:", color="green")
        with b2:
            st.badge(f"{v['not_found']} not in the documents", icon=":material/help:", color="gray")
        with b3:
            if v["failed"]:
                st.badge(f"{v['failed']} need review", icon=":material/warning:", color="orange")
        with b4:
            st.badge(f"{len(changes)} addendum changes", icon=":material/edit_note:", color="blue")

        # headline cards
        cards = st.columns(4)
        for col, name in zip(cards, ["Due Date", "Bid Submission Type", "company_name", "contact_info"]):
            f = fields.get(name, {})
            with col.container(border=True, height="stretch"):
                st.caption(LABELS[name])
                val = clean(f.get("value"))
                st.markdown(f"**{val[:160] + ('...' if len(val) > 160 else '')}**" if val else "_Not stated_")
                if f.get("sources"):
                    s0 = f["sources"][0]
                    st.caption(f":material/description: {doc_label(s0['file'], s0['doc_type'])} · p.{s0['page']}")
                if name == "Due Date" and str(f.get("notes", "")).lower().find("addendum") >= 0:
                    st.caption(":material/edit_note: Updated by an addendum")

        summary = fields.get("Bid Summary", {})
        if summary.get("value"):
            with st.container(border=True):
                st.markdown("**Summary**")
                st.markdown(summary["value"])
                written_by_ai = str(summary.get("notes", "")).startswith("generated by")
                st.caption("Written by AI (Gemini) from the verified facts" if written_by_ai
                           else "Assembled from the verified facts")

        st.subheader("All key facts")
        st.caption("Each fact links to the document and page it was taken from. Verified = the value was found "
                   "in its cited passage. Not in the documents = the system did not guess.")
        for group, names in GROUPS:
            rows = []
            for name in names:
                f = fields.get(name)
                if not f:
                    continue
                s0 = f["sources"][0] if f["sources"] else None
                rows.append({"Fact": LABELS.get(name, name), "Value": clean(f["value"]) or "—",
                             "Status": status_of(f), "Confidence": round(100 * f["confidence"]),
                             "Source": f"{doc_label(s0['file'], s0['doc_type'])} · p.{s0['page']}" if s0 else "—"})
            st.markdown(f"**{group}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Fact": st.column_config.TextColumn(width="small"),
                "Value": st.column_config.TextColumn(width="large"),
                "Status": st.column_config.TextColumn(width="small"),
                "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%",
                                                              width="small"),
                "Source": st.column_config.TextColumn(width="medium"),
            })

        st.subheader("Check the evidence")
        st.caption("Pick a fact to see the exact passage it came from.")
        names = [n for _, group_names in GROUPS for n in group_names if n in fields]
        field = st.selectbox("Fact", names, format_func=lambda n: LABELS.get(n, n), key="evidence_field")
        f = fields[field]
        st.markdown(f"**{LABELS.get(field, field)}:** {clean(f['value']) or '_Not stated in the documents_'}")
        st.caption(f"{status_of(f)} · confidence {round(100 * f['confidence'])}% · {f['notes']}")
        st.html(PASSAGE_CSS)
        for s in f["sources"]:
            with st.container(border=True):
                st.markdown(f":material/description: **{doc_label(s['file'], s['doc_type'])}** · page {s['page']}"
                            f" · {DOC_TYPES.get(s['doc_type'], s['doc_type'])}")
                st.html(f"<div class='passage'>{highlight(s['evidence'], (f['value'] or '').split(';')[:1])}</div>")

        if changes:
            with st.expander(f"What the addendums changed ({len(changes)})"):
                st.dataframe(pd.DataFrame([{"Addendum": c["addendum"], "Page": c["page"], "Change": c["change"]}
                                           for c in changes]), hide_index=True, width="stretch")
        st.download_button("Download all facts (JSON)", json.dumps(record, indent=2, ensure_ascii=False),
                           file_name=f"{bid}.json", mime="application/json", icon=":material/download:")

# ------------------------------------------------------------------ search
SEARCH_EXAMPLES = ["bid bond", "delivery within days of award", "required affidavits", "Latitude 5550",
                   "pre-proposal meeting", "invoice payment"]


def _use_search_example() -> None:
    if st.session_state.get("search_example"):
        st.session_state.search_q = st.session_state.search_example


with tab_search:
    st.markdown("Type a few words. Each result shows the passage, the document and the page.")
    st.pills("Try an example", SEARCH_EXAMPLES, key="search_example", on_change=_use_search_example)
    q = st.text_input("Search the bid documents", key="search_q", placeholder="e.g. bid bond")
    c1, c2 = st.columns([1, 2])
    s_bid = c1.selectbox("Bid", ["All bids", *bid_ids()],
                         format_func=lambda b: b if b == "All bids" else bid_label(b), key="search_bid")
    s_types = c2.multiselect("Document types (optional)", list(DOC_TYPES), format_func=DOC_TYPES.get,
                             key="search_types")
    with st.expander("Advanced search settings"):
        s_mode = st.segmented_control("Search method", list(SEARCH_METHODS), format_func=SEARCH_METHODS.get,
                                      default="hybrid", required=True, key="search_mode",
                                      help="Smart search combines exact keywords with meaning, and ranks the "
                                           "right page first most often (see How well it works).")
        s_k = st.slider("Number of results", 1, 10, 5, key="search_k")
        s_rerank = st.toggle("Re-rank with a second model", key="search_rerank",
                             help="Slower, and on our test set it ranked the right page first less often.")
    if q:
        hits = service.search(q, None if s_bid == "All bids" else s_bid, int(s_k), s_types or None,
                              s_mode or "hybrid", s_rerank)
        if not hits:
            st.warning("No matching passages. Try fewer or different words.", icon=":material/search_off:")
        else:
            st.caption(f"{len(hits)} passages, best match first")
            st.html(PASSAGE_CSS)
            terms = re.findall(r"[A-Za-z0-9.\-]+", q)
            for h in hits:
                with st.container(border=True):
                    head, tags = st.columns([3, 2], vertical_alignment="center")
                    head.markdown(f"**{h['rank']}. {doc_label(h['file'], h['doc_type'])}** · page {h['page']}")
                    with tags:
                        t1, t2 = st.columns(2)
                        t1.badge(h["bid_id"], icon=":material/folder:", color="gray")
                        t2.badge(DOC_TYPES.get(h["doc_type"], h["doc_type"]), color="blue")
                    st.html(f"<div class='passage'>{highlight(h['text'][:900], terms)}</div>")
                    st.caption("Best match" if h["rank"] == 1 else "Related passage",
                               help=f"Raw ranking score: {h['score']} (only meaningful for comparing results "
                                    "of the same search).")

# ------------------------------------------------------------------ ask
ASK_EXAMPLES = ["When is the submission deadline for Bid1 after all addendums?",
                "Which affidavits are required for the Dell laptop bid?",
                "What is the delivery requirement for Bid2?",
                "Who is the procurement contact for Bid1?",
                "Is a bid bond required for Bid1 or Bid2?"]
ANSWER_SOURCE = {"llm": ":material/smart_toy: Written by AI from the passages below",
                 "record": ":material/fact_check: From the verified key facts",
                 "extractive": ":material/format_quote: Closest sentences from the documents",
                 "not_found": ":material/search_off: Nothing relevant was found in the documents"}


def _use_ask_example() -> None:
    if st.session_state.get("ask_example"):
        st.session_state.pending_question = st.session_state.ask_example
        st.session_state.ask_example = None


def render_answer(r: dict) -> None:
    kind = r.get("answer_type", "extractive")
    if kind == "record" and r.get("facts"):
        for f in r["facts"]:
            st.markdown(f"**{f['bid_id']} · {LABELS.get(f['field'], f['field'])}**")
            st.markdown(clean(f["value"]) if f["value"] else "_Not stated in the documents._")
            if f["file"]:
                st.caption(f":material/description: {doc_label(f['file'])} · page {f['page']}"
                           + (f" · {f['note']}" if f["note"] else ""))
        if r.get("supporting"):
            with st.expander("Supporting sentences from the documents"):
                st.markdown(r["supporting"])
    elif kind == "not_found":
        st.markdown("I couldn't find this in the documents.")
    else:
        st.markdown(r["answer"])
    st.caption(ANSWER_SOURCE.get(kind, ""))
    if r.get("citations"):
        with st.expander(f"Sources ({len(r['citations'])})"):
            for c in r["citations"]:
                st.markdown(f"**[{c['n']}]** {doc_label(c['file'])} · page {c['page']} · `{c['bid_id']}`")
    if r.get("trace_url"):
        st.markdown(f"[See how this answer was produced (Langfuse trace)]({r['trace_url']})")


with tab_ask:
    st.markdown("Ask in plain English. Answers come only from the documents and show where they came from. "
                "Mention a bid (\"Bid2\", \"the Dell laptop bid\") to focus on it, or \"both bids\" to compare.")
    st.pills("Try a question", ASK_EXAMPLES, key="ask_example", on_change=_use_ask_example)
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_answer(msg["result"])
    prompt = st.chat_input("Ask about the bids...") or st.session_state.pop("pending_question", None)
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.spinner("Searching the documents..."):
            result = service.ask(prompt)
        st.session_state.chat.append({"role": "assistant", "result": result})
        st.rerun()
    if st.session_state.chat and st.button("Clear the conversation", icon=":material/delete:"):
        st.session_state.chat = []
        st.rerun()

# ------------------------------------------------------------------ add a bid (not in the public demo)
if tab_add is not None:
    with tab_add:
        st.markdown("Upload the files of **one** bid (portal page, RFP, addendums, spec sheets, affidavits). "
                    "They are saved as a new bid, indexed, and read by the AI agents. No code changes needed.")
        new_id = st.text_input("Name for this bid", placeholder="e.g. Bid3")
        files = st.file_uploader("Bid documents (PDF or HTML)", type=["pdf", "html", "htm"],
                                 accept_multiple_files=True)
        if st.button("Index and extract", icon=":material/rocket_launch:", disabled=not (new_id and files)):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", new_id):
                st.error("Use letters, digits, '-' or '_' only (max 40 characters).")
            elif (service.FOLDERS / new_id).exists():
                st.error(f"A bid named {new_id} already exists.")
            else:
                folder = service.FOLDERS / new_id
                folder.mkdir(parents=True)
                for f in files:
                    (folder / Path(f.name).name).write_bytes(f.getvalue())  # basename only: no path traversal
                with st.status(f"Processing {new_id}...", expanded=True) as status:
                    st.write(service.index_folder(folder)["status"])
                    result, _ = service.extract(folder)
                    status.update(label=f"{new_id}: {result['validation']}", state="complete")
                st.success(f"Done. Open **Key facts** and choose {new_id}.")

# ------------------------------------------------------------------ how well it works
with tab_eval:
    st.markdown("Two tests show how reliable the system is. Both use answers checked by hand against the "
                "documents.")
    st.subheader("1. Does search find the right page?")
    rpath = ROOT / "eval" / "retrieval_results.json"
    if rpath.exists():
        rows = json.loads(rpath.read_text(encoding="utf-8"))["configs"]
        n_q = rows[0]["questions"] if rows else 0
        st.markdown(f"We asked **{n_q} test questions** whose correct page we know. The chart shows how often "
                    "each search method ranks the correct page **first**.")
        import altair as alt
        dark = theme_type() == "dark"
        hi, lo = ("#3987e5", "#184f95") if dark else ("#2a78d6", "#86b6ef")  # validated: same hue, both themes
        ink, muted, grid = ("#f0efec", "#898781", "#2c2c2a") if dark else ("#0b0b0b", "#898781", "#e1e0d9")
        df = pd.DataFrame([{"Method": METHOD_NAMES.get(r["config"], r["config"]),
                            "Top-1": round(100 * r["Recall@1"]), "Top-3": round(100 * r["Recall@3"]),
                            "MRR": r["MRR"]} for r in rows])
        df["best"] = df["Top-1"] == df["Top-1"].max()
        df["label"] = df["Top-1"].astype(str) + "%"
        base = alt.Chart(df).encode(
            y=alt.Y("Method:N", sort="-x", title=None),
            x=alt.X("Top-1:Q", title="Correct page ranked first (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("Method:N"), alt.Tooltip("Top-1:Q", title="Ranked first (%)"),
                     alt.Tooltip("Top-3:Q", title="In top 3 (%)"), alt.Tooltip("MRR:Q", format=".3f")])
        bars = base.mark_bar(cornerRadiusEnd=4, size=20).encode(
            color=alt.condition(alt.datum.best, alt.value(hi), alt.value(lo)))
        labels = base.mark_text(align="left", dx=6, color=ink).encode(text="label:N")
        chart = (bars + labels).properties(height=230).configure_axis(
            labelColor=muted, titleColor=muted, gridColor=grid, domainColor=grid, tickColor=grid,
        ).configure_view(strokeWidth=0)
        st.altair_chart(chart, width="stretch", theme=None)
        best = df.loc[df["Top-1"].idxmax(), "Method"]
        st.caption(f"**{best}** is the default because it ranks the right page first most often. The extra "
                   "re-ranker finds every right page within the top 5 but ranks it first less often, so it is off.")
        with st.expander("Full results table"):
            st.dataframe(pd.DataFrame(rows).rename(columns={"config": "Configuration"}), hide_index=True,
                         width="stretch")
            st.caption("Recall@k: share of questions with the right page in the top k. MRR: average of 1/rank of "
                       "the first right page. `python -m bid_intel eval-retrieval`")

    st.subheader("2. Are the extracted facts correct?")
    epath = ROOT / "eval" / "extraction_results.json"
    if epath.exists():
        res = json.loads(epath.read_text(encoding="utf-8"))
        st.markdown("Every fact was compared with an answer written by hand from the documents. Facts that are "
                    "genuinely ambiguous in the documents are not scored.")
        for mode, per in res.items():
            cols = st.columns(len(per) + 1)
            total_c = sum(r["correct"] for r in per.values())
            total_n = sum(r["scored"] for r in per.values())
            cols[0].metric(f"All bids ({mode})", f"{total_c} / {total_n}", f"{total_c / total_n:.0%} correct",
                           delta_color="off")
            for col, (b, r) in zip(cols[1:], per.items()):
                col.metric(b, f"{r['correct']} / {r['scored']}", f"{r['accuracy']:.0%} correct", delta_color="off")
        st.caption("The rule-based extractors were written while reading these two bids, so this score is "
                   "optimistic; the AI extraction layer is what generalises to new bids. "
                   "`python -m bid_intel eval-extraction`")
