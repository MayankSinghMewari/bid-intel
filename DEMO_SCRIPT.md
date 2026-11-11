# Demo script (5 to 10 minutes)

Before recording: activate the virtual environment, run `python -m bid_intel tracing-check`, and start the UI
with `streamlit run app.py` (http://localhost:8501).

1. **Problem (30 s).** Bid teams read long RFPs, addendums and portal pages to find due dates, bonds,
   required affidavits and specs, and values change between versions. The system finds cited answers and
   extracts a structured, cited record per bid.

2. **Architecture (90 s).** README "Architecture" section and `docs/agent_graph.md`: ingestion, hybrid
   search (BM25 + Chroma embeddings), LangGraph agents with parallel field groups, the validator feedback
   loop, the optional LLM stage, addendum reconciliation.

3. **Extraction (2 min).** UI, Extraction tab: Bid1. Show Due Date (taken from Addendum 2, original date in
   the note), the evidence passage for a field, the addendum changes. Switch to Bid2: models, part numbers,
   affidavits, delivery. Or on the CLI: `python -m bid_intel run --bid data/bid_folders/Bid1`.

4. **Search and Ask (2 min).** Search tab: "bid bond" filtered to Bid1 and to RFP documents. Ask tab:
   "Which affidavits are required for the Dell laptop bid?", "Compare the warranty requirements of both
   bids." Open the citations and the Langfuse trace link.

5. **Observability (60 s).** In Langfuse, open an `extract-bid` trace: agents per field group, retriever
   and evaluator steps, retries flagged, generations with tokens.

6. **Unseen bid (45 s).** Add bid tab: upload a folder, index and extract without code changes.

7. **Evaluation (60 s).** Evaluation tab: retrieval configurations (hybrid best on MRR; re-ranker did not
   help) and extraction accuracy with the LLM-field count. State the limits: 20 questions, labels by the
   author, rules tuned on these two bids.

8. **Tests and limits (30 s).** `pytest -q` (offline). Known limitations from the README.
