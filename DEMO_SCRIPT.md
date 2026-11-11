# Demo Script (5 to 10 minutes)

Record this as a screen capture, or run it live. Each step lists the command, what to say, and what the viewer should see.

1. **Problem (30 s).** Bid documents are long and the answers are scattered across questions. The system finds cited answers and extracts structured values, each with a confidence score.

2. **Architecture (60 s).** Open `docs/architecture.svg`. Walk through: ingestion, then the BM25 search engine, then four agents on a message bus, then the JSON output.

3. **Ingest (45 s).**
   `python -m bid_intel ingest data/bids/JA-207652/addendum_1.pdf --bid JA-207652`
   Point out: 39 items parsed from 6 pages, with numbering checked.

4. **Ask (90 s).** Run three questions and show citations:
   - `python -m bid_intel ask --bid JA-207652 "Is accidental damage protection required?"` (cites Q9)
   - `python -m bid_intel ask --bid JA-207652 "Do the AIO desktops need a webcam?"` (cites Q23)
   - `python -m bid_intel ask --bid JA-207652 "Why are 65W chargers required?"` (cites Q38, which is the vendor's question, so mention the lower confidence)

5. **Extraction and trace (120 s).**
   `python -m bid_intel extract --bid JA-207652 --trace logs/agent_trace.json`
   Open `data/bids/JA-207652/extraction.json`. Show one field with its citation, evidence, and confidence. Then open `logs/agent_trace.json` and show the retriever, extractor, and verifier messages for one field.

6. **Evaluation (60 s).** Open `eval/report.md`. Explain Recall@k and MRR, the six configurations, and why the best and default configurations are close. State the limitations plainly.

7. **API (45 s).** `python -m bid_intel serve`, then open `/docs` in a browser and run `/bids/JA-207652/ask`.

8. **Tests (20 s).** `pytest -q` shows 17 passing.

9. **Limitations and next steps (60 s).** One document tested. Rules are hand-written. Ask mode is extractive, with no LLM generation. Dense retrieval, OCR, and comparison agents are not built. Docker and CI are written but not run here.

Before recording, run the setup steps in a clean virtual environment to confirm that the commands work on your machine.
