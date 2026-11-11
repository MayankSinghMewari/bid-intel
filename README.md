# RFP Intelligence Platform: RAG Search Engine and Multi-Agent Extraction

Indexes bid folders (HTML bid pages and PDFs, including addendums), searches them with hybrid retrieval and cited results, and runs a multi-agent pipeline that writes one JSON record per bid with the 20 required fields, each with sources and a confidence score.

Tested on the two provided folders, `data/bid_folders/Bid1` (Dallas ISD JA-207652: bid page, RFP, Addendum 1, Addendum 2) and `data/bid_folders/Bid2` (Maryland Treasurer BPM044557: bid page, PORFP, specs, two affidavits). Nothing in the code names these bids; a new folder with a similar structure runs with the same command.

## Setup

Python 3.10+. `pdftotext` (poppler) is used when installed, otherwise `pypdf`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, about 70 MB). Embeddings are cached in `.cache/`, so later runs only embed new chunks.

LLM (the generation step of RAG): copy `.env.example` to `.env` and set `GEMINI_API_KEY` (free key from https://aistudio.google.com/apikey; model `GEMINI_MODEL`, default `gemini-3.5-flash`) or `ANTHROPIC_API_KEY`. `.env` is git-ignored. With a key, Q&A answers and the Bid Summary field are written by the LLM from retrieved passages / validated fields only, and every answer must cite passages as [n] or it is rejected. Without a key, or when the LLM call fails (timeout, 503, quota), the system falls back to the validated extraction record and cited sentences, and the trace records the reason. Tests run with the LLM disabled.

## How to run

```bash
# Extraction mode: ONE COMMAND per bid folder -> outputs/<bid>.json and logs/<bid>_trace.json
python -m bid_intel run --bid data/bid_folders/Bid1
python -m bid_intel run --bid data/bid_folders/Bid2

# Search (hybrid, with metadata filters)
python -m bid_intel search "bid bond" --bid-id Bid2
python -m bid_intel search "due date" --bid-id Bid1 --doc-type addendum

# Question-answering mode (cited answers; names like "Bid1" in the question restrict the search)
python -m bid_intel ask-bids "What is the submission deadline for Bid1 after all addendums?"

# Retrieval evaluation (writes eval/bid_report.md and eval/bid_results.json)
python -m bid_intel eval-bids

# REST API on http://127.0.0.1:8000 (docs at /docs)
python -m bid_intel serve
#   POST /index {"folder": "data/bid_folders/Bid1"}
#   GET  /search?q=bid+bond&bid_id=Bid2&top_k=5&doc_type=rfp
#   POST /ask {"question": "Which affidavits are required for the Dell laptop bid?"}
#   POST /extract/Bid1

# Tests
pytest -q
```

## Deliverables in this repository

| Deliverable | Location |
|---|---|
| JSON output, one per bid | `outputs/Bid1.json`, `outputs/Bid2.json` |
| Retrieval evaluation report | `eval/bid_report.md` (questions: `eval/bid_questions.json`) |
| Sample Q&A log (12 questions) | `logs/bid_qa_log.md` |
| Agent trace of a full extraction run | `logs/Bid1_trace.json` (also `logs/Bid2_trace.json`) |
| Architecture diagram | "Architecture" section below (the earlier prototype diagram is in `docs/architecture.svg`) |

## Architecture

```
Bid folder (HTML + PDF)
   |
[IngestionAgent] docs.py: detect doc type, parse HTML (BeautifulSoup) / PDF (pdftotext), strip repeated
   |              headers/footers, fix hyphenation, attach metadata, log unreadable/empty files
   v
[Chunking: 180-word windows, 40-word overlap, per page] -> [BM25 index] + [dense index: bge-small]
   ^                                                             (incremental: new bids are appended)
   | search tool: hybrid RRF, metadata filters, query expansion, optional cross-encoder rerank
   |
[Orchestrator] plans field groups -> runs them in parallel (thread pool)
   |-> RetrievalAgent    query expansion + filters (bid, doc type), widens on retry
   |-> ExtractionAgent   one specialist per group: identity, dates, commercial, product
   |-> ValidatorAgent    citation present, value found verbatim in cited evidence, date format
   |      rejected -> back to RetrievalAgent (max 2 retries)
   |-> ReconcileAgent    lists addendum changes, applies the latest change (e.g. extended due date)
   v
outputs/<bid>.json (fields + sources + confidence + addendum_changes + validation) and trace
```

Shared state is a Pydantic `BidState` (plan, evidence, draft fields, validation, trace). Agent messages are Pydantic models (`FieldResult`, `Source`, `TraceStep`). Every step logs agent, action, input, output and latency.

## Design decisions

**Ingestion.** HTML bid pages are parsed into label/value facts (solicitation number, title, closing date, contact, issuing organization), which are the most reliable source for those fields. PDFs are read page by page so every chunk keeps its page number. Lines repeated on more than half of a document's pages are removed as headers/footers. Document type comes from the file name (`addendum N`, `affidavit`, `spec`, `.html` = bid page, otherwise `rfp`). Files that fail to parse, or pages with no text (scanned), are recorded in the ingestion log and skipped instead of stopping the run.

**Chunking.** 180-word windows with 40-word overlap, never crossing a page boundary. RFPs mix short form fields, clauses and spec lists; 180 words holds one clause or a short table while keeping citations to a single page. The overlap keeps sentences that straddle a boundary retrievable.

**Embedding model.** `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU, no API key). Small enough to run on a laptop, and strong on retrieval benchmarks for its size.

**Retrieval.** BM25 (pure Python, keeps negations and numbers such as model and part numbers) plus dense cosine search, merged with reciprocal rank fusion (k=60). Metadata filters restrict by bid, document type and addendum number. Query expansion adds RFP synonyms ("deadline" -> due date, closing date, submission). A cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) is available but off by default, because on this evaluation it lowered MRR and cost about 5 s per query.

**Agent framework.** Own orchestration in `bid_intel/bid_agents.py` rather than LangGraph or CrewAI: the flow is a fixed plan with a bounded retry loop, which is short to write and easy to trace and test, and it adds no dependencies.

**Extraction.** Each of the 20 fields (`bid_intel/fields.py`) declares a retrieval query, preferred document types, bid-page fact keys, and a deterministic rule that reads the value from a retrieved passage. Rules use general RFP vocabulary (dates, "bid bond", "authorized reseller", "Master Contract", affidavit names), not text from these two bids. Extraction is deterministic and every value points at the passage it came from. The cost is recall: values phrased in unexpected ways are missed and reported as "Not found in documents".

**Addendum reconciliation.** Addendum sentences with change verbs (extend, revise, replace, new due date ...) are collected into `addendum_changes`. If an addendum states a new due date, the latest one overrides the base value and the note records the previous value.

**Validation and confidence.** The validator rejects values without a citation, values not found verbatim in the cited evidence, and malformed dates. Rejected fields are retried with a wider search. Confidence: 0.9 for bid-page facts, 0.55 + 0.35/rank for rule matches, 0.93 for addendum overrides, multiplied by 0.6 if validation fails. These are heuristics, not calibrated probabilities.

**Prompts.** The LLM prompts are in `bid_intel/llm.py`. They require answers only from the numbered passages, an explicit "Not found in documents", and a citation on every claim; answers that cite passages that were not provided are rejected.

## Retrieval evaluation

20 questions (10 per bid) with gold file and page. Each search is filtered to the question's bid.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | ms/query |
|---|---|---|---|---|---|
| bm25 | 0.65 | 0.9 | 0.9 | 0.758 | 0.9 |
| bm25 + query expansion | 0.65 | 0.9 | 0.95 | 0.777 | 0.9 |
| dense (bge-small) | 0.75 | 0.9 | 0.95 | 0.835 | 59.1 |
| **hybrid RRF (default)** | **0.85** | 0.9 | 0.95 | **0.879** | 24.7 |
| hybrid RRF + cross-encoder rerank | 0.65 | 0.95 | 1.0 | 0.804 | 5235.7 |

Hybrid search gives the best top-1 accuracy. The re-ranker finds every gold passage within 5 but ranks it first less often. With 20 questions, one question is 0.05, so small differences are not significant.

## Extraction results

| | Passed validation | Failed validation | Not found in documents |
|---|---|---|---|
| Bid1 | 14 | 0 | 6 |
| Bid2 | 16 | 0 | 4 |

"Not found" is reported instead of guessing. For Bid1, Delivery Date, Payment Terms, Bid Bond, MFG, Model_no and Part_no are genuinely absent: the RFP asks vendors to propose their own make and model ("Proposed Device Make and Model") and gives no delivery or payment terms. Rules include exclusion patterns so that look-alike clauses (for example, prime-vendor payment to subcontractors) are not reported as payment terms.

Examples: Bid1 due date "July 9, 2024 at 2:00 PM CST", taken from Addendum 2 with the original bid-page date in the note; Bid2 delivery "Delivery within 45 days of Award", MFG "Dell", cooperative "Desktop, Laptop and Tablet 2015 Master Contract, 060B5400007", models "Latitude 5550; WD22TB4", specs "CPU: Intel Core Ultra 5 125U; RAM: 16 GB DDR5; Storage: 256 GB NVMe SSD; Display: 15.6\" FHD ...".

Validation of normalised fields (Term of Bid, Product Specification) checks each component value against the cited passage, since the combined string does not appear verbatim.

**Q&A answers.** The Q&A agent first answers from the validated extraction record when the question maps to a field (deadline, bond, affidavits, delivery, contact ...), citing the field's source, then adds the best-matching retrieved sentences. Open questions ("compare the warranty requirements") are answered from retrieved sentences only; with `ANTHROPIC_API_KEY` set, Claude writes the answer from the same passages.

## Known limitations

- Field extraction is rule-based; the LLM writes Q&A answers and the Bid Summary. LLM-based field extraction (Pydantic structured output behind the same validator) is the next step.
- The free Gemini tier rate-limits (429) and returns 503 under load; calls retry with backoff and then fall back. `logs/bid_qa_log.md` was generated with Gemini; `outputs/Bid2.json` has a template summary because the quota ran out during that run.
- Rule-based extraction will miss values phrased in ways the rules do not cover on an unseen bid; those fields come back as "Not found in documents" rather than wrong values.
- Bid1's RFP specifies several device tiers; Product Specification reports one tier.
- Tables are read as text in layout order; there is no cell-level table extraction. No OCR (scanned pages are logged and skipped).
- The 20 evaluation questions and their labels were written by the author.
- The Docker image and CI workflow were not run locally.
- `docs/addendum_prototype.md` documents the earlier Addendum-1-only prototype (the `ingest`, `ask`, `extract`, `eval` commands), which is still included.
