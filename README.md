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

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, about 70 MB) and embeds every chunk into a persistent Chroma vector store at `.cache/chroma` (about a minute for Bid1's 63-page RFP on a 4-core CPU). Later runs only embed new or changed chunks.

LLM (the generation step of RAG): copy `.env.example` to `.env` and set `GEMINI_API_KEY` (free key from https://aistudio.google.com/apikey; model `GEMINI_MODEL`, default `gemini-3.5-flash`) or `ANTHROPIC_API_KEY`. `.env` is git-ignored. With a key, Q&A answers and the Bid Summary field are written by the LLM from retrieved passages / validated fields only, and every answer must cite passages as [n] or it is rejected. Without a key, or when the LLM call fails (timeout, 503, quota), the system falls back to the validated extraction record and cited sentences, and the trace records the reason. Tests run with the LLM disabled.

## Run with Docker (no Python setup)

The image is built, smoke-tested and published by CI on every push to `main`, with the search index and the
embedding model already inside it.

```bash
docker run -p 8000:8000 ghcr.io/mayanksinghmewari/bid-intel                     # REST API, docs at /docs
docker run -p 8501:8501 ghcr.io/mayanksinghmewari/bid-intel     streamlit run app.py --server.address 0.0.0.0                                 # web UI
docker run -p 8501:8501 -e BID_INTEL_DEMO=1 ghcr.io/mayanksinghmewari/bid-intel     streamlit run app.py --server.address 0.0.0.0                                 # read-only public demo
# optional: -e GEMINI_API_KEY=... for LLM answers, -e LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL for tracing
```

## How to run

```bash
# Web UI on http://localhost:8501 : Extraction (20 fields + evidence), Search, Ask (cited chat),
# Add bid (upload a new folder -> index + extract), Evaluation
streamlit run app.py
```

```bash
# Extraction mode: ONE COMMAND per bid folder -> outputs/<bid>.json and logs/<bid>_trace.json
python -m bid_intel run --bid data/bid_folders/Bid1
python -m bid_intel run --bid data/bid_folders/Bid2

# Search (hybrid, with metadata filters)
python -m bid_intel search "bid bond" --bid-id Bid2
python -m bid_intel search "due date" --bid-id Bid1 --doc-type addendum

# Question-answering mode (cited answers; names like "Bid1" in the question restrict the search)
python -m bid_intel ask "What is the submission deadline for Bid1 after all addendums?"

# Retrieval evaluation (writes eval/retrieval_report.md and eval/retrieval_results.json)
python -m bid_intel eval-retrieval

# Extraction accuracy vs hand-checked labels, rules only vs rules + LLM (eval/extraction_report.md)
python -m bid_intel eval-extraction                 # --modes rules  (no LLM calls)

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
| Retrieval evaluation report | `eval/retrieval_report.md` (questions: `eval/retrieval_questions.json`) |
| Sample Q&A log (12 questions) | `logs/qa_log.md` |
| Agent trace of a full extraction run | `logs/Bid1_trace.json` (also `logs/Bid2_trace.json`) |
| Web UI (bonus) | `app.py` (`streamlit run app.py`) |
| Architecture diagram | "Architecture" section below; agent graph generated from the code: `docs/agent_graph.md` |

## Architecture

Search engine and agents (the agent part is a LangGraph state graph; diagram generated from the code in [`docs/agent_graph.md`](docs/agent_graph.md) with `python -m bid_intel graph`):

```
Bid folder (HTML + PDF)
   |
 ingest      ingest.py: detect doc type, parse HTML (BeautifulSoup) / PDF (pdftotext), strip repeated headers,
   |         fix hyphenation, attach metadata, log unreadable/empty files; index into
   |         BM25 (in memory) + Chroma (bge-small embeddings + metadata, persistent, incremental)
 plan        orchestrator splits the 19 extracted fields into 4 groups: identity, dates, commercial, product
   |
   |  Send x4 (parallel)
 rule_extract   one specialist per group: RetrievalAgent (hybrid search, metadata filters) + field rules
   |
 validate    ValidatorAgent: citation present, value in cited evidence, date format
   |--- rejected fields, attempts left -> Send back to rule_extract with wider retrieval   (feedback loop)
   |--- LLM available -> Send x4 (parallel) llm_extract: structured output over retrieved evidence,
   |                     quote must be verbatim in a cited passage, rejected fields re-asked once
   |                     -> llm_merge (validated LLM value replaces the rule value)
   v
 reconcile   ReconcileAgent: addendum change log; latest addendum overrides (e.g. extended due date)
 summarize   Bid Summary (LLM from validated fields, template fallback)
 finalize    -> outputs/<bid>.json + logs/<bid>_trace.json
```

Graph state (`bid_intel/graph.py`, `GraphState`) is the run's working memory: plan, draft fields, attempt counts, validated rounds, retry list and LLM results; parallel branches return partial updates that reducers merge. The final record is the Pydantic `BidState`; agent messages are Pydantic models (`FieldResult`, `Source`, `TraceStep`, `FieldValue`). Every agent step is logged with input, output and latency (and tokens for LLM calls).

## Design decisions

**Ingestion.** HTML bid pages are parsed into label/value facts (solicitation number, title, closing date, contact, issuing organization), which are the most reliable source for those fields. PDFs are read page by page so every chunk keeps its page number. Lines repeated on more than half of a document's pages are removed as headers/footers. Document type comes from the file name (`addendum N`, `affidavit`, `spec`, `.html` = bid page, otherwise `rfp`). Files that fail to parse, or pages with no text (scanned), are recorded in the ingestion log and skipped instead of stopping the run.

**Chunking.** 180-word windows with 40-word overlap, never crossing a page boundary. RFPs mix short form fields, clauses and spec lists; 180 words holds one clause or a short table while keeping citations to a single page. The overlap keeps sentences that straddle a boundary retrievable.

**Embedding model.** `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU, no API key). Small enough to run on a laptop, and strong on retrieval benchmarks for its size.

**Vector store.** Chroma (`bid_intel/vectorstore.py`), persistent on disk, cosine HNSW index. Each vector is stored with its chunk's metadata (bid_id, file_name, doc_type, page, addendum_number), so metadata filters such as "only Bid2 specs" run inside the vector search. Indexing is incremental: before embedding, the store compares chunk ids and a hash of each chunk's text with what it already holds, embeds only new or changed chunks, and deletes chunks of a re-indexed bid that no longer exist. Re-running a bid embeds nothing (about 5 s, all of it PDF parsing). Embeddings are computed by our own model and passed to Chroma explicitly, so the embedding model is a single, configurable choice (`BID_INTEL_EMBED_MODEL`). BM25 stays in memory because its statistics are corpus-wide and rebuilding it is cheap. Migrating from the earlier in-memory NumPy index left every retrieval metric unchanged, which confirms the store returns the same rankings.

**Retrieval.** BM25 (pure Python, keeps negations and numbers such as model and part numbers) plus dense cosine search, merged with reciprocal rank fusion (k=60). Metadata filters restrict by bid, document type and addendum number. Query expansion adds RFP synonyms ("deadline" -> due date, closing date, submission). A cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) is available but off by default, because on this evaluation it lowered MRR and cost about 5 s per query.

**Agent framework.** LangGraph (`bid_intel/graph.py`). The workflow is a state graph: field groups fan out in parallel with `Send`, the validator's verdict is a conditional edge that loops rejected fields back to extraction (bounded by `MAX_RETRIES`), and the LLM stage is a conditional branch that is skipped when no LLM is available. The agents are plain Python classes the nodes call, so they are unit-tested without the graph, and the graph is tested end to end offline (rules path, and LLM path with a scripted fake LLM). The earlier hand-written orchestrator produced identical field values on both bids, which was checked when switching. LangGraph was chosen over CrewAI/AutoGen because this workflow is a fixed plan with explicit routing, not open-ended agent conversation; a graph makes the routing and the feedback loop visible and testable.

**Extraction.** Two layers of specialists per field group (identity, dates, commercial, product):

1. *Rule specialists* (`bid_intel/fields.py`): each field declares a retrieval query, preferred document types, bid-page fact keys and a deterministic rule with exclusion patterns. Free, fast, reproducible, and the fallback when no LLM is available.
2. *LLM specialists* (`bid_intel/llm_extract.py`): one structured-output call per group (4 calls per bid, not 20). Evidence is gathered with the search engine as a tool: passages the rule specialists already cited (shared state), the bid page, the top passages for each field's keyword query (suits BM25), its natural-language question (suits embeddings), and the best addendum passage (addenda override and clarify the RFP). The LLM must return a Pydantic schema per field: value or null, cited passage numbers, a verbatim quote, and a note. The validator rejects values whose quote does not appear in the cited passages (whitespace/quote/case normalised), invalid citations and malformed dates; rejected fields are sent back once with the reason, and only those fields are re-asked. A validated LLM value replaces the rule value (confidence 0.95 if both agree, 0.85 otherwise, and the note records the rule value when they differ); otherwise the rule value stays.

The LLM layer is what generalises to unseen bids: the rules were written while reading these two bids, so their accuracy here overstates how they would do on a new one.

**Addendum reconciliation.** Addendum sentences with change verbs (extend, revise, replace, new due date ...) are collected into `addendum_changes`. If an addendum states a new due date, the latest one overrides the base value and the note records the previous value.

**Validation and confidence.** The validator rejects values without a citation, values not found verbatim in the cited evidence, and malformed dates. Rejected fields are retried with a wider search. Confidence: 0.9 for bid-page facts, 0.55 + 0.35/rank for rule matches, 0.93 for addendum overrides, multiplied by 0.6 if validation fails. These are heuristics, not calibrated probabilities.

**Observability.** Every run writes a local JSON trace (`logs/<bid>_trace.json`). With Langfuse keys in `.env` (free cloud account; see `.env.example`), runs are also traced to Langfuse (`bid_intel/observability.py`), instrumented following the official [Langfuse skill](https://github.com/langfuse/skills) and audited against the [tracing best practices](https://langfuse.com/docs/observability/best-practices) on real traces fetched back with `langfuse-cli`:

- One trace per unit of work: `extract-bid` (tag `extraction`) and `answer-question` (tag `qa`). Names are stable and verb-first; the bid id and options are metadata. Environment (`BID_INTEL_ENV`, default `development`) and release (git commit) are set on every trace.
- Each graph node is a typed observation and the agent work nests under it: `extract-<group>-fields` (agent) > `search-evidence` (retriever) + `apply-field-rules` (tool); `validate-fields` > `validate-field` (evaluator, rejections at WARNING); `llm-extract-<group>-fields` (agent) > `gather-evidence` (retriever) + `extract-fields` (generation) + `check-quote` (evaluator); `reconcile-addenda` and `summarize-bid` (chain) > `generate-text` (generation).
- Generations record the model, the prompt as role-labelled messages, the output, and token usage including thinking tokens; failed calls (timeouts, 429 quota) are marked ERROR with the reason.
- Orchestrator decisions (plan, retry, route, merge) are events; agent work is not duplicated as events.
- The LangChain callback handler was tried first and replaced: it typed every node as a generic chain, added routing internals as spans, and could not nest the agents' work under the nodes.

`python -m bid_intel run` prints the trace link; `python -m bid_intel tracing-check` verifies keys, environment and release. The CLI calls `shutdown()` at exit; nothing flushes per request. Without keys tracing is a no-op, tests always run with it off, and tracing errors never stop a run. Bid documents are public procurement records, so no masking is applied; API keys are never part of traced inputs.

**Web UI.** `app.py` (Streamlit) calls the same service layer as the CLI and the REST API. The Extraction tab shows the 20 fields with confidence, validation status, method (rules, LLM, addendum) and the cited evidence per field; Search exposes hybrid search with bid and document-type filters, mode and re-ranking; Ask is a chat whose answers cite passages and link the Langfuse trace; Add bid uploads a new folder (file names reduced to their basename), indexes it incrementally and runs the agents; Evaluation shows the retrieval and extraction results. `.streamlit/config.toml` binds the server to localhost because the app has no login and its chat spends the LLM quota. A headless test (`tests/test_app.py`, Streamlit AppTest) renders every tab and checks search and chat results offline.

**CI/CD.** `.github/workflows/ci.yml` has two jobs. `test` installs the dependencies, runs the offline test suite and the retrieval evaluation, and uploads the report. `docker` runs only if the tests pass: it builds the image, starts the container, calls `/health`, a filtered search and a question, checks that the container runs as a non-root user, and only then publishes `ghcr.io/mayanksinghmewari/bid-intel:latest` and `:<commit>` (pushes to `main` only; pull requests build and test without publishing). `.dockerignore` keeps `.env`, `.venv`, caches and `.git` out of the image. The index is built at image build time, so containers answer immediately. `BID_INTEL_DEMO=1` turns the UI into a safe public demo: LLM and tracing forced off, uploads and re-runs hidden.

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

**Extraction accuracy** (`eval/extraction_report.md`, labels in `eval/extraction_gold.json`, checked against the documents; ambiguous fields are marked skip with the reason): rules only score 35/36 scored fields (97%). The only miss is Bid1 Delivery Date, which is answered in Addendum 1 Q4 ("quotes from awarded vendors starting in September 2024"). The report counts how many fields actually came from the LLM in each run, so a run where the LLM was unavailable cannot be mistaken for LLM accuracy. During development the LLM values were qualitatively richer (submission "iSupplier portal (preferred) or sealed envelope", pre-bid meeting "non-mandatory", product quantities) and evaluation exposed three issues that were fixed generally: missing spec evidence (now shared from the rule specialists), model vs SKU confusion (clearer field descriptions), and keyword-only retrieval missing Q&A addenda (natural-language and addendum-restricted queries). The free Gemini quota ran out before a clean LLM run; re-run `eval-extraction` to add the LLM row.

Validation of normalised fields (Term of Bid, Product Specification) checks each component value against the cited passage, since the combined string does not appear verbatim.

**Q&A answers.** The Q&A agent first answers from the validated extraction record when the question maps to a field (deadline, bond, affidavits, delivery, contact ...), citing the field's source, then adds the best-matching retrieved sentences. Open questions ("compare the warranty requirements") are answered from retrieved sentences only; with `ANTHROPIC_API_KEY` set, Claude writes the answer from the same passages.

## Repository layout

```
app.py                 Streamlit UI
bid_intel/
  ingest.py            bid-folder ingestion: HTML/PDF parsing, cleaning, metadata, chunking
  bm25.py              lexical index
  embeddings.py        bge-small embeddings (fastembed)
  vectorstore.py       Chroma vector store: metadata filters, incremental upsert
  search.py            hybrid search: BM25 + dense (RRF), filters, query expansion, re-ranking
  fields.py            the 20 fields: queries, rules, descriptions
  agents.py            agents (retrieval, rule extraction, validator, reconciler), Orchestrator, shared state
  llm_extract.py       LLM extraction specialists: structured output, quote validation, feedback loop
  graph.py             LangGraph workflow wiring the agents
  llm.py               LLM providers (Gemini / Anthropic), retries, circuit breaker, cited answers
  observability.py     Langfuse tracing
  service.py           service layer used by the CLI, API and UI: index, search, ask, extract
  cli.py, api.py       command line and REST API
  eval_retrieval.py    Recall@k / MRR across search configurations
  eval_extraction.py   field accuracy vs hand-checked labels
data/bid_folders/      one folder per bid (Bid1, Bid2)
outputs/               one JSON record per bid
eval/                  question sets, gold labels, reports
logs/                  agent traces, sample Q&A log
docs/agent_graph.md    agent graph (generated)
tests/                 offline test suite
```

## Known limitations

- LLM extraction is implemented and tested offline (validator and feedback loop), but the free-tier quota ran out before a clean accuracy run with the LLM; the committed outputs are rule-based plus Gemini summaries.
- After a quota error the LLM is switched off for 10 minutes (circuit breaker) and every step falls back to rules or extractive answers; the trace records why.
- The free Gemini tier rate-limits (429) and returns 503 under load; calls retry with backoff and then fall back. `logs/qa_log.md` and both bid summaries were generated with Gemini.
- Rule-based extraction will miss values phrased in ways the rules do not cover on an unseen bid; those fields come back as "Not found in documents" rather than wrong values.
- Bid1's RFP specifies several device tiers; Product Specification reports one tier.
- Tables are read as text in layout order; there is no cell-level table extraction. No OCR (scanned pages are logged and skipped).
- The 20 evaluation questions and their labels were written by the author.
