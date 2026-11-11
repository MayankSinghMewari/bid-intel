# Bid Intelligence

Ingests public-sector bid documents, searches them with cited retrieval, and runs a multi-agent extraction that writes one JSON file per bid with values, citations and confidence scores.

Tested on Dallas ISD RFP JA-207652 Addendum No. 1 (39 vendor questions). This is the only document in the repository. Results on other bids are unknown.

## Contents
- `bid_intel/`: ingestion, BM25 search, schema, agents, service layer, CLI, API
- `tests/`: 17 tests (ingestion, retrieval, extraction, API)
- `eval/`: question set, retrieval evaluation report, configuration results
- `logs/`: sample Q&A log (12 questions), full agent trace for one extraction run
- `data/bids/JA-207652/`: source PDF, parsed items, extraction JSON
- `docs/architecture.svg`: architecture diagram (`architecture.md` has the Mermaid version)
- `DEMO_SCRIPT.md`: outline for the 5 to 10 minute walkthrough

## Setup

Requirements: Python 3.10+, and `pdftotext` (poppler) if available. Without it the code falls back to `pypdf`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
# 1. Ingest a bid document (writes data/bids/<BID>/items.json)
python -m bid_intel ingest data/bids/JA-207652/addendum_1.pdf --bid JA-207652

# 2. Search and get cited passages (CLI)
python -m bid_intel ask --bid JA-207652 "Is accidental damage protection required?"

# 3. Run the multi-agent extraction (writes data/bids/<BID>/extraction.json)
python -m bid_intel extract --bid JA-207652 --trace logs/agent_trace.json

# 4. Retrieval evaluation across six configurations (writes eval/report.md)
python -m bid_intel eval --bid JA-207652

# 5. HTTP API on http://127.0.0.1:8000 (interactive docs at /docs)
python -m bid_intel serve
curl -X POST localhost:8000/bids/JA-207652/ask -H 'content-type: application/json' \
     -d '{"question":"When will the RFP be awarded?"}'
curl -X POST localhost:8000/bids/JA-207652/extract
curl localhost:8000/bids/JA-207652/extraction

# Tests
pytest -q
```

## Modes
| Mode | Command | Output |
|---|---|---|
| Ingestion | `ingest` | `items.json` per bid folder |
| RAG search (CLI) | `ask` | cited passages, confidence |
| Multi-agent extraction | `extract` | `extraction.json` and an agent trace |
| Evaluation | `eval` | `eval/results.json`, `eval/report.md` |
| API | `serve` | JSON over HTTP |

## Design decisions

**Ingestion.** `pdftotext -layout` keeps the numbered list structure. The parser accepts a question number only when it is the next expected number (1, 2, 3 ...). That stops numbered lines inside answers, such as "5 business days", from being read as new questions. Header and footer lines are removed. The parser stops at the signature block.

**Chunking.** Three strategies are supported and evaluated:
- `qa` (default): one chunk per Q&A pair. This matches how the District answers, keeps each citation exact, and gives 39 chunks.
- `window:40:20` and `window:80:40`: overlapping word windows. Each chunk records every question number it spans, so citations stay correct.

I chose `qa` as the default because citations map directly to a question number. The evaluation shows the window chunking with k1=1.5, b=0.3 scores marginally higher (MRR 0.986 against 0.972), a difference of one question, so the choice is not conclusive.

**Embedding model.** None in this version. Retrieval is BM25, implemented in pure Python so it runs with no model download and no network. Dense embeddings are the natural next step for paraphrased questions. The addendum uses similar vocabulary to vendor questions, which is why lexical search works well here. This is a limitation for other bids, not a finding.

**Retrieval strategy.** BM25 with stopword removal. Negation words ("not", "no") and version numbers ("3.1") are kept, because they change the meaning of RFP answers. Recall@k and MRR use gold labels from `eval/questions.json`.

**Agent framework.** No framework. Four roles (Orchestrator, Retriever, Extractor, Verifier) pass messages on a shared in-memory bus, and every message is recorded in the trace. A framework like LangGraph or AutoGen would add dependencies without changing the behavior, and the roles are small enough to read in one file.

**Extraction rules.** Each field in `bid_intel/schema.py` has a retrieval query and a regular-expression rule with a named `value` group. Rules are deterministic, so extraction is reproducible and fully auditable. The cost is that a new bid with different wording needs new rules. An LLM extractor could replace the rule step, with the verifier unchanged.

**Verification.** The verifier re-checks each candidate with three tests: the cited span appears verbatim in the cited chunk, the cited question numbers exist in the document, and the field rule gives the same value on that chunk. Be clear about what this does: it confirms the citation and the rule are consistent. It does not check that the rule captures the right meaning. Semantic checking needs an LLM or a human.

**Confidence.** `0.4 x retrieval strength + 0.4 x evidence location + 0.2 x verified`. Evidence location is 1.0 if the value comes from the District's answer and 0.5 if it appears only in the vendor's question. Example: the charger wattage (65W) appears only in the vendor's question, so its confidence is 0.8, not 1.0. This is a documented heuristic, not a calibrated probability. The mean reported value (0.98) is high because the rules are strict on this document, not because the system is reliable on other bids.

**Prompts.** No LLM prompts exist in this version, so there are none to document. The rules in `schema.py` play that role.

## Known limitations and assumptions
- One document tested. The schema is written for this addendum's wording and will miss values phrased differently in other bids.
- Extraction rules are hand-written. Each new bid needs schema work or an LLM extractor.
- Ask mode is extractive. It returns the best passage with a citation and does not generate an answer.
- Confidence is a heuristic.
- Gold labels were written by the team that built the system.
- Dense retrieval, OCR of scanned PDFs, the comparison agent, the go/no-go recommendation, caching, and cost tracking are not implemented.
- The Docker image and CI workflow were written but not run in this environment because Docker is not available here.
- Field values in the example output come from a 2024 addendum. Dates and requirements in the current bid may differ.

## Repository layout
```
bid_intel/   package (ingest, bm25, retrieval, schema, agents, service, cli, api, evaluation)
tests/       pytest suite
eval/        questions.json, report.md, results.json
logs/        qa_log.md, agent_trace.json
data/bids/   per-bid folders
docs/        architecture diagram
```
