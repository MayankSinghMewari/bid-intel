"""LLM extraction specialists: one structured-output call per field group, over retrieved evidence only.

Flow for one group (identity, dates, commercial, product):
  1. Evidence: for each field, the search engine (the agents' tool) returns its top passages for the
     field's query, filtered to the bid; the bid page is always included. Passages are numbered [n].
  2. Extraction: the LLM returns a Pydantic-validated JSON object: per field a value (or null), the
     passage numbers it relies on, and a verbatim quote from one of those passages.
  3. Validation: the quote must appear in a cited passage (normalised whitespace, quotes and case),
     citations must exist, dates must look like dates. Rejected fields are sent back to the LLM once
     with the validator's reason (feedback loop). Values that still fail are discarded.
"""
from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from . import llm
from .fields import FieldSpec
from .search import BidSearch, SearchHit

PASSAGES_PER_FIELD = 3   # per keyword query
QUESTION_PASSAGES = 2    # per natural-language question
MAX_PASSAGES = 22        # per group call; about 22 x 180 words keeps prompts near 7k tokens

# Natural-language form of each field. Field queries are keyword bags that suit BM25; dense retrieval
# matches questions better (and vendor Q&A addenda are written as questions).
QUESTIONS = {
    "Bid Number": "What is the solicitation or RFP number?",
    "Title": "What is the title of this bid?",
    "Due Date": "When are proposals due, and was the due date changed?",
    "Bid Submission Type": "How must proposals be submitted?",
    "Term of Bid": "How long is the contract term and what renewal options exist?",
    "Pre Bid Meeting": "Is there a pre-bid or pre-proposal meeting, and when?",
    "Installation": "Are installation, imaging or deployment services required?",
    "Bid Bond Requirement": "Is a bid bond or bid security required?",
    "Delivery Date": "When do the products need to be delivered?",
    "Payment Terms": "What are the payment terms and how should invoices be submitted?",
    "Any Additional Documentation Required": "Which forms, affidavits or certificates must be submitted?",
    "MFG for Registration": "Must the bidder be an authorized reseller of a specific manufacturer?",
    "Contract or Cooperative to use": "Which cooperative or master contract applies?",
    "Model_no": "Which product models are requested?",
    "Part_no": "What are the part or SKU numbers?",
    "Product": "Which products and quantities are requested?",
    "contact_info": "Who is the procurement contact?",
    "company_name": "Which organization issued this bid?",
    "Product Specification": "What are the minimum technical specifications of the devices?",
}

# What each field means (assignment section 8), given to the LLM verbatim
DESCRIPTIONS = {
    "Bid Number": "Official solicitation / RFP identifier.",
    "Title": "Full title of the bid or RFP.",
    "Due Date": "Final submission deadline including time and time zone, after any addendum changes.",
    "Bid Submission Type": "How the bid is submitted (electronic portal, sealed paper, email, etc.).",
    "Term of Bid": "Contract duration and any renewal options.",
    "Pre Bid Meeting": "Date, time, location/format, and whether attendance is mandatory; null if none.",
    "Installation": "Whether installation / deployment / imaging services are required, and what they include.",
    "Bid Bond Requirement": "Required bid bond or bid security with amount or percentage, 'Not required' if the "
                            "documents say so, otherwise null.",
    "Delivery Date": "Required delivery date, delivery window after award or order, or when deliveries are "
                     "expected to start (including answers in addendum Q&A).",
    "Payment Terms": "Payment terms to the vendor (e.g. Net 30) and invoicing conditions. Not payments a prime "
                     "vendor makes to its subcontractors.",
    "Any Additional Documentation Required": "Affidavits, forms, certificates, insurance, W-9, etc. that must be "
                                             "submitted. Semicolon-separated list.",
    "MFG for Registration": "Manufacturer(s) the bidder must be registered or authorized with; null if no "
                            "specific manufacturer is named.",
    "Contract or Cooperative to use": "Referenced cooperative or state contract vehicle, if any.",
    "Model_no": "Product model name/number(s), e.g. 'Latitude 5550', 'WD22TB4' (the model, not a SKU, SI# or "
                "configuration code); null if the issuer does not specify models.",
    "Part_no": "Part / SKU / item numbers (e.g. SI#, SKU, P/N codes) of the requested products.",
    "Product": "Product name(s) and quantity.",
    "contact_info": "Procurement contact: name, email, phone.",
    "company_name": "Issuing organization / agency name.",
    "Product Specification": "Key technical specifications: processor, memory (RAM), storage, display, operating "
                             "system, warranty. If several device tiers are specified, give the main values and say "
                             "they vary by tier. General rules such as 'must be new' are not specifications.",
}

SYSTEM = """You are an extraction specialist for public-sector bid documents (RFPs, addendums, bid portal pages, \
spec sheets, affidavits). Extract the requested fields using ONLY the numbered passages.

Rules:
- If a field is not stated in the passages, set value to null. Never guess or use outside knowledge.
- citations: the passage numbers that support the value. Every non-null value needs at least one.
- quote: copy one short span (at most 200 characters) EXACTLY, character for character, from a cited passage. \
It must contain the key fact of the value.
- An addendum overrides the original RFP. If an addendum changes a value, use the addendum value and \
mention the change in note.
- Dates: include the time and time zone when stated.
- Lists: separate items with "; ".
- Keep values concise (under 60 words)."""


class FieldValue(BaseModel):
    name: str = Field(description="field name exactly as requested")
    value: str | None = Field(description="extracted value, or null if not in the passages")
    citations: list[int] = Field(description="supporting passage numbers")
    quote: str | None = Field(description="exact span copied from a cited passage")
    note: str | None = Field(default=None, description="optional: addendum change or ambiguity")


class GroupExtraction(BaseModel):
    fields: list[FieldValue]


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def gather_evidence(engine: BidSearch, bid_id: str, specs: list[FieldSpec],
                    seed_chunk_ids: list[str] | None = None) -> list[SearchHit]:
    """Retrieval for a group: passages the rule specialists already cited (shared state), the bid page,
    then the top passages per field from the search engine."""
    seen: dict[str, SearchHit] = {}
    for cid in seed_chunk_ids or []:
        i = engine._index.get(cid)
        if i is not None:
            seen.setdefault(cid, SearchHit(engine.chunks[i], 1.0, 0))
    for c in engine.chunks:
        if c.bid_id == bid_id and c.doc_type == "bid_page" and c.chunk_id not in seen:
            seen[c.chunk_id] = SearchHit(c, 1.0, 0)
    has_addenda = any(c.bid_id == bid_id and c.doc_type == "addendum" for c in engine.chunks)
    for spec in specs:
        q = QUESTIONS.get(spec.name, spec.query)
        searches = [engine.search(spec.query, top_k=PASSAGES_PER_FIELD, bid_id=bid_id),
                    engine.search(q, top_k=QUESTION_PASSAGES, bid_id=bid_id, expand=False)]
        if has_addenda:  # addenda override and clarify the RFP, so always look there too
            searches.append(engine.search(q, top_k=1, bid_id=bid_id, doc_types=["addendum"], expand=False))
        for hits in searches:
            for h in hits:
                seen.setdefault(h.chunk.chunk_id, h)
    return list(seen.values())[:MAX_PASSAGES]


def _prompt(specs: list[FieldSpec], passages: list[SearchHit], feedback: dict[str, str] | None = None) -> str:
    blocks = [f"[{i}] (file: {h.chunk.file_name}, page {h.chunk.page}, type: {h.chunk.doc_type}"
              + (f", addendum {h.chunk.addendum_number}" if h.chunk.addendum_number else "") + f")\n{h.chunk.text}"
              for i, h in enumerate(passages, 1)]
    fields = "\n".join(f"- {s.name}: {DESCRIPTIONS.get(s.name, '')}" for s in specs)
    text = "PASSAGES\n\n" + "\n\n".join(blocks) + f"\n\nFIELDS TO EXTRACT\n{fields}"
    if feedback:
        text += ("\n\nYOUR PREVIOUS ANSWER FOR THESE FIELDS WAS REJECTED BY THE VALIDATOR. Fix it: copy the quote "
                 "exactly from a cited passage, or set value to null if the passages do not state it.\n"
                 + "\n".join(f"- {k}: {v}" for k, v in feedback.items()))
    return text


def validate(fv: FieldValue, spec: FieldSpec, passages: list[SearchHit]) -> tuple[bool, str]:
    if fv.value is None:
        return True, "not stated"
    if not fv.citations:
        return False, "value has no citation"
    bad = [n for n in fv.citations if not 1 <= n <= len(passages)]
    if bad:
        return False, f"cited passage numbers do not exist: {bad}"
    if not fv.quote or len(fv.quote.strip()) < 3:
        return False, "missing supporting quote"
    cited_text = " ".join(_norm(passages[n - 1].chunk.text) for n in fv.citations)
    if _norm(fv.quote) not in cited_text:
        return False, "quote not found verbatim in the cited passages (possible hallucination)"
    if spec.kind == "date" and not re.search(r"\d{4}|\d{1,2}/\d{1,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec",
                                             fv.value, re.I):
        return False, f"not a date: {fv.value!r}"
    return True, "ok"


def extract_group(engine: BidSearch, bid_id: str, group: str, specs: list[FieldSpec], log,
                  seed_chunk_ids: list[str] | None = None) -> dict:
    """Returns {field name: (FieldValue | None, passages, validated, reason, attempts)}; never raises."""
    from . import observability
    t0 = time.perf_counter()
    with observability.span("gather-evidence", "retriever",
                            input={"fields": [s.name for s in specs], "seed_chunks": len(seed_chunk_ids or [])}) as sp:
        passages = gather_evidence(engine, bid_id, specs, seed_chunk_ids)
        observability.set_output(sp, [f"[{i}] {h.chunk.file_name} p.{h.chunk.page} ({h.chunk.doc_type})"
                                      for i, h in enumerate(passages, 1)])
    log("retriever", "gather_evidence", {"group": group, "fields": [s.name for s in specs]},
        {"passages": [h.chunk.chunk_id for h in passages]}, t0)
    by_name = {s.name: s for s in specs}
    results: dict[str, tuple] = {}
    pending, feedback = list(specs), None
    for attempt in (1, 2):
        t1 = time.perf_counter()
        try:
            out = llm._complete(SYSTEM, _prompt(pending, passages, feedback), schema=GroupExtraction,
                                timeout_s=90)
        except Exception as exc:  # timeout, quota, invalid JSON: keep earlier results, rules fill the rest
            log(f"llm_extractor:{group}", "extract_error", {"attempt": attempt},
                {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}, t1)
            break
        got = {fv.name: fv for fv in out["parsed"].fields if fv.name in by_name}
        feedback = {}
        for spec in pending:
            fv = got.get(spec.name)
            with observability.span("check-quote", "evaluator",
                                    input={"field": spec.name, "value": fv.value if fv else None,
                                           "citations": fv.citations if fv else [],
                                           "quote": fv.quote if fv else None}) as ev:
                if fv is None:
                    ok, reason = False, "field missing from the LLM output"
                else:
                    ok, reason = validate(fv, spec, passages)
                observability.set_output(ev, {"passed": ok, "reason": reason}, level=None if ok else "WARNING")
            results[spec.name] = (fv, passages, ok, reason, attempt)
            if not ok:
                feedback[spec.name] = reason
        log(f"llm_extractor:{group}", "extract", {"attempt": attempt, "fields": [s.name for s in pending]},
            {"model": out["model"], "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
             "thinking_tokens": out.get("thinking_tokens"),
             "values": {k: (v[0].value if v[0] else None) for k, v in results.items() if k in got},
             "rejected": feedback}, t1)
        pending = [by_name[n] for n in feedback]
        if not pending:
            break
    return results
