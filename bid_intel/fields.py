"""The 20 required bid fields. Each field declares how to retrieve evidence (query, preferred document
types, HTML fact keys) and a deterministic rule that pulls the value out of a retrieved passage.
Rules use generic RFP vocabulary, not wording from a specific bid."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE = rf"(?:{MONTHS}\.?\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{1,2}}-[A-Z]{{3}}-\d{{4}})"
TIME = r"(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)"
TZ = r"(?:[ECMP][SD]?T|Eastern|Central|Mountain|Pacific)"
DATETIME = rf"{DATE}(?:\s*(?:at|@|,)?\s*{TIME})?(?:\s*{TZ})?"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+(?=[A-Z0-9])", re.sub(r"\s+", " ", text)) if s.strip()]


def _first_sentence(pattern: str, flags=re.I, max_len=320):
    def rule(text: str) -> str | None:
        for s in _sentences(text):
            if re.search(pattern, s, flags):
                return s[:max_len]
        return None
    return rule


def _regex(pattern: str, group: int | str = 1, flags=re.I):
    def rule(text: str) -> str | None:
        m = re.search(pattern, re.sub(r"\s+", " ", text), flags)
        return m.group(group).strip(" .,;") if m else None
    return rule


def _any(*rules):
    def rule(text: str) -> str | None:
        for r in rules:
            v = r(text)
            if v:
                return v
        return None
    return rule


def _all_matches(patterns: list[str], sep="; "):
    def rule(text: str) -> str | None:
        flat = re.sub(r"\s+", " ", text)
        found: list[str] = []
        for p in patterns:
            for m in re.finditer(p, flat, re.I):
                v = m.group(1 if m.groups() else 0).strip(" .,;")
                if v and v.lower() not in [f.lower() for f in found]:
                    found.append(v)
        return sep.join(found) if found else None
    return rule


def _term(text: str) -> str | None:
    flat = re.sub(r"\s+", " ", text)
    m = re.search(r"Initial Term:?\s*(\d+)", flat, re.I)
    if m:
        renewals = re.findall(r"Renewal\s*\d+:?\s*(\d+)", flat, re.I)
        tail = f"; {len(renewals)} renewal option(s) of {', '.join(renewals)} year(s)" if renewals else ""
        return f"Initial term {m.group(1)} year(s){tail}"
    return _first_sentence(r"(initial|contract|agreement)\s+(term|period)[^.]{0,120}\b(year|month)s?\b"
                           r"|\b(year|month)s?\b[^.]{0,60}renewal")(text)


def _bond(text: str) -> str | None:
    for s in _sentences(text):
        if re.search(r"bid (bond|security|guarant)", s, re.I):
            if re.search(r"\bnot\b[^.]{0,30}required|no bid (bond|security)", s, re.I):
                return "Not required"
            amt = re.search(r"(\d+(?:\.\d+)?\s*%|\$\s?[\d,]+(?:\.\d{2})?)", s)
            if amt or re.search(r"\b(shall|must)\b[^.]{0,40}(submit|include|provide|furnish)", s, re.I):
                return s[:320]
    return None


DOC_PATTERNS = [
    r"\b((?:Contract|Mercury|Non-?Collusion|Bidder'?s?|Notarized)\s+Affidavit)",
    r"\b(Form\s+1295)", r"\b(W-?9)\b", r"\b(Certificate of (?:Insurance|Interested Parties))",
    r"\b(Conflict of Interest Questionnaire)", r"\b(Letter of Authori[sz]ation)",
    r"\b(MBE (?:Participation )?(?:Worksheet|forms?))", r"\b(Felony Conviction Notice)",
    r"\b(signed addend(?:um|a)(?: acknowledg\w+)?)", r"\b(acknowledge and return[^.]{0,20}addendum)",
    r"\b(Debarment[^.]{0,30}Certification)", r"\b(House Bill \d+[^.]{0,40}(?:verification|certification))",
]

SPEC_PATTERNS = [
    ("CPU", r"((?:Intel|AMD|Apple|MediaTek|Qualcomm)\b[^\n;]{0,45}?(?:processor|Processor|N\d{3}|Ryzen \d|A\d{2})[^\n;]{0,20})"),
    ("CPU", r"(Processor:?\s[^\n;]{3,60})"),
    ("RAM", r"(\d+\s?GB[^\n;]{0,25}?(?:DDR\d\w*|LPDDR\d\w*)(?:\s+Memory)?)"),
    ("Storage", r"(\d+\s?(?:GB|TB)[^\n;]{0,40}?(?:SSD|eMMC|NVMe|Storage))"),
    ("Display", r"(\d{2}(?:\.\d)?(?:\"|-inch| inch)[^\n;]{0,40})"),
    ("Battery", r"(\d-cell[^\n;]{0,20}Wh)"),
    ("Wireless", r"(Wi-?Fi\s?\d\w?[^\n;]{0,25})"),
    ("OS", r"(Windows \d+ (?:Pro|Home|Education)|Chrome ?OS|iPadOS)"),
    ("Warranty", r"(\d\s?(?:Year|Yr)s?[^\n;]{0,40}(?:Warranty|ProSupport|Support))"),
]
SKU_CODE = re.compile(r"\s+\d{3}-[A-Z0-9]{4}\b")


def _specs(text: str) -> str | None:
    found: dict[str, str] = {}
    for label, pat in SPEC_PATTERNS:
        if label in found:
            continue
        m = re.search(pat, text, re.I if label != "CPU" else 0)
        if m:
            found[label] = re.sub(r"\s+", " ", SKU_CODE.sub("", m.group(1))).strip(" ,.;")
    return "; ".join(f"{k}: {v}" for k, v in found.items()) if len(found) >= 2 else None


@dataclass
class FieldSpec:
    name: str
    group: str                       # field group: extraction specialists run groups in parallel
    query: str
    rule: Callable[[str], str | None]
    doc_types: list[str] | None = None
    fact_keys: list[str] = field(default_factory=list)   # HTML bid-page facts to use first
    multi: bool = False             # gather values from several passages (lists of documents)
    kind: str = "text"              # "date" fields get a format check from the validator
    keep_filter: bool = False       # keep the doc-type filter on retries (e.g. contract term is never in an affidavit)


FIELDS: list[FieldSpec] = [
    FieldSpec("Bid Number", "identity", "solicitation number RFP number bid number",
              _any(_regex(r"(?:RFP|IFB|RFQ|PORFP|Solicitation|Bid)\s*(?:No\.?|Number|#)?\s*:?\s*#?([A-Z]{1,5}-?\d{4,}[A-Z0-9-]*)"),
                   _regex(r"\b([A-Z]{2,4}-\d{5,})\b", flags=0)),
              fact_keys=["solicitation number"]),
    FieldSpec("Title", "identity", "title of the request for proposal",
              _regex(r"(?:RFP|IFB|Solicitation)\s+[A-Z]{1,5}-?\d{4,}\s+([A-Z][^\n.]{5,90})"),
              fact_keys=["title"]),
    FieldSpec("Due Date", "dates", "proposal due date and time closing date submission deadline",
              _regex(rf"(?:due|closing|deadline|submit(?:ted)?\s+no later than)[^.]{{0,60}}?({DATETIME})"),
              fact_keys=["closing date"], kind="date"),
    FieldSpec("Bid Submission Type", "dates", "how proposals are submitted electronically sealed portal email",
              _first_sentence(r"(accepted|submitted|submit|submission)[^.]{0,120}(electronic|e-?procurement|portal|"
                              r"sealed|online|eMMA|email|iSupplier|hand deliver)")),
    FieldSpec("Term of Bid", "commercial", "initial term of contract renewal options years",
              _term, doc_types=["rfp", "addendum", "bid_page"], keep_filter=True),
    FieldSpec("Pre Bid Meeting", "dates", "pre-bid pre-proposal meeting conference date mandatory",
              _first_sentence(r"pre-?\s?(bid|proposal)\s+(meeting|conference)[^.]{0,200}"),
              fact_keys=["prebid conference"]),
    FieldSpec("Installation", "product", "installation imaging deployment setup services required",
              _first_sentence(r"\b(installation|imaging|deployment|asset tagging|set-?up)\b")),
    FieldSpec("Bid Bond Requirement", "commercial", "bid bond bid security guarantee amount percent required",
              _bond),
    FieldSpec("Delivery Date", "dates", "delivery within days of award shipping lead time",
              _any(_regex(r"(deliver(?:y|ed)?\s+within\s+[\w\s()]{1,25}?days?\s+(?:of|after|from)\s+(?:award|receipt[\w\s]{0,20}|order))"),
                   _first_sentence(r"deliver(y|ed)?\b[^.]{0,40}\bwithin\b[^.]{0,30}\b(days|weeks)\b"))),
    FieldSpec("Payment Terms", "commercial", "payment terms net 30 invoice invoicing",
              _any(_regex(r"\b(net\s*\d{2})\b"),
                   _first_sentence(r"(payment|paid)[^.]{0,80}within[^.]{0,30}days|invoices?\b[^.]{0,80}"
                                   r"(submitted|emailed|sent|within)"))),
    FieldSpec("Any Additional Documentation Required", "commercial",
              "required forms affidavit certificate insurance W-9 Form 1295 submit with proposal",
              _all_matches(DOC_PATTERNS), multi=True),
    FieldSpec("MFG for Registration", "product", "authorized reseller manufacturer authorization registered OEM",
              _any(_regex(r"authori[sz]ed\s+(?:reseller|dealer|partner|distributor)\s+(?:for|of)\s+(?:the\s+)?([A-Z][\w&.-]+)"),
                   _regex(r"Manufacturer\s+Name:?\s+([A-Z][\w&.-]+)"))),
    FieldSpec("Contract or Cooperative to use", "commercial",
              "cooperative purchasing interlocal master contract vehicle piggyback",
              _any(_regex(r"((?:Desktop|Hardware)[^.]{0,60}Master Contract,?\s*\d{3}[A-Z0-9]+)"),
                   _regex(r"(member of (?:the )?[A-Z][^.]{5,90}(?:Cooperative|Consortium)[^.]{0,30})"),
                   _regex(r"((?:TIPS|BuyBoard|DIR-[A-Z]+-\d+|Sourcewell|OMNIA|NASPO)[^.]{0,60})", flags=0))),
    FieldSpec("Model_no", "product", "model number product model",
              _all_matches([r"\b((?:Latitude|Precision|OptiPlex|ThinkPad|ThinkCentre|EliteBook|ProBook|Chromebook|"
                            r"Surface Pro|Surface Laptop|MacBook (?:Air|Pro)|iPad)\s?[A-Z]?\d{2,4}[A-Za-z]?)\b",
                            r"Dock\s*[-–]\s*([A-Z]{2}\d{2}[A-Z]{2,3}\d?)\b"]),
              doc_types=["rfp", "specs", "bid_page"], multi=True),
    FieldSpec("Part_no", "product", "part number SKU item number",
              _all_matches([r"\b(?:SI#|SKU|Part\s*(?:No\.?|#|Number)|P/N)\s*:?\s*([A-Z0-9][A-Z0-9-]{3,})"]),
              doc_types=["rfp", "specs", "bid_page"], multi=True),
    FieldSpec("Product", "product", "product description quantity laptops desktops devices",
              _first_sentence(r"\b(laptops?|desktops?|tablets?|chromebooks?|monitors?|devices?|docks?)\b[^.]{0,200}"
                              r"(\bqty\b|quantit|\b\d+\b)"),
              fact_keys=["description"]),
    FieldSpec("contact_info", "identity", "procurement officer point of contact email phone",
              _all_matches([r"\b([\w.+-]+@[\w-]+(?:\.[\w-]+)+)", r"(\(?\d{3}\)?[-. ]\d{3}-\d{4})"]),
              fact_keys=["contact information"]),
    FieldSpec("company_name", "identity", "issuing organization agency district office",
              _regex(r"\b((?:[A-Z][\w']+\s){1,4}(?:Independent School District|School District|County|"
                     r"Treasurer's Office|Department of [A-Z]\w+|University))", flags=0),
              fact_keys=["issuing organization"]),
    FieldSpec("Bid Summary", "summary", "scope of work purpose of this solicitation", lambda t: None),
    FieldSpec("Product Specification", "product", "minimum specifications processor memory storage display warranty",
              _specs, doc_types=["specs", "rfp"]),
]
