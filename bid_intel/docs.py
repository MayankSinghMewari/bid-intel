"""Generic bid-folder ingestion: HTML bid pages and PDFs (RFP, addendums, specs, affidavits).

Every page becomes one or more chunks carrying metadata:
bid_id, file_name, doc_type, addendum_number, page, chunk_id.
Nothing here is specific to a particular bid, so a new folder works without code changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("bid_intel.docs")

CHUNK_WORDS = 180   # roughly one clause or one short table; small enough for precise citations
OVERLAP_WORDS = 40  # keeps sentences that straddle a boundary retrievable from both chunks


@dataclass
class DocChunk:
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: str
    page: int
    text: str
    addendum_number: int | None = None
    facts: dict = field(default_factory=dict)  # key/value pairs parsed from HTML bid tables


def doc_type_of(name: str) -> tuple[str, int | None]:
    n = name.lower()
    m = re.search(r"addend(?:um|a)\s*(?:no\.?\s*)?(\d+)", n)
    if m:
        return "addendum", int(m.group(1))
    if n.endswith((".html", ".htm")):
        return "bid_page", None
    if "affidavit" in n or "certif" in n or "form" in n:
        return "affidavit", None
    if "spec" in n:
        return "specs", None
    return "rfp", None


def _clean(text: str) -> str:
    text = text.replace("​", " ").replace("\xad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _pdf_pages(path: Path) -> list[str]:
    if shutil.which("pdftotext"):
        out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True)
        if out.returncode == 0:
            return out.stdout.decode("utf-8", errors="ignore").split("\f")
    from pypdf import PdfReader
    return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop header/footer lines that repeat on more than half of the pages."""
    if len(pages) < 4:
        return pages
    counts: dict[str, int] = {}
    for p in pages:
        for line in {l.strip() for l in p.splitlines() if l.strip()}:
            counts[line] = counts.get(line, 0) + 1
    repeated = {l for l, c in counts.items() if c > len(pages) / 2 and len(l) < 120}
    return ["\n".join(l for l in p.splitlines() if l.strip() not in repeated) for p in pages]


def _html_page(path: Path) -> tuple[str, dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        t.decompose()
    lines = [re.sub(r"\s+", " ", s).strip() for s in soup.get_text("\n").splitlines()]
    lines = [l for l in lines if l]
    # Bid portals render "label \n value" pairs; keep the block between Basic Information and Disclaimer
    start = next((i for i, l in enumerate(lines) if l.lower() == "basic information"), 0)
    end = next((i for i, l in enumerate(lines) if l.lower().startswith("disclaimer")), len(lines))
    body = lines[start:end]
    labels = {"reference number", "issuing organization", "solicitation type", "solicitation number",
              "title", "location", "purchase type", "piggyback contract", "publication", "closing date",
              "prebid conference", "questions due by", "description", "attendance", "event date",
              "event type", "source id"}
    facts: dict[str, str] = {}
    for i, l in enumerate(body[:-1]):
        if l.lower() in labels and body[i + 1].lower() not in labels and l.lower() not in facts:
            facts[l.lower()] = body[i + 1]
    if "contact information" in [l.lower() for l in body]:
        j = [l.lower() for l in body].index("contact information")
        facts["contact information"] = "; ".join(body[j + 1:j + 4])
    text = "\n".join(f"{l}" for l in body)
    return text, facts


def _windows(words: list[str]) -> list[str]:
    if len(words) <= CHUNK_WORDS:
        return [" ".join(words)] if words else []
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step)]


def ingest_folder(folder: Path, bid_id: str | None = None) -> tuple[list[DocChunk], list[dict]]:
    """Parse every supported file in a bid folder. Returns (chunks, ingestion log)."""
    bid_id = bid_id or folder.name
    chunks: list[DocChunk] = []
    report: list[dict] = []
    for path in sorted(folder.iterdir()):
        if path.is_dir():
            continue
        doc_type, add_no = doc_type_of(path.name)
        entry = {"file": path.name, "doc_type": doc_type, "addendum_number": add_no}
        try:
            if path.suffix.lower() in (".html", ".htm"):
                text, facts = _html_page(path)
                pages, page_facts = [text], facts
            elif path.suffix.lower() == ".pdf":
                pages, page_facts = _strip_repeated_lines(_pdf_pages(path)), {}
            else:
                entry["status"] = "skipped: unsupported type"
                report.append(entry)
                continue
        except Exception as exc:  # unreadable file: log it and keep going
            entry["status"] = f"error: {type(exc).__name__}: {exc}"
            log.warning("failed to parse %s: %s", path.name, exc)
            report.append(entry)
            continue
        empty = 0
        n_before = len(chunks)
        for pno, raw in enumerate(pages, start=1):
            text = _clean(raw)
            if len(text.split()) < 5:
                empty += 1
                continue
            for j, win in enumerate(_windows(text.split())):
                cid = f"{bid_id}:{path.stem[:30]}-{hashlib.sha1(path.name.encode()).hexdigest()[:6]}:p{pno}:{j}"
                chunks.append(DocChunk(cid, bid_id, path.name, doc_type,
                                       pno, win, add_no, page_facts if j == 0 and pno == 1 else {}))
        entry.update(pages=len(pages), empty_pages=empty, chunks=len(chunks) - n_before,
                     status="ok" if len(chunks) > n_before else "warning: no text (scanned? OCR not enabled)")
        report.append(entry)
    return chunks, report


def save_index(chunks: list[DocChunk], report: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    (out_dir / "ingestion_log.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_index(out_dir: Path) -> list[DocChunk]:
    return [DocChunk(**d) for d in json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))]
