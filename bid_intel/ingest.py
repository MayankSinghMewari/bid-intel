"""Ingestion: PDF or text -> pages -> numbered Q&A items -> retrieval chunks."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

HEADER_RES = [
    re.compile(r"^\s*ADDENDUM No\. \d+\s*$"),
    re.compile(r"^\s*RFP JA-\d+.*$"),
    re.compile(r"^\s*Page \d+ \| \d+\s*$"),
]
STOP_RE = re.compile(r"^\s*(The information in this Addendum|END OF ADDENDUM)")
QUESTION_RE = re.compile(r"^\s{0,8}(\d{1,2})\.\s+(\S.*)$")
ANSWER_RE = re.compile(r"^\s*Answer:\s*(.*)$")
WS_RE = re.compile(r"\s+")


@dataclass
class QAItem:
    qnum: int
    page: int
    question: str
    answer: str

    @property
    def citation(self) -> str:
        return f"Addendum No. 1, Q{self.qnum}, p. {self.page}"


@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    text: str
    qnums: list[int]
    page: int


def _clean(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def read_pages(path: Path) -> list[str]:
    """Return the document as a list of page strings (pdftotext, pypdf fallback)."""
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8").split("\f")
    if shutil.which("pdftotext"):
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, check=True,
        ).stdout
    else:
        from pypdf import PdfReader
        out = "\f".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(str(path)).pages)
    return out.split("\f")


def parse_qa(pages: list[str]) -> list[QAItem]:
    """Parse numbered questions and 'Answer:' blocks. Question numbers must run 1, 2, 3 ...
    which stops numbered lines inside answers from being mistaken for new questions."""
    records: list[dict] = []
    current: dict | None = None
    expected = 1
    state = None
    done = False
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            if not line.strip() or any(r.match(line) for r in HEADER_RES):
                continue
            if STOP_RE.match(line):
                done = True
                break
            m = QUESTION_RE.match(line)
            if m and int(m.group(1)) == expected:
                if current:
                    records.append(current)
                current = {"qnum": expected, "page": page_no, "question": [m.group(2).strip()], "answer": []}
                expected += 1
                state = "q"
                continue
            a = ANSWER_RE.match(line)
            if a and current is not None:
                state = "a"
                current["answer"].append(a.group(1).strip())
                continue
            if current is None:
                continue
            (current["question"] if state == "q" else current["answer"]).append(line.strip())
        if done:
            break
    if current:
        records.append(current)
    return [
        QAItem(
            qnum=r["qnum"],
            page=r["page"],
            question=_clean(" ".join(r["question"])),
            answer=_clean(" ".join(x for x in r["answer"] if x)),
        )
        for r in records
    ]


def build_chunks(items: list[QAItem], chunking: str) -> list[Chunk]:
    """chunking = 'qa' (one chunk per Q&A pair) or 'window:<size>:<stride>' (word windows)."""
    if chunking == "qa":
        return [
            Chunk(f"qa-{it.qnum:02d}", "qa", f"{it.question} Answer: {it.answer}", [it.qnum], it.page)
            for it in items
        ]
    kind, size_s, stride_s = chunking.split(":")
    if kind != "window":
        raise ValueError(f"unknown chunking: {chunking}")
    size, stride = int(size_s), int(stride_s)
    words: list[tuple[str, int, int]] = []
    for it in items:
        for w in f"{it.question} Answer: {it.answer}".split():
            words.append((w, it.qnum, it.page))
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        seg = words[start:start + size]
        chunks.append(Chunk(
            chunk_id=f"win{size}-{len(chunks):03d}",
            strategy=chunking,
            text=" ".join(w for w, _, _ in seg),
            qnums=sorted({q for _, q, _ in seg}),
            page=seg[0][2],
        ))
        if start + size >= len(words):
            break
        start += stride
    return chunks


def ingest(source: Path, bid_id: str, bid_dir: Path) -> dict:
    """Parse a bid document and store its Q&A items under the bid folder."""
    pages = read_pages(source)
    items = parse_qa(pages)
    meta = {"bid_id": bid_id, "source": source.name, "pages": len(pages), "items": len(items)}
    bid_dir.mkdir(parents=True, exist_ok=True)
    (bid_dir / "items.json").write_text(
        json.dumps({"meta": meta, "items": [asdict(i) for i in items]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def load_items(bid_dir: Path) -> tuple[dict, list[QAItem]]:
    data = json.loads((bid_dir / "items.json").read_text(encoding="utf-8"))
    return data["meta"], [QAItem(**i) for i in data["items"]]
