"""Retrieval evaluation over the bid folders: Recall@k and MRR for several search configurations.

Gold labels are (file, page) pairs. Each question names a file fragment and an anchor phrase; the gold
pages are every page of that file whose text contains the anchor. A retrieved chunk is relevant when its
file and page match a gold pair, which keeps labels independent of the chunk size being tested."""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

from .ingest import ingest_folder
from .search import BidSearch

CONFIGS = [
    {"name": "bm25", "mode": "bm25", "expand": False, "rerank": False},
    {"name": "bm25 + query expansion", "mode": "bm25", "expand": True, "rerank": False},
    {"name": "dense (bge-small)", "mode": "dense", "expand": False, "rerank": False},
    {"name": "hybrid RRF", "mode": "hybrid", "expand": True, "rerank": False},
    {"name": "hybrid RRF + cross-encoder rerank", "mode": "hybrid", "expand": True, "rerank": True},
]


def gold_pages(chunks, q) -> set[tuple[str, int]]:
    anchor = q["anchor"].lower()
    return {(c.file_name, c.page) for c in chunks
            if q["bid_id"] == c.bid_id and q["file"].lower() in c.file_name.lower()
            and anchor in " ".join(c.text.split()).lower()}


def run(folders: list[Path], questions_path: Path, out_dir: Path, ks=(1, 3, 5)) -> dict:
    chunks = []
    for f in folders:
        chunks += ingest_folder(f)[0]
    engine = BidSearch(chunks)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    golds = {q["id"]: gold_pages(chunks, q) for q in questions}
    unlabeled = [qid for qid, g in golds.items() if not g]
    rows = []
    for cfg in CONFIGS:
        rec = {k: [] for k in ks}
        rr, t0 = [], time.perf_counter()
        for q in questions:
            gold = golds[q["id"]]
            if not gold:
                continue
            hits = engine.search(q["question"], top_k=max(ks), bid_id=q["bid_id"], mode=cfg["mode"],
                                 expand=cfg["expand"], rerank=cfg["rerank"])
            ranks = [h.rank for h in hits if (h.chunk.file_name, h.chunk.page) in gold]
            for k in ks:
                rec[k].append(1.0 if ranks and ranks[0] <= k else 0.0)
            rr.append(1.0 / ranks[0] if ranks else 0.0)
        n = len(rr)
        rows.append({"config": cfg["name"], **{f"Recall@{k}": round(mean(rec[k]), 3) for k in ks},
                     "MRR": round(mean(rr), 3), "questions": n,
                     "ms_per_query": round((time.perf_counter() - t0) * 1000 / max(n, 1), 1)})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_results.json").write_text(json.dumps({"configs": rows, "unlabeled": unlabeled}, indent=2))
    n1 = sum(q["bid_id"] == "Bid1" for q in questions)
    lines = ["# Retrieval Evaluation: Bid1 + Bid2", "",
             f"{len(questions)} questions ({n1} Bid1, {len(questions) - n1} Bid2) in `eval/retrieval_questions.json`. "
             "A hit is a retrieved chunk from the gold file and page. Each search is filtered to the question's bid.",
             "", "Recall@k = share of questions with at least one gold passage in the top k. "
             "MRR = mean reciprocal rank of the first gold passage.", "",
             "| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | ms/query |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['Recall@1']} | {r['Recall@3']} | {r['Recall@5']} | "
                     f"{r['MRR']} | {r['ms_per_query']} |")
    if unlabeled:
        lines += ["", f"Questions skipped because their anchor was not found: {unlabeled}"]
    lines += ["", "Limitations: 20 questions written by the system's author. With this sample size a difference "
              "of one question (0.05) is not significant."]
    (out_dir / "retrieval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"configs": rows, "unlabeled": unlabeled}
