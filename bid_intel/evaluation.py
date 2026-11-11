"""Retrieval evaluation across chunking and BM25 configurations, plus report writers."""
from __future__ import annotations

import json
from pathlib import Path

from . import service
from .ingest import build_chunks
from .retrieval import SearchEngine, evaluate

CHUNKINGS = ["qa", "window:40:20", "window:80:40"]
BM25_SETTINGS = [(1.2, 0.75), (1.5, 0.3)]


def run_evaluation(bid_id: str, questions_path: Path, out_dir: Path) -> dict:
    _, items = service.load_items(service.bid_dir(bid_id))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    rows = []
    for chunking in CHUNKINGS:
        for k1, b in BM25_SETTINGS:
            engine = SearchEngine(build_chunks(items, chunking), k1=k1, b=b)
            metrics = evaluate(engine, questions)
            rows.append({"chunking": chunking, "k1": k1, "b": b, "chunks": len(engine.chunks), **metrics})
    best = max(rows, key=lambda r: (r["MRR"], r["Recall@3"]))

    # Per-question misses for the best configuration (gold chunk not in top 3)
    best_engine = SearchEngine(build_chunks(items, best["chunking"]), k1=best["k1"], b=best["b"])
    misses = []
    for q in questions:
        hits = best_engine.search(q["question"], k=3)
        gold = set(q["gold"])
        if not any(set(h.chunk.qnums) & gold for h in hits):
            top = hits[0].chunk.qnums if hits else []
            misses.append({"id": q["id"], "question": q["question"], "gold": q["gold"], "top1_qnums": top})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"configs": rows, "best": best, "misses": misses}, indent=2))
    (out_dir / "report.md").write_text(_report(bid_id, questions, rows, best, misses, len(items)), encoding="utf-8")
    return {"best": best, "configs_tested": len(rows), "questions": len(questions)}


def _report(bid_id, questions, rows, best, misses, n_items) -> str:
    lines = [
        f"# Retrieval Evaluation: {bid_id}",
        "",
        f"Question set: {len(questions)} natural-language questions, hand-labelled with the Addendum "
        "question numbers that answer them (`eval/questions.json`).",
        "",
        "Metrics:",
        "- **Recall@k**: fraction of gold question numbers covered by the top k returned chunks, averaged over questions.",
        "- **MRR**: mean reciprocal rank of the first returned chunk containing a gold question.",
        "",
        "## Configurations tested",
        "",
        "| Chunking | k1 | b | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['chunking']} | {r['k1']} | {r['b']} | {r['chunks']} | "
                     f"{r['Recall@1']:.3f} | {r['Recall@3']:.3f} | {r['Recall@5']:.3f} | {r['MRR']:.3f} |")
    lines += [
        "",
        f"**Best by MRR:** chunking `{best['chunking']}`, k1={best['k1']}, b={best['b']} "
        f"(MRR {best['MRR']:.3f}, Recall@3 {best['Recall@3']:.3f}).",
        "",
        "## Misses for the best configuration (gold not in top 3)",
        "",
    ]
    if misses:
        lines += ["| Question | Gold | Top-1 retrieved |", "|---|---|---|"]
        for m in misses:
            lines.append(f"| {m['question']} | {m['gold']} | {m['top1_qnums']} |")
    else:
        lines.append("None.")
    lines += [
        "",
        "## Limitations",
        f"- One document ({n_items} question/answer items) and {len(questions)} questions. Results are indicative, not statistically robust.",
        "- Gold labels were written by the same team that built the system, so they may favour the chunking unit.",
        "- Retrieval is lexical (BM25). Dense embeddings were not evaluated in this version.",
        "- The best and default configurations differ by one question, so the choice of chunking is not conclusive.",
    ]
    return "\n".join(lines) + "\n"


def write_qa_log(bid_id: str, questions: list[dict], path: Path, n: int = 12) -> None:
    lines = [f"# Sample Q&A Log: {bid_id}", "",
             "Answers are extractive: the top-ranked Addendum passage is returned with its citation. "
             "No LLM generates text in this version.", ""]
    for i, q in enumerate(questions[:n], start=1):
        res = service.ask(bid_id, q["question"], k=2)
        top = res["citations"][0] if res["citations"] else None
        lines += [
            f"## Q{i}. {q['question']}",
            "",
            f"**Answer (extractive):** {res['answer'] or 'No answer found.'}",
            "",
            f"**Citation:** {top['text'] if top else 'none'}  ",
            f"**Confidence:** {res['confidence']}  ",
            f"**Gold (eval label):** {q['gold']}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
