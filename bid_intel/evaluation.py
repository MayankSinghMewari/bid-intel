"""Retrieval evaluation across chunking and BM25 configurations, plus report writers."""
from __future__ import annotations

import json
from pathlib import Path

from . import embeddings, service
from .ingest import build_chunks
from .retrieval import SearchEngine, evaluate

CHUNKINGS = ["qa", "window:40:20", "window:80:40"]
BM25_SETTINGS = [(1.2, 0.75), (1.5, 0.3)]
DENSE_MODES = ["dense", "hybrid"]  # run when fastembed is installed; hybrid uses BM25 k1=1.2, b=0.75


def run_evaluation(bid_id: str, questions_path: Path, out_dir: Path) -> dict:
    _, items = service.load_items(service.bid_dir(bid_id))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    configs = [("bm25", c, k1, b) for c in CHUNKINGS for k1, b in BM25_SETTINGS]
    if embeddings.available():
        configs += [(m, c, 1.2, 0.75) for m in DENSE_MODES for c in CHUNKINGS]
    rows = []
    for mode, chunking, k1, b in configs:
        engine = SearchEngine(build_chunks(items, chunking), k1=k1, b=b, mode=mode)
        metrics = evaluate(engine, questions)
        rows.append({"mode": mode, "chunking": chunking, "k1": k1 if mode != "dense" else None,
                     "b": b if mode != "dense" else None, "chunks": len(engine.chunks), **metrics})
    best = max(rows, key=lambda r: (r["MRR"], r["Recall@3"]))

    # Per-question misses for the best configuration (gold chunk not in top 3)
    best_engine = SearchEngine(build_chunks(items, best["chunking"]), k1=best["k1"] or 1.2,
                               b=best["b"] if best["b"] is not None else 0.75, mode=best["mode"])
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
        f"Retrieval modes: `bm25` (lexical), `dense` (local embeddings, `{embeddings.MODEL_NAME}`), "
        "`hybrid` (reciprocal rank fusion of BM25 and dense, k=60).",
        "",
        "| Mode | Chunking | k1 | b | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        k1 = r['k1'] if r['k1'] is not None else "-"
        b = r['b'] if r['b'] is not None else "-"
        lines.append(f"| {r['mode']} | {r['chunking']} | {k1} | {b} | {r['chunks']} | "
                     f"{r['Recall@1']:.3f} | {r['Recall@3']:.3f} | {r['Recall@5']:.3f} | {r['MRR']:.3f} |")
    lines += [
        "",
        f"**Best by MRR:** mode `{best['mode']}`, chunking `{best['chunking']}`, k1={best['k1']}, b={best['b']} "
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
        "- The vendor questions and District answers share vocabulary, which favours lexical search. "
        "Dense and hybrid retrieval matter more for paraphrased questions on other bids.",
        "- The best and default configurations differ by one question, so the choice of chunking is not conclusive.",
    ]
    return "\n".join(lines) + "\n"


def write_qa_log(bid_id: str, questions: list[dict], path: Path, n: int = 12) -> None:
    rows = [(q, service.ask(bid_id, q["question"], k=3)) for q in questions[:n]]
    methods = sorted({r["method"] for _, r in rows})
    lines = [f"# Sample Q&A Log: {bid_id}", "",
             "Each question is answered from the top 3 retrieved passages. When Anthropic credentials are "
             "available, Claude writes the answer from those passages only and must cite question numbers "
             "(answers citing passages that were not retrieved are rejected). Otherwise the top passage is "
             "returned verbatim.", "", "Methods used in this run: " + "; ".join(f"`{m}`" for m in methods), ""]
    for i, (q, res) in enumerate(rows, start=1):
        cites = "; ".join(c["text"] for c in res["citations"]) or "none"
        hit = any(set(c["qnums"]) & set(q["gold"]) for c in res["citations"][:1])
        lines += [
            f"## Q{i}. {q['question']}",
            "",
            f"**Answer:** {res['answer'] or 'No answer found.'}",
            "",
            f"**Retrieved (ranked):** {cites}  ",
            f"**Confidence:** {res['confidence']}  ",
            f"**Gold (eval label):** {q['gold']} | top-1 correct: {'yes' if hit else 'no'}  ",
            f"**Method:** {res['method']}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("
".join(lines), encoding="utf-8")
