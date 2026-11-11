"""Extraction accuracy against hand-written gold labels (eval/extraction_gold.json).

A field is correct when its value satisfies the label: every expected group matched (any string of a
group, case-insensitive), or null for fields that are absent from the documents. Fields marked skip
(ambiguous in the documents, or generated text) are not scored. Runs each mode on the same index, so
"rules" and "rules + LLM" are compared on identical retrieval."""
from __future__ import annotations

import json
from pathlib import Path


def score_field(value: str | None, gold: dict) -> bool | None:
    if "skip" in gold:
        return None
    v = (value or "").lower()
    if gold.get("absent"):
        return value is None or any(a.lower() in v for a in gold.get("also_accept", []))
    if value is None:
        return False
    return all(any(alt.lower() in v for alt in group) for group in gold["expect"])


def score_bid(result: dict, gold: dict) -> dict:
    rows, correct, scored = [], 0, 0
    for name, label in gold.items():
        if name.startswith("_"):
            continue
        value = result["fields"].get(name, {}).get("value")
        ok = score_field(value, label)
        if ok is not None:
            scored += 1
            correct += ok
        rows.append({"field": name, "value": value, "correct": ok})
    llm_fields = sum(1 for f in result["fields"].values() if str(f.get("notes", "")).startswith("LLM extraction"))
    return {"correct": correct, "scored": scored, "accuracy": round(correct / scored, 3) if scored else None,
            "llm_fields": llm_fields, "rows": rows}


def run(folders: list[Path], gold_path: Path, out_dir: Path, modes=("rules", "rules+llm")) -> dict:
    from . import service
    from .agents import Orchestrator

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    service.ensure_indexed()
    report: dict = {}
    for mode in modes:
        for folder in folders:
            if folder.name not in gold:
                continue
            orch = Orchestrator(folder, engine=service.engine(), use_llm=(mode != "rules"))
            report.setdefault(mode, {})[folder.name] = score_bid(orch.run(), gold[folder.name])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extraction_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    (out_dir / "extraction_report.md").write_text(_markdown(report), encoding="utf-8")
    return {m: {b: f"{r['correct']}/{r['scored']} (LLM fields: {r['llm_fields']})" for b, r in per.items()}
            for m, per in report.items()}


def _markdown(report: dict) -> str:
    modes = list(report)
    bids_ = list(next(iter(report.values())))
    lines = ["# Extraction Accuracy", "",
             "Gold labels: `eval/extraction_gold.json` (checked by hand against the documents). A field is correct "
             "when the value contains the expected facts, or is null when the documents do not state it. Ambiguous "
             "fields and the generated summary are not scored.", "",
             "| Mode | " + " | ".join(bids_) + " | Total |", "|---|" + "---|" * (len(bids_) + 1)]
    for m in modes:
        c = sum(report[m][b]["correct"] for b in bids_)
        n = sum(report[m][b]["scored"] for b in bids_)
        cells = [f"{report[m][b]['correct']}/{report[m][b]['scored']}" for b in bids_]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {c}/{n} ({c / n:.0%}) |")
        if m != "rules":
            used = {b: report[m][b].get("llm_fields", 0) for b in bids_}
            lines.append(f"|  LLM-sourced fields | " + " | ".join(str(used[b]) for b in bids_) + " | |")
            if not any(used.values()):
                lines += ["", "> **Warning:** no field came from the LLM in this run (no key, quota exhausted or "
                          "errors), so this row shows the rule-based fallback, not LLM accuracy.", ""]
    lines += ["", "## Per field", "", "| Bid | Field | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    mark = {True: "ok", False: "**wrong**", None: "-"}
    for b in bids_:
        for i, row in enumerate(report[modes[0]][b]["rows"]):
            cells = [mark[report[m][b]["rows"][i]["correct"]] for m in modes]
            lines.append(f"| {b} | {row['field']} | " + " | ".join(cells) + " |")
    if len(modes) > 1:
        lines += ["", "## Values that differ between modes", ""]
        for b in bids_:
            for i, row in enumerate(report[modes[0]][b]["rows"]):
                vals = [report[m][b]["rows"][i]["value"] for m in modes]
                if len(set(map(str, vals))) > 1 and row["correct"] is not None or \
                        any(report[m][b]["rows"][i]["correct"] is False for m in modes):
                    lines.append(f"- **{b} / {row['field']}**")
                    for m, v in zip(modes, vals):
                        lines.append(f"  - {m}: {str(v)[:220] if v is not None else 'null'}")
    return "\n".join(lines) + "\n"
