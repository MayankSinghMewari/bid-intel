"""Command-line interface (python -m bid_intel <command>).

  run --bid <folder>        index a bid folder, run the agents, write outputs/<bid>.json + logs/<bid>_trace.json
  index --bid <folder>      index a bid folder only (incremental)
  search "<query>"          hybrid search with metadata filters, cited passages
  ask "<question>"          cited answer over the indexed bids
  eval-retrieval            Recall@k / MRR across search configurations
  eval-extraction           field accuracy vs hand-checked labels (rules vs rules + LLM)
  qalog                     regenerate logs/qa_log.md from the sample questions
  graph                     regenerate docs/agent_graph.md from the LangGraph
  tracing-check             verify Langfuse keys
  serve                     REST API (FastAPI)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from . import observability
    try:
        return _main(argv)
    finally:
        observability.shutdown()  # short-lived process: send queued traces before exit


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bid_intel", description="RFP intelligence: hybrid RAG search and "
                                                               "multi-agent extraction over bid folders")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="index a bid folder, run the agents, write outputs/<bid>.json")
    r.add_argument("--bid", required=True, type=Path, help="path to a bid folder, e.g. data/bid_folders/Bid1")
    r.add_argument("--trace", type=Path, help="agent trace path (default logs/<bid>_trace.json)")

    ix = sub.add_parser("index", help="index a bid folder (only new or changed chunks are embedded)")
    ix.add_argument("--bid", required=True, type=Path)

    sr = sub.add_parser("search", help="hybrid search with metadata filters")
    sr.add_argument("query")
    sr.add_argument("--bid-id", default=None, help="restrict to one bid, e.g. Bid2")
    sr.add_argument("--doc-type", action="append", help="bid_page, rfp, addendum, specs, affidavit (repeatable)")
    sr.add_argument("-k", type=int, default=5)
    sr.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    sr.add_argument("--rerank", action="store_true", help="add the cross-encoder re-ranker (slower)")

    a = sub.add_parser("ask", aliases=["ask-bids"], help="cited answer to a question over the bids")
    a.add_argument("question")
    a.add_argument("--bid-id", default=None)

    sub.add_parser("eval-retrieval", aliases=["eval-bids"], help="Recall@k / MRR across search configurations")
    ee = sub.add_parser("eval-extraction", help="field accuracy vs eval/extraction_gold.json")
    ee.add_argument("--modes", nargs="+", default=["rules", "rules+llm"], choices=["rules", "rules+llm"])
    sub.add_parser("qalog", aliases=["bid-qalog"], help="write logs/qa_log.md from the sample questions")
    sub.add_parser("graph", help="write docs/agent_graph.md (Mermaid, generated from the LangGraph)")
    sub.add_parser("tracing-check", help="check the Langfuse keys and connection")

    sv = sub.add_parser("serve", help="start the REST API")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside Docker")
    return p


ALIASES = {"ask-bids": "ask", "eval-bids": "eval-retrieval", "bid-qalog": "qalog"}


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cmd = ALIASES.get(args.cmd, args.cmd)
    from . import service

    if cmd == "run":
        result, trace = service.extract(args.bid)
        tpath = args.trace or service.ROOT / "logs" / f"{result['bid_id']}_trace.json"
        tpath.parent.mkdir(parents=True, exist_ok=True)
        tpath.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        for k, v in result["fields"].items():
            src = f"{v['sources'][0]['file']} p.{v['sources'][0]['page']}" if v["sources"] else "-"
            print(f"{k:38} {str(v['value'])[:70]:70} conf={v['confidence']:<5} {src}")
        print(json.dumps(result["validation"]))
        print(f"wrote {service.OUT / (result['bid_id'] + '.json')} and {tpath}")
        if service.LAST_TRACE_URL:
            print(f"Langfuse trace: {service.LAST_TRACE_URL}")
    elif cmd == "index":
        print(json.dumps(service.index_folder(args.bid), indent=2))
    elif cmd == "search":
        for h in service.search(args.query, args.bid_id, args.k, args.doc_type, args.mode, args.rerank):
            print(f"#{h['rank']} [{h['bid_id']}] {h['file']} p.{h['page']} ({h['doc_type']}) score={h['score']}")
            print("   " + " ".join(h["text"][:300].split()))
    elif cmd == "ask":
        print(json.dumps(service.ask(args.question, args.bid_id), indent=2, ensure_ascii=False))
    elif cmd == "eval-retrieval":
        from . import eval_retrieval
        folders = sorted(f for f in service.FOLDERS.iterdir() if f.is_dir())
        print(json.dumps(eval_retrieval.run(folders, service.ROOT / "eval" / "retrieval_questions.json",
                                            service.ROOT / "eval"), indent=2))
    elif cmd == "eval-extraction":
        from . import eval_extraction
        folders = sorted(f for f in service.FOLDERS.iterdir() if f.is_dir())
        print(json.dumps(eval_extraction.run(folders, service.ROOT / "eval" / "extraction_gold.json",
                                             service.ROOT / "eval", tuple(args.modes)), indent=2))
    elif cmd == "qalog":
        for i, r in enumerate(service.write_qa_log(service.ROOT / "logs" / "qa_log.md"), 1):
            print(f"Q{i} [{r['method'][:55]}] {' '.join(r['answer'].split())[:140]}")
    elif cmd == "graph":
        from .graph import mermaid
        out = service.ROOT / "docs" / "agent_graph.md"
        out.write_text("# Agent graph (generated by `python -m bid_intel graph`)\n\n"
                       "Solid edges always run; dotted edges are conditional (routing decided at run time). "
                       "`rule_extract` and `llm_extract` run once per field group in parallel (LangGraph `Send`)."
                       "\n\n```mermaid\n" + mermaid() + "```\n", encoding="utf-8")
        print(f"wrote {out}")
    elif cmd == "tracing-check":
        from . import observability
        print(json.dumps(observability.check(), indent=2))
    elif cmd == "serve":
        import uvicorn
        uvicorn.run("bid_intel.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
