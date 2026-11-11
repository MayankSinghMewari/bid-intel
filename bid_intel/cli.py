"""Command-line interface.
  python -m bid_intel ingest <pdf> --bid JA-207652
  python -m bid_intel ask --bid JA-207652 "question"
  python -m bid_intel extract --bid JA-207652 [--trace logs/agent_trace.json]
  python -m bid_intel eval
  python -m bid_intel serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import service
from .ingest import ingest as ingest_doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bid_intel")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="parse a bid document into Q&A items")
    s.add_argument("source", type=Path)
    s.add_argument("--bid", required=True)

    a = sub.add_parser("ask", help="search a bid and return cited passages")
    a.add_argument("question")
    a.add_argument("--bid", required=True)
    a.add_argument("-k", type=int, default=3)
    a.add_argument("--chunking", default=service.DEFAULT_CHUNKING)
    a.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=None,
                   help=f"retrieval mode (default: {service.DEFAULT_MODE})")
    a.add_argument("--no-llm", action="store_true", help="return the extractive passage, skip LLM generation")

    e = sub.add_parser("extract", help="run the multi-agent extraction")
    e.add_argument("--bid", required=True)
    e.add_argument("--trace", type=Path, help="write the agent trace to this file")

    v = sub.add_parser("eval", help="retrieval evaluation across configurations")
    v.add_argument("--bid", default="JA-207652")
    v.add_argument("--questions", type=Path, default=service.ROOT / "eval" / "questions.json")

    ql = sub.add_parser("qalog", help="write the sample Q&A log from the eval questions")
    ql.add_argument("--bid", default="JA-207652")
    ql.add_argument("-n", type=int, default=12)
    ql.add_argument("--out", type=Path, default=service.ROOT / "logs" / "qa_log.md")

    r = sub.add_parser("run", help="ONE COMMAND: index a bid folder, run the agents, write outputs/<bid>.json")
    r.add_argument("--bid", required=True, type=Path, help="path to a bid folder, e.g. data/bid_folders/Bid1")
    r.add_argument("--trace", type=Path, help="write the agent trace here (default logs/<bid>_trace.json)")

    sr = sub.add_parser("search", help="hybrid search across bid folders with metadata filters")
    sr.add_argument("query")
    sr.add_argument("--bid-id", default=None, help="restrict to one bid, e.g. Bid2")
    sr.add_argument("--doc-type", action="append", help="rfp, addendum, specs, affidavit, bid_page")
    sr.add_argument("-k", type=int, default=5)
    sr.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    sr.add_argument("--rerank", action="store_true", help="add the cross-encoder re-ranker (slower; did not help MRR here)")

    ab = sub.add_parser("ask-bids", help="answer a question over all bid folders with citations")
    ab.add_argument("question")
    ab.add_argument("--bid-id", default=None)

    sub.add_parser("eval-bids", help="retrieval evaluation over the bid folders (Recall@k, MRR)")

    ee = sub.add_parser("eval-extraction", help="field accuracy vs eval/extraction_gold.json")
    ee.add_argument("--modes", nargs="+", default=["rules", "rules+llm"], choices=["rules", "rules+llm"])

    sub.add_parser("bid-qalog", help="write logs/bid_qa_log.md from the sample questions")

    sv = sub.add_parser("serve", help="start the HTTP API")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside Docker")

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        meta = ingest_doc(args.source, args.bid, service.BIDS_DIR / args.bid)
        print(json.dumps(meta, indent=2))
    elif args.cmd == "ask":
        print(json.dumps(service.ask(args.bid, args.question, args.k, args.chunking, args.mode, not args.no_llm), indent=2, ensure_ascii=False))
    elif args.cmd == "extract":
        result, trace = service.extract(args.bid)
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            args.trace.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
        print(f"wrote {service.BIDS_DIR / args.bid / 'extraction.json'}")
    elif args.cmd == "eval":
        from .evaluation import run_evaluation
        summary = run_evaluation(args.bid, args.questions, service.ROOT / "eval")
        print(json.dumps(summary, indent=2))
    elif args.cmd == "run":
        from . import bids
        result, trace = bids.extract(args.bid)
        tpath = args.trace or service.ROOT / "logs" / f"{result['bid_id']}_trace.json"
        tpath.parent.mkdir(parents=True, exist_ok=True)
        tpath.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        for k, v in result["fields"].items():
            src = f"{v['sources'][0]['file']} p.{v['sources'][0]['page']}" if v["sources"] else "-"
            print(f"{k:38} {str(v['value'])[:70]:70} conf={v['confidence']:<5} {src}")
        print(json.dumps(result["validation"]))
        print(f"wrote {bids.OUT / (result['bid_id'] + '.json')} and {tpath}")
    elif args.cmd == "search":
        from . import bids
        for h in bids.search(args.query, args.bid_id, args.k, args.doc_type, args.mode, args.rerank):
            print(f"#{h['rank']} [{h['bid_id']}] {h['file']} p.{h['page']} ({h['doc_type']}) score={h['score']}")
            print("   " + " ".join(h["text"][:300].split()))
    elif args.cmd == "ask-bids":
        from . import bids
        print(json.dumps(bids.ask(args.question, args.bid_id), indent=2, ensure_ascii=False))
    elif args.cmd == "eval-bids":
        from . import bid_eval, bids
        folders = sorted(f for f in bids.FOLDERS.iterdir() if f.is_dir())
        print(json.dumps(bid_eval.run(folders, service.ROOT / "eval" / "bid_questions.json",
                                      service.ROOT / "eval"), indent=2))
    elif args.cmd == "eval-extraction":
        from . import bids, extraction_eval
        folders = sorted(f for f in bids.FOLDERS.iterdir() if f.is_dir())
        print(json.dumps(extraction_eval.run(folders, service.ROOT / "eval" / "extraction_gold.json",
                                             service.ROOT / "eval", tuple(args.modes)), indent=2))
    elif args.cmd == "bid-qalog":
        from . import bids
        for i, r in enumerate(bids.write_qa_log(service.ROOT / "logs" / "bid_qa_log.md"), 1):
            print(f"Q{i} [{r['method'][:55]}] {' '.join(r['answer'].split())[:140]}")
    elif args.cmd == "qalog":
        from .evaluation import write_qa_log
        qs = json.loads((service.ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
        write_qa_log(args.bid, qs, args.out, args.n)
        print(f"wrote {args.out}")
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("bid_intel.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
