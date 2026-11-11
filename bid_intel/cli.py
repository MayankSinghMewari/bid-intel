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

    e = sub.add_parser("extract", help="run the multi-agent extraction")
    e.add_argument("--bid", required=True)
    e.add_argument("--trace", type=Path, help="write the agent trace to this file")

    v = sub.add_parser("eval", help="retrieval evaluation across configurations")
    v.add_argument("--bid", default="JA-207652")
    v.add_argument("--questions", type=Path, default=service.ROOT / "eval" / "questions.json")

    sv = sub.add_parser("serve", help="start the HTTP API")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        meta = ingest_doc(args.source, args.bid, service.BIDS_DIR / args.bid)
        print(json.dumps(meta, indent=2))
    elif args.cmd == "ask":
        print(json.dumps(service.ask(args.bid, args.question, args.k, args.chunking), indent=2, ensure_ascii=False))
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
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("bid_intel.api:app", host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
