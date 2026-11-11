"""REST API (FastAPI). Start with: python -m bid_intel serve   (interactive docs at /docs)

  GET  /health
  GET  /bids                         bid folders and whether an extraction exists
  POST /index        {"bid_id"}      index a folder under data/bid_folders (incremental)
  GET  /search?q=&bid_id=&top_k=&doc_type=&mode=&rerank=
  POST /ask          {"question", "bid_id"?}
  POST /extract/{bid_id}             run the agents, return the record and the trace
  GET  /extraction/{bid_id}          stored record

Folders are addressed by name and must live under data/bid_folders, so API callers cannot make the
server read arbitrary paths.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import service

app = FastAPI(title="RFP Intelligence API", version="1.0.0")
BID_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _folder(bid_id: str) -> Path:
    if not BID_ID.match(bid_id):
        raise HTTPException(status_code=400, detail="bid_id: letters, digits, '-' or '_' only")
    folder = service.FOLDERS / bid_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"no bid folder named {bid_id} under data/bid_folders")
    return folder


class IndexRequest(BaseModel):
    bid_id: str = Field(description="name of a folder under data/bid_folders, e.g. Bid1")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    bid_id: str | None = Field(default=None, description="restrict to one bid; default: inferred from the question")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/bids")
def list_bids() -> list[dict]:
    return [{"bid_id": f.name, "extracted": (service.OUT / f"{f.name}.json").exists()}
            for f in sorted(service.FOLDERS.iterdir()) if f.is_dir()]


@app.post("/index")
def index(req: IndexRequest) -> dict:
    return service.index_folder(_folder(req.bid_id))


@app.get("/search")
def search(q: str = Query(min_length=2), bid_id: str | None = None, top_k: int = Query(5, ge=1, le=20),
           doc_type: list[str] | None = Query(None), mode: Literal["bm25", "dense", "hybrid"] = "hybrid",
           rerank: bool = False) -> list[dict]:
    if bid_id is not None:
        _folder(bid_id)
    return service.search(q, bid_id, top_k, doc_type, mode, rerank)


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.bid_id is not None:
        _folder(req.bid_id)
    return service.ask(req.question, req.bid_id)


@app.post("/extract/{bid_id}")
def extract(bid_id: str) -> dict:
    result, trace = service.extract(_folder(bid_id))
    return {"result": result, "trace": trace}


@app.get("/extraction/{bid_id}")
def stored_extraction(bid_id: str) -> dict:
    _folder(bid_id)
    path = service.OUT / f"{bid_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no extraction yet; POST /extract/{bid_id}")
    return json.loads(path.read_text(encoding="utf-8"))
