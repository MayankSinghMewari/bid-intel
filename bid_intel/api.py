"""HTTP API (FastAPI). Start with: python -m bid_intel serve"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from typing import Literal

from pydantic import BaseModel, Field

from . import service

app = FastAPI(title="Bid Intelligence API", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=3, ge=1, le=10)
    mode: Literal["bm25", "dense", "hybrid"] | None = None
    use_llm: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/bids/{bid_id}/ask")
def ask(bid_id: str, req: AskRequest) -> dict:
    try:
        return service.ask(bid_id, req.question, req.k, mode=req.mode, use_llm=req.use_llm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/bids/{bid_id}/extract")
def extract(bid_id: str) -> dict:
    try:
        result, trace = service.extract(bid_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"result": result, "trace": trace}


@app.get("/bids/{bid_id}/extraction")
def stored_extraction(bid_id: str) -> dict:
    path = service.BIDS_DIR / bid_id / "extraction.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no extraction yet; POST /bids/{id}/extract")
    return json.loads(path.read_text(encoding="utf-8"))


# ---- bid-folder endpoints (assignment spec: POST /index, GET /search, POST /ask) ----
from pathlib import Path  # noqa: E402

from fastapi import Query  # noqa: E402

from . import bids  # noqa: E402


class IndexRequest(BaseModel):
    folder: str = Field(description="path to a bid folder, e.g. data/bid_folders/Bid1")
    bid_id: str | None = None


class BidAskRequest(BaseModel):
    question: str = Field(min_length=3)
    bid_id: str | None = None


@app.post("/index")
def index(req: IndexRequest) -> dict:
    folder = Path(req.folder)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"folder not found: {req.folder}")
    return bids.index_folder(folder, req.bid_id)


@app.get("/search")
def search(q: str, bid_id: str | None = None, top_k: int = Query(5, ge=1, le=20),
           doc_type: list[str] | None = Query(None), mode: Literal["bm25", "dense", "hybrid"] = "hybrid",
           rerank: bool = False) -> list[dict]:
    return bids.search(q, bid_id, top_k, doc_type, mode, rerank)


@app.post("/ask")
def ask_bids(req: BidAskRequest) -> dict:
    return bids.ask(req.question, req.bid_id)


@app.post("/extract/{bid_id}")
def extract_bid(bid_id: str) -> dict:
    folder = bids.FOLDERS / bid_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"no bid folder {bid_id}")
    result, trace = bids.extract(folder)
    return {"result": result, "trace": trace}
