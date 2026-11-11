"""HTTP API (FastAPI). Start with: python -m bid_intel serve"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import service

app = FastAPI(title="Bid Intelligence API", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/bids/{bid_id}/ask")
def ask(bid_id: str, req: AskRequest) -> dict:
    try:
        return service.ask(bid_id, req.question, req.k)
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
