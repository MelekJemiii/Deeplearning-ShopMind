"""Retrieval API: the knowledge-base search tool used by the n8n RAG Agent.

It reuses the exact code that was evaluated in Step 6 (same embedding model, same query
format, same collection), so production retrieval = evaluated retrieval.

  GET  /health   -> dependencies status + collection size
  POST /search   -> top-k chunks for a query, with optional metadata filters and a score floor
Run locally:  uvicorn shopmind_rag.api:app --app-dir src --port 8000
"""
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import SETTINGS
from .embedding import get_embedder
from .logging_utils import get_logger
from . import sparse
from .vectorstore import collection_name, get_client, search, search_hybrid

log = get_logger("api")
RCFG = SETTINGS["retrieval"]
STATE: dict = {}


class Filters(BaseModel):
    device_category: list[Literal["laptop", "smartphone", "tablet", "accessory", "general"]] | None = None
    source_type: list[Literal["wikipedia", "team_written", "manufacturer_pdf"]] | None = None
    language: list[Literal["fr", "en"]] | None = None
    doc_id: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000, description="Natural-language search query")
    top_k: int = Field(default=RCFG["top_k"], ge=1, le=RCFG["max_top_k"])
    min_score: float = Field(default=RCFG["min_score"], ge=0.0, le=1.0)
    # Flat single-value filters: easy for an LLM tool call to fill ("any" = no filter)
    source_type: Literal["any", "wikipedia", "team_written", "manufacturer_pdf"] = "any"
    device_category: Literal["any", "laptop", "smartphone", "tablet", "accessory", "general"] = "any"
    # Full filter object (lists) for programmatic callers
    filters: Filters | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    heading_path: list[str]
    source_type: str
    source_url: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[Chunk]
    dropped_below_min_score: int
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["embedder"] = get_embedder(RCFG["provider"])
    STATE["client"] = get_client()
    STATE["mode"] = RCFG.get("mode", "dense")
    STATE["collection"] = collection_name(RCFG["strategy"], RCFG["provider"], STATE["mode"])
    try:  # warm-up: loads the embedding model now instead of on the first user query
        STATE["embedder"].embed_query("warm-up", use_cache=False)
        log.info("Embedder warmed up (%s)", STATE["embedder"].model)
    except Exception as e:
        log.warning("Warm-up failed (embedding service down?): %s", e)
    yield


app = FastAPI(title="ShopMind KB Retrieval API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    client, name = STATE["client"], STATE["collection"]
    try:
        points = client.count(name).count if client.collection_exists(name) else None
    except Exception as e:
        raise HTTPException(503, f"vector store unavailable: {e}")
    return {"status": "ok" if points else "degraded", "collection": name, "points": points, "mode": STATE["mode"],
            "embedding": {"provider": RCFG["provider"], "model": STATE["embedder"].model}}


@app.post("/search", response_model=SearchResponse)
def search_kb(req: SearchRequest):
    t0 = time.perf_counter()
    try:
        vec = STATE["embedder"].embed_query(req.query, use_cache=False)
    except Exception as e:
        raise HTTPException(503, f"embedding service unavailable: {e}")
    flt = {k: v for k, v in (req.filters.model_dump() if req.filters else {}).items() if v}
    for field in ("source_type", "device_category"):
        value = getattr(req, field)
        if value != "any":
            flt[field] = [value]
    if STATE["mode"] == "hybrid":
        hits = search_hybrid(STATE["client"], STATE["collection"], vec, sparse.query_vector(req.query), req.top_k,
                             SETTINGS["hybrid"]["prefetch_k"], flt or None)
    else:
        hits = search(STATE["client"], STATE["collection"], vec, req.top_k, flt or None)
    kept = [h for h in hits if h["score"] >= req.min_score]
    ms = round(1000 * (time.perf_counter() - t0), 1)
    log.info("search %r filters=%s -> %d/%d kept (%.0f ms)", req.query[:60], flt, len(kept), len(hits), ms)
    return SearchResponse(query=req.query, collection=STATE["collection"],
                          results=[Chunk(**{k: h.get(k, "" if k != "heading_path" else []) for k in Chunk.model_fields}
                                         | {"score": round(h["score"], 4)}) for h in kept],
                          dropped_below_min_score=len(hits) - len(kept), latency_ms=ms)
