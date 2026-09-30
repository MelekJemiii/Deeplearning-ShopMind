"""API tests without any external service: fake embedder + on-disk local Qdrant."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    qd = tmp_path_factory.mktemp("qdrant")
    os.environ["QDRANT_URL"] = f"path:{qd}"
    os.environ["EMBEDDING_PROVIDER"] = "fake"
    from shopmind_rag import api
    from shopmind_rag.config import SETTINGS
    from shopmind_rag.embedding import get_embedder
    from shopmind_rag.vectorstore import collection_name, get_client, recreate, upsert
    SETTINGS["retrieval"]["provider"] = "fake"
    SETTINGS["retrieval"]["rerank"] = False   # base tests: first-stage retrieval only
    chunks = [
        {"chunk_id": "g1#B000", "doc_id": "g1", "title": "Guide data science", "heading_path": ["Guide", "RAM"],
         "source_type": "team_written", "source_url": "", "device_category": "laptop", "language": "fr",
         "topic": "use_case_guide", "text": "16 Go de RAM minimum pour la data science"},
        {"chunk_id": "p1#B000", "doc_id": "p1", "title": "Legion 5 specs", "heading_path": ["Legion 5", "Graphics"],
         "source_type": "manufacturer_pdf", "source_url": "", "device_category": "laptop", "language": "en",
         "topic": "spec_sheet", "text": "RTX 5060 laptop GPU 115W TGP 8GB GDDR7"},
        {"chunk_id": "w1#B000", "doc_id": "w1", "title": "Smartphone", "heading_path": ["Smartphone"],
         "source_type": "wikipedia", "source_url": "", "device_category": "smartphone", "language": "en",
         "topic": "devices", "text": "A smartphone is a mobile phone with advanced computing"},
    ]
    emb, qc = get_embedder("fake"), get_client()
    name = collection_name("B", "fake")
    recreate(qc, name, emb.dims)
    upsert(qc, name, chunks, emb.embed_documents([(c["title"], c["text"]) for c in chunks]))
    qc.close()
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["points"] == 3


def test_search_returns_best_chunk_first(client):
    r = client.post("/search", json={"query": "RAM data science", "min_score": 0}).json()
    assert r["results"][0]["chunk_id"] == "g1#B000"
    assert r["results"][0]["heading_path"] == ["Guide", "RAM"]


def test_filters(client):
    r = client.post("/search", json={"query": "RAM data science", "min_score": 0,
                                     "filters": {"source_type": ["manufacturer_pdf"]}}).json()
    assert [x["doc_id"] for x in r["results"]] == ["p1"]


def test_flat_filter_for_llm_tools(client):
    r = client.post("/search", json={"query": "RAM data science", "min_score": 0, "source_type": "team_written"}).json()
    assert [x["doc_id"] for x in r["results"]] == ["g1"]
    r = client.post("/search", json={"query": "RAM", "min_score": 0, "source_type": "any"}).json()
    assert len(r["results"]) == 3


def test_min_score_drops_unrelated(client):
    r = client.post("/search", json={"query": "pizza restaurant tunis", "min_score": 0.5}).json()
    assert r["results"] == [] and r["dropped_below_min_score"] >= 1


def test_validation(client):
    assert client.post("/search", json={"query": "x"}).status_code == 422             # too short
    assert client.post("/search", json={"query": "RAM", "top_k": 50}).status_code == 422  # over max_top_k
    assert client.post("/search", json={"query": "RAM", "filters": {"source_type": ["blog"]}}).status_code == 422


def test_hybrid_search_ranks_exact_token(tmp_path):
    """Hybrid: an exact spec token (15IAX11) that dense hashing barely sees is found via BM25."""
    from shopmind_rag import sparse
    from shopmind_rag.embedding import get_embedder
    from shopmind_rag.vectorstore import QdrantClient, recreate_hybrid, search_hybrid, upsert_hybrid
    docs = [{"chunk_id": f"d{i}", "doc_id": f"d{i}", "title": t, "text": x} for i, (t, x) in enumerate([
        ("Legion 5 15IAX11", "Graphics RTX 5060 115W TGP"),
        ("Gaming guide", "Choose a gaming laptop with a good GPU and 144 Hz screen"),
        ("Laptop", "A laptop is a portable personal computer")])]
    emb, qc = get_embedder("fake"), QdrantClient(path=str(tmp_path))
    texts = [f"{d['title']}\n\n{d['text']}" for d in docs]
    recreate_hybrid(qc, "h", emb.dims)
    upsert_hybrid(qc, "h", docs, emb.embed_documents([(d["title"], d["text"]) for d in docs]),
                  [sparse.doc_vector(t, sparse.avg_doc_len(texts)) for t in texts])
    hits = search_hybrid(qc, "h", emb.embed_query("15IAX11 TGP"), sparse.query_vector("15IAX11 TGP"), 3, 10)
    assert hits[0]["chunk_id"] == "d0"
    assert -1.0 <= hits[0]["score"] <= 1.0 and "fusion_score" in hits[0]
    qc.close()


def test_rerank_mode_filters_on_rerank_score(client):
    from shopmind_rag import api
    from shopmind_rag.rerank import FakeReranker
    api.STATE["reranker"] = FakeReranker()
    try:
        r = client.post("/search", json={"query": "RAM data science", "min_score": 0}).json()
        assert r["results"][0]["chunk_id"] == "g1#B000" and r["results"][0]["rerank_score"] > 0
        # chunks with no word overlap get rerank_score 0 < min_rerank_score -> dropped
        assert all(x["rerank_score"] >= api.RCFG["min_rerank_score"] for x in r["results"])
    finally:
        api.STATE["reranker"] = None
