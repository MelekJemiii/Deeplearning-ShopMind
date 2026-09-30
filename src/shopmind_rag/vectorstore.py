"""Qdrant helpers: one collection per chunking strategy, deterministic point ids (idempotent upserts)."""
import uuid

from qdrant_client import QdrantClient, models

from .config import SETTINGS, env

CFG = SETTINGS["vector_store"]


def get_client() -> QdrantClient:
    url = env("QDRANT_URL")
    if url == ":memory:":
        return QdrantClient(":memory:")
    if url.startswith("path:"):  # local on-disk mode (tests, no server)
        return QdrantClient(path=url[5:])
    return QdrantClient(url=url, api_key=env("QDRANT_API_KEY", required=False) or None)


def collection_name(strategy: str, provider: str, mode: str = "dense") -> str:
    # provider (and mode) in the name: collections built differently can coexist and be compared
    base = f"{CFG['collection_prefix']}_{provider}_{strategy}"
    return base if mode == "dense" else f"{base}_{mode}"


def point_id(chunk_id: str) -> str:
    # same chunk_id -> same point id: re-indexing overwrites instead of duplicating
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def recreate(client: QdrantClient, name: str, dims: int):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE))
    for field in CFG["payload_indexes"]:
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)


def upsert(client: QdrantClient, name: str, chunks: list[dict], vectors):
    bs = CFG["upsert_batch_size"]
    for i in range(0, len(chunks), bs):
        client.upsert(name, points=[
            models.PointStruct(id=point_id(c["chunk_id"]), vector=v.tolist(), payload=c)
            for c, v in zip(chunks[i:i + bs], vectors[i:i + bs])])


def _filter(filters: dict | None):
    if not filters:
        return None
    return models.Filter(must=[models.FieldCondition(key=f, match=models.MatchAny(any=v if isinstance(v, list) else [v]))
                               for f, v in filters.items()])


# ---------- hybrid collections: named dense vector + BM25 sparse vector ----------

def recreate_hybrid(client: QdrantClient, name: str, dims: int):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=dims, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    for field in CFG["payload_indexes"]:
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)


def upsert_hybrid(client: QdrantClient, name: str, chunks: list[dict], vectors, sparse: list[tuple]):
    bs = CFG["upsert_batch_size"]
    for i in range(0, len(chunks), bs):
        client.upsert(name, points=[
            models.PointStruct(id=point_id(c["chunk_id"]), payload=c, vector={
                "dense": v.tolist(), "bm25": models.SparseVector(indices=sp[0], values=sp[1])})
            for c, v, sp in zip(chunks[i:i + bs], vectors[i:i + bs], sparse[i:i + bs])])


def search_hybrid(client: QdrantClient, name: str, vector, sparse: tuple, k: int, prefetch_k: int,
                  filters: dict | None = None) -> list[dict]:
    """RRF fusion of dense and BM25 candidates. 'score' = dense cosine (comparable to dense mode,
    so min_score thresholds keep their meaning); 'fusion_score' = RRF score used for ranking."""
    import numpy as np
    flt = _filter(filters)
    res = client.query_points(
        name,
        prefetch=[models.Prefetch(query=vector.tolist(), using="dense", limit=prefetch_k, filter=flt),
                  models.Prefetch(query=models.SparseVector(indices=sparse[0], values=sparse[1]),
                                  using="bm25", limit=prefetch_k, filter=flt)],
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True, with_vectors=["dense"])
    q = vector / (np.linalg.norm(vector) or 1)
    out = []
    for p in res.points:
        d = np.asarray(p.vector["dense"], dtype=np.float32)
        out.append({**p.payload, "score": float(q @ (d / (np.linalg.norm(d) or 1))), "fusion_score": p.score})
    return out


def search(client: QdrantClient, name: str, vector, k: int, filters: dict | None = None) -> list[dict]:
    res = client.query_points(name, query=vector.tolist(), limit=k, query_filter=_filter(filters), with_payload=True)
    return [{"score": p.score, **p.payload} for p in res.points]
