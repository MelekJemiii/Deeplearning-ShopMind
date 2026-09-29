"""Qdrant helpers: one collection per chunking strategy, deterministic point ids (idempotent upserts)."""
import uuid

from qdrant_client import QdrantClient, models

from .config import SETTINGS, env

CFG = SETTINGS["vector_store"]


def get_client() -> QdrantClient:
    url = env("QDRANT_URL")
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, api_key=env("QDRANT_API_KEY", required=False) or None)


def collection_name(strategy: str) -> str:
    return f"{CFG['collection_prefix']}_{strategy}"


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


def search(client: QdrantClient, name: str, vector, k: int, filters: dict | None = None) -> list[dict]:
    flt = None
    if filters:
        flt = models.Filter(must=[models.FieldCondition(key=f, match=models.MatchAny(any=v if isinstance(v, list) else [v]))
                                  for f, v in filters.items()])
    res = client.query_points(name, query=vector.tolist(), limit=k, query_filter=flt, with_payload=True)
    return [{"score": p.score, **p.payload} for p in res.points]
