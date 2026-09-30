"""Cross-encoder reranking: retrieve many candidates (recall), then reorder them (precision).

A bi-encoder (bge-m3) embeds query and chunk SEPARATELY, then compares vectors: fast, but coarse.
A cross-encoder (bge-reranker-v2-m3) reads query AND chunk TOGETHER and outputs a relevance score:
too slow to scan the whole KB, but precise on a short candidate list.
Served by Hugging Face Text Embeddings Inference (TEI) in Docker: POST /rerank.
"""
import os
import re

import requests

from .config import SETTINGS, env
from .logging_utils import get_logger

log = get_logger("rerank")
CFG = SETTINGS["rerank"]


class Reranker:
    def scores(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score for each text, in the SAME order as `texts`."""
        raise NotImplementedError

    def rerank(self, query: str, hits: list[dict], k: int) -> list[dict]:
        if not hits:
            return []
        s = self.scores(query, [h["text"] for h in hits])
        for h, score in zip(hits, s):
            h["rerank_score"] = round(float(score), 4)
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:k]


class TEIReranker(Reranker):
    def __init__(self):
        self.url = env("RERANKER_URL", required=False) or "http://localhost:8081"

    def scores(self, query, texts):
        out = [0.0] * len(texts)
        bs = CFG["batch_size"]
        for i in range(0, len(texts), bs):  # TEI limits the number of texts per request
            r = requests.post(f"{self.url}/rerank", timeout=CFG["timeout_s"],
                              json={"query": query, "texts": texts[i:i + bs], "truncate": True})
            r.raise_for_status()
            for item in r.json():  # [{"index": i, "score": s}, ...] sorted by score
                out[i + item["index"]] = item["score"]
        return out


class FakeReranker(Reranker):
    """Word-overlap scorer: lets tests run without the TEI service."""

    def scores(self, query, texts):
        q = set(re.findall(r"\w+", query.lower()))
        return [len(q & set(re.findall(r"\w+", t.lower()))) / (len(q) or 1) for t in texts]


def get_reranker(provider: str | None = None) -> Reranker:
    provider = provider or os.getenv("RERANK_PROVIDER") or CFG["provider"]
    return {"tei": TEIReranker, "fake": FakeReranker}[provider]()
