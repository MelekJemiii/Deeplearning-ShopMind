"""Embedding providers with an on-disk cache.

The SAME model, dimensions and formats must be used at indexing time and at query time
(in n8n too): vectors from different models/formats are not comparable.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .config import ROOT, SETTINGS, env
from .logging_utils import get_logger

log = get_logger("embedding")
CFG = SETTINGS["embedding"]


class EmbeddingCache:
    """Append-only JSONL cache: key = sha256(provider|model|dims|input text)."""

    def __init__(self, provider: str, model: str, dims: int):
        d = ROOT / CFG["cache_dir"]
        d.mkdir(parents=True, exist_ok=True)
        # provider in the key: test vectors (fake) must never be served as real Gemini vectors
        self.path = d / f"cache_{provider}_{model}_{dims}.jsonl"
        self.prefix = f"{provider}|{model}|{dims}|"
        self.data: dict[str, list[float]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.data[rec["k"]] = rec["v"]

    def key(self, text: str) -> str:
        return hashlib.sha256((self.prefix + text).encode("utf-8")).hexdigest()

    def get(self, text: str):
        return self.data.get(self.key(text))

    def put_many(self, texts: list[str], vectors: list[list[float]]):
        with open(self.path, "a", encoding="utf-8") as f:
            for t, v in zip(texts, vectors):
                k = self.key(t)
                self.data[k] = v
                f.write(json.dumps({"k": k, "v": v}) + "\n")


class DailyQuotaExceeded(RuntimeError):
    """Requests-per-day quota reached: retrying today is pointless (resets at midnight Pacific Time)."""


class RateLimiter:
    """Sliding 60 s window over texts and (estimated) tokens sent."""

    def __init__(self, rpm: int, tpm: int, window_s: float = 60.0):
        self.rpm, self.tpm, self.window = rpm, tpm, window_s
        self.events: list[tuple[float, int, int]] = []  # (time, n_texts, n_tokens)

    def wait(self, n_texts: int, n_tokens: int):
        while True:
            now = time.monotonic()
            self.events = [e for e in self.events if now - e[0] < self.window]
            used_r = sum(e[1] for e in self.events)
            used_t = sum(e[2] for e in self.events)
            if used_r + n_texts <= self.rpm and used_t + n_tokens <= self.tpm:
                self.events.append((now, n_texts, n_tokens))
                return
            wait = self.window - (now - self.events[0][0]) + 0.1
            log.info("  throttling %.0fs (window: %d texts, %d tokens)", wait, used_r, used_t)
            time.sleep(wait)


def estimate_tokens(text: str) -> int:
    # ~4 chars/token for EN/FR, +20% margin: the limiter must err on the safe side
    return int(len(text) / 4 * 1.2) + 1


class Embedder:
    provider = "base"

    def __init__(self):
        self.model, self.dims = CFG["model"], CFG["dimensions"]
        self.cache = EmbeddingCache(self.provider, self.model, self.dims)
        self.api_calls = 0
        rl = CFG.get("rate_limit") or {}
        self.limiter = RateLimiter(rl.get("requests_per_minute", 10**9), rl.get("tokens_per_minute", 10**12)) \
            if self.provider != "fake" else None

    # ----- formatting (task instructions for gemini-embedding-2) -----
    def format_document(self, title: str, text: str) -> str:
        return CFG["document_format"].format(title=title or "none", text=text)

    def format_query(self, query: str) -> str:
        return CFG["query_format"].format(query=query)

    # ----- public API -----
    def embed_documents(self, items: list[tuple[str, str]]) -> np.ndarray:
        return self._embed([self.format_document(t, x) for t, x in items])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([self.format_query(query)])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if self.cache.get(t) is None]
        if todo:
            log.info("Embedding %d new texts (%d cached)", len(todo), len(set(texts)) - len(todo))
        bs = CFG["batch_size"]
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            if self.limiter:
                self.limiter.wait(len(batch), sum(estimate_tokens(t) for t in batch))
            vecs = self._call_with_retry(batch)
            if len(vecs) != len(batch):  # guard: gemini-embedding-2 can AGGREGATE inputs into one vector
                raise RuntimeError(f"expected {len(batch)} embeddings, got {len(vecs)}")
            self.cache.put_many(batch, vecs)
            if len(todo) > bs:
                log.info("  %d/%d", min(i + bs, len(todo)), len(todo))
        return np.array([self.cache.get(t) for t in texts], dtype=np.float32)

    def _call_with_retry(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(1, CFG["max_retries"] + 1):
            try:
                self.api_calls += 1
                return self._call(batch)
            except Exception as e:  # rate limit / transient server errors -> exponential backoff
                code, msg = getattr(e, "code", None), str(e)
                if "PerDay" in msg or "per day" in msg.lower():
                    raise DailyQuotaExceeded(msg[:300]) from e
                if code in (429, 500, 502, 503, 504) or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = min(60, 5 * 2 ** (attempt - 1))
                    log.warning("API %s, retry in %ss (attempt %d/%d)", code or "error", wait, attempt,
                                CFG["max_retries"])
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("embedding API still failing after retries")

    def _call(self, batch: list[str]) -> list[list[float]]:
        raise NotImplementedError


class GeminiEmbedder(Embedder):
    provider = "gemini"

    def __init__(self):
        super().__init__()
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=env("GEMINI_API_KEY"))

    def _call(self, batch):
        t = self.types
        # One Content object per text -> one embedding per text (a flat list would be aggregated)
        res = self.client.models.embed_content(
            model=self.model,
            contents=[t.Content(parts=[t.Part.from_text(text=x)]) for x in batch],
            config=t.EmbedContentConfig(output_dimensionality=self.dims))
        return [e.values for e in res.embeddings]


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing vectors: lets the whole pipeline run without an API key."""
    provider = "fake"

    def _call(self, batch):
        out = []
        for text in batch:
            v = np.zeros(self.dims, dtype=np.float32)
            for w in text.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


def get_embedder() -> Embedder:
    # EMBEDDING_PROVIDER env var overrides the config (e.g. "fake" in tests / CI)
    provider = os.getenv("EMBEDDING_PROVIDER") or CFG["provider"]
    return {"gemini": GeminiEmbedder, "fake": FakeEmbedder}[provider]()
