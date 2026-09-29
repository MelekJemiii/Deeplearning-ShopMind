"""
Step 5 — Embed the chunks of each strategy and index them in Qdrant.

Input:  data/chunks/chunks_<S>.jsonl
Output: Qdrant collections shopmind_kb_<S> (vectors + full chunk payload, keyword indexes for filters)
        embeddings cached in data/embeddings/ (rerunning costs nothing)
Ends with a sanity check: one test query per strategy, top-3 printed.

Prerequisites: docker compose up -d qdrant ; GEMINI_API_KEY and QDRANT_URL in .env
Usage: python scripts/07_embed_index.py [--strategy A]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import ROOT, SETTINGS  # noqa: E402
from shopmind_rag.embedding import get_embedder  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag.vectorstore import collection_name, get_client, recreate, search, upsert  # noqa: E402

log = get_logger("embed_index")
SANITY_QUERY = "Combien de RAM faut-il pour faire de la data science ?"


def main():
    only = sys.argv[sys.argv.index("--strategy") + 1] if "--strategy" in sys.argv else None
    emb, client = get_embedder(), get_client()
    log.info("Embedder: %s (%s, %d dims) | Qdrant: %s", emb.provider, emb.model, emb.dims,
             type(client).__name__)
    for s in SETTINGS["chunking"]["strategies"]:
        if only and s != only:
            continue
        path = ROOT / SETTINGS["chunking"]["output_dir"] / f"chunks_{s}.jsonl"
        chunks = [json.loads(line) for line in open(path, encoding="utf-8")]
        vectors = emb.embed_documents([(c["title"], c["text"]) for c in chunks])
        name = collection_name(s)
        recreate(client, name, emb.dims)
        upsert(client, name, chunks, vectors)
        count = client.count(name).count
        log.info("[%s] %d chunks -> %d points in '%s'", s, len(chunks), count, name)
        if count != len(chunks):
            log.error("[%s] point count mismatch (duplicate chunk_ids?)", s)
        hits = search(client, name, emb.embed_query(SANITY_QUERY), 3)
        log.info("[%s] sanity '%s'", s, SANITY_QUERY)
        for h in hits:
            log.info("   %.3f  %-16s %s", h["score"], h["chunk_id"], h["text"][:80].replace("\n", " "))
    log.info("Embedding API calls this run: %d", emb.api_calls)


if __name__ == "__main__":
    main()
