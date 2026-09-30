"""
Step 5 — Embed the chunks of each strategy and index them in Qdrant.

Input:  data/chunks/chunks_<S>.jsonl
Output: Qdrant collections shopmind_kb_<S> (vectors + full chunk payload, keyword indexes for filters)
        embeddings cached in data/embeddings/ (rerunning costs nothing)
Ends with a sanity check: one test query per strategy, top-3 printed.

Prerequisites: docker compose up -d qdrant ollama ; docker compose exec ollama ollama pull bge-m3
Usage: python scripts/07_embed_index.py [--strategy A] [--hybrid]
       --hybrid also builds <collection>_hybrid (dense + BM25 sparse), reusing cached dense vectors
       set EMBEDDING_PROVIDER=gemini  (Windows)  to index with another provider
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import ROOT, SETTINGS  # noqa: E402
from shopmind_rag.embedding import DailyQuotaExceeded, get_embedder  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag import sparse  # noqa: E402
from shopmind_rag.vectorstore import (collection_name, get_client, recreate, recreate_hybrid, search,  # noqa: E402
                                      search_hybrid, upsert, upsert_hybrid)

log = get_logger("embed_index")
SANITY_QUERY = "Combien de RAM faut-il pour faire de la data science ?"


def main():
    only = sys.argv[sys.argv.index("--strategy") + 1] if "--strategy" in sys.argv else None
    hybrid = "--hybrid" in sys.argv
    emb, client = get_embedder(), get_client()
    log.info("Embedder: %s (%s, %d dims) | Qdrant: %s", emb.provider, emb.model, emb.dims,
             type(client).__name__)
    for s in SETTINGS["chunking"]["strategies"]:
        if only and s != only:
            continue
        path = ROOT / SETTINGS["chunking"]["output_dir"] / f"chunks_{s}.jsonl"
        chunks = [json.loads(line) for line in open(path, encoding="utf-8")]
        try:
            vectors = emb.embed_documents([(c["title"], c["text"]) for c in chunks])
        except DailyQuotaExceeded:
            cached = sum(emb.cache.get(emb.format_document(c["title"], c["text"])) is not None for c in chunks)
            log.error("[%s] DAILY embedding quota reached: %d/%d chunks embedded and cached. "
                      "Quota resets at midnight Pacific Time; rerun then, cached chunks are not re-sent.",
                      s, cached, len(chunks))
            sys.exit(2)
        name = collection_name(s, emb.provider)
        recreate(client, name, emb.dims)
        upsert(client, name, chunks, vectors)
        count = client.count(name).count
        log.info("[%s] %d chunks -> %d points in '%s'", s, len(chunks), count, name)
        if count != len(chunks):
            log.error("[%s] point count mismatch (duplicate chunk_ids?)", s)
        qvec = emb.embed_query(SANITY_QUERY)
        hits = search(client, name, qvec, 3)
        log.info("[%s] sanity '%s'", s, SANITY_QUERY)
        for h in hits:
            log.info("   %.3f  %-16s %s", h["score"], h["chunk_id"], h["text"][:80].replace("\n", " "))

        if hybrid:
            docs = [f"{c['title']}\n\n{c['text']}" for c in chunks]
            avgdl = sparse.avg_doc_len(docs)
            hname = collection_name(s, emb.provider, "hybrid")
            recreate_hybrid(client, hname, emb.dims)
            upsert_hybrid(client, hname, chunks, vectors, [sparse.doc_vector(d, avgdl) for d in docs])
            log.info("[%s] hybrid collection '%s': %d points (avg doc length %.0f tokens)", s, hname,
                     client.count(hname).count, avgdl)
            for h in search_hybrid(client, hname, qvec, sparse.query_vector(SANITY_QUERY), 3,
                                   SETTINGS["hybrid"]["prefetch_k"]):
                log.info("   dense %.3f  rrf %.4f  %-16s", h["score"], h["fusion_score"], h["chunk_id"])
    log.info("Embedding API calls this run: %d", emb.api_calls)


if __name__ == "__main__":
    main()
