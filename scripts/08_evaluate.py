"""
Step 6 — Evaluate retrieval quality on the frozen test set.

For every (embedding provider, chunking strategy) collection and every test query:
  retrieve top-K chunks, find the rank of the first RELEVANT chunk
  (chunk from a relevant doc AND containing that doc's evidence, normalized match).
Metrics (in-scope queries): Hit@k, MRR, plus a doc-level Hit@5 diagnostic
  (right document, wrong chunk?). Breakdown per query type.
Out-of-scope queries: top-1 similarity compared with in-scope ones -> can a score
  threshold reject questions the KB cannot answer?
Latency: query embedding + vector search, per query.

Outputs (versioned, they are report evidence):
  eval/results/<run>.json         full per-query details (top-5 ids + scores)
  eval/results/per_query_<run>.csv
  eval/results/summary.md         comparison tables across runs

Usage: python scripts/08_evaluate.py                       # config provider, strategies A and B, dense
       python scripts/08_evaluate.py --providers ollama gemini --strategies A
       python scripts/08_evaluate.py --strategies A B --modes dense hybrid
       python scripts/08_evaluate.py --strategies B --modes dense hybrid dense_rerank hybrid_rerank
"""
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import ROOT, SETTINGS  # noqa: E402
from shopmind_rag.embedding import get_embedder  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag.rerank import get_reranker  # noqa: E402
from shopmind_rag.text import contains_evidence  # noqa: E402
from shopmind_rag import sparse  # noqa: E402
from shopmind_rag.vectorstore import collection_name, get_client, search, search_hybrid  # noqa: E402

log = get_logger("evaluate")
EV = SETTINGS["evaluation"]
KS = EV["k_values"]
K_MAX = max(KS)


def arg_list(flag, default):
    if flag not in sys.argv:
        return default
    i, out = sys.argv.index(flag) + 1, []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        out.append(sys.argv[i]); i += 1
    return out


def first_relevant_rank(hits, relevant) -> tuple[int | None, int | None]:
    """(rank of first relevant chunk, rank of first chunk from a relevant doc), 1-based."""
    ev = {r["doc_id"]: r["evidence"] for r in relevant}
    chunk_rank = doc_rank = None
    for i, h in enumerate(hits, start=1):
        if h["doc_id"] in ev:
            doc_rank = doc_rank or i
            if contains_evidence(h["text"], ev[h["doc_id"]]):
                chunk_rank = i
                break
    return chunk_rank, doc_rank


def evaluate(provider, strategy, queries, client, mode="dense"):
    """mode: dense | hybrid | dense_rerank | hybrid_rerank"""
    emb = get_embedder(provider)
    base, use_rerank = mode.replace("_rerank", ""), mode.endswith("_rerank")
    reranker = get_reranker() if use_rerank else None
    n_first = max(K_MAX, SETTINGS["rerank"]["candidates"]) if use_rerank else K_MAX
    name = collection_name(strategy, provider, base)
    if not client.collection_exists(name):
        log.warning("SKIP %s: collection not found (run 07_embed_index.py)", name)
        return None
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        vec = emb.embed_query(q["query"])
        t1 = time.perf_counter()
        if base == "hybrid":
            hits = search_hybrid(client, name, vec, sparse.query_vector(q["query"]), n_first,
                                 max(n_first, SETTINGS["hybrid"]["prefetch_k"]))
        else:
            hits = search(client, name, vec, n_first)
        t2 = time.perf_counter()
        if reranker:
            hits = reranker.rerank(q["query"], hits, K_MAX)
        t3 = time.perf_counter()
        rank, doc_rank = first_relevant_rank(hits, q.get("relevant") or [])
        rows.append({"id": q["id"], "type": q["type"], "lang": q["lang"], "query": q["query"],
                     "rank": rank, "doc_rank": doc_rank, "top1_score": round(hits[0]["score"], 4) if hits else 0,
                     "top1_rerank_score": hits[0].get("rerank_score") if hits else None,
                     "embed_ms": round(1000 * (t1 - t0), 1), "search_ms": round(1000 * (t2 - t1), 1),
                     "rerank_ms": round(1000 * (t3 - t2), 1),
                     "top5": [(h["chunk_id"], round(h["score"], 4)) for h in hits[:5]]})
    run = f"{provider}_{strategy}" + ("" if mode == "dense" else f"_{mode}")
    return {"run": run, "provider": provider, "strategy": strategy, "mode": mode,
            "model": emb.model, "collection": name, "rows": rows, "metrics": metrics(rows)}


def metrics(rows):
    ins = [r for r in rows if r["type"] != "out_of_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]

    def block(rs):
        n = len(rs)
        out = {f"hit@{k}": round(sum(r["rank"] is not None and r["rank"] <= k for r in rs) / n, 3) for k in KS}
        out["mrr"] = round(sum(1 / r["rank"] for r in rs if r["rank"]) / n, 3)
        out["doc_hit@5"] = round(sum(r["doc_rank"] is not None and r["doc_rank"] <= 5 for r in rs) / n, 3)
        out["n"] = n
        return out

    by_type = defaultdict(list)
    for r in ins:
        by_type[r["type"]].append(r)
    m = {"overall": block(ins), "by_type": {t: block(rs) for t, rs in sorted(by_type.items())}}
    top1_in = [r["top1_score"] for r in ins]
    top1_ok = [r["top1_score"] for r in ins if r["rank"] == 1]
    top1_oos = [r["top1_score"] for r in oos]
    if oos:
        m["out_of_scope"] = {
            "oos_top1_max": max(top1_oos), "oos_top1_mean": round(statistics.mean(top1_oos), 3),
            "in_scope_top1_min": min(top1_in), "in_scope_top1_mean": round(statistics.mean(top1_in), 3),
            "correct_top1_min": min(top1_ok) if top1_ok else None,
            # fraction of in-scope queries that a threshold just above the worst OOS score would wrongly reject
            "in_scope_rejected_at_oos_max": round(sum(s <= max(top1_oos) for s in top1_in) / len(top1_in), 3)}
    m["latency_ms"] = {"embed_mean": round(statistics.mean(r["embed_ms"] for r in rows), 1),
                       "search_mean": round(statistics.mean(r["search_ms"] for r in rows), 1),
                       "rerank_mean": round(statistics.mean(r["rerank_ms"] for r in rows), 1)}
    rr_in = [r["top1_rerank_score"] for r in ins if r["top1_rerank_score"] is not None]
    rr_oos = [r["top1_rerank_score"] for r in oos if r["top1_rerank_score"] is not None]
    if rr_in and rr_oos:  # does the cross-encoder score separate answerable from unanswerable better?
        m["out_of_scope_rerank"] = {
            "oos_top1_max": max(rr_oos), "in_scope_top1_min": min(rr_in),
            "in_scope_rejected_at_oos_max": round(sum(x <= max(rr_oos) for x in rr_in) / len(rr_in), 3)}
    return m


def write_outputs(results):
    out = ROOT / EV["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for res in results:
        (out / f"{res['run']}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(out / f"per_query_{res['run']}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "type", "lang", "rank", "doc_rank", "top1_score", "query", "top1_chunk"])
            for r in res["rows"]:
                w.writerow([r["id"], r["type"], r["lang"], r["rank"] or "", r["doc_rank"] or "", r["top1_score"],
                            r["query"], r["top5"][0][0] if r["top5"] else ""])

    md = [f"# Retrieval evaluation\n\n_Generated {datetime.now():%Y-%m-%d %H:%M}, "
          f"{len(results[0]['rows'])} queries, top-{K_MAX} retrieval, no filters, no reranking. "
          f"Hybrid = dense + BM25 fused with RRF; *_rerank = top-{SETTINGS['rerank']['candidates']} candidates "
          f"reordered by {SETTINGS['rerank']['model']}. Similarity columns use the dense cosine._\n",
          "## Overall (in-scope queries)\n",
          "| Run | Model | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR | Doc Hit@5 |",
          "|---|---|" + "---|" * (len(KS) + 2)]
    for r in results:
        o = r["metrics"]["overall"]
        md.append(f"| {r['run']} | {r['model']} | " + " | ".join(f"{o[f'hit@{k}']:.2f}" for k in KS)
                  + f" | {o['mrr']:.3f} | {o['doc_hit@5']:.2f} |")
    types = sorted({t for r in results for t in r["metrics"]["by_type"]})
    md += ["\n## Hit@5 by query type\n", "| Type | n | " + " | ".join(r["run"] for r in results) + " |",
           "|---|---|" + "---|" * len(results)]
    for t in types:
        n = results[0]["metrics"]["by_type"][t]["n"]
        md.append(f"| {t} | {n} | " + " | ".join(f"{r['metrics']['by_type'][t]['hit@5']:.2f}" for r in results) + " |")
    md += ["\n## Out-of-scope detection (top-1 similarity)\n",
           "| Run | OOS max | OOS mean | In-scope min | In-scope mean | In-scope rejected if threshold = OOS max |",
           "|---|---|---|---|---|---|"]
    for r in results:
        o = r["metrics"].get("out_of_scope")
        if o:
            md.append(f"| {r['run']} | {o['oos_top1_max']:.3f} | {o['oos_top1_mean']:.3f} | {o['in_scope_top1_min']:.3f} "
                      f"| {o['in_scope_top1_mean']:.3f} | {o['in_scope_rejected_at_oos_max']:.0%} |")
    rr = [r for r in results if "out_of_scope_rerank" in r["metrics"]]
    if rr:
        md += ["\n## Out-of-scope detection with the reranker score (top-1)\n",
               "| Run | OOS max | In-scope min | In-scope rejected if threshold = OOS max |", "|---|---|---|---|"]
        for r in rr:
            o = r["metrics"]["out_of_scope_rerank"]
            md.append(f"| {r['run']} | {o['oos_top1_max']:.3f} | {o['in_scope_top1_min']:.3f} "
                      f"| {o['in_scope_rejected_at_oos_max']:.0%} |")
    md += ["\n## Latency (mean per query)\n", "| Run | Query embedding (ms) | Vector search (ms) | Rerank (ms) |",
           "|---|---|---|---|"]
    for r in results:
        lat = r["metrics"]["latency_ms"]
        md.append(f"| {r['run']} | {lat['embed_mean']} | {lat['search_mean']} | {lat['rerank_mean']} |")
    md.append("\n_Embedding latency is ~0 when query vectors come from the cache (reruns)._")
    md += ["\n## Misses (no relevant chunk in top 5)\n"]
    for r in results:
        miss = [x for x in r["rows"] if x["type"] != "out_of_scope" and not (x["rank"] and x["rank"] <= 5)]
        md.append(f"**{r['run']}**: " + (", ".join(f"{x['id']} ({x['type']}, rank {x['rank'] or '>10'}, "
                                                    f"top1 {x['top5'][0][0]})" for x in miss) or "none"))
        md.append("")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("Results written to %s", out)


def main():
    queries = yaml.safe_load(open(ROOT / "eval" / "test_queries.yaml", encoding="utf-8"))["queries"]
    providers = arg_list("--providers", [SETTINGS["embedding"]["provider"]])
    strategies = arg_list("--strategies", list(SETTINGS["chunking"]["strategies"]))
    modes = arg_list("--modes", ["dense"])
    client, results = get_client(), []
    runs = [(p, s, m) for p in providers for s in strategies for m in modes]
    for p, s, mode in runs:
        res = evaluate(p, s, queries, client, mode)
        if not res:
            continue
        o = res["metrics"]["overall"]
        log.info("[%s] Hit@1 %.2f  Hit@3 %.2f  Hit@5 %.2f  Hit@10 %.2f  MRR %.3f  DocHit@5 %.2f",
                 res["run"], o["hit@1"], o["hit@3"], o["hit@5"], o["hit@10"], o["mrr"], o["doc_hit@5"])
        results.append(res)
    if results:
        write_outputs(results)


if __name__ == "__main__":
    main()
