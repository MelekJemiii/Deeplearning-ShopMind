# Retrieval evaluation

_Generated 2026-09-30 02:13, 31 queries, top-10 retrieval, no filters, no reranking. Hybrid = dense + BM25 fused with RRF; *_rerank = top-20 candidates reordered by BAAI/bge-reranker-v2-m3. Similarity columns use the dense cosine._

## Overall (in-scope queries)

| Run | Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Doc Hit@5 |
|---|---|---|---|---|---|---|---|
| ollama_B | bge-m3 | 0.82 | 0.93 | 0.96 | 0.96 | 0.882 | 0.96 |
| ollama_B_hybrid | bge-m3 | 0.75 | 0.96 | 1.00 | 1.00 | 0.854 | 1.00 |
| ollama_B_dense_rerank | bge-m3 | 0.86 | 1.00 | 1.00 | 1.00 | 0.923 | 1.00 |
| ollama_B_hybrid_rerank | bge-m3 | 0.86 | 1.00 | 1.00 | 1.00 | 0.923 | 1.00 |

## Hit@5 by query type

| Type | n | ollama_B | ollama_B_hybrid | ollama_B_dense_rerank | ollama_B_hybrid_rerank |
|---|---|---|---|---|---|
| compatibility | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| cross_lingual | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| definition | 7 | 1.00 | 1.00 | 1.00 | 1.00 |
| spec_lookup | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| use_case | 9 | 0.89 | 1.00 | 1.00 | 1.00 |

## Out-of-scope detection (top-1 similarity)

| Run | OOS max | OOS mean | In-scope min | In-scope mean | In-scope rejected if threshold = OOS max |
|---|---|---|---|---|---|
| ollama_B | 0.641 | 0.502 | 0.591 | 0.701 | 11% |
| ollama_B_hybrid | 0.641 | 0.473 | 0.424 | 0.677 | 21% |
| ollama_B_dense_rerank | 0.641 | 0.472 | 0.424 | 0.689 | 14% |
| ollama_B_hybrid_rerank | 0.641 | 0.472 | 0.424 | 0.689 | 14% |

## Out-of-scope detection with the reranker score (top-1)

| Run | OOS max | In-scope min | In-scope rejected if threshold = OOS max |
|---|---|---|---|
| ollama_B_dense_rerank | 0.154 | 0.172 | 0% |
| ollama_B_hybrid_rerank | 0.154 | 0.172 | 0% |

## Latency (mean per query)

| Run | Query embedding (ms) | Vector search (ms) | Rerank (ms) |
|---|---|---|---|
| ollama_B | 0.3 | 52.8 | 0.0 |
| ollama_B_hybrid | 0.1 | 62.5 | 0.0 |
| ollama_B_dense_rerank | 0.2 | 32.1 | 640.3 |
| ollama_B_hybrid_rerank | 0.2 | 57.7 | 492.2 |

_Embedding latency is ~0 when query vectors come from the cache (reruns)._

## Misses (no relevant chunk in top 5)

**ollama_B**: q06 (use_case, rank >10, top1 kb_017#B006)

**ollama_B_hybrid**: none

**ollama_B_dense_rerank**: none

**ollama_B_hybrid_rerank**: none

