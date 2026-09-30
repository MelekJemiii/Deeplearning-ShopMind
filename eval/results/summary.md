# Retrieval evaluation

_Generated 2026-09-30 01:45, 31 queries, top-10 retrieval, no filters, no reranking. Hybrid = dense + BM25 fused with RRF; its similarity columns use the dense cosine._

## Overall (in-scope queries)

| Run | Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Doc Hit@5 |
|---|---|---|---|---|---|---|---|
| ollama_A | bge-m3 | 0.71 | 0.89 | 0.93 | 0.96 | 0.810 | 0.93 |
| ollama_A_hybrid | bge-m3 | 0.71 | 0.96 | 1.00 | 1.00 | 0.830 | 1.00 |
| ollama_B | bge-m3 | 0.82 | 0.93 | 0.96 | 0.96 | 0.882 | 0.96 |
| ollama_B_hybrid | bge-m3 | 0.64 | 0.96 | 1.00 | 1.00 | 0.801 | 1.00 |

## Hit@5 by query type

| Type | n | ollama_A | ollama_A_hybrid | ollama_B | ollama_B_hybrid |
|---|---|---|---|---|---|
| compatibility | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| cross_lingual | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| definition | 7 | 1.00 | 1.00 | 1.00 | 1.00 |
| spec_lookup | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| use_case | 9 | 0.78 | 1.00 | 0.89 | 1.00 |

## Out-of-scope detection (top-1 similarity)

| Run | OOS max | OOS mean | In-scope min | In-scope mean | In-scope rejected if threshold = OOS max |
|---|---|---|---|---|---|
| ollama_A | 0.621 | 0.484 | 0.572 | 0.691 | 7% |
| ollama_A_hybrid | 0.607 | 0.454 | 0.449 | 0.670 | 18% |
| ollama_B | 0.641 | 0.502 | 0.591 | 0.701 | 11% |
| ollama_B_hybrid | 0.641 | 0.473 | 0.424 | 0.664 | 25% |

## Latency (mean per query)

| Run | Query embedding (ms) | Vector search (ms) |
|---|---|---|
| ollama_A | 0.1 | 17.4 |
| ollama_A_hybrid | 0.0 | 24.2 |
| ollama_B | 0.1 | 15.5 |
| ollama_B_hybrid | 0.0 | 23.5 |

_Embedding latency is ~0 when query vectors come from the cache (reruns)._

## Misses (no relevant chunk in top 5)

**ollama_A**: q03 (use_case, rank 7, top1 kb_041#A012), q06 (use_case, rank >10, top1 kb_017#A005)

**ollama_A_hybrid**: none

**ollama_B**: q06 (use_case, rank >10, top1 kb_017#B006)

**ollama_B_hybrid**: none

