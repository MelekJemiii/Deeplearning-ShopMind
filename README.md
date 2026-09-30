# ShopMind AI — RAG (Knowledge Base) component

Technical knowledge base + retrieval for the ShopMind AI multi-agent assistant.

## Stack
- Python 3.11+ (ingestion, chunking, evaluation)
- Qdrant (vector store) — Docker
- Ollama + bge-m3 (local embeddings) — Docker
- n8n (orchestration) — Docker

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then fill in the values
docker compose up -d            # starts Qdrant + Ollama + n8n
docker compose exec ollama ollama pull bge-m3   # once: download the embedding model
```
- Qdrant dashboard: http://localhost:6333/dashboard
- n8n: http://localhost:5678

## Pipeline
| Step | Command | Output |
|---|---|---|
| 1. Collect Wikipedia | `python scripts/01_download_wikipedia.py` | `data/markdown/`, `data/kb_metadata.csv` |
| 1b. Add team guides | `python scripts/02_add_guides.py` | `data/markdown/guide_*.md`, metadata rows |
| 1c. Convert PDFs | `python scripts/03_convert_pdfs.py` | `data/markdown/pdf_*.md`, metadata rows |
| 2. Validate test set | `python scripts/04_validate_testset.py` | checks `eval/test_queries.yaml` labels |
| 3. Clean | `python scripts/05_clean.py` then `python scripts/04_validate_testset.py --processed` | `data/processed/*.md`, `cleaning_report.csv` |
| 4. Chunk (A + B) | `python scripts/06_chunk.py` | `data/chunks/chunks_{A,B}.jsonl`, `stats_{A,B}.json` |
| 5. Embed + index | `python scripts/07_embed_index.py` | Qdrant collections `shopmind_kb_<provider>_<A|B>` |
| 6. Evaluate retrieval | `python scripts/08_evaluate.py` | `eval/results/summary.md` + per-run JSON/CSV |
| 6b. Hybrid (dense + BM25) | `python scripts/07_embed_index.py --hybrid` then `python scripts/08_evaluate.py --strategies A B --modes dense hybrid` | `*_hybrid` collections, 4-run comparison |
| 6c. Reranking | `docker compose up -d reranker` then `python scripts/08_evaluate.py --strategies B --modes dense hybrid dense_rerank hybrid_rerank` | 4-run comparison incl. cross-encoder |
| 7. Retrieval API | `docker compose up -d --build rag-api` | `POST http://localhost:8000/search` (docs: `/docs`) |
| 3. Chunk | *(next step)* | `data/processed/` |
| 4. Embed + index | *(next step)* | Qdrant collections |
| 5. Evaluate | *(next step)* | `eval/results/` |

All parameters (paths, chunk sizes, models, top-k) live in `config/settings.yaml`. Secrets live in `.env` (never committed).

## Licensing notes
- Wikipedia content: CC BY-SA 4.0 (attribution in `data/kb_metadata.csv`).
- Manufacturer PDFs: public documents, used internally for academic purposes; not redistributed (excluded from git).
- `pymupdf4llm` / PyMuPDF is **AGPL-3.0**: fine for the academic project. For a commercial closed-source deployment,
  either buy a PyMuPDF commercial license or switch the converter to an MIT/Apache tool (e.g. pdfplumber, docling).
  The conversion is isolated in `scripts/03_convert_pdfs.py`, so swapping it does not affect the rest of the pipeline.

## Tests
```bash
pytest -q          # API tests run without any service (fake embedder + local Qdrant)
```
