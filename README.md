# ShopMind AI — RAG (Knowledge Base) component

Technical knowledge base + retrieval for the ShopMind AI multi-agent assistant.

## Stack
- Python 3.11+ (ingestion, chunking, evaluation)
- Qdrant (vector store) — Docker
- n8n (orchestration) — Docker

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then fill in the values
docker compose up -d            # starts Qdrant + n8n
```
- Qdrant dashboard: http://localhost:6333/dashboard
- n8n: http://localhost:5678

## Pipeline
| Step | Command | Output |
|---|---|---|
| 1. Collect Wikipedia | `python scripts/01_download_wikipedia.py` | `data/markdown/`, `data/kb_metadata.csv` |
| 1b. Add team guides | `python scripts/02_add_guides.py` | `data/markdown/guide_*.md`, metadata rows |
| 2. Clean / convert PDFs | *(next step)* | |
| 3. Chunk | *(next step)* | `data/processed/` |
| 4. Embed + index | *(next step)* | Qdrant collections |
| 5. Evaluate | *(next step)* | `eval/results/` |

All parameters (paths, chunk sizes, models, top-k) live in `config/settings.yaml`. Secrets live in `.env` (never committed).
