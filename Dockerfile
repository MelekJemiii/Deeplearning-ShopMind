# Retrieval API image: serves POST /search for the n8n RAG Agent
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY config ./config
COPY src ./src
RUN useradd --create-home app && mkdir -p logs && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "shopmind_rag.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
