FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY ui/ ui/
COPY models/ models/
COPY data/ data/

# Run as a non-root user (uid 1000 is what Hugging Face Spaces expects) and
# make the data/models folders writable for uploads, the database and retraining.
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health')"

# One process per container: scale by adding containers (docker compose --scale api=N)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
