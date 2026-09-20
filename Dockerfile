# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code edits do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY data/training_prompts.json data/eval_length_adversarial.json ./data/

# Train the classifier at build time so a fresh container can route on first
# request instead of falling back to the MODERATE tier.
RUN python scripts/train_classifier.py

# Run as a non-root user; it needs to own /app/data to write the SQLite file.
RUN useradd --create-home --uid 1000 autopilot \
    && chown -R autopilot:autopilot /app
USER autopilot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
