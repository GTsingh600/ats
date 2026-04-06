FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml /app/
COPY __init__.py client.py engine.py graders.py inference.py models.py planner.py tasks.py /app/
COPY server /app/server
COPY scripts /app/scripts

RUN pip install --no-cache-dir \
    "fastapi>=0.128.0" \
    "openai>=2.30.0" \
    "openenv-core[core]>=0.2.3" \
    "pydantic>=2.12.0" \
    "uvicorn>=0.41.0"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
