FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DATA_CACHE_DIR=/app/.cache/market_data
ENV JUGAAD_CACHE_DIR=/app/.cache/jugaad
ENV J_CACHE_DIR=/app/.cache/jugaad

WORKDIR /app

# ── 1. System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Upgrade pip ─────────────────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# ── 3. Copy dep files + src skeleton, then install ────────────────────────────
#    pip needs src/ to exist to resolve the package, but the cache is still
#    valid on rebuilds as long as pyproject.toml hasn't changed
COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

# ── 4. Copy remaining assets (docs, env example) ──────────────────────────────
COPY docs ./docs
COPY .env.example ./

# ── 5. Pre-create cache dirs ───────────────────────────────────────────────────
RUN mkdir -p /app/.cache/market_data /app/.cache/jugaad

EXPOSE 8501
CMD ["streamlit", "run", "src/indian_stock_pipeline/ui/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]