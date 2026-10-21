# JARVIS: API + web UI in one CPU container. The LLM is either the host's Ollama
# (docker compose service `jarvis`, host network) or a hosted provider (`jarvis-cpu`).
#
#   docker compose build
#   docker compose up -d jarvis            # http://127.0.0.1:8000/ui
#
# Base images are pinned by digest: a tag like 3.12-slim-trixie is rebuilt for OS
# patches, so the digest is what makes a build reproducible. To take new patches,
# `docker pull` the tag and update the digest here.

# ---- build stage: resolve nothing, just install the hash-locked set ----------------
FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM python:3.12-slim-trixie@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47 AS build

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY requirements.txt .
# --require-hashes: every wheel must match the sha256 recorded by `make lock`, so a
# tampered or swapped package on the index fails the build. The cache mount keeps
# downloads between builds without putting them in the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv pip sync --python /opt/venv/bin/python --require-hashes requirements.txt

# ---- runtime stage: no compiler, no uv, no build cache -----------------------------
FROM python:3.12-slim-trixie@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47

LABEL org.opencontainers.image.title="jarvis-rag" \
      org.opencontainers.image.description="Local RAG agent: API + web UI" \
      org.opencontainers.image.source="https://github.com/Tesseract-coder/jarvis-rag"

COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY jarvis_rag/ jarvis_rag/

# Everything writable lives under /app/storage and /app/data (volumes) or /tmp (tmpfs),
# so the container can run with a read-only root filesystem.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    HF_HOME=/app/storage/models/hf \
    GRADIO_TEMP_DIR=/tmp/gradio \
    GRADIO_ANALYTICS_ENABLED=False \
    STORAGE_DIR=/app/storage \
    DATA_DIR=/app/data \
    API_PORT=8000

# Non-root. Compose overrides the uid to match yours so bind-mounted files stay yours.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin jarvis \
 && mkdir -p /app/storage /app/data && chown jarvis /app/storage /app/data
USER jarvis

EXPOSE 8000
# No curl in slim images; the standard library is enough. /health is 503 when Ollama is
# unreachable, so "unhealthy" means "can't answer questions", not just "process alive".
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"API_PORT\"]}/health', timeout=4)"]

CMD ["python", "-m", "jarvis_rag.ui", "--with-api"]
