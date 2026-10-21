# Hugging Face Docker Space: JARVIS on the free CPU tier.
# Assembled into build/hf-space by `make space`; tested locally with `make space-test`.
#
# Differences from the main Dockerfile, all forced by how Spaces work:
#   - port 7860 (README app_port), container runs as uid 1000
#   - the disk is wiped on every restart, so the demo index is BUILT INTO THE IMAGE
#   - no Ollama: FastEmbed (bge-small) embeddings in-process, a hosted LLM via secrets
#   - read-only knowledge base: visitors chat, they don't upload or crawl
# Secrets (Space settings → Variables and secrets): GROQ_API_KEY, API_KEY, UI_PASSWORD.

FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM python:3.12-slim-trixie@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv pip sync --python /opt/venv/bin/python --require-hashes /tmp/requirements.txt

FROM python:3.12-slim-trixie@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47
COPY --from=build /opt/venv /opt/venv

# Spaces run the container as uid 1000: create that user and give it /app.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin user \
 && mkdir -p /app && chown user /app
USER user
WORKDIR /app
COPY --chown=user jarvis_rag/ jarvis_rag/
COPY --chown=user learn/sample_docs/ learn/sample_docs/
COPY --chown=user eval/ eval/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/storage/models/hf \
    GRADIO_ANALYTICS_ENABLED=False \
    STORAGE_DIR=/app/storage \
    DATA_DIR=/app/data \
    API_HOST=0.0.0.0 \
    API_PORT=7860 \
    LLM_PROVIDER=groq \
    LLM_FALLBACK= \
    EMBED_PROVIDER=fastembed \
    FASTEMBED_MODEL=BAAI/bge-small-en-v1.5 \
    EMBED_QUERY_PREFIX="Represent this sentence for searching relevant passages: " \
    EMBED_DOCUMENT_PREFIX= \
    COLLECTION_NAME=jarvis-bge \
    KB_READ_ONLY=true \
    MEMORY_BACKEND=sqlite

# Bake the knowledge base and every model download into the image, so a restart (or a
# cold start after the Space slept) needs no network and no re-ingest. Then GATE the
# build on the retrieval eval: a missing document or a broken embedder fails here,
# instead of shipping a demo that quietly answers worse. (The first version of this
# image shipped without its five Markdown docs and nothing complained.) The gate tests
# the mode that is served (RETRIEVAL_MODE=rerank); hybrid alone scores lower here.
RUN python -m jarvis_rag.demo_data \
 && python -m jarvis_rag.ingest --path data/docs --path data/code --csv data/csv \
 && python eval/run_eval.py --modes rerank --min-hit5 0.95

EXPOSE 7860
CMD ["python", "-m", "jarvis_rag.ui", "--with-api"]
