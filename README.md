# jarvis-rag

**JARVIS**: a 100% free, locally-runnable LLM agent with RAG over your PDFs, Word docs,
code, websites and CSVs. Built in phases as a learn-by-building project; teaching notes
live in [`docs/`](docs/).

> Status: **Phase 1: from scratch** ✅

| Phase | Notes |
|---|---|
| 0: Environment | [docs/phase-0.md](docs/phase-0.md) |
| 1: LLM calls, embeddings, vector DB, RAG and an agent loop with no frameworks | [docs/phase-1.md](docs/phase-1.md) |

## Quick start

Prerequisites: [Ollama](https://ollama.com) running locally, [uv](https://docs.astral.sh/uv/), `make`.

```bash
make setup          # .venv (Python 3.12) + pinned deps + .env
make pull-models    # qwen3:4b-instruct, qwen3:1.7b, nomic-embed-text
make check          # environment doctor
make test           # no GPU/Ollama needed
```

## Learn by building

`learn/` rebuilds each core idea with nothing but HTTP calls and numpy:

```bash
.venv/bin/python learn/01_tokens_and_llm_call.py     # tokens, speed, temperature, context window
.venv/bin/python learn/02_embeddings_cosine.py       # embeddings + cosine similarity by hand
.venv/bin/python learn/03_vector_db_from_scratch.py  # vector DB in ~60 lines, brute force vs IVF
.venv/bin/python learn/04_rag_without_framework.py   # chunk -> embed -> retrieve -> cite
.venv/bin/python learn/05_agent_loop_from_scratch.py # the tool-calling loop behind every agent
.venv/bin/python learn/06_chunking_experiments.py    # chunk size/overlap vs hit@k and MRR
```

All configuration lives in `.env` (see [`.env.example`](.env.example)).

## Default models (tuned for a 4 GB GPU)

| Role | Model | Where it runs |
|---|---|---|
| Chat | `qwen3:4b-instruct` (Q4_K_M) | GPU, all layers, 8k context |
| Fallback chat | `qwen3:1.7b` | GPU |
| Embeddings | `nomic-embed-text` (768-dim) | CPU, so it never evicts the chat model |

With 8 GB+ VRAM, try `CHAT_MODEL=qwen3:8b`; with 12 GB+, `qwen3:14b`.
