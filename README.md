# jarvis-rag

**JARVIS**: a 100% free, locally-runnable LLM agent with RAG over your PDFs, Word docs,
code, websites and CSVs. Built in phases as a learn-by-building project; teaching notes
live in [`docs/`](docs/).

> Status: **Phase 0: environment** ✅ ([notes](docs/phase-0.md))

## Quick start

Prerequisites: [Ollama](https://ollama.com) running locally, [uv](https://docs.astral.sh/uv/), `make`.

```bash
make setup          # .venv (Python 3.12) + pinned deps + .env
make pull-models    # qwen3:4b-instruct, qwen3:1.7b, nomic-embed-text
make check          # environment doctor
make test           # no GPU/Ollama needed
```

All configuration lives in `.env` (see [`.env.example`](.env.example)).

## Default models (tuned for a 4 GB GPU)

| Role | Model | Where it runs |
|---|---|---|
| Chat | `qwen3:4b-instruct` (Q4_K_M) | GPU, all layers, 8k context |
| Fallback chat | `qwen3:1.7b` | GPU |
| Embeddings | `nomic-embed-text` (768-dim) | CPU, so it never evicts the chat model |

With 8 GB+ VRAM, try `CHAT_MODEL=qwen3:8b`; with 12 GB+, `qwen3:14b`.
