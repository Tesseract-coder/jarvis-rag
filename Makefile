# JARVIS task runner. `make help` lists targets.

PY := .venv/bin/python

# Model names come from .env when present, so `make pull-models` pulls what the app uses.
-include .env
CHAT_MODEL ?= qwen3:4b-instruct
FALLBACK_CHAT_MODEL ?= qwen3:1.7b
EMBED_MODEL ?= nomic-embed-text

.PHONY: help setup lock pull-models check test

help:
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## create .venv (Python 3.12) and install pinned deps; create .env
	uv venv --python 3.12 .venv
	uv pip sync requirements.txt --python $(PY)
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

lock: ## re-resolve requirements.in -> requirements.txt
	uv pip compile requirements.in -o requirements.txt --python-version 3.12

pull-models: ## pull the chat, fallback and embedding models into Ollama
	ollama pull $(CHAT_MODEL)
	ollama pull $(FALLBACK_CHAT_MODEL)
	ollama pull $(EMBED_MODEL)

check: ## environment doctor: GPU, Ollama, models, VRAM, smoke tests
	$(PY) -m jarvis_rag.check_env

test: ## run the test suite (no GPU or Ollama needed)
	$(PY) -m pytest -q
