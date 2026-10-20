# JARVIS task runner. `make help` lists targets.

PY := .venv/bin/python

# Model names come from .env when present, so `make pull-models` pulls what the app uses.
-include .env
CHAT_MODEL ?= qwen3:4b-instruct
FALLBACK_CHAT_MODEL ?= qwen3:1.7b
EMBED_MODEL ?= nomic-embed-text
DATA_DIR ?= data

.PHONY: help setup lock pull-models check test demo-data ingest sources eval eval-answers chat eval-agent api

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

demo-data: ## write the Halcyon demo corpus (md, pdf, docx, csv) into data/
	$(PY) -m jarvis_rag.demo_data

ingest: ## ingest data/docs, data/code and data/csv (idempotent; re-run any time)
	$(PY) -m jarvis_rag.ingest --path $(DATA_DIR)/docs --path $(DATA_DIR)/code --csv $(DATA_DIR)/csv

sources: ## list what is in the knowledge base
	$(PY) -m jarvis_rag.ingest --list

eval: ## retrieval quality: hit@k / MRR for similarity, mmr, hybrid, rerank
	$(PY) eval/run_eval.py

eval-answers: ## + generate answers and grade them with the LLM as judge (slow)
	$(PY) eval/run_eval.py --modes $(or $(RETRIEVAL_MODE),rerank) --answers

chat: ## talk to JARVIS in the terminal (memory: storage/memory.db)
	$(PY) -m jarvis_rag.agent

eval-agent: ## grade the agent: tool choice, numbers, citations, abstention (~4 min)
	$(PY) eval/run_agent_eval.py

api: ## HTTP API on API_HOST:API_PORT (docs at /docs)
	$(PY) -m jarvis_rag.api

test: ## run the test suite (no GPU or Ollama needed)
	$(PY) -m pytest -q
