# JARVIS task runner. `make help` lists targets.

PY := .venv/bin/python

# Model names come from .env when present, so `make pull-models` pulls what the app uses.
-include .env
CHAT_MODEL ?= qwen3:4b-instruct
FALLBACK_CHAT_MODEL ?= qwen3:1.7b
EMBED_MODEL ?= nomic-embed-text
DATA_DIR ?= data

.PHONY: help setup lock pull-models check test demo-data ingest sources eval eval-answers chat eval-agent api ui app providers docker-build docker-up docker-down docker-logs docker-ingest space space-test space-push train-setup train-data train export-reader eval-reader

help:
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## create .venv (Python 3.12) and install pinned deps; create .env
	uv venv --python 3.12 .venv
	uv pip sync requirements.txt --python $(PY)
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

lock: ## re-resolve requirements.in -> requirements.txt
	uv pip compile requirements.in -o requirements.txt --python-version 3.12 --generate-hashes

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

ui: ## web UI on UI_HOST:UI_PORT, talking to the API at API_URL (run `make api` too)
	$(PY) -m jarvis_rag.ui

app: ## one process: API on API_PORT with the web UI at /ui
	$(PY) -m jarvis_rag.ui --with-api

providers: ## list models your API keys can use (+ free OpenRouter models); PING=1 to test calls
	$(PY) -m jarvis_rag.providers $(if $(PING),--ping)

docker-build: ## build the jarvis-rag image (CPU; hash-locked deps, pinned base images)
	docker compose build jarvis

docker-up: ## run API + UI in Docker on 127.0.0.1:8000, using the host's Ollama
	docker compose up -d jarvis && echo "UI: http://127.0.0.1:8000/ui   API docs: http://127.0.0.1:8000/docs"

docker-down: ## stop the containers
	docker compose --profile cpu down

docker-logs: ## follow the container logs
	docker compose logs -f jarvis

docker-ingest: ## ingest data/ from inside the image (stop docker-up first, or use POST /ingest/file)
	docker compose run --rm jarvis python -m jarvis_rag.ingest --path data/docs --path data/code --csv data/csv

space: ## assemble the Hugging Face Space bundle in build/hf-space (nothing is uploaded)
	deploy/build_space.sh

space-test: space ## build the Space image and run it locally the way HF does (uid 1000, port 7860)
	docker build -t jarvis-space build/hf-space
	@echo "run: docker run --rm -p 127.0.0.1:7860:7860 --user 1000 -e API_KEY=... -e UI_PASSWORD=... -e GROQ_API_KEY=... jarvis-space"

space-push: space ## create/update the Space (needs: hf auth login, SPACE=user/name, .space-secrets); private unless PUBLIC=1
	@test -n "$(SPACE)" || (echo "usage: make space-push SPACE=your-hf-user/jarvis-rag [PUBLIC=1]" && exit 2)
	@test -f .space-secrets || (echo "missing .space-secrets: cp deploy/hf-space/secrets.example .space-secrets and fill it in" && exit 2)
	.venv/bin/hf auth whoami
	.venv/bin/hf repos create $(SPACE) --type space --sdk docker $(if $(PUBLIC),--public,--private) --secrets-file .space-secrets --exist-ok
	.venv/bin/hf upload $(SPACE) build/hf-space . --repo-type space --commit-message "Deploy JARVIS $$(git rev-parse --short HEAD)"
	@echo "Building on HF: https://huggingface.co/spaces/$(SPACE)  (app: https://$$(echo $(SPACE) | tr '/' '-').hf.space/ui/)"

train-setup: ## Phase 10: separate .venv-train with Unsloth + PyTorch/CUDA wheels (~6 GB, no system CUDA needed)
	uv venv .venv-train --python 3.12
	uv pip sync requirements-train.txt --python .venv-train/bin/python --require-hashes

train-data: ## build the reader dataset from docs/ + learn/ with a Groq teacher (needs GROQ_API_KEY; resumable)
	$(PY) finetune/make_dataset.py

train: ## QLoRA fine-tune Qwen3-1.7B as the reader (~4 h on a GTX 1650; unloads Ollama models first)
	-curl -s localhost:11434/api/ps | $(PY) -c "import json,sys,urllib.request as u; [u.urlopen(u.Request('http://localhost:11434/api/generate', json.dumps({'model': m['name'], 'keep_alive': 0}).encode())) for m in json.load(sys.stdin)['models']]"
	systemd-inhibit --what=sleep:idle:handle-lid-switch --why="QLoRA training" .venv-train/bin/python finetune/train.py --epochs 1  # a suspend kills the CUDA context

export-reader: ## merge LoRA -> GGUF -> Q4_K_M (builds llama-quantize, CPU) -> ollama create jarvis-reader:1.7b
	.venv-train/bin/python finetune/export.py

eval-reader: ## compare readers on identical Halcyon contexts (needs GROQ_API_KEY for the judge)
	$(PY) finetune/eval_reader.py --models qwen3:1.7b jarvis-reader:1.7b qwen3:4b-instruct

test: ## run the test suite (no GPU or Ollama needed)
	$(PY) -m pytest -q tests
