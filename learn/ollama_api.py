"""The entire "framework" the learn/ scripts use: three HTTP calls to Ollama.

POST /api/generate   prompt in  -> text out          (01)
POST /api/chat       messages in -> message out      (01, 04, 05)
POST /api/embed      texts in    -> vectors out      (02-06)

Model names and GPU options come from .env via jarvis_rag.config, so these
scripts run against exactly what the app will use.
"""

import importlib
import sys
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `jarvis_rag` importable when run as `python learn/0X_*.py`

from jarvis_rag.config import get_settings  # noqa: E402

S = get_settings()
CHAT_MODEL = S.chat_model
EMBED_MODEL = S.embed_model
CHAT_OPTIONS = {"num_ctx": S.num_ctx} | ({"num_gpu": S.num_gpu} if S.num_gpu is not None else {})
SAMPLE_DOCS = ROOT / "learn" / "sample_docs"

client = httpx.Client(base_url=S.ollama_base_url, timeout=600)


def generate(prompt: str, model: str = CHAT_MODEL, raw: bool = False, **options) -> dict:
    body = {"model": model, "prompt": prompt, "raw": raw, "stream": False,
            "think": S.reasoning, "options": CHAT_OPTIONS | options}
    return client.post("/api/generate", json=body).raise_for_status().json()


def chat(messages: list[dict], model: str = CHAT_MODEL, tools: list[dict] | None = None, **options) -> dict:
    body = {"model": model, "messages": messages, "stream": False,
            "think": S.reasoning, "options": CHAT_OPTIONS | options}
    if tools:
        body["tools"] = tools
    return client.post("/api/chat", json=body).raise_for_status().json()


def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    body = {"model": model, "input": texts, "options": {"num_gpu": S.embed_num_gpu}}
    return np.array(client.post("/api/embed", json=body).raise_for_status().json()["embeddings"], dtype=np.float32)


def tok_per_s(resp: dict, phase: str = "eval") -> float:
    """Ollama reports durations in nanoseconds. phase: 'eval' (generation) or 'prompt_eval'."""
    return resp[f"{phase}_count"] / (resp[f"{phase}_duration"] / 1e9)


def load_script(name: str):
    """Import a sibling script whose filename starts with a digit, e.g. load_script('03_vector_db_from_scratch')."""
    return importlib.import_module(name)


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
