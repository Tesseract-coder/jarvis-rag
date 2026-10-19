from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis_rag.config import Settings


def make(**env) -> Settings:
    # _env_file=None: never let a developer's real .env leak into tests.
    return Settings(_env_file=None, **env)


def test_defaults_target_a_4gb_gpu():
    s = make()
    assert s.agent_name == "JARVIS"
    assert s.address_user_as == ""
    assert s.llm_provider == "ollama"
    assert s.chat_model == "qwen3:4b-instruct"
    assert s.embed_model == "nomic-embed-text"
    assert s.num_gpu == 99
    assert s.embed_num_gpu == 0
    assert s.reasoning is False
    assert (s.chunk_size, s.chunk_overlap) == (800, 120)
    assert (s.code_chunk_size, s.code_chunk_overlap) == (1200, 150)


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "llama3.1:8b")
    monkeypatch.setenv("NUM_CTX", "4096")
    monkeypatch.setenv("ADDRESS_USER_AS", "sir")
    s = make()
    assert s.chat_model == "llama3.1:8b"
    assert s.num_ctx == 4096
    assert s.address_user_as == "sir"


def test_env_file_is_read(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AGENT_NAME=FRIDAY\nSTORAGE_DIR=/tmp/jarvis-store\n")
    s = Settings(_env_file=env)
    assert s.agent_name == "FRIDAY"
    assert s.chroma_dir == Path("/tmp/jarvis-store/chroma")
    assert s.sqlite_path == Path("/tmp/jarvis-store/jarvis.db")


@pytest.mark.parametrize("value", ["auto", "", "None"])
def test_num_gpu_auto_means_let_ollama_decide(monkeypatch, value):
    monkeypatch.setenv("NUM_GPU", value)
    assert make().num_gpu is None


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        make()
