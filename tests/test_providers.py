"""Phase 7: provider switch and the embedding fingerprint. Constructing the
hosted chat clients makes no network call, so all of this runs offline."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from jarvis_rag import llm
from jarvis_rag.config import Settings
from jarvis_rag.llm import FastEmbedEmbeddings, MissingAPIKey, get_chat_model, get_embeddings, get_fallback_model
from jarvis_rag.vectorstore import FINGERPRINT_KEY, EmbeddingMismatch, KnowledgeBase


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.mark.parametrize("provider,var", [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"),
                                          ("openrouter", "OPENROUTER_API_KEY")])
def test_missing_key_explains_where_to_get_one(provider, var):
    with pytest.raises(MissingAPIKey, match=var) as e:
        get_chat_model(settings(llm_provider=provider))
    assert "https://" in str(e.value)


def test_groq():
    m = get_chat_model(settings(llm_provider="groq", groq_api_key="gk", reasoning_effort="low", llm_max_retries=5))
    assert isinstance(m, ChatGroq)
    assert (m.model_name, m.groq_api_key.get_secret_value(), m.max_retries, m.reasoning_effort) == \
        ("openai/gpt-oss-20b", "gk", 5, "low")


def test_gemini_accepts_either_env_name(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-var")
    s = settings(llm_provider="gemini")
    m = get_chat_model(s)
    assert isinstance(m, ChatGoogleGenerativeAI) and m.model.endswith("gemini-3.5-flash-lite")
    assert s.gemini_api_key.get_secret_value() == "from-google-var"


def test_openrouter_uses_the_openai_protocol():
    m = get_chat_model(settings(llm_provider="openrouter", openrouter_api_key="ok", openrouter_model="x/y:free"))
    assert isinstance(m, ChatOpenAI)
    assert (m.model_name, m.openai_api_base) == ("x/y:free", llm.OPENROUTER_BASE_URL)
    assert m.default_headers["X-Title"] == "jarvis-rag"


@pytest.mark.parametrize("provider,key", [("groq", "groq_api_key"), ("gemini", "gemini_api_key"),
                                          ("openrouter", "openrouter_api_key")])
def test_every_provider_binds_our_tools_offline(provider, key):
    @tool
    def calculator(expression: str) -> str:
        """Evaluate arithmetic."""
        return expression

    bound = get_chat_model(settings(llm_provider=provider, **{key: "k"})).bind_tools([calculator])
    assert bound is not None  # schema conversion happens here; a bad schema would raise


def test_fallback_model():
    assert get_fallback_model(settings()) is None
    assert get_fallback_model(settings(llm_provider="groq", groq_api_key="k", llm_fallback="groq")) is None
    assert isinstance(get_fallback_model(settings(llm_provider="groq", groq_api_key="k", llm_fallback="ollama")),
                      ChatOllama)


def test_fastembed_provider_is_wrapped_with_prefixes(monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, name, cache_dir=None):
            self.name = name

        def embed(self, texts, batch_size=1):
            import numpy as np
            for t in texts:
                yield np.array([float(len(t)), 1.0])

    import fastembed
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    e = get_embeddings(settings(embed_provider="fastembed"))
    assert isinstance(e.inner, FastEmbedEmbeddings) and e.inner.model.name == "nomic-ai/nomic-embed-text-v1.5"
    assert e.inner.batch_size == 1
    assert e.embed_query("abc") == [float(len("search_query: abc")), 1.0]
    assert e.embed_documents(["x", "yy"]) == [[18.0, 1.0], [19.0, 1.0]]


# --------------------------------------------------------------------------- embedding fingerprint

class Flipped(DeterministicFakeEmbedding):
    """Same dimension, different "model": every vector points the other way."""

    def embed_documents(self, texts):
        return [[-x for x in v] for v in super().embed_documents(texts)]

    def embed_query(self, text):
        return [-x for x in super().embed_query(text)]


def kb(tmp_path, emb, name="fingerprint"):
    return KnowledgeBase(settings(storage_dir=tmp_path, collection_name=name), embeddings=emb)


def test_fingerprint_adopted_then_enforced(tmp_path):
    first = kb(tmp_path, DeterministicFakeEmbedding(size=16))
    assert FINGERPRINT_KEY not in (first.collection.metadata or {})  # opening needs no model
    first.search("anything")
    assert FINGERPRINT_KEY in first.collection.metadata

    kb(tmp_path, DeterministicFakeEmbedding(size=16)).search("same model: fine")
    with pytest.raises(EmbeddingMismatch, match="dim 16 vs 32"):
        kb(tmp_path, DeterministicFakeEmbedding(size=32)).search("other dimension")
    with pytest.raises(EmbeddingMismatch, match="probe similarity -1.000"):
        kb(tmp_path, Flipped(size=16)).search("same dimension, other space")


def test_fingerprint_guards_ingest_too(tmp_path):
    from langchain_core.documents import Document

    kb(tmp_path, DeterministicFakeEmbedding(size=16)).search("x")
    doc = Document("hello", metadata={"source": "a.md", "chunk_id": "c1"})
    with pytest.raises(EmbeddingMismatch):
        kb(tmp_path, Flipped(size=16)).upsert_source("a.md", [doc])
    assert kb(tmp_path, DeterministicFakeEmbedding(size=16)).count() == 0  # nothing was written
