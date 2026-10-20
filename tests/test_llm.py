import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from jarvis_rag.config import Settings
from jarvis_rag.llm import PrefixedEmbeddings, get_chat_model, get_embeddings


def test_chat_model_carries_settings():
    # Constructing ChatOllama does not contact the server.
    s = Settings(_env_file=None, chat_model="qwen3:1.7b", num_ctx=2048, num_gpu=None)
    m = get_chat_model(s)
    assert isinstance(m, ChatOllama)
    assert (m.model, m.num_ctx, m.num_gpu, m.reasoning) == ("qwen3:1.7b", 2048, None, False)


def test_chat_model_override_name():
    assert get_chat_model(Settings(_env_file=None), model="llama3.1:8b").model == "llama3.1:8b"


def test_embeddings_run_on_cpu_by_default():
    e = get_embeddings(Settings(_env_file=None))
    assert isinstance(e.inner, OllamaEmbeddings)
    assert (e.inner.model, e.inner.num_gpu) == ("nomic-embed-text", 0)
    assert (e.query_prefix, e.document_prefix) == ("search_query: ", "search_document: ")


def test_prefixes_are_applied_per_side():
    class Recorder(Embeddings):
        def embed_documents(self, texts):
            self.seen = texts
            return [[0.0]] * len(texts)

        def embed_query(self, text):
            self.seen = [text]
            return [0.0]

    rec = Recorder()
    e = PrefixedEmbeddings(rec, "q: ", "d: ")
    e.embed_documents(["a", "b"])
    assert rec.seen == ["d: a", "d: b"]
    e.embed_query("c")
    assert rec.seen == ["q: c"]


def test_cloud_providers_not_wired_yet():
    with pytest.raises(NotImplementedError):
        get_chat_model(Settings(_env_file=None, llm_provider="groq"))
    with pytest.raises(NotImplementedError):
        get_embeddings(Settings(_env_file=None, embed_provider="fastembed"))


def test_fakes_used_by_later_phases_work_offline():
    # Every later phase tests against these instead of Ollama.
    emb = DeterministicFakeEmbedding(size=768)
    assert len(emb.embed_query("car")) == 768
    assert emb.embed_query("car") == emb.embed_query("car")
    chat = GenericFakeChatModel(messages=iter([AIMessage(content="At your service.")]))
    assert chat.invoke("hello").content == "At your service."
