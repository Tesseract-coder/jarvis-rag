"""Model factories. The rest of the package asks for "a chat model" or
"an embedder" and never imports a provider class directly.

Phase 0 wires Ollama only; Phase 7 adds the cloud providers and FastEmbed.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from jarvis_rag.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None, model: str | None = None) -> BaseChatModel:
    s = settings or get_settings()
    if s.llm_provider != "ollama":
        raise NotImplementedError(f"LLM_PROVIDER={s.llm_provider} arrives in Phase 7")
    return ChatOllama(
        model=model or s.chat_model,
        base_url=s.ollama_base_url,
        num_ctx=s.num_ctx,
        num_gpu=s.num_gpu,
        temperature=s.temperature,
        reasoning=s.reasoning,
        keep_alive=s.keep_alive,
    )


class PrefixedEmbeddings(Embeddings):
    """Adds the model's task prefixes, so callers never have to remember them.

    nomic-embed-text was trained on "search_query: ..." vs "search_document: ...";
    embedding both sides without them costs retrieval quality (learn/02 showed it).
    """

    def __init__(self, inner: Embeddings, query_prefix: str = "", document_prefix: str = ""):
        self.inner, self.query_prefix, self.document_prefix = inner, query_prefix, document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.embed_provider != "ollama":
        raise NotImplementedError(f"EMBED_PROVIDER={s.embed_provider} arrives in Phase 7")
    inner = OllamaEmbeddings(
        model=s.embed_model,
        base_url=s.ollama_base_url,
        num_gpu=s.embed_num_gpu,
        keep_alive=s.keep_alive,
    )
    return PrefixedEmbeddings(inner, s.embed_query_prefix, s.embed_document_prefix)
