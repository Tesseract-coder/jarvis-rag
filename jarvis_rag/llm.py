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


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.embed_provider != "ollama":
        raise NotImplementedError(f"EMBED_PROVIDER={s.embed_provider} arrives in Phase 7")
    return OllamaEmbeddings(
        model=s.embed_model,
        base_url=s.ollama_base_url,
        num_gpu=s.embed_num_gpu,
        keep_alive=s.keep_alive,
    )
