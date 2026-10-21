"""Model factories. The rest of the package asks for "a chat model" or
"an embedder" and never imports a provider class directly.

    LLM_PROVIDER=ollama       local, private, free (default)
    LLM_PROVIDER=groq         hosted, free tier: fast open models (GROQ_API_KEY)
    LLM_PROVIDER=gemini       hosted, free tier: Google Gemini (GEMINI_API_KEY)
    LLM_PROVIDER=openrouter   hosted, ":free" models from many labs (OPENROUTER_API_KEY)
    EMBED_PROVIDER=ollama | fastembed (in-process ONNX on CPU, no server needed)

Hosted providers receive every prompt, and that means the retrieved chunks of your
documents. Keep sensitive corpora on Ollama.
"""

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from jarvis_rag.config import Settings, get_settings

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
KEY_HELP = {
    "groq": ("GROQ_API_KEY", "https://console.groq.com/keys"),
    "gemini": ("GEMINI_API_KEY", "https://aistudio.google.com/apikey"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/settings/keys"),
}


class MissingAPIKey(RuntimeError):
    pass


def api_key(s: Settings, provider: str) -> str:
    secret = {"groq": s.groq_api_key, "gemini": s.gemini_api_key, "openrouter": s.openrouter_api_key}[provider]
    if secret is None:
        var, url = KEY_HELP[provider]
        raise MissingAPIKey(f"LLM_PROVIDER={provider} needs {var} in .env (free key: {url})")
    return secret.get_secret_value()


def model_name(s: Settings, provider: str | None = None) -> str:
    p = provider or s.llm_provider
    return {"ollama": s.chat_model, "groq": s.groq_model, "gemini": s.gemini_model,
            "openrouter": s.openrouter_model}[p]


def get_chat_model(settings: Settings | None = None, model: str | None = None,
                   provider: str | None = None) -> BaseChatModel:
    s = settings or get_settings()
    p = provider or s.llm_provider
    name = model or model_name(s, p)
    # Hosted free tiers answer 429 when you exceed requests/minute; the SDKs retry
    # with backoff (honouring Retry-After) up to LLM_MAX_RETRIES times.
    common = dict(temperature=s.temperature, max_retries=s.llm_max_retries, timeout=s.llm_timeout)
    effort = {"reasoning_effort": s.reasoning_effort} if s.reasoning_effort else {}
    if p == "ollama":
        return ChatOllama(model=name, base_url=s.ollama_base_url, num_ctx=s.num_ctx, num_gpu=s.num_gpu,
                          temperature=s.temperature, reasoning=s.reasoning, keep_alive=s.keep_alive)
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=name, api_key=api_key(s, "groq"), **common, **effort)
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=name, api_key=api_key(s, "gemini"), **common,
                                      **({"thinking_level": s.reasoning_effort} if s.reasoning_effort else {}))
    if p == "openrouter":
        from langchain_openai import ChatOpenAI

        # OpenRouter speaks the OpenAI API; the headers are its optional app attribution.
        return ChatOpenAI(model=name, base_url=OPENROUTER_BASE_URL, api_key=api_key(s, "openrouter"),
                          default_headers={"X-Title": "jarvis-rag"}, **common, **effort)
    raise ValueError(f"unknown LLM_PROVIDER {p!r}")


def get_fallback_model(s: Settings) -> BaseChatModel | None:
    """LLM_FALLBACK=ollama: when the hosted model fails (rate limit after retries,
    outage, no network) the agent retries that step on the local model."""
    if not s.llm_fallback or s.llm_fallback == s.llm_provider:
        return None
    return get_chat_model(s, provider=s.llm_fallback)


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


class FastEmbedEmbeddings(Embeddings):
    """FastEmbed (ONNX Runtime, CPU) inside this process: no Ollama needed for
    embeddings, which is what a free CPU-only host (HF Spaces, Phase 9) needs.
    The model downloads once into storage/models. Prefixes are added by
    PrefixedEmbeddings, not here, so both providers embed identical strings.

    batch_size=1 on purpose: a batch is padded to its longest text, and nomic accepts
    8k tokens. Measured on 60 of our chunks (Ryzen 5600H): batch 32 = 488 ms/chunk and
    2.8 GB peak RSS, batch 1 = 133 ms/chunk and 1.1 GB. Same lesson as the reranker."""

    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 1):
        from fastembed import TextEmbedding  # lazy: ~1 s import, ONNX session

        self.model = TextEmbedding(model_name, cache_dir=cache_dir)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.embed_provider == "fastembed":
        inner = FastEmbedEmbeddings(s.fastembed_model, cache_dir=str(s.models_dir), batch_size=s.fastembed_batch_size)
    elif s.embed_provider == "ollama":
        inner = OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url, num_gpu=s.embed_num_gpu,
                                 keep_alive=s.keep_alive)
    else:
        raise ValueError(f"unknown EMBED_PROVIDER {s.embed_provider!r}")
    return PrefixedEmbeddings(inner, s.embed_query_prefix, s.embed_document_prefix)
