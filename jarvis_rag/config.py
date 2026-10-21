"""All runtime configuration, loaded from environment variables and `.env`.

Nothing else in the package should read os.environ directly: import
`get_settings()` instead, so every knob is documented in one place.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from typing import Annotated

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Persona
    agent_name: str = "JARVIS"
    address_user_as: str = ""

    # Providers
    llm_provider: Literal["ollama", "groq", "gemini", "openrouter"] = "ollama"
    embed_provider: Literal["ollama", "fastembed"] = "ollama"

    # Hosted providers (Phase 7): free tiers, keys from each provider's console.
    # Your prompts, including retrieved document chunks, are sent to the provider.
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    gemini_api_key: SecretStr | None = Field(default=None,
                                             validation_alias=AliasChoices("gemini_api_key", "google_api_key"))
    gemini_model: str = "gemini-3.5-flash-lite"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "google/gemma-4-31b-it:free"
    llm_fallback: Literal["ollama", "groq", "gemini", "openrouter"] | None = None  # used when the main LLM errors
    llm_max_retries: int = 3
    llm_timeout: float = 120.0
    reasoning_effort: str | None = None  # hosted reasoning models: "low" | "medium" | "high"
    fastembed_model: str = "nomic-ai/nomic-embed-text-v1.5"  # same vectors as Ollama's nomic-embed-text
    fastembed_batch_size: int = 1  # bigger batches pad to the longest chunk: slower and far more RAM

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:4b-instruct"
    fallback_chat_model: str = "qwen3:1.7b"
    embed_model: str = "nomic-embed-text"
    num_ctx: int = 8192
    num_gpu: int | None = 99  # None = let Ollama auto-fit layers
    embed_num_gpu: int = 0
    temperature: float = 0.2
    reasoning: bool = False
    keep_alive: int = 1800  # seconds
    # Instruction prefixes the embedding model was trained with (nomic-embed-text's
    # are below; set both to "" for models without them). Queries and documents
    # are embedded differently on purpose.
    embed_query_prefix: str = "search_query: "
    embed_document_prefix: str = "search_document: "

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    code_chunk_size: int = 1200
    code_chunk_overlap: int = 150

    # Vector store
    collection_name: str = "jarvis"

    # Retrieval (Phase 3; eval/run_eval.py compares the modes)
    retrieval_mode: Literal["similarity", "mmr", "hybrid", "rerank"] = "rerank"
    retrieval_k: int = 5
    fetch_k: int = 20  # candidates per retriever before fusion / MMR / reranking
    mmr_lambda: float = 0.5  # 1 = pure relevance, 0 = pure diversity
    bm25_weight: float = 0.5  # hybrid: share of BM25 in rank fusion (vector gets the rest)
    rrf_c: int = 60  # rank-fusion constant from the RRF paper
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Ingestion limits
    max_file_mb: float = 20.0
    url_max_pages: int = 25
    user_agent: str = "jarvis-rag (self-hosted RAG ingester)"

    # Agent (Phase 4)
    memory_backend: Literal["sqlite", "memory"] = "sqlite"  # sqlite = conversations survive restarts
    auto_search: bool = True  # search the knowledge base before every turn (small models forget to)
    agent_max_model_calls: int = 8  # per question: stops a confused small model looping on tools
    history_max_tokens: int = 3000  # older turns are dropped from the prompt (kept on disk)
    tool_output_max_chars: int = 4000
    sql_max_rows: int = 200
    sql_timeout_s: float = 5.0
    allow_ingest_tool: bool = False  # let the agent write notes into the knowledge base
    fetch_allow_private: bool = False  # let fetch_url reach localhost / LAN addresses

    # HTTP API (Phase 5)
    api_host: str = "127.0.0.1"  # 0.0.0.0 exposes it to your network: then API_KEY is required
    api_port: int = 8000
    api_key: SecretStr | None = None  # clients send it as the X-API-Key header
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:7860", "http://127.0.0.1:7860",
                                                    "http://localhost:3000"]
    max_message_chars: int = 4000
    max_crawl_depth: int = 2

    # Web UI (Phase 6)
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860
    api_url: str = "http://127.0.0.1:8000"  # where the standalone UI finds the API
    ui_username: str = "jarvis"
    ui_password: SecretStr | None = None  # required when UI_HOST is not localhost

    # Paths
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")

    @field_validator("num_gpu", mode="before")
    @classmethod
    def _auto_num_gpu(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "auto", "none"):
            return None
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        return [o.strip() for o in v.split(",") if o.strip()] if isinstance(v, str) else v

    @field_validator("api_key", "ui_password", "groq_api_key", "gemini_api_key", "openrouter_api_key",
                     "llm_fallback", "reasoning_effort", mode="before")
    @classmethod
    def _empty_key_is_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.storage_dir / "jarvis.db"

    @property
    def memory_path(self) -> Path:
        # Separate from jarvis.db on purpose: run_sql can read every table in jarvis.db,
        # and conversation history is not something the model should query.
        return self.storage_dir / "memory.db"

    @property
    def models_dir(self) -> Path:
        return self.storage_dir / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
