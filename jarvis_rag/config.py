"""All runtime configuration, loaded from environment variables and `.env`.

Nothing else in the package should read os.environ directly: import
`get_settings()` instead, so every knob is documented in one place.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Persona
    agent_name: str = "JARVIS"
    address_user_as: str = ""

    # Providers
    llm_provider: Literal["ollama", "groq", "gemini", "openrouter"] = "ollama"
    embed_provider: Literal["ollama", "fastembed"] = "ollama"

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

    # Paths
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")

    @field_validator("num_gpu", mode="before")
    @classmethod
    def _auto_num_gpu(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "auto", "none"):
            return None
        return v

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
