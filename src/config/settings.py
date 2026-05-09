"""Centralized configuration management using Pydantic settings."""

from typing import List, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")
    openai_embedding_model: str = Field(default="text-embedding-3-large")

    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model")
    gemini_embedding_model: str = Field(default="models/embedding-001")

    default_llm_provider: Literal["openai", "gemini"] = Field(default="gemini")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    vector_store_type: Literal["pinecone", "faiss"] = Field(default="faiss")

    # Pinecone
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_environment: str = Field(default="us-east-1")
    pinecone_index_name: str = Field(default="cogitx-rag")
    pinecone_dimension: int = Field(default=1536)

    # FAISS
    faiss_index_path: str = Field(default="./data/faiss_index")
    faiss_dimension: int = Field(default=1536)


class GraphSettings(BaseSettings):
    """Neo4j graph database configuration."""

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="cogitx-password")
    neo4j_database: str = Field(default="neo4j")


class MemorySettings(BaseSettings):
    """Memory and caching configuration."""

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    cache_ttl: int = Field(default=3600)

    enable_short_term_memory: bool = Field(default=True)
    enable_long_term_memory: bool = Field(default=True)
    enable_structured_memory: bool = Field(default=True)


class ProcessingSettings(BaseSettings):
    """Document processing and chunking configuration."""

    max_workers: int = Field(default=10, description="Maximum parallel workers")
    chunk_size: int = Field(default=512, description="Default chunk size in tokens")
    overlap_size: int = Field(default=50, description="Chunk overlap size")

    # Search thresholds
    semantic_threshold_chunk_score: float = Field(default=0.25)
    ensemble_threshold_score: float = Field(default=0.00)

    # Deduplication thresholds
    jaccard_similarity_threshold: float = Field(default=0.85)
    contained_ratio: float = Field(default=0.85)
    fuzzy_threshold: int = Field(default=85)

    # Embedding settings
    embedding_batch_size_gpu: int = Field(default=64, description="Batch size for GPU")
    embedding_batch_size_cpu: int = Field(default=32, description="Batch size for CPU")
    embedding_workers_gpu: int = Field(default=2, description="Workers for GPU")
    embedding_workers_cpu: int = Field(default=4, description="Workers for CPU")


class RetrievalSettings(BaseSettings):
    """Retrieval and ranking configuration."""

    # Vector search
    vector_top_k: int = Field(default=10)
    vector_similarity_threshold: float = Field(default=0.7)

    # BM25 search
    bm25_top_k: int = Field(default=10)
    bm25_k1: float = Field(default=1.5)
    bm25_b: float = Field(default=0.75)

    # Graph traversal
    graph_max_depth: int = Field(default=3)
    graph_max_results: int = Field(default=20)

    # Hybrid retrieval weights
    hybrid_vector_weight: float = Field(default=0.5)
    hybrid_bm25_weight: float = Field(default=0.3)
    hybrid_graph_weight: float = Field(default=0.2)
    rerank_top_k: int = Field(default=5)


class ContextSettings(BaseSettings):
    """Context building configuration."""

    max_context_length: int = Field(default=8000)
    context_compression_enabled: bool = Field(default=True)
    compression_ratio: float = Field(default=0.6)


class QuerySettings(BaseSettings):
    """Query understanding configuration."""

    query_expansion_enabled: bool = Field(default=True)
    query_decomposition_enabled: bool = Field(default=False)
    max_sub_queries: int = Field(default=3)


class APISettings(BaseSettings):
    """API server configuration."""

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_reload: bool = Field(default=False)
    cors_origins: List[str] = Field(default=["*"])
    upload_dir: str = Field(default="./uploaded_docs")
    cache_dir: str = Field(default="/tmp/document_cache")


class SlackSettings(BaseSettings):
    """Slack integration configuration."""

    slack_bot_token: str = Field(default="")
    slack_app_token: str = Field(default="")
    slack_signing_secret: str = Field(default="")
    slack_enabled: bool = Field(default=False)


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    log_level: str = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(default="json")
    log_file: str = Field(default="./logs/cogitx-rag.log")


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration."""

    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    debug: bool = Field(default=False)
    environment: Literal["development", "staging", "production"] = Field(default="development")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    api: APISettings = Field(default_factory=APISettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)


# Global settings instance
settings = Settings()
