"""Core Pydantic models for data validation."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from src.core.types_and_exception import EntityType, RelationType, RetrieverType, Metadata


class Document(BaseModel):
    """Document model for ingestion and storage."""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: Metadata = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    chunks: Optional[List["DocumentChunk"]] = Field(None, description="Document chunks")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """Chunked document segment."""

    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    embedding: Optional[List[float]] = Field(None, description="Chunk embedding vector")
    metadata: Metadata = Field(default_factory=dict, description="Chunk metadata")
    chunk_index: int = Field(..., description="Position in parent document")
    start_char: Optional[int] = Field(None, description="Start character position")
    end_char: Optional[int] = Field(None, description="End character position")


class Entity(BaseModel):
    """Knowledge graph entity."""

    id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    embedding: Optional[List[float]] = Field(None, description="Entity embedding")


class Relation(BaseModel):
    """Knowledge graph relation."""

    id: str = Field(..., description="Unique relation identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: RelationType = Field(..., description="Relation type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relation properties")
    weight: float = Field(default=1.0, description="Relation strength/weight")


class Query(BaseModel):
    """Query model for RAG system."""

    text: str = Field(..., description="Query text")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    top_k: Optional[int] = Field(None, description="Number of results to retrieve")
    retriever_types: Optional[List[RetrieverType]] = Field(
        None, description="Specific retrievers to use"
    )
    include_graph: bool = Field(default=True, description="Include graph traversal")
    include_memory: bool = Field(default=True, description="Include memory context")


class RetrievalResult(BaseModel):
    """Single retrieval result."""

    id: str = Field(..., description="Result identifier")
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Relevance score")
    source: RetrieverType = Field(..., description="Retriever source")
    metadata: Metadata = Field(default_factory=dict, description="Result metadata")
    entities: Optional[List[Entity]] = Field(None, description="Associated entities")
    relations: Optional[List[Relation]] = Field(None, description="Associated relations")


class RAGResponse(BaseModel):
    """RAG system response."""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[RetrievalResult] = Field(
        default_factory=list, description="Source documents"
    )
    confidence: float = Field(default=0.0, description="Answer confidence score")
    metadata: Metadata = Field(default_factory=dict, description="Response metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    retrieval_stats: Dict[str, int] = Field(
        default_factory=dict, description="Retrieval statistics"
    )


class ContextWindow(BaseModel):
    """Context window for LLM."""

    query: str = Field(..., description="Query text")
    retrieved_contexts: List[str] = Field(..., description="Retrieved text chunks")
    graph_context: Optional[str] = Field(None, description="Graph-based context")
    memory_context: Optional[str] = Field(None, description="Memory context")
    system_prompt: str = Field(..., description="System prompt")
    total_tokens: int = Field(..., description="Total token count")


class MemoryEntry(BaseModel):
    """Memory entry for caching."""

    key: str = Field(..., description="Memory key")
    value: Any = Field(..., description="Memory value")
    type: str = Field(..., description="Memory type")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, description="Access frequency")


class SearchRequest(BaseModel):
    """Search request model"""

    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, description="Number of results to return")
