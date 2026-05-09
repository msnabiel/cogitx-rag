"""Core types and exceptions"""

from enum import Enum
from typing import TypeVar, Dict, Any


class VectorStoreType(str, Enum):
    PINECONE = "pinecone"
    FAISS = "faiss"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"

class RetrieverType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    BM25 = "bm25"
    HYBRID = "hybrid"

class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    STRUCTURED = "structured"

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    DATE = "DATE"
    DOCUMENT = "DOCUMENT"

class RelationType(str, Enum):
    MENTIONS = "MENTIONS"
    RELATES_TO = "RELATES_TO"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"
    WORKS_FOR = "WORKS_FOR"
    CREATED_BY = "CREATED_BY"
    OCCURRED_AT = "OCCURRED_AT"
    SIMILAR_TO = "SIMILAR_TO"

class RankingStrategy(str, Enum):
    SIMILARITY = "similarity"
    RRF = "rrf"
    WEIGHTED = "weighted"
    CROSS_ENCODER = "cross_encoder"

T = TypeVar("T")
Metadata = Dict[str, Any]


class CogitXException(Exception):
    """Base exception"""

class ConfigurationError(CogitXException):
    """Configuration error"""

class VectorStoreError(CogitXException):
    """Vector store error"""

class GraphDatabaseError(CogitXException):
    """Graph database error"""

class EmbeddingError(CogitXException):
    """Embedding error"""

class RetrievalError(CogitXException):
    """Retrieval error"""

class LLMError(CogitXException):
    """LLM error"""

class MemoryError(CogitXException):
    """Memory error"""

class ValidationError(CogitXException):
    """Validation error"""

class EntityExtractionError(CogitXException):
    """Entity extraction error"""
