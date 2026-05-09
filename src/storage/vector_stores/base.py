"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from core.models import DocumentChunk, RetrievalResult
from core.types import Metadata


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    def __init__(self, dimension: int, index_name: str):
        """
        Initialize vector store.

        Args:
            dimension: Embedding vector dimension
            index_name: Name of the vector index
        """
        self.dimension = dimension
        self.index_name = index_name

    @abstractmethod
    async def create_index(self, **kwargs) -> None:
        """Create or initialize the vector index."""
        pass

    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Metadata]] = None,
    ) -> None:
        """
        Insert or update vectors in the store.

        Args:
            ids: List of unique identifiers
            embeddings: List of embedding vectors
            metadata: Optional metadata for each vector
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Metadata]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters
            score_threshold: Minimum similarity score

        Returns:
            List of (id, score, metadata) tuples
        """
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """
        Delete vectors by IDs.

        Args:
            ids: List of vector IDs to delete
        """
        pass

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Tuple[List[float], Metadata]]:
        """
        Retrieve vector and metadata by ID.

        Args:
            id: Vector ID

        Returns:
            Tuple of (embedding, metadata) or None
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Get total number of vectors in store.

        Returns:
            Vector count
        """
        pass

    async def upsert_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Convenience method to upsert document chunks.

        Args:
            chunks: List of document chunks with embeddings
        """
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        metadata = [
            {
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        await self.upsert(ids=ids, embeddings=embeddings, metadata=metadata)
