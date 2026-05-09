"""Long-term semantic memory using vector store."""

from typing import List, Optional
from loguru import logger
from vector_stores.base import BaseVectorStore


class SemanticMemory:
    """Long-term semantic memory for storing important information."""

    def __init__(self, vector_store: BaseVectorStore):
        """
        Initialize semantic memory.

        Args:
            vector_store: Vector store for semantic storage
        """
        self.vector_store = vector_store
        logger.info("Initialized SemanticMemory")

    async def store(
        self,
        key: str,
        content: str,
        embedding: List[float],
        metadata: dict = None,
    ) -> None:
        """
        Store information in long-term semantic memory.

        Args:
            key: Unique identifier
            content: Content to store
            embedding: Content embedding
            metadata: Additional metadata
        """
        metadata = metadata or {}
        metadata["content"] = content
        metadata["type"] = "semantic_memory"

        await self.vector_store.upsert(
            ids=[key],
            embeddings=[embedding],
            metadata=[metadata],
        )

        logger.debug(f"Stored in semantic memory: {key}")

    async def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[dict]:
        """
        Retrieve relevant memories.

        Args:
            query_embedding: Query embedding
            top_k: Number of memories to retrieve

        Returns:
            List of relevant memories
        """
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters={"type": "semantic_memory"},
        )

        memories = []
        for doc_id, score, metadata in results:
            memories.append({
                "id": doc_id,
                "content": metadata.get("content", ""),
                "score": score,
                "metadata": metadata,
            })

        logger.info(f"Retrieved {len(memories)} semantic memories")
        return memories
