"""Pinecone vector store implementation."""

from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
from vector_stores.base import BaseVectorStore
from core.types import Metadata
from core.exceptions import VectorStoreError
from config.settings import settings


class PineconeVectorStore(BaseVectorStore):
    """Pinecone-based vector store for cloud-hosted retrieval."""

    def __init__(
        self,
        api_key: str = None,
        environment: str = None,
        index_name: str = None,
        dimension: int = None,
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Index name
            dimension: Embedding dimension
        """
        api_key = api_key or settings.vector_store.pinecone_api_key
        environment = environment or settings.vector_store.pinecone_environment
        index_name = index_name or settings.vector_store.pinecone_index_name
        dimension = dimension or settings.vector_store.pinecone_dimension

        if not api_key:
            raise VectorStoreError("Pinecone API key not provided")

        super().__init__(dimension=dimension, index_name=index_name)

        self.environment = environment
        self.pc = Pinecone(api_key=api_key)
        self.index = None

        logger.info(f"Initialized Pinecone vector store: {index_name}")

    async def create_index(
        self,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        """
        Create Pinecone index.

        Args:
            metric: Similarity metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider
            region: Cloud region
        """
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Pinecone index already exists: {self.index_name}")

            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            raise VectorStoreError(f"Index creation failed: {str(e)}")

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Metadata]] = None,
        batch_size: int = 100,
    ) -> None:
        """Insert or update vectors."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        try:
            # Prepare vectors for upsert
            vectors = []
            for i, (id, embedding) in enumerate(zip(ids, embeddings)):
                vector_data = {"id": id, "values": embedding}

                if metadata and i < len(metadata):
                    vector_data["metadata"] = metadata[i]

                vectors.append(vector_data)

            # Upsert in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)

            logger.info(f"Upserted {len(vectors)} vectors to Pinecone")

        except Exception as e:
            logger.error(f"Pinecone upsert error: {e}")
            raise VectorStoreError(f"Upsert failed: {str(e)}")

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Metadata]]:
        """Search for similar vectors."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        try:
            # Build filter query
            filter_query = None
            if filters:
                filter_query = filters

            # Search
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_query,
                include_metadata=True,
            )

            results = []
            for match in response.matches:
                score = float(match.score)

                # Apply score threshold
                if score_threshold and score < score_threshold:
                    continue

                metadata = match.metadata if hasattr(match, "metadata") else {}
                results.append((match.id, score, metadata))

            logger.info(f"Pinecone search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            raise VectorStoreError(f"Search failed: {str(e)}")

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")

        except Exception as e:
            logger.error(f"Pinecone delete error: {e}")
            raise VectorStoreError(f"Delete failed: {str(e)}")

    async def get_by_id(self, id: str) -> Optional[Tuple[List[float], Metadata]]:
        """Retrieve vector by ID."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        try:
            response = self.index.fetch(ids=[id])

            if id in response.vectors:
                vector_data = response.vectors[id]
                embedding = vector_data.values
                metadata = vector_data.metadata if hasattr(vector_data, "metadata") else {}
                return (embedding, metadata)

            return None

        except Exception as e:
            logger.error(f"Pinecone fetch error: {e}")
            return None

    async def count(self) -> int:
        """Get vector count."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count

        except Exception as e:
            logger.error(f"Pinecone count error: {e}")
            return 0

    async def delete_all(self) -> None:
        """Delete all vectors in the index."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        try:
            self.index.delete(delete_all=True)
            logger.info(f"Deleted all vectors from Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Pinecone delete_all error: {e}")
            raise VectorStoreError(f"Delete all failed: {str(e)}")
