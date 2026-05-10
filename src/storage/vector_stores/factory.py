"""Factory for creating vector store instances."""

from loguru import logger
from .base import BaseVectorStore
from .faiss_store import FAISSVectorStore
from .pinecone_store import PineconeVectorStore
from ...core.types_and_exception import VectorStoreType, ConfigurationError
from ...config.settings import settings


class VectorStoreFactory:
    """Factory for creating and managing vector store instances."""

    @staticmethod
    def create(
        store_type: VectorStoreType = None,
        **kwargs,
    ) -> BaseVectorStore:
        """
        Create a vector store instance.

        Args:
            store_type: Type of vector store to create
            **kwargs: Additional arguments for the vector store

        Returns:
            Vector store instance

        Raises:
            ConfigurationError: If store type is invalid
        """
        store_type = store_type or settings.vector_store.vector_store_type

        logger.info(f"Creating vector store: {store_type}")

        if store_type == VectorStoreType.FAISS:
            return FAISSVectorStore(**kwargs)

        elif store_type == VectorStoreType.PINECONE:
            return PineconeVectorStore(**kwargs)

        else:
            raise ConfigurationError(f"Unsupported vector store type: {store_type}")

    @staticmethod
    async def get_or_create(
        store_type: VectorStoreType = None,
        create_if_missing: bool = True,
        **kwargs,
    ) -> BaseVectorStore:
        """
        Get existing or create new vector store.

        Args:
            store_type: Type of vector store
            create_if_missing: Create index if it doesn't exist
            **kwargs: Additional arguments

        Returns:
            Vector store instance
        """
        store = VectorStoreFactory.create(store_type=store_type, **kwargs)

        # Try to load existing index
        if hasattr(store, "load_index"):
            loaded = await store.load_index()
            if not loaded and create_if_missing:
                await store.create_index()
        elif create_if_missing:
            await store.create_index()

        return store
