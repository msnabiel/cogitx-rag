"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Union
from loguru import logger


class BaseEmbedding(ABC):
    """Abstract embedding provider interface."""

    def __init__(self, model_name: str, dimension: int):
        """
        Initialize embedding provider.

        Args:
            model_name: Name of the embedding model
            dimension: Embedding vector dimension
        """
        self.model_name = model_name
        self.dimension = dimension
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input text strings

        Returns:
            List of embedding vectors
        """
        pass

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Flexible embedding method for single or batch input.

        Args:
            text: Single text string or list of texts

        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(text, str):
            return await self.embed_text(text)
        else:
            return await self.embed_batch(text)

    def validate_dimension(self, embedding: List[float]) -> bool:
        """
        Validate embedding dimension.

        Args:
            embedding: Embedding vector

        Returns:
            True if dimension matches expected dimension
        """
        return len(embedding) == self.dimension

    async def embed_query(self, query: str) -> List[float]:
        return await self.embed_text(query)
