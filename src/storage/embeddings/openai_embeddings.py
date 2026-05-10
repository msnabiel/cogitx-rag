"""OpenAI embeddings implementation."""

from typing import List
from openai import AsyncOpenAI
from loguru import logger
from embeddings.base import BaseEmbedding
from core.exceptions import EmbeddingError
from config.settings import settings


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        dimension: int = 1536,
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model_name: Model name (defaults to settings)
            dimension: Embedding dimension
        """
        api_key = api_key or settings.llm.openai_api_key
        model_name = model_name or settings.llm.openai_embedding_model

        if not api_key:
            raise EmbeddingError("OpenAI API key not provided")

        super().__init__(model_name=model_name, dimension=dimension)
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            embedding = response.data[0].embedding

            if not self.validate_dimension(embedding):
                raise EmbeddingError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}"
                )

            return embedding

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Maximum batch size for API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )

                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings using OpenAI")
            return all_embeddings

        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}")

    async def embed(self, text):
        if isinstance(text, str):
            return await self.embed_text(text)
        return await self.embed_batch(text)
