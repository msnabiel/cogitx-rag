"""Google Gemini embeddings implementation."""

from typing import List
import google.generativeai as genai
from loguru import logger
from embeddings.base import BaseEmbedding
from core.exceptions import EmbeddingError
from config.settings import settings


class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embedding provider."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        dimension: int = 768,
    ):
        """
        Initialize Gemini embeddings.

        Args:
            api_key: Gemini API key (defaults to settings)
            model_name: Model name (defaults to settings)
            dimension: Embedding dimension (Gemini default is 768)
        """
        api_key = api_key or settings.llm.gemini_api_key
        model_name = model_name or settings.llm.gemini_embedding_model

        if not api_key:
            raise EmbeddingError("Gemini API key not provided")

        super().__init__(model_name=model_name, dimension=dimension)

        # Configure Gemini
        genai.configure(api_key=api_key)

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            embedding = result["embedding"]

            if not self.validate_dimension(embedding):
                raise EmbeddingError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}"
                )

            return embedding

        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Maximum batch size

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document",
                )

                # Handle both single and batch results
                if isinstance(result["embedding"][0], list):
                    embeddings = result["embedding"]
                else:
                    embeddings = [result["embedding"]]

                all_embeddings.extend(embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings using Gemini")
            return all_embeddings

        except Exception as e:
            logger.error(f"Gemini batch embedding error: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}")

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding optimized for query (uses retrieval_query task type).

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query",
            )
            return result["embedding"]

        except Exception as e:
            logger.error(f"Gemini query embedding error: {e}")
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}")
