"""Local sentence-transformer embedding providers."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from .base import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    """Local embedding provider backed by SentenceTransformer."""

    def __init__(self, model_name: str, dimension: int = 384, device: str = "cpu"):
        super().__init__(model_name=model_name, dimension=dimension)
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    async def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)[0].tolist()
        if not self.validate_dimension(embedding):
            logger.warning(f"Local embedding dimension mismatch: {len(embedding)} != {self.dimension}")
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class LocalDualEmbedding(BaseEmbedding):
    """Local dual embedding provider using BGE + MiniLM."""

    def __init__(
        self,
        bge_model_name: str,
        mini_model_name: str,
        device: str = "cpu",
    ):
        bge_model = SentenceTransformer(bge_model_name, device=device)
        mini_model = SentenceTransformer(mini_model_name, device=device)
        dimension = bge_model.get_sentence_embedding_dimension() + mini_model.get_sentence_embedding_dimension()
        super().__init__(model_name=f"{bge_model_name}+{mini_model_name}", dimension=dimension)
        self.bge_model = bge_model
        self.mini_model = mini_model
        self.device = device

    async def embed_text(self, text: str) -> List[float]:
        bge_embedding = self.bge_model.encode([text], normalize_embeddings=True)[0]
        mini_embedding = self.mini_model.encode([text], normalize_embeddings=True)[0]
        embedding = np.concatenate([bge_embedding, mini_embedding]).tolist()
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        bge_embeddings = self.bge_model.encode(texts, normalize_embeddings=True)
        mini_embeddings = self.mini_model.encode(texts, normalize_embeddings=True)
        combined = np.concatenate([bge_embeddings, mini_embeddings], axis=1)
        return combined.tolist()
