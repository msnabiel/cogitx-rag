"""HuggingFace embedding generation utilities"""

import time
import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddings:
    """Generate embeddings using HuggingFace models"""

    def __init__(self, batch_size: int = 32, device: str = "cpu"):
        """
        Initialize embedding generator

        Args:
            batch_size: Batch size for embedding generation
            device: Device to use (cuda or cpu)
        """
        self.batch_size = batch_size
        self.device = device

    def generate_embeddings(
        self,
        texts: List[str],
        model: SentenceTransformer,
        model_name: str = "model",
        prefix: str = None
    ) -> np.ndarray:
        """
        Generate embeddings with optimized batch processing

        Args:
            texts: List of texts to embed
            model: Sentence transformer model
            model_name: Name for logging
            prefix: Optional prefix for each text (e.g., "passage: ")

        Returns:
            numpy array of embeddings
        """
        logger.info(f"Generating {model_name} embeddings for {len(texts)} texts...")
        start_time = time.time()

        # Add prefix if specified
        input_texts = texts
        if prefix:
            input_texts = [f"{prefix}{t}" for t in texts]

        # Use batch processing for better performance
        embeddings = model.encode(
            input_texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        generation_time = time.time() - start_time
        logger.info(f"{model_name} embeddings generated in {generation_time:.2f} seconds")
        return embeddings

    def generate_bge_embeddings(
        self,
        texts: List[str],
        model: SentenceTransformer
    ) -> np.ndarray:
        """Generate BGE embeddings"""
        return self.generate_embeddings(texts, model, "BGE")

    def generate_all_mini_embeddings(
        self,
        texts: List[str],
        model: SentenceTransformer
    ) -> np.ndarray:
        """Generate All-MiniLM embeddings"""
        return self.generate_embeddings(texts, model, "All-MiniLM")

    def generate_bge_large_embeddings(
        self,
        texts: List[str],
        model: SentenceTransformer
    ) -> np.ndarray:
        """Generate BGE Large (1024-dim) embeddings"""
        return self.generate_embeddings(texts, model, "BGE-Large-1024")

    def generate_e5_embeddings(
        self,
        texts: List[str],
        model: SentenceTransformer
    ) -> np.ndarray:
        """Generate E5 embeddings with passage prefix"""
        return self.generate_embeddings(texts, model, "E5-Large-1024", prefix="passage: ")
