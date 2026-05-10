"""FAISS vector store implementation."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from loguru import logger
from .base import BaseVectorStore
from ...core.types_and_exception import Metadata, VectorStoreError
from ...config.settings import settings


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for local/fast retrieval."""

    def __init__(
        self,
        dimension: int = None,
        index_name: str = "faiss_index",
        index_path: str = None,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_name: Index name
            index_path: Path to save/load index
        """
        dimension = dimension or settings.vector_store.faiss_dimension
        index_path = index_path or settings.vector_store.faiss_index_path

        super().__init__(dimension=dimension, index_name=index_name)

        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / f"{index_name}.index"
        self.metadata_file = self.index_path / f"{index_name}_metadata.pkl"

        self.index: Optional[faiss.Index] = None
        self.id_map: Dict[int, str] = {}  # Internal ID -> External ID
        self.metadata_map: Dict[str, Metadata] = {}  # External ID -> Metadata
        self.next_id = 0

        logger.info(f"Initialized FAISS vector store: {self.index_name}")

    async def create_index(self, use_gpu: bool = False) -> None:
        """
        Create FAISS index.

        Args:
            use_gpu: Use GPU for FAISS (requires faiss-gpu)
        """
        try:
            # Create IndexFlatIP (Inner Product) for cosine similarity
            # Alternatively, use IndexFlatL2 for L2 distance
            self.index = faiss.IndexFlatIP(self.dimension)

            # Optionally use IVF for faster search on large datasets
            # nlist = 100  # number of clusters
            # quantizer = faiss.IndexFlatIP(self.dimension)
            # self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

            if use_gpu and faiss.get_num_gpus() > 0:
                logger.info("Using GPU for FAISS")
                self.index = faiss.index_cpu_to_all_gpus(self.index)

            logger.info(f"Created FAISS index with dimension {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise VectorStoreError(f"Index creation failed: {str(e)}")

    async def load_index(self) -> bool:
        """
        Load existing index from disk.

        Returns:
            True if loaded successfully
        """
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                self.index = faiss.read_index(str(self.index_file))

                with open(self.metadata_file, "rb") as f:
                    data = pickle.load(f)
                    self.id_map = data["id_map"]
                    self.metadata_map = data["metadata_map"]
                    self.next_id = data["next_id"]

                logger.info(f"Loaded FAISS index from {self.index_file}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            return False

    async def save_index(self) -> None:
        """Save index to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))

                with open(self.metadata_file, "wb") as f:
                    pickle.dump(
                        {
                            "id_map": self.id_map,
                            "metadata_map": self.metadata_map,
                            "next_id": self.next_id,
                        },
                        f,
                    )

                logger.info(f"Saved FAISS index to {self.index_file}")

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise VectorStoreError(f"Index save failed: {str(e)}")

    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Metadata]] = None,
    ) -> None:
        """Insert or update vectors."""
        if self.index is None:
            await self.create_index()

        try:
            # Convert to numpy array and normalize for cosine similarity
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)  # Normalize for cosine similarity with Inner Product

            # Add to FAISS index
            internal_ids = list(range(self.next_id, self.next_id + len(ids)))
            self.index.add(vectors)

            # Update mappings
            for i, external_id in enumerate(ids):
                internal_id = internal_ids[i]
                self.id_map[internal_id] = external_id

                if metadata and i < len(metadata):
                    self.metadata_map[external_id] = metadata[i]
                else:
                    self.metadata_map[external_id] = {}

            self.next_id += len(ids)

            # Auto-save after upsert
            await self.save_index()

            logger.info(f"Upserted {len(ids)} vectors to FAISS")

        except Exception as e:
            logger.error(f"FAISS upsert error: {e}")
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
            await self.load_index()
            if self.index is None:
                return []

        try:
            # Normalize query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # Search
            scores, internal_ids = self.index.search(query_vector, top_k)

            results = []
            for i, internal_id in enumerate(internal_ids[0]):
                if internal_id == -1:  # FAISS returns -1 for empty results
                    continue

                score = float(scores[0][i])

                # Apply score threshold
                if score_threshold and score < score_threshold:
                    continue

                external_id = self.id_map.get(internal_id)
                if external_id:
                    metadata = self.metadata_map.get(external_id, {})

                    # Apply metadata filters
                    if filters:
                        if not all(metadata.get(k) == v for k, v in filters.items()):
                            continue

                    results.append((external_id, score, metadata))

            logger.info(f"FAISS search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            raise VectorStoreError(f"Search failed: {str(e)}")

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors (requires index rebuild in FAISS)."""
        # FAISS doesn't support direct deletion - need to rebuild index
        logger.warning("FAISS delete requires index rebuild - not implemented")
        # TODO: Implement by rebuilding index without deleted vectors
        pass

    async def get_by_id(self, id: str) -> Optional[Tuple[List[float], Metadata]]:
        """Get vector by ID (not efficiently supported by FAISS)."""
        metadata = self.metadata_map.get(id)
        if metadata:
            return (None, metadata)  # FAISS doesn't store original vectors efficiently
        return None

    async def count(self) -> int:
        """Get vector count."""
        if self.index is None:
            await self.load_index()
        return self.index.ntotal if self.index else 0
