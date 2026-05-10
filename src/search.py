"""Search methods for RAG system"""

import time
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any
from fastapi import HTTPException

from src.utils.chunking_strategies import merge_chunks, _find_overlap, _jaccard_similarity, _is_contained_with_ratio
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Load thresholds from settings
SEMANTIC_THRESHOLD_CHUNK_SCORE = settings.processing.semantic_threshold_chunk_score
ENSEMBLE_THRESHOLD_SCORE = settings.processing.ensemble_threshold_score
JACCARD_SIMILARITY_THRESHOLD = settings.processing.jaccard_similarity_threshold
CONTAINED_RATIO = settings.processing.contained_ratio


class DocumentChunk:
    """Represents a chunk of text from a document"""
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
        self.chunk_id = None
        self.embedding = None
        self.semantic_score = 0.0
        self.lexical_score = 0.0
        self.combined_score = 0.0
        self.relevance_score = 0.0


class SearchResult:
    """Container for search results"""
    def __init__(self, chunk, semantic_score=0.0, lexical_score=0.0, combined_score=0.0, search_strategy="ensemble"):
        self.chunk = chunk
        self.semantic_score = semantic_score
        self.lexical_score = lexical_score
        self.combined_score = combined_score
        self.search_strategy = search_strategy
        self.rank = 0


def merge_chunks_search(results: List[SearchResult], min_overlap: int = 5) -> List[SearchResult]:
    """
    Merge SearchResult chunks if they overlap by at least `min_overlap` characters.
    Keeps the highest combined_score when merging.
    """
    texts = [(r.chunk.text, r.combined_score, r) for r in results]
    merged = True

    while merged:
        merged = False
        new_texts = []
        skip = set()

        for i in range(len(texts)):
            if i in skip:
                continue
            merged_text, merged_score, orig_result = texts[i]

            for j in range(i + 1, len(texts)):
                if j in skip:
                    continue

                t2, score2, _ = texts[j]

                # Containment check (keep longer, best score)
                if _is_contained_with_ratio(merged_text, t2, min_ratio=CONTAINED_RATIO):
                    if len(t2) > len(merged_text):
                        merged_text = t2
                    merged_score = max(merged_score, score2)
                    skip.add(j)
                    merged = True
                    continue

                # High-similarity check via Jaccard (near-duplicates)
                jacc = _jaccard_similarity(merged_text, t2)
                if jacc >= JACCARD_SIMILARITY_THRESHOLD:
                    if len(t2) > len(merged_text):
                        merged_text = t2
                    merged_score = max(merged_score, score2)
                    skip.add(j)
                    merged = True
                    continue

                # Forward overlap
                overlap = _find_overlap(merged_text, t2, min_overlap)
                if overlap > 0:
                    logger.info(f"➡️ Merging '{merged_text[-20:]}' + '{t2[:20]}' (overlap={overlap})")
                    merged_text = merged_text + t2[overlap:]
                    merged_score = max(merged_score, score2)
                    skip.add(j)
                    merged = True
                    continue

                # Reverse overlap
                overlap = _find_overlap(t2, merged_text, min_overlap)
                if overlap > 0:
                    logger.info(f"➡️ Merging '{t2[-20:]}' + '{merged_text[:20]}' (overlap={overlap})")
                    merged_text = t2 + merged_text[overlap:]
                    merged_score = max(merged_score, score2)
                    skip.add(j)
                    merged = True

            # Create new SearchResult with merged text
            new_result = SearchResult(
                chunk=DocumentChunk(merged_text),
                combined_score=merged_score,
                search_strategy=orig_result.search_strategy
            )
            # Assign stable chunk_id
            new_result.chunk.chunk_id = f"merged_{hashlib.md5(merged_text.encode('utf-8')).hexdigest()[:8]}"
            new_texts.append((merged_text, merged_score, new_result))

        texts = new_texts

    return [t[2] for t in texts]


class SearchMethods:
    """Search methods for ensemble retrieval"""

    def __init__(self, faiss_index, bm25, chunks, bge_model, all_mini_model, all_mini_embeddings, vector_store=None):
        """
        Initialize search methods

        Args:
            faiss_index: FAISS index for semantic search
            bm25: BM25 index for lexical search
            chunks: List of document chunks
            bge_model: BGE embedding model
            all_mini_model: All-MiniLM embedding model
            all_mini_embeddings: Pre-computed All-MiniLM embeddings
        """
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.chunks = chunks
        self.bge_model = bge_model
        self.all_mini_model = all_mini_model
        self.all_mini_embeddings = all_mini_embeddings
        self.vector_store = vector_store
        self.chunk_lookup = {getattr(chunk, "chunk_id", None): chunk for chunk in chunks if getattr(chunk, "chunk_id", None)}

    async def semantic_search(self, query: str, top_k: int = 10, score_threshold: float = SEMANTIC_THRESHOLD_CHUNK_SCORE) -> List[SearchResult]:
        """Pure semantic search using FAISS (combined BGE + Intfloat)"""
        # Generate both embeddings
        embedding_bge = self.bge_model.encode([query], normalize_embeddings=True)
        embedding_all_mini = self.all_mini_model.encode([query], normalize_embeddings=True)

        # Concatenate to match the FAISS index format
        query_embedding = np.concatenate([embedding_bge, embedding_all_mini], axis=1)

        if self.vector_store is not None:
            results = await self.vector_store.search(
                query_embedding=query_embedding[0].tolist(),
                top_k=min(top_k * 3, len(self.chunks)),
                score_threshold=score_threshold,
            )

            search_results = []
            for external_id, score, metadata in results:
                chunk = self.chunk_lookup.get(external_id)
                if chunk is None:
                    chunk = DocumentChunk(
                        text=metadata.get("content", ""),
                        metadata=metadata or {},
                    )
                    chunk.chunk_id = external_id
                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        semantic_score=float(score),
                        search_strategy="semantic",
                    )
                )
            return search_results

        if not self.faiss_index:
            raise HTTPException(status_code=500, detail="FAISS index not built")

        # Search
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k*3, len(self.chunks))
        )

        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if score < score_threshold:
                logger.info(f"Dropping chunk {idx} | Score: {score:.3f}")
            if idx < len(self.chunks) and score >= score_threshold:
                chunk = self.chunks[idx]
                result = SearchResult(
                    chunk=chunk,
                    semantic_score=float(score),
                    search_strategy="semantic"
                )
                results.append(result)

        return results

    def lexical_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Pure lexical search using BM25"""
        if not self.bm25:
            raise HTTPException(status_code=500, detail="BM25 index not built")

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            result = SearchResult(
                chunk=chunk,
                lexical_score=float(scores[idx]),
                search_strategy="lexical"
            )
            results.append(result)

        return results

    async def ensemble_search(self, query: str, top_k: int = 10, score_threshold: float = ENSEMBLE_THRESHOLD_SCORE) -> List[SearchResult]:
        """Ensemble search using multiple embedding models"""
        start_time = time.time()
        if (self.faiss_index is None and self.vector_store is None) or not self.bm25:
            raise HTTPException(status_code=500, detail="Indices not built")

        # Get results from different models
        bge_results = await self.semantic_search(query, top_k * 3) # Semantic (BGE+AllMini) weight
        bm25_results = self.lexical_search(query, top_k * 3)

        # Combine results using reciprocal rank fusion
        combined_scores = {}

        for i, result in enumerate(bge_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"chunk": result.chunk, "scores": []}
            combined_scores[chunk_id]["scores"].append(1.0 / (i + 20))  # BGE 1024 + Intfloat 1024 weight

        for i, result in enumerate(bm25_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"chunk": result.chunk, "scores": []}
            combined_scores[chunk_id]["scores"].append(1.0 / (i + 40))  # BM25 weight

        # Calculate final scores
        results = []
        for chunk_id, data in combined_scores.items():
            final_score = sum(data["scores"])
            logger.info(f"Chunk {chunk_id} | Score: {final_score:.3f} | Text preview: {data['chunk'].text[:50]}")
            if final_score >= score_threshold:  # <-- filter low-scoring chunks
                result = SearchResult(
                        chunk=data["chunk"],
                        combined_score=final_score,
                        search_strategy="ensemble"
                    )
                results.append(result)

        # Merge duplicates / overlapping chunks
        results = merge_chunks_search(results)
        # Sort by final score and return top-k
        results.sort(key=lambda x: x.combined_score, reverse=True)
        logger.info(f"Ensemble search took {time.time() - start_time:.2f} seconds")
        return results[:top_k]

    def _all_mini_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using All-MiniLM model with pre-computed embeddings"""
        if not self.chunks or self.all_mini_embeddings is None:
            return []

        # Encode query with All-MiniLM
        query_embedding = self.all_mini_model.encode([query], normalize_embeddings=True)

        # Calculate similarities with pre-computed All-MiniLM embeddings
        similarities = np.dot(self.all_mini_embeddings, query_embedding[0])

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            result = SearchResult(
                chunk=chunk,
                semantic_score=float(similarities[idx]),
                search_strategy="all_mini"
            )
            results.append(result)

        return results


__all__ = ['SearchMethods', 'SearchResult', 'DocumentChunk', 'merge_chunks_search']
