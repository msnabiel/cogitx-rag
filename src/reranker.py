"""Reranking functions for search results"""

import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder based reranking"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Loaded cross-encoder: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.model = None

    def rerank(self, query: str, results: List, top_k: Optional[int] = None) -> List:
        if not self.model or not results:
            return results
        try:
            pairs = [[query, result.content if hasattr(result, 'content') else result.chunk.text] for result in results]
            scores = self.model.predict(pairs)
            for result, score in zip(results, scores):
                result.score = float(score)
            reranked = sorted(results, key=lambda x: x.score, reverse=True)
            return reranked[:top_k] if top_k else reranked
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results[:top_k] if top_k else results


class Reranker:
    """Reranking utilities for search results"""

    @staticmethod
    def combine_and_rerank(query: str, semantic_results: List, lexical_results: List) -> List:
        """Combine and rerank search results"""
        # Create a mapping of chunk_id to results
        combined_results = {}

        # Add semantic results
        for i, result in enumerate(semantic_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result
            else:
                # Update with better semantic score
                if result.semantic_score > combined_results[chunk_id].semantic_score:
                    combined_results[chunk_id].semantic_score = result.semantic_score

        # Add lexical results
        for i, result in enumerate(lexical_results):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result
            else:
                # Update with better lexical score
                if result.lexical_score > combined_results[chunk_id].lexical_score:
                    combined_results[chunk_id].lexical_score = result.lexical_score

        # Calculate combined scores with normalization
        results = list(combined_results.values())

        # Normalize semantic scores
        if results:
            max_semantic = max(r.semantic_score for r in results) if any(r.semantic_score > 0 for r in results) else 1.0
            min_semantic = min(r.semantic_score for r in results)
            semantic_range = max_semantic - min_semantic if max_semantic > min_semantic else 1.0

            # Normalize lexical scores
            max_lexical = max(r.lexical_score for r in results) if any(r.lexical_score > 0 for r in results) else 1.0
            min_lexical = min(r.lexical_score for r in results)
            lexical_range = max_lexical - min_lexical if max_lexical > min_lexical else 1.0

            for result in results:
                # Normalize scores
                norm_semantic = (result.semantic_score - min_semantic) / semantic_range if semantic_range > 0 else 0.0
                norm_lexical = (result.lexical_score - min_lexical) / lexical_range if lexical_range > 0 else 0.0

                # Calculate combined score with weights
                result.combined_score = (norm_semantic * 0.6) + (norm_lexical * 0.4)
                result.search_strategy = "hybrid"

        # Apply additional reranking factors
        results = Reranker.apply_reranking_factors(query, results)

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results

    @staticmethod
    def apply_reranking_factors(query: str, results: List) -> List:
        """Apply additional reranking factors"""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        for result in results:
            chunk_text_lower = result.chunk.text.lower()
            chunk_tokens = set(chunk_text_lower.split())

            # Factor 1: Query term density
            overlap_tokens = query_tokens.intersection(chunk_tokens)
            term_density = len(overlap_tokens) / len(query_tokens) if query_tokens else 0.0

            # Factor 2: Chunk length penalty (prefer medium-length chunks)
            chunk_length = len(result.chunk.text.split())
            length_penalty = 1.0
            if chunk_length < 10:
                length_penalty = 0.7  # Too short
            elif chunk_length > 200:
                length_penalty = 0.8  # Too long

            # Factor 3: Position bonus (prefer chunks from beginning of document)
            position_bonus = 1.0
            if result.chunk.metadata.get("start_idx", 0) < 1000:
                position_bonus = 1.1  # Slight bonus for early chunks

            # Apply factors to combined score
            result.combined_score *= (1.0 + term_density * 0.2) * length_penalty * position_bonus

        return results


__all__ = ['Reranker', 'CrossEncoderReranker']
