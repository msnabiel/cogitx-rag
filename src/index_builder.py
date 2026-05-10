"""Index building for FAISS and BM25"""

import time
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from src.search import SearchMethods

logger = logging.getLogger(__name__)


def build_indices(
    chunks,
    embedding_provider,
    embedding_workers,
    update_globals_fn,
    vector_store=None,
):
    """
    Build FAISS and BM25 indices with parallel embedding generation

    Args:
        chunks: List of DocumentChunk objects
        embedding_provider: Selected embedding provider
        update_globals_fn: Callback to update global state

    Returns:
        SearchMethods instance
    """
    texts = [chunk.text for chunk in chunks]

    logger.info(f"Building indices for {len(chunks)} chunks with parallel embedding generation...")
    start_time = time.time()

    combined_embeddings = asyncio.run(embedding_provider.embed_batch(texts))
    combined_embeddings = np.array(combined_embeddings, dtype=np.float32)

    # Store embeddings in chunks
    for i, chunk in enumerate(chunks):
        chunk.embedding = combined_embeddings[i]
        chunk.chunk_id = f"chunk_{i}"

    # Build unified FAISS index
    logger.info("Building unified FAISS index with combined embeddings...")
    dimension = combined_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(combined_embeddings.astype('float32'))

    # Build BM25 index
    logger.info("Building BM25 index...")
    tokenized_texts = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    # Initialize search methods
    if vector_store is not None:
        logger.info("Upserting chunks into Pinecone vector store...")
        ids = [chunk.chunk_id for chunk in chunks]
        metadata = []
        for chunk in chunks:
            chunk_meta = dict(getattr(chunk, "metadata", {}) or {})
            chunk_meta["content"] = chunk.text
            chunk_meta["chunk_id"] = chunk.chunk_id
            metadata.append(chunk_meta)
        asyncio.run(vector_store.upsert(ids=ids, embeddings=combined_embeddings.tolist(), metadata=metadata))

    search_methods = SearchMethods(
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    # Update global state via callback
    update_globals_fn(
        chunks=chunks,
        bge_embeddings=combined_embeddings,
        all_mini_embeddings=combined_embeddings,
        combined_embeddings=combined_embeddings,
        faiss_index=faiss_index,
        bm25=bm25,
        search_methods=search_methods
    )

    embedding_time = time.time() - start_time
    logger.info(f"Indices built successfully in {embedding_time:.2f} seconds")

    return search_methods


__all__ = ['build_indices']
