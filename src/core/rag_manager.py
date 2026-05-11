"""Centralized RAG state and logic management."""

import os
from types import SimpleNamespace
from loguru import logger
from src.search import SearchMethods
from src.query_processor import ProcessQuery
from src.core.models import Query

class RAGManager:
    """Manages RAG state, indexing, and query processing."""

    def __init__(self, settings, embedding_provider, rag_state_manager, llm_client, conversation_memory, system_prompt, pinecone_store=None):
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.rag_state_manager = rag_state_manager
        self.llm_client = llm_client
        self.conversation_memory = conversation_memory
        self.system_prompt = system_prompt
        self.pinecone_store = pinecone_store

        # State
        self.faiss_index = None
        self.bm25 = None
        self.chunks = []
        self.local_embeddings_1 = None
        self.local_embeddings_2 = None
        self.indexed_vectors = None
        self.search_methods = None
        self.query_processor = None

    def update_state(self, **kwargs):
        """Update internal RAG state."""
        self.chunks = kwargs.get('chunks', self.chunks)
        self.local_embeddings_1 = kwargs.get('local_embeddings_1', self.local_embeddings_1)
        self.local_embeddings_2 = kwargs.get('local_embeddings_2', self.local_embeddings_2)
        self.indexed_vectors = kwargs.get('indexed_vectors', self.indexed_vectors)
        self.faiss_index = kwargs.get('faiss_index', self.faiss_index)
        self.bm25 = kwargs.get('bm25', self.bm25)
        self.search_methods = kwargs.get('search_methods', self.search_methods)
        self.query_processor = None # Reset processor to force recreation with new search methods

    def clear_state(self):
        """Clear all in-memory and persisted state."""
        self.chunks = []
        self.local_embeddings_1 = None
        self.local_embeddings_2 = None
        self.indexed_vectors = None
        self.faiss_index = None
        self.bm25 = None
        self.search_methods = None
        self.query_processor = None

    def load_state(self):
        """Load state from persistence."""
        loaded = self.rag_state_manager.load()
        if loaded:
            self.faiss_index = loaded["faiss_index"]
            self.bm25 = loaded["bm25"]
            self.chunks = loaded["chunks"]
            self.local_embeddings_1 = loaded["local_embeddings_1"]
            self.local_embeddings_2 = loaded["local_embeddings_2"]
            self.indexed_vectors = loaded["combined_embeddings"]
            self.search_methods = SearchMethods(
                faiss_index=self.faiss_index,
                bm25=self.bm25,
                chunks=self.chunks,
                embedding_provider=self.embedding_provider,
                vector_store=self.pinecone_store,
            )
            return True
        return False

    def get_query_processor(self):
        """Get or initialize query processor."""
        if self.query_processor is None and self.search_methods is not None:
            self.query_processor = ProcessQuery(
                self.search_methods,
                self.llm_client,
                generation_config=None,
                memory_manager=self.conversation_memory,
                system_prompt=self.system_prompt,
            )
        return self.query_processor

    async def get_or_reload_processor(self):
        """Ensure query processor is ready, reloading if necessary."""
        processor = self.get_query_processor()
        if processor is None:
            if self.load_state():
                processor = self.get_query_processor()
        return processor
