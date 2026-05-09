"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from loguru import logger
from embeddings.openai_embeddings import OpenAIEmbedding
from embeddings.gemini_embeddings import GeminiEmbedding
from vector_stores.factory import VectorStoreFactory
from graph.neo4j_client import Neo4jClient
from graph.entity_extractor import EntityExtractor
from graph.graph_builder import GraphBuilder
from graph.graph_retriever import GraphRetriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from reranking.cross_encoder import CrossEncoderReranker
from memory.cache import ShortTermMemory
from llm.gemini_client import GeminiClient
from llm.openai_client import OpenAIClient
from pipeline.rag_pipeline import RAGPipeline
from pipeline.ingestion import IngestionPipeline
from config.settings import settings


class Dependencies:
    """Singleton dependencies container."""

    _instance = None

    def __init__(self):
        if Dependencies._instance is not None:
            raise Exception("Use Dependencies.get_instance()")

        self.embedding_model = None
        self.vector_store = None
        self.neo4j_client = None
        self.graph_builder = None
        self.graph_retriever = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.reranker = None
        self.short_term_memory = None
        self.llm_client = None
        self.rag_pipeline = None
        self.ingestion_pipeline = None

        Dependencies._instance = self

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """Initialize all dependencies."""
        logger.info("Initializing dependencies...")

        # Embedding model
        if settings.llm.default_llm_provider == "gemini":
            self.embedding_model = GeminiEmbedding()
        else:
            self.embedding_model = OpenAIEmbedding()

        # Vector store
        self.vector_store = await VectorStoreFactory.get_or_create()

        # Neo4j and graph components
        try:
            self.neo4j_client = Neo4jClient()
            await self.neo4j_client.connect()

            entity_extractor = EntityExtractor()
            self.graph_builder = GraphBuilder(self.neo4j_client, entity_extractor)
            self.graph_retriever = GraphRetriever(self.neo4j_client, entity_extractor)
        except Exception as e:
            logger.warning(f"Graph components not initialized: {e}")

        # Retrievers
        self.vector_retriever = VectorRetriever(self.vector_store, self.embedding_model)

        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.load_index()

        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            graph_retriever=self.graph_retriever,
        )

        # Reranker
        try:
            self.reranker = CrossEncoderReranker()
        except Exception as e:
            logger.warning(f"Reranker not initialized: {e}")

        # Memory
        try:
            self.short_term_memory = ShortTermMemory()
            await self.short_term_memory.connect()
        except Exception as e:
            logger.warning(f"Memory not initialized: {e}")

        # LLM
        if settings.llm.default_llm_provider == "gemini":
            self.llm_client = GeminiClient()
        else:
            self.llm_client = OpenAIClient()

        # RAG Pipeline
        self.rag_pipeline = RAGPipeline(
            retriever=self.hybrid_retriever,
            llm=self.llm_client,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            short_term_memory=self.short_term_memory,
        )

        # Ingestion Pipeline
        self.ingestion_pipeline = IngestionPipeline(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            graph_builder=self.graph_builder,
            bm25_retriever=self.bm25_retriever,
        )

        logger.info("Dependencies initialized successfully")


# Dependency injection functions
@lru_cache()
def get_dependencies() -> Dependencies:
    """Get dependencies singleton."""
    return Dependencies.get_instance()


async def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline instance."""
    deps = get_dependencies()
    return deps.rag_pipeline


async def get_ingestion_pipeline() -> IngestionPipeline:
    """Get ingestion pipeline instance."""
    deps = get_dependencies()
    return deps.ingestion_pipeline
