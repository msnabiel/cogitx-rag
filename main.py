import os
import torch
from sentence_transformers import SentenceTransformer
import asyncio
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.storage.memory.cache import DocumentCache
from src.utils.logger import setup_logging, log_request_middleware
from src.utils.prompt_loader import load_prompt
from src.utils.chunking_strategies import chunk_text as chunk_text_strategy
from src.config.settings import settings
from src.storage.embeddings.local_embeddings import LocalEmbedding
from src.storage.embeddings.local_embeddings import LocalDualEmbedding
from src.storage.embeddings.openai_embeddings import OpenAIEmbedding
from src.storage.embeddings.gemini_embeddings import GeminiEmbedding
from src.index_builder import build_indices as build_indices_fn
from src.ingestion import DocumentProcessor
from src.query_processor import ProcessQuery
from src.api.routes import create_router
from src.storage.memory.state_manager import RAGStateManager, ConversationMemoryManager
from src.core.models import Query
from src.llm.gemini_client import GeminiClient
from src.llm.openai_client import OpenAIClient
from src.integrations.slack.bot import start_slack_bot
from src.integrations.telegram.bot import start_telegram_bot
from src.storage.vector_stores.pinecone_store import PineconeVectorStore

logger = setup_logging()
logger.info("=== COGITX-RAG SYSTEM STARTING ===\n")

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load env
load_dotenv(dotenv_path=".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Directories
UPLOAD_DIR = settings.api.upload_dir
CACHE_DIR = settings.api.cache_dir
STATE_DIR = os.getenv("RAG_STATE_DIR", "./data/rag_state")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

def build_embedding_provider():
    provider = settings.embeddings.embedding_provider
    if provider == "openai":
        return OpenAIEmbedding(
            model_name=settings.embeddings.openai_embedding_model,
            dimension=settings.vector_store.pinecone_dimension if settings.vector_store.vector_store_type == "pinecone" else settings.vector_store.faiss_dimension,
        )
    if provider == "gemini":
        return GeminiEmbedding(
            model_name=settings.embeddings.gemini_embedding_model,
            dimension=settings.vector_store.pinecone_dimension if settings.vector_store.vector_store_type == "pinecone" else settings.vector_store.faiss_dimension,
        )
    if provider == "local_single":
        return LocalEmbedding(
            model_name=settings.embeddings.local_embedding_model_1,
            dimension=settings.embeddings.local_dimension,
            device=DEVICE,
        )
    return LocalDualEmbedding(
        bge_model_name=settings.embeddings.local_embedding_model_1,
        mini_model_name=settings.embeddings.local_embedding_model_2,
        device=DEVICE,
    )

embedding_provider = build_embedding_provider()

# Extract model for chunking if available
embedding_model = getattr(embedding_provider, "bge_model", getattr(embedding_provider, "model", None))

# Global state - RAG indices and chunks
faiss_index = None
bm25 = None
chunks = []
local_embeddings_1 = None
local_embeddings_2 = None
indexed_vectors = None
search_methods = None
query_processor = None

# Cache
document_cache = DocumentCache(cache_dir=CACHE_DIR, logger=logger)
rag_state_manager = RAGStateManager(STATE_DIR, logger)

# LLM config
system_prompt = load_prompt("prompts/system_prompt.txt")

def build_llm_client():
    provider = settings.llm.default_llm_provider
    if provider == "openai":
        return OpenAIClient()
    return GeminiClient(api_key=GEMINI_API_KEY)

llm_client = build_llm_client()

conversation_memory = ConversationMemoryManager(
    CACHE_DIR,
    logger,
    window_size=settings.memory.conversation_window_size,
    overflow_threshold=settings.memory.overflow_summary_threshold,
    llm_client=llm_client,
)

pinecone_store = None
if settings.vector_store.vector_store_type == "pinecone":
    pinecone_store = PineconeVectorStore(dimension=embedding_provider.dimension)

def update_global_state(**kwargs):
    """Update global RAG state after indexing"""
    global chunks, local_embeddings_1, local_embeddings_2, indexed_vectors
    global faiss_index, bm25, search_methods
    chunks = kwargs['chunks']
    local_embeddings_1 = kwargs['local_embeddings_1']
    local_embeddings_2 = kwargs['local_embeddings_2']
    indexed_vectors = kwargs['indexed_vectors']
    faiss_index = kwargs['faiss_index']
    bm25 = kwargs['bm25']
    search_methods = kwargs['search_methods']

def clear_rag_state():
    """Clear persisted and in-memory RAG retrieval state."""
    global chunks, local_embeddings_1, local_embeddings_2, indexed_vectors
    global faiss_index, bm25, search_methods, query_processor

    chunks = []
    local_embeddings_1 = None
    local_embeddings_2 = None
    indexed_vectors = None
    faiss_index = None
    bm25 = None
    search_methods = None
    query_processor = None

    # Clear document cache registry to allow re-indexing
    document_cache.clear_all()

    for path in [
        rag_state_manager.faiss_file,
        rag_state_manager.bm25_file,
        rag_state_manager.chunks_file,
        rag_state_manager.local_embeddings_1_file,
        rag_state_manager.local_embeddings_2_file,
        rag_state_manager.combined_embeddings_file,
    ]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"Failed to remove state file {path}: {e}")

    return {
        "state_cleared": True,
        "message": "RAG state cleared",
    }

async def build_indices(input_chunks):
    """Build FAISS + BM25 indices, merging with existing chunks if any."""
    global chunks
    
    # Merge new chunks with existing ones
    all_chunks = chunks + input_chunks
    
    workers = settings.processing.embedding_workers_gpu if DEVICE == "cuda" else settings.processing.embedding_workers_cpu
    search_state = await build_indices_fn(
        chunks=all_chunks,
        embedding_provider=embedding_provider,
        embedding_workers=workers,
        update_globals_fn=update_global_state,
        vector_store=pinecone_store,
    )
    if settings.vector_store.vector_store_type != "pinecone":
        rag_state_manager.save(
            faiss_index=faiss_index,
            bm25=bm25,
            chunks=chunks,
            local_embeddings_1=local_embeddings_1,
            local_embeddings_2=local_embeddings_2,
            combined_embeddings=indexed_vectors,
        )
    return search_state

def get_query_processor():
    """Get or initialize query processor"""
    global query_processor
    if query_processor is None and search_methods is not None:
        query_processor = ProcessQuery(
            search_methods,
            llm_client,
            generation_config=None,
            memory_manager=conversation_memory,
            system_prompt=system_prompt,
        )
    return query_processor

async def _get_or_reload_query_processor():
    """Ensure query processor is ready, reloading from state if necessary."""
    processor = get_query_processor()
    if processor is None:
        global search_methods, chunks, local_embeddings_1, local_embeddings_2, indexed_vectors, faiss_index, bm25
        if search_methods is None:
            loaded = rag_state_manager.load()
            if loaded:
                faiss_index = loaded["faiss_index"]
                bm25 = loaded["bm25"]
                chunks = loaded["chunks"]
                local_embeddings_1 = loaded["local_embeddings_1"]
                local_embeddings_2 = loaded["local_embeddings_2"]
                indexed_vectors = loaded["combined_embeddings"]
                from src.search import SearchMethods
                search_methods = SearchMethods(
                    faiss_index=faiss_index,
                    bm25=bm25,
                    chunks=chunks,
                    embedding_provider=embedding_provider,
                    vector_store=pinecone_store,
                )
                processor = get_query_processor()
    return processor

async def query_rag_for_slack(query: Query):
    """Run the current query workflow and adapt it for Slack."""
    processor = await _get_or_reload_query_processor()
    if processor is None:
        raise RuntimeError("No document is available for retrieval yet. Upload a readable PDF in this thread first.")
    result = await processor.process(query.text, session_id=query.session_id)
    return SimpleNamespace(
        answer=result.answer,
        confidence=result.confidence,
        processing_time_ms=0.0,
        sources=result.sources,
        citations=result.citations,
    )

async def ingest_and_query_for_slack(file_path: str, query_text: str, session_id: str):
    """Ingest a local file and answer a query against the refreshed index."""
    file_url = f"file:/{os.path.abspath(file_path)}"
    # Force re-ingestion if we have no index currently
    force_ingest = search_methods is None
    ingestion_results = await document_processor.ingest_documents_async([file_url], force=force_ingest)
    
    processor = await _get_or_reload_query_processor()
    if processor is None:
        raise RuntimeError("No documents were indexed from the uploaded file.")
    
    result = await processor.process(query_text, session_id=session_id)
    return {
        "url": file_url,
        "ingestion_results": ingestion_results,
        "answer": result.answer,
        "citations": result.citations,
        "confidence": result.confidence,
    }

async def ingest_files_and_query_for_slack(file_paths: list[str], query_text: str, session_id: str):
    """Ingest multiple local files and answer a query against the refreshed index."""
    file_urls = [f"file:/{os.path.abspath(path)}" for path in file_paths]
    # Force re-ingestion if we have no index currently
    force_ingest = search_methods is None
    ingestion_results = await document_processor.ingest_documents_async(file_urls, force=force_ingest)

    processor = await _get_or_reload_query_processor()
    if processor is None:
        raise RuntimeError("No documents were indexed from the uploaded files.")

    result = await processor.process(query_text, session_id=session_id)
    return {
        "urls": file_urls,
        "ingestion_results": ingestion_results,
        "answer": result.answer,
        "citations": result.citations,
        "confidence": result.confidence,
    }

# Initialize document processor
document_processor = DocumentProcessor(
    lambda text: chunk_text_strategy(text, model=embedding_model),
    build_indices,
    document_cache
)

loaded_state = rag_state_manager.load()
if loaded_state:
    from src.search import SearchMethods
    faiss_index = loaded_state["faiss_index"]
    bm25 = loaded_state["bm25"]
    chunks = loaded_state["chunks"]
    local_embeddings_1 = loaded_state["local_embeddings_1"]
    local_embeddings_2 = loaded_state["local_embeddings_2"]
    indexed_vectors = loaded_state["combined_embeddings"]
    search_methods = SearchMethods(
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        embedding_provider=embedding_provider,
        vector_store=pinecone_store,
    )
    query_processor = get_query_processor()
else:
    logger.info("No persisted RAG state found. Clearing document ingestion registry to ensure fresh starts.")
    document_cache.clear_registry()

# FastAPI app
app = FastAPI(title="CogitX-RAG API", description="Production-grade RAG system", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(
    create_router(
        UPLOAD_DIR,
        document_cache,
        document_processor,
        lambda: search_methods,
        lambda: chunks,
        get_query_processor,
        clear_rag_state,
    ),
    prefix="/api/v1",
)
app.middleware("http")(log_request_middleware)


@app.on_event("startup")
async def startup_slack_bot():
    slack_enabled = settings.slack.slack_enabled
    logger.info(f"Slack enabled resolved to {slack_enabled}")
    if slack_enabled:
        logger.info("Slack enabled; starting Slack bot in background")
        asyncio.create_task(
            start_slack_bot(
                query_rag_for_slack,
                ingest_and_query_for_slack,
                ingest_files_and_query_for_slack,
                enabled=slack_enabled,
            )
        )


@app.on_event("startup")
async def startup_telegram_bot():
    telegram_enabled = settings.telegram.telegram_enabled
    logger.info(f"Telegram enabled resolved to {telegram_enabled}")
    if telegram_enabled:
        logger.info("Telegram enabled; starting Telegram bot in background")
        asyncio.create_task(
            start_telegram_bot(
                query_rag_for_slack,
                ingest_files_and_query_for_slack,
                enabled=telegram_enabled,
            )
        )

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", settings.api.api_host)
    port = int(os.getenv("API_PORT", settings.api.api_port))
    reload_enabled = settings.environment != "production"
    logger.info("=== STARTING UVICORN SERVER ===")
    logger.info(f"Server will start on http://{host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        workers=1 if reload_enabled else settings.api.api_workers,
        log_config=None,
        access_log=True,
    )
