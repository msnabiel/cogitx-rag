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
from src.core.rag_manager import RAGManager
from src.llm.gemini_client import GeminiClient
from src.llm.openai_client import OpenAIClient
from src.integrations.slack.bot import start_slack_bot
from src.integrations.slack.handlers import SlackHandler
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

# Core RAG Manager
rag_manager = RAGManager(
    settings=settings,
    embedding_provider=embedding_provider,
    rag_state_manager=rag_state_manager,
    llm_client=llm_client,
    conversation_memory=conversation_memory,
    system_prompt=system_prompt,
    pinecone_store=pinecone_store,
)

async def build_indices(input_chunks):
    """Build FAISS + BM25 indices, merging with existing chunks if any."""
    # Merge new chunks with existing ones
    all_chunks = rag_manager.chunks + input_chunks
    
    workers = settings.processing.embedding_workers_gpu if DEVICE == "cuda" else settings.processing.embedding_workers_cpu
    search_state = await build_indices_fn(
        chunks=all_chunks,
        embedding_provider=embedding_provider,
        embedding_workers=workers,
        update_globals_fn=rag_manager.update_state,
        vector_store=pinecone_store,
    )
    if settings.vector_store.vector_store_type != "pinecone":
        rag_state_manager.save(
            faiss_index=rag_manager.faiss_index,
            bm25=rag_manager.bm25,
            chunks=rag_manager.chunks,
            local_embeddings_1=rag_manager.local_embeddings_1,
            local_embeddings_2=rag_manager.local_embeddings_2,
            combined_embeddings=rag_manager.indexed_vectors,
        )
    return search_state

def clear_rag_state():
    """Clear persisted and in-memory RAG retrieval state."""
    rag_manager.clear_state()

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

# Initialize document processor
document_processor = DocumentProcessor(
    lambda text: chunk_text_strategy(text, model=embedding_model),
    build_indices,
    document_cache
)

# Initialize Slack Handler
slack_handler = SlackHandler(document_processor, rag_manager)

# Load initial state
if not rag_manager.load_state():
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
        lambda: rag_manager.search_methods,
        lambda: rag_manager.chunks,
        rag_manager.get_query_processor,
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
                slack_handler.query_rag,
                slack_handler.ingest_and_query,
                slack_handler.ingest_files_and_query,
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
                slack_handler.query_rag, # Reusing slack handler logic as it's generic enough
                slack_handler.ingest_files_and_query,
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
