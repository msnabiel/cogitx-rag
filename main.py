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
from src.storage.embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from src.index_builder import build_indices as build_indices_fn
from src.ingestion import DocumentProcessor
from src.query_processor import ProcessQuery
from src.api.routes import create_router
from src.storage.memory.state_manager import RAGStateManager, ConversationMemoryManager
from src.core.models import Query
from src.llm.gemini_client import GeminiClient
from src.llm.openai_client import OpenAIClient
from src.integrations.slack.bot import start_slack_bot

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

# Processing config
EMBEDDING_BATCH_SIZE = settings.processing.embedding_batch_size_gpu if DEVICE == "cuda" else settings.processing.embedding_batch_size_cpu
EMBEDDING_WORKERS = settings.processing.embedding_workers_gpu if DEVICE == "cuda" else settings.processing.embedding_workers_cpu

# Initialize models
bge_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
all_mini_model = SentenceTransformer('all-MiniLM-L6-v2')
if DEVICE == "cuda":
    bge_model = bge_model.to(DEVICE)
    all_mini_model = all_mini_model.to(DEVICE)

embedding_generator = HuggingFaceEmbeddings(batch_size=EMBEDDING_BATCH_SIZE, device=DEVICE)

# Global state - RAG indices and chunks
faiss_index = None
bm25 = None
chunks = []
bge_embeddings = None
all_mini_embeddings = None
combined_embeddings = None
search_methods = None
query_processor = None

# Cache
document_cache = DocumentCache(cache_dir=CACHE_DIR, logger=logger)
rag_state_manager = RAGStateManager(STATE_DIR, logger)
conversation_memory = ConversationMemoryManager(CACHE_DIR, logger, window_size=6)

# LLM config
system_prompt = load_prompt("prompts/system_prompt.txt")

def build_llm_client():
    provider = settings.llm.default_llm_provider
    if provider == "openai":
        return OpenAIClient()
    return GeminiClient(api_key=GEMINI_API_KEY)

llm_client = build_llm_client()

def update_global_state(**kwargs):
    """Update global RAG state after indexing"""
    global chunks, bge_embeddings, all_mini_embeddings, combined_embeddings
    global faiss_index, bm25, search_methods
    chunks = kwargs['chunks']
    bge_embeddings = kwargs['bge_embeddings']
    all_mini_embeddings = kwargs['all_mini_embeddings']
    combined_embeddings = kwargs['combined_embeddings']
    faiss_index = kwargs['faiss_index']
    bm25 = kwargs['bm25']
    search_methods = kwargs['search_methods']

def build_indices(input_chunks):
    """Build FAISS + BM25 indices"""
    search_state = build_indices_fn(
        chunks=input_chunks,
        embedding_generator=embedding_generator,
        bge_model=bge_model,
        all_mini_model=all_mini_model,
        embedding_workers=EMBEDDING_WORKERS,
        update_globals_fn=update_global_state
    )
    rag_state_manager.save(
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        bge_embeddings=bge_embeddings,
        all_mini_embeddings=all_mini_embeddings,
        combined_embeddings=combined_embeddings,
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

async def query_rag_for_slack(query: Query):
    """Run the current query workflow and adapt it for Slack."""
    processor = get_query_processor()
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
    ingestion_results = await document_processor.ingest_documents_async([file_url])
    processor = get_query_processor()
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
    ingestion_results = []
    for file_url in file_urls:
        ingestion_results.extend(await document_processor.ingest_documents_async([file_url]))

    processor = get_query_processor()
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
    lambda text: chunk_text_strategy(text, model=bge_model),
    build_indices,
    document_cache
)

loaded_state = rag_state_manager.load()
if loaded_state:
    from src.search import SearchMethods
    faiss_index = loaded_state["faiss_index"]
    bm25 = loaded_state["bm25"]
    chunks = loaded_state["chunks"]
    bge_embeddings = loaded_state["bge_embeddings"]
    all_mini_embeddings = loaded_state["all_mini_embeddings"]
    combined_embeddings = loaded_state["combined_embeddings"]
    search_methods = SearchMethods(
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        bge_model=bge_model,
        all_mini_model=all_mini_model,
        all_mini_embeddings=all_mini_embeddings,
    )

# FastAPI app
app = FastAPI(title="CogitX-RAG API", description="Production-grade RAG system", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(create_router(UPLOAD_DIR, document_cache, document_processor, lambda: search_methods, lambda: chunks, get_query_processor), prefix="/api/v1")
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
