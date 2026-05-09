import os
import torch
from sentence_transformers import SentenceTransformer
import google.genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.storage.memory.cache import DocumentCache
from config import generation_config as gen_config
from src.utils.logger import setup_logging, log_request_middleware
from src.utils.prompt_loader import load_prompt
from src.utils.chunking_strategies import chunk_text as chunk_text_strategy
from src.config.settings import settings
from src.storage.embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from src.index_builder import build_indices as build_indices_fn
from src.ingestion import DocumentProcessor
from src.query_processor import ProcessQuery
from src.api.routes import create_router

logger = setup_logging()
logger.info("=== COGITX-RAG SYSTEM STARTING ===\n")

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load env
load_dotenv(dotenv_path=".env.local")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Directories
UPLOAD_DIR = settings.api.upload_dir
CACHE_DIR = settings.api.cache_dir
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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

# Gemini config
system_prompt = load_prompt("prompts/system_prompt.txt")
gen_config['system_instruction'] = system_prompt
generation_config = types.GenerateContentConfig(**gen_config)
gemini_client = google.genai.Client(api_key=GEMINI_API_KEY)

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
    return build_indices_fn(
        chunks=input_chunks,
        embedding_generator=embedding_generator,
        bge_model=bge_model,
        all_mini_model=all_mini_model,
        embedding_workers=EMBEDDING_WORKERS,
        update_globals_fn=update_global_state
    )

def get_query_processor():
    """Get or initialize query processor"""
    global query_processor
    if query_processor is None and search_methods is not None:
        query_processor = ProcessQuery(search_methods, gemini_client, generation_config)
    return query_processor

# Initialize document processor
document_processor = DocumentProcessor(
    lambda text: chunk_text_strategy(text, model=bge_model),
    build_indices,
    document_cache
)

# FastAPI app
app = FastAPI(title="CogitX-RAG API", description="Production-grade RAG system", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(create_router(UPLOAD_DIR, document_cache, document_processor, lambda: search_methods, lambda: chunks, get_query_processor), prefix="/api/v1")
app.middleware("http")(log_request_middleware)

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
