"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from config.settings import settings
from config.logging_config import setup_logging
from api.dependencies import Dependencies
from api.routes import query, ingest, health


# Setup logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting CogitX-RAG API...")

    deps = Dependencies.get_instance()
    await deps.initialize()

    logger.info("CogitX-RAG API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down CogitX-RAG API...")

    if deps.rag_pipeline:
        await deps.rag_pipeline.close()

    logger.info("CogitX-RAG API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="CogitX-RAG",
    description="Graph-based RAG system with dual-path retrieval",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "CogitX-RAG",
        "version": "0.1.0",
        "description": "Graph-based RAG system with dual-path retrieval",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        reload=settings.api.api_reload,
        workers=settings.api.api_workers if not settings.api.api_reload else 1,
    )
