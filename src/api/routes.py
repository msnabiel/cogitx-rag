"""API routes"""

import os
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File
import logging

from src.core.models import SearchRequest

logger = logging.getLogger(__name__)


def create_router(
    upload_dir,
    document_cache,
    document_processor,
    get_search_methods_fn,
    get_chunks_fn,
    get_query_processor_fn
):
    """Create unified API router"""
    router = APIRouter()

    @router.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        try:
            logger.info(f"Uploading file: {file.filename}")
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(upload_dir, unique_filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            file_url = f"file:/{os.path.abspath(file_path)}"
            logger.info(f"File uploaded successfully: {file_url}")
            ingestion_results = await document_processor.ingest_documents_async([file_url])
            return {
                "url": file_url,
                "filename": file.filename,
                "size": len(content),
                "status": "success",
                "ingestion_results": ingestion_results
            }
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise HTTPException(status_code=500, detail=f"Upload error: {e}")

    @router.post("/search")
    async def search_documents(request: SearchRequest):
        try:
            chunks = get_chunks_fn()
            if not chunks:
                raise HTTPException(status_code=400, detail="No documents indexed. Please ingest documents first.")
            search_methods = get_search_methods_fn()
            results = search_methods.ensemble_search(request.query, request.top_k)
            result_chunks = [result.chunk for result in results]
            scores = [result.combined_score for result in results]
            formatted_results = []
            for i, (chunk, score) in enumerate(zip(result_chunks, scores)):
                formatted_results.append({
                    "rank": i + 1,
                    "content": chunk.text,
                    "metadata": chunk.metadata,
                    "score": float(score),
                    "chunk_id": chunk.chunk_id
                })
            return {
                "query": request.query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {e}")

    @router.post("/ingest")
    async def ingest_documents(file_paths: List[str]):
        try:
            results = await document_processor.ingest_documents_async(file_paths)
            return {"results": results}
        except Exception as e:
            logger.error(f"Error in ingestion: {e}")
            raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

    @router.post("/query")
    async def process_query(query: str, session_id: str | None = None):
        try:
            processor = get_query_processor_fn()
            if processor is None:
                raise HTTPException(status_code=400, detail="No documents indexed. Please ingest documents first.")
            answer = await processor.process(query, session_id=session_id)
            return {"query": query, "answer": answer}
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing error: {e}")

    @router.get("/cache/stats")
    async def get_cache_stats():
        return document_cache.get_stats()

    @router.post("/cache/clear")
    async def clear_cache():
        success = document_cache.clear_all()
        return {
            "success": success,
            "message": "All cache entries cleared" if success else "Failed to clear cache"
        }

    @router.post("/cache/clear-expired")
    async def clear_expired_cache():
        return {
            "success": True,
            "cleared_entries": 0,
            "message": "No cache expiry configured - no entries to clear"
        }

    return router


__all__ = ['create_router']
