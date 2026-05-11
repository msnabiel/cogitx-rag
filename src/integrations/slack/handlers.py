"""Slack integration handlers for CogitX-RAG"""

import os
from types import SimpleNamespace
from src.core.models import Query

class SlackHandler:
    """Adapts RAG capabilities for Slack bot requirements."""
    
    def __init__(self, document_processor, rag_manager):
        self.document_processor = document_processor
        self.rag_manager = rag_manager

    async def query_rag(self, query: Query):
        """Run the current query workflow and adapt it for Slack."""
        processor = await self.rag_manager.get_or_reload_processor()
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

    async def ingest_and_query(self, file_path: str, query_text: str, session_id: str):
        """Ingest a local file and answer a query against the refreshed index."""
        file_url = f"file:/{os.path.abspath(file_path)}"
        # Force re-ingestion if we have no processor currently
        processor = await self.rag_manager.get_or_reload_processor()
        force_ingest = processor is None
        
        ingestion_results = await self.document_processor.ingest_documents_async([file_url], force=force_ingest)
        
        # Get processor again
        processor = await self.rag_manager.get_or_reload_processor()
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

    async def ingest_files_and_query(self, file_paths: list[str], query_text: str, session_id: str):
        """Ingest multiple local files and answer a query against the refreshed index."""
        file_urls = [f"file:/{os.path.abspath(path)}" for path in file_paths]
        
        processor = await self.rag_manager.get_or_reload_processor()
        force_ingest = processor is None
        
        ingestion_results = await self.document_processor.ingest_documents_async(file_urls, force=force_ingest)

        processor = await self.rag_manager.get_or_reload_processor()
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
