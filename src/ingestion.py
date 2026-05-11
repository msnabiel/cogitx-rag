"""Document ingestion and processing utilities"""

import os
import logging
import asyncio
import hashlib
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.text_extractor import TextExtractor
from src.utils.text_cleaner import CleaningOptions

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document ingestion and processing"""

    def __init__(self, chunk_text_fn, build_indices_fn, document_cache):
        """
        Initialize with functions and cache

        Args:
            chunk_text_fn: Function to chunk text
            build_indices_fn: Function to build indices
            document_cache: DocumentCache instance
        """
        self.chunk_text = chunk_text_fn
        self.build_indices = build_indices_fn
        self.document_cache = document_cache
        self.text_extractor = TextExtractor()

    async def ingest_documents_async(self, doc_inputs: List[str], force: bool = False) -> List:
        """
        Ingest documents using the text extraction service

        Args:
            doc_inputs: List of document paths or URLs
            force: Whether to force re-ingestion even if already indexed

        Returns:
            List of IngestionResult objects
        """
        all_chunks = []
        results = []

        # Process local files only
        local_files = doc_inputs
        if local_files:
            logger.info(f"Processing {len(local_files)} local files using parallel method...")
            try:
                local_chunks = self.process_documents_parallel(local_files, force=force)
                all_chunks.extend(local_chunks)
                for file in local_files:
                    results.append({"source": file, "success": True})
            except Exception as e:
                logger.error(f"Failed to process local files: {e}")
                for file in local_files:
                    results.append({"source": file, "success": False})

        # Check if anything worked
        if not all_chunks:
            logger.warning("No new chunks extracted from any document.")
        else:
            await self.build_indices(all_chunks)
            logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(doc_inputs)} documents")

        return results

    def ingest_documents(self, file_paths: List[str], force: bool = False):
        """Synchronous wrapper for document ingestion"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.ingest_documents_async(file_paths, force=force))
        finally:
            loop.close()

    def process_documents_parallel(self, inputs: List[str], max_workers: int = 10, force: bool = False):
        """
        Process multiple documents (local files only) in parallel using direct extraction

        Args:
            inputs: List of local file paths
            max_workers: Maximum number of parallel workers
            force: Whether to force re-ingestion even if already indexed

        Returns:
            List of DocumentChunk objects
        """
        def process_single_input(input_source: str):
            try:
                if not (input_source.startswith("file://") or os.path.exists(input_source)):
                    logger.error(f"Invalid input: {input_source}")
                    return []

                local_path = input_source.replace("file:/", "")
                logger.info(f"📄 Processing local file: {local_path}")

                with open(local_path, "rb") as f:
                    file_bytes = f.read()

                content_hash = hashlib.sha256(file_bytes).hexdigest()
                cached_data = self.document_cache.get(content_hash)

                if cached_data:
                    logger.info(f"🔄 Reusing cached extracted text for hash {content_hash}")
                    text = cached_data["data"]["text"]
                    metadata = cached_data["data"]["metadata"]
                else:
                    extraction_result = self.text_extractor.extract_text_from_bytes(
                        file_bytes,
                        os.path.basename(local_path),
                        cleaning_options=CleaningOptions(
                            normalize_unicode=True,
                            clean_whitespace=True,
                            preserve_structure=True,
                            max_length=100000,
                            enable_ocr=True,
                        )
                    )
                    text = extraction_result.text
                    metadata = extraction_result.metadata
                    metadata["content_hash"] = content_hash

                    cache_data = {
                        "text": text,
                        "metadata": metadata,
                        "processing_time": metadata.get("processing_time", 0)
                    }
                    self.document_cache.set(content_hash, cache_data)

                logger.info(f"✅ Extracted {len(text)} characters from {input_source}")
                logger.info(f"Extracted preview: {text[:500]}")
                if not text.strip():
                    logger.warning(f"No content extracted from {input_source}")
                    return []

                if not force and self.document_cache.is_ingested(content_hash):
                    logger.info(f"⏭️ Skipping already indexed document hash {content_hash}")
                    return []

                # Create chunks
                chunks = self.chunk_text(text)
                for chunk in chunks:
                    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                        chunk.metadata["content_hash"] = content_hash
                self.document_cache.mark_ingested(content_hash)
                return chunks

            except Exception as e:
                logger.error(f"Processing failed for {input_source}: {e}")
                return []

        # Parallel execution
        all_chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_input = {executor.submit(process_single_input, inp): inp for inp in inputs}
            for future in as_completed(future_to_input):
                input_source = future_to_input[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    logger.info(f"✅ Finished processing {input_source} → {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Unhandled exception for {input_source}: {e}")

        return all_chunks


__all__ = ['DocumentProcessor']
