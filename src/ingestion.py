"""Document ingestion and processing utilities"""

import os
import logging
import asyncio
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

    async def ingest_documents_async(self, doc_inputs: List[str]) -> List:
        """
        Ingest documents using the text extraction service

        Args:
            doc_inputs: List of document paths or URLs

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
                local_chunks = self.process_documents_parallel(local_files)
                all_chunks.extend(local_chunks)
                for file in local_files:
                    results.append({"source": file, "success": True})
            except Exception as e:
                logger.error(f"Failed to process local files: {e}")
                for file in local_files:
                    results.append({"source": file, "success": False})

        # Check if anything worked
        if not all_chunks:
            logger.warning("No chunks extracted from any document.")
        else:
            self.build_indices(all_chunks)
            logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(doc_inputs)} documents")

        return results

    def ingest_documents(self, file_paths: List[str]):
        """Synchronous wrapper for document ingestion"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.ingest_documents_async(file_paths))
        finally:
            loop.close()

    def process_documents_parallel(self, inputs: List[str], max_workers: int = 10):
        """
        Process multiple documents (local files only) in parallel using direct extraction

        Args:
            inputs: List of local file paths
            max_workers: Maximum number of parallel workers

        Returns:
            List of DocumentChunk objects
        """
        def process_single_input(input_source: str):
            try:
                # Check cache first
                cached_data = self.document_cache.get(input_source)
                if cached_data:
                    logger.info(f"🔄 Using cached data for {input_source}")
                    text = cached_data["data"]["text"]
                    metadata = cached_data["data"]["metadata"]
                else:
                    # Process local file only
                    if input_source.startswith("file://") or os.path.exists(input_source):
                        # normalize local file path
                        local_path = input_source.replace("file:/", "")
                        logger.info(f"📄 Processing local file: {local_path}")

                        with open(local_path, 'rb') as f:
                            file_bytes = f.read()

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

                        cache_data = {
                            "text": text,
                            "metadata": metadata,
                            "processing_time": metadata.get('processing_time', 0)
                        }
                        self.document_cache.set(input_source, cache_data)

                    else:
                        logger.error(f"Invalid input: {input_source}")
                        return []

                logger.info(f"✅ Extracted {len(text)} characters from {input_source}")

                if not text.strip():
                    logger.warning(f"No content extracted from {input_source}")
                    return []

                # Create chunks
                return self.chunk_text(text)

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
