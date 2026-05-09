"""Text processing utilities."""

import re
from typing import List
from loguru import logger


class TextProcessor:
    """Utilities for text processing and chunking."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)

        return text.strip()

    @staticmethod
    def chunk_by_sentences(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[str]:
        """
        Chunk text by sentences with overlap.

        Args:
            text: Input text
            chunk_size: Target chunk size in characters
            overlap: Overlap size in characters

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('. '.join(current_chunk) + '.')

                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk[-1:])  # Last sentence as overlap
                current_chunk = [overlap_text, sentence] if len(overlap_text) < overlap else [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        logger.info(f"Chunked text into {len(chunks)} chunks")
        return chunks

    @staticmethod
    def chunk_by_tokens(
        text: str,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ) -> List[str]:
        """
        Chunk text by token count (rough estimate: 1 token ≈ 4 chars).

        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap in tokens

        Returns:
            List of chunks
        """
        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4

        return TextProcessor.chunk_by_sentences(text, max_chars, overlap_chars)
