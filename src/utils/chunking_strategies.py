"""Text chunking strategies for document processing"""

import re
import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import constants from settings
from src.config.settings import settings

CHUNK_SIZE = settings.processing.chunk_size
OVERLAP_SIZE = settings.processing.overlap_size
JACCARD_SIMILARITY_THRESHOLD = settings.processing.jaccard_similarity_threshold
CONTAINED_RATIO = settings.processing.contained_ratio



@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    metadata: dict = None
    semantic_score: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _tokenize_for_similarity(text: str) -> set:
    return set(text.lower().split())


def _jaccard_similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    tokens_a = _tokenize_for_similarity(a)
    tokens_b = _tokenize_for_similarity(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)

    return len(intersection) / len(union) if union else 0.0


def _is_contained_with_ratio(a: str, b: str, min_ratio: float = CONTAINED_RATIO) -> bool:
    """Check if text a is contained in text b with minimum ratio"""
    tokens_a = _tokenize_for_similarity(a)
    tokens_b = _tokenize_for_similarity(b)

    if not tokens_a or not tokens_b:
        return False

    intersection = tokens_a.intersection(tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))

    return (len(intersection) / smaller) >= min_ratio if smaller > 0 else False


def chunk_text(text: str, model=None) -> List[DocumentChunk]:
    """Enhanced text chunking with multiple strategies"""
    chunks = []

    # Strategy 1: Semantic chunking (sentence-based)
    semantic_chunks = _semantic_chunking(text)
    chunks.extend(semantic_chunks)

    # Strategy 2: Hierarchical chunking (paragraph-based)
    hierarchical_chunks = _hierarchical_chunking(text)
    chunks.extend(hierarchical_chunks)

    # Strategy 3: Fixed-size chunking (fallback)
    if len(chunks) < 3:  # If other strategies didn't produce enough chunks
        fixed_chunks = _fixed_size_chunking(text)
        chunks.extend(fixed_chunks)

    # Remove duplicates and sort by position
    unique_chunks = _deduplicate_chunks(chunks)
    unique_chunks.sort(key=lambda x: x.metadata.get("start_idx", 0))

    for chunk in unique_chunks:
        if isinstance(chunk.metadata, dict):
            chunk.metadata.update(_extract_position_metadata(chunk.text))

    # Apply dynamic chunk resizing
    resized_chunks = _dynamic_resize(unique_chunks, model=model)

    # Merge overlapping/similar chunks
    merged_chunks = merge_chunks(resized_chunks, min_overlap=5)
    logger.info(f"Merged chunks: input={len(resized_chunks)} → output={len(merged_chunks)}")
    return merged_chunks


def _semantic_chunking(text: str) -> List[DocumentChunk]:
    """Chunk text based on semantic boundaries (sentences)"""

    # Split by sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    chunks = []

    current_chunk = []
    current_length = 0
    start_idx = 0

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence.split())

        if current_length + sentence_length > CHUNK_SIZE and current_chunk:
            # Create chunk from current sentences
            chunk_text = " ".join(current_chunk)
            chunk = DocumentChunk(
                text=chunk_text,
                metadata={
                    "start_idx": start_idx,
                    "end_idx": start_idx + current_length,
                    "chunk_type": "semantic",
                    "num_sentences": len(current_chunk)
                }
            )
            chunks.append(chunk)

            # Start new chunk with overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
            start_idx = start_idx + len(overlap_sentences) * 10  # Approximate
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk = DocumentChunk(
            text=chunk_text,
            metadata={
                "start_idx": start_idx,
                "end_idx": start_idx + current_length,
                "chunk_type": "semantic",
                "num_sentences": len(current_chunk)
            }
        )
        chunks.append(chunk)

    return chunks


def _hierarchical_chunking(text: str) -> List[DocumentChunk]:
    """Chunk text based on hierarchical structure (paragraphs, sections)"""
    # Split by paragraph boundaries
    paragraphs = text.split('\n\n')
    chunks = []

    current_chunk = []
    current_length = 0
    start_idx = 0

    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_length = len(paragraph.split())

        if current_length + paragraph_length > CHUNK_SIZE * 1.5 and current_chunk:
            # Create chunk from current paragraphs
            chunk_text = "\n\n".join(current_chunk)
            chunk = DocumentChunk(
                text=chunk_text,
                metadata={
                    "start_idx": start_idx,
                    "end_idx": start_idx + current_length,
                    "chunk_type": "hierarchical",
                    "num_paragraphs": len(current_chunk)
                }
            )
            chunks.append(chunk)

            # Start new chunk
            current_chunk = [paragraph]
            current_length = paragraph_length
            start_idx = start_idx + current_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length

    # Add final chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunk = DocumentChunk(
            text=chunk_text,
            metadata={
                "start_idx": start_idx,
                "end_idx": start_idx + current_length,
                "chunk_type": "hierarchical",
                "num_paragraphs": len(current_chunk)
            }
        )
        chunks.append(chunk)

    return chunks


def _fixed_size_chunking(text: str) -> List[DocumentChunk]:
    """Traditional fixed-size chunking with overlap"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), CHUNK_SIZE - OVERLAP_SIZE):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)

        chunk = DocumentChunk(
            text=chunk_text,
            metadata={
                "start_idx": i,
                "end_idx": i + len(chunk_words),
                "chunk_type": "fixed_size"
            }
        )
        chunks.append(chunk)

    return chunks


def _dynamic_resize(chunks: List[DocumentChunk],
                    min_chunk_size: int = 150,
                    max_chunk_size: int = CHUNK_SIZE,
                    model=None) -> List[DocumentChunk]:
    """Dynamically resize chunks based on semantic importance"""
    adjusted = []
    for chunk in chunks:
        word_count = len(chunk.text.split())
        if word_count < min_chunk_size:
            adjusted.append(chunk)
        else:
            adjusted.append(chunk)
    return adjusted


def _deduplicate_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Remove duplicate chunks based on content similarity"""
    unique_chunks: List[DocumentChunk] = []

    for chunk in chunks:
        candidate_text = chunk.text.strip()
        if len(candidate_text) <= 10:
            continue
        skip_add = False
        replace_index = None

        for idx, existing in enumerate(unique_chunks):
            existing_text = existing.text.strip()
            # Exact match
            if candidate_text.lower() == existing_text.lower():
                skip_add = True
                break
            # Containment deduplication (prefer longer)
            if _is_contained_with_ratio(candidate_text, existing_text, min_ratio=CONTAINED_RATIO):
                if len(candidate_text) > len(existing_text):
                    replace_index = idx
                else:
                    skip_add = True
                break
            # High Jaccard similarity (near-duplicate)
            if _jaccard_similarity(candidate_text, existing_text) >= JACCARD_SIMILARITY_THRESHOLD:
                if len(candidate_text) > len(existing_text):
                    replace_index = idx
                else:
                    skip_add = True
                break

        if replace_index is not None:
            unique_chunks[replace_index] = chunk
        elif not skip_add:
            unique_chunks.append(chunk)

    return unique_chunks


def merge_chunks(chunks: List[DocumentChunk], min_overlap: int = 5) -> List[DocumentChunk]:
    """Merge overlapping or similar chunks"""
    if not chunks:
        return []

    merged = [chunks[0]]

    for current in chunks[1:]:
        last_merged = merged[-1]

        # Check for overlap
        overlap_len = _find_overlap(last_merged.text, current.text, min_overlap)

        if overlap_len >= min_overlap:
            # Merge chunks
            merged_text = last_merged.text + current.text[overlap_len:]
            merged_metadata = {
                **last_merged.metadata,
                "merged": True,
                "merge_count": last_merged.metadata.get("merge_count", 1) + 1
            }
            merged[-1] = DocumentChunk(
                text=merged_text,
                metadata=merged_metadata,
                semantic_score=max(last_merged.semantic_score, current.semantic_score)
            )
        else:
            merged.append(current)

    return merged


def _find_overlap(a: str, b: str, min_overlap: int = 5) -> int:
    """Find the length of overlapping text between end of a and start of b"""
    best_overlap = 0
    for i in range(min_overlap, min(len(a), len(b)) + 1):
        if a[-i:] == b[:i]:
            best_overlap = i
    return best_overlap


def _extract_position_metadata(text: str) -> dict:
    """Extract page and line markers embedded by the extractor."""
    markers = re.findall(r"\[Page (\d+) \| Line (\d+)\]", text)
    if not markers:
        line_markers = re.findall(r"\[Line (\d+)\]", text)
        if not line_markers:
            return {}
        lines = [int(num) for num in line_markers]
        return {
            "start_line": min(lines),
            "end_line": max(lines),
        }

    pages = [int(page) for page, _ in markers]
    lines = [int(line) for _, line in markers]
    return {
        "start_page": min(pages),
        "end_page": max(pages),
        "start_line": min(lines),
        "end_line": max(lines),
    }
