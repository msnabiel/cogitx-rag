"""Text cleaning and deduplication utilities"""

import re
import unicodedata
import logging
from functools import lru_cache
from typing import Dict, List
from dataclasses import dataclass, field
from rapidfuzz import fuzz

from src.config.settings import settings

logger = logging.getLogger(__name__)
FUZZY_THRESHOLD = settings.processing.fuzzy_threshold

@dataclass
class CleaningOptions:
    """Text extraction and cleaning configuration"""
    normalize_unicode: bool = True
    clean_whitespace: bool = True
    preserve_structure: bool = True
    max_length: int = 100000
    enable_ocr: bool = True
    fix_ocr_errors: bool = False
    combine_digital_and_ocr: bool = True
    confidence_threshold: float = 0.6
    language: str = 'eng'
    preserve_formatting: bool = True
    remove_metadata: bool = False
    custom_patterns: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate options"""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


class TextCleaner:
    """Improved text cleaner with caching and better performance"""

    def __init__(self, options: CleaningOptions):
        self.options = options
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.whitespace_pattern = re.compile(r'\s+')
        self.line_break_pattern = re.compile(r'\n{3,}')
        self.bullet_pattern = re.compile(r'^[\s]*[•\-\*]\s+', re.MULTILINE)

        if self.options.fix_ocr_errors:
            self.ocr_patterns = [
                (re.compile(r'\bl\b'), 'I'),
                (re.compile(r'\|'), 'l'),
                (re.compile(r'(?<=[a-z])O(?=[a-z])'), 'o'),
            ]

    @lru_cache(maxsize=1000)
    def normalize_unicode_cached(self, text: str) -> str:
        """Cached Unicode normalization"""
        return unicodedata.normalize('NFKC', text)

    def clean_text(self, text: str, is_ocr: bool = False) -> str:
        """
        Clean text with all configured options

        Args:
            text: Raw text to clean
            is_ocr: Whether this is OCR-extracted text

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        original_length = len(text)

        if self.options.normalize_unicode:
            text = self.normalize_unicode_cached(text)

        if is_ocr and self.options.fix_ocr_errors:
            for pattern, replacement in self.ocr_patterns:
                text = pattern.sub(replacement, text)

        if self.options.clean_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
            text = self.line_break_pattern.sub('\n\n', text)

        for pattern, replacement in self.options.custom_patterns.items():
            text = re.sub(pattern, replacement, text)

        if self.options.max_length and len(text) > self.options.max_length:
            logger.warning(f"Text truncated from {len(text)} to {self.options.max_length} chars")
            truncated = text[:self.options.max_length]
            if ' ' in truncated:
                text = truncated.rsplit(' ', 1)[0] + "..."
            else:
                text = truncated

        logger.debug(f"Text cleaned: {original_length} -> {len(text)} chars")
        return text.strip()


def clean_ocr_text(text: str) -> str:
    """Quick OCR text cleaning utility function"""
    cleaner = TextCleaner(CleaningOptions(fix_ocr_errors=True))
    return cleaner.clean_text(text, is_ocr=True)

def fuzzy_matching(texts: List[str], min_ratio: int = FUZZY_THRESHOLD) -> List[str]:
    """
    Deduplicate a list of texts using fuzzy matching.

    Args:
        texts: List of strings to deduplicate.
        min_ratio: Minimum similarity (0-100) to consider two texts duplicates.

    Returns:
        List of unique texts.
    """
    unique_texts: List[str] = []

    for text in texts:
        text_norm = text.lower().strip()
        skip = False
        replace_index = None

        for idx, existing in enumerate(unique_texts):
            existing_norm = existing.lower().strip()
            if fuzz.token_set_ratio(text_norm, existing_norm) >= min_ratio:
                if len(text) > len(existing):
                    replace_index = idx
                else:
                    skip = True
                break

        if replace_index is not None:
            unique_texts[replace_index] = text
        elif not skip:
            unique_texts.append(text)

    return unique_texts


__all__ = ['CleaningOptions', 'TextCleaner', 'clean_ocr_text', 'fuzzy_matching']
