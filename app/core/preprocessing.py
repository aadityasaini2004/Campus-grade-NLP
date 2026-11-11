
"""
CampusGuide - NLP Text Preprocessing Engine

Responsibilities:
1. Normalize Unicode characters.
2. Remove unwanted control characters.
3. Fix words split across PDF lines.
4. Normalize whitespace and paragraphs.
5. Tokenize words and sentences using NLTK.
6. Preserve PDF document metadata.
7. Generate useful text statistics.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from nltk.tokenize import PunktSentenceTokenizer, wordpunct_tokenize


# =========================================================
# TEXT PREPROCESSOR
# =========================================================

class TextPreprocessor:
    """Clean and prepare extracted PDF text for NLP."""

    VERSION = "1.0"

    def __init__(self, lowercase: bool = False):
        """
        Initialize the preprocessor.

        Args:
            lowercase:
                Whether cleaned text should be converted
                to lowercase. Disabled by default to
                preserve names, acronyms and terminology.
        """

        self.lowercase = lowercase

        # No external NLTK corpus download is required.
        self.sentence_tokenizer = PunktSentenceTokenizer()

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    @staticmethod
    def _validate_text(text: str) -> None:
        """Validate text input."""

        if not isinstance(text, str):
            raise TypeError("Expected text input as a string.")

    # -----------------------------------------------------
    # UNICODE NORMALIZATION
    # -----------------------------------------------------

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode using NFKC.

        Also removes soft hyphens and zero-width spaces.
        """

        text = unicodedata.normalize("NFKC", text)

        text = text.replace("\u00ad", "")
        text = text.replace("\u200b", "")

        return text

    # -----------------------------------------------------
    # CONTROL CHARACTER REMOVAL
    # -----------------------------------------------------

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Remove unwanted control and formatting characters."""

        return "".join(
            character
            for character in text
            if (
                character in "\n\r\t"
                or unicodedata.category(character)[0] != "C"
            )
        )

    # -----------------------------------------------------
    # BROKEN WORD CORRECTION
    # -----------------------------------------------------

    @staticmethod
    def fix_line_break_hyphenation(text: str) -> str:
        """
        Join words split across PDF lines.

        Example:
            univer-
            sity

        Becomes:
            university
        """

        return re.sub(
            r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)",
            "",
            text
        )

    # -----------------------------------------------------
    # WHITESPACE NORMALIZATION
    # -----------------------------------------------------

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace while preserving paragraph
        boundaries.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Separate paragraphs using blank lines.
        paragraphs = re.split(r"\n\s*\n+", text.strip())

        cleaned_paragraphs = []

        for paragraph in paragraphs:

            # Join wrapped PDF lines into continuous text.
            lines = [
                re.sub(r"\s+", " ", line).strip()
                for line in paragraph.splitlines()
            ]

            cleaned_paragraph = " ".join(
                line for line in lines if line
            ).strip()

            if cleaned_paragraph:
                cleaned_paragraphs.append(cleaned_paragraph)

        return "\n\n".join(cleaned_paragraphs)

    # -----------------------------------------------------
    # MAIN CLEANING PIPELINE
    # -----------------------------------------------------

    def clean_text(
        self,
        text: str,
        lowercase: bool | None = None
    ) -> str:
        """
        Apply the complete text cleaning pipeline.

        Args:
            text: Extracted PDF text.
            lowercase: Optional lowercase override.

        Returns:
            Cleaned text.
        """

        self._validate_text(text)

        if not text.strip():
            return ""

        # Step 1: Unicode normalization.
        text = self.normalize_unicode(text)

        # Step 2: Remove unwanted characters.
        text = self.remove_control_characters(text)

        # Step 3: Fix broken words.
        text = self.fix_line_break_hyphenation(text)

        # Step 4: Normalize whitespace.
        text = self.normalize_whitespace(text)

        # Step 5: Optional lowercase conversion.
        use_lowercase = (
            self.lowercase if lowercase is None else lowercase
        )

        if use_lowercase:
            text = text.lower()

        return text.strip()

    # -----------------------------------------------------
    # WORD TOKENIZATION
    # -----------------------------------------------------

    def tokenize_words(self, text: str) -> list[str]:
        """
        Split text into word and punctuation tokens.

        Example:
            'Hello, CampusGuide!'

        Returns:
            ['Hello', ',', 'CampusGuide', '!']
        """

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return []

        return wordpunct_tokenize(cleaned_text)

    # -----------------------------------------------------
    # SENTENCE TOKENIZATION
    # -----------------------------------------------------

    def tokenize_sentences(self, text: str) -> list[str]:
        """
        Split cleaned text into sentences using NLTK.
        """

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return []

        return self.sentence_tokenizer.tokenize(cleaned_text)

    # -----------------------------------------------------
    # PAGE-LEVEL PREPROCESSING
    # -----------------------------------------------------

    def preprocess_pages(
        self,
        pages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Preprocess pages returned by PDFDocumentLoader.

        Preserves document metadata and adds processing
        statistics to each page.

        Args:
            pages: Extracted PDF page records.

        Returns:
            Processed page records.
        """

        if not isinstance(pages, list):
            raise TypeError("pages must be a list.")

        processed_pages = []

        for index, page in enumerate(pages):

            if not isinstance(page, dict):
                raise TypeError(
                    f"Page at index {index} must be a dictionary."
                )

            if not isinstance(page.get("text"), str):
                raise ValueError(
                    f"Page at index {index} has invalid text."
                )

            metadata = page.get("metadata")

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Page at index {index} has invalid metadata."
                )

            original_text = page["text"]

            # Clean the extracted page text.
            cleaned_text = self.clean_text(original_text)

            # Skip empty pages safely.
            if not cleaned_text:
                continue

            # Copy original metadata instead of modifying
            # the input page.
            processed_metadata = metadata.copy()

            processed_metadata.update({
                "original_char_count": len(original_text),
                "processed_char_count": len(cleaned_text),
                "preprocessing_version": self.VERSION,
            })

            processed_pages.append({
                "text": cleaned_text,
                "metadata": processed_metadata,
            })

        return processed_pages

    # -----------------------------------------------------
    # TEXT STATISTICS
    # -----------------------------------------------------

    def get_statistics(self, text: str) -> dict[str, int]:
        """
        Generate useful statistics for cleaned text.
        """

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return {
                "characters": 0,
                "words": 0,
                "sentences": 0,
                "paragraphs": 0,
            }

        words = self.tokenize_words(cleaned_text)
        sentences = self.tokenize_sentences(cleaned_text)

        paragraphs = [
            paragraph
            for paragraph in cleaned_text.split("\n\n")
            if paragraph.strip()
        ]

        return {
            "characters": len(cleaned_text),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len(paragraphs),
        }
