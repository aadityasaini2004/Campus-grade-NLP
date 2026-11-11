
"""
CampusGuide - Intelligent Text Chunking Engine

Responsibilities:
1. Split text using paragraph and sentence boundaries.
2. Handle oversized sentences and paragraphs.
3. Generate configurable text chunks.
4. Preserve context using overlapping chunks.
5. Generate deterministic chunk identifiers.
6. Preserve original document and page metadata.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.config import settings


# =========================================================
# CUSTOM EXCEPTION
# =========================================================

class ChunkingError(Exception):
    """Raised when text chunking fails."""


# =========================================================
# TEXT CHUNKER
# =========================================================

class TextChunker:
    """
    Split processed document pages into smaller chunks.

    Chunk sizes and overlaps are measured in characters.
    Sentence and paragraph boundaries are preferred.
    """

    VERSION = "1.0"

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else settings.chunk_size
        )

        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunk_overlap
        )

        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

    # -----------------------------------------------------
    # SENTENCE SPLITTING
    # -----------------------------------------------------

    @staticmethod
    def _split_sentences(paragraph: str) -> list[str]:
        """
        Split a paragraph into sentences.

        Basic punctuation-based segmentation is used to avoid
        requiring external NLP model downloads.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph.strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # -----------------------------------------------------
    # OVERSIZED TEXT SPLITTING
    # -----------------------------------------------------

    def _split_long_text(self, text: str) -> list[str]:
        """
        Split a sentence or segment that exceeds chunk_size.

        Splitting prefers whitespace boundaries to avoid
        cutting words in half.
        """

        if len(text) <= self.chunk_size:
            return [text]

        pieces: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + self.chunk_size,
                text_length
            )

            if end < text_length:

                boundary = text.rfind(" ", start, end)

                # Prefer a word boundary if it is reasonably close.
                if boundary > start:
                    end = boundary

            piece = text[start:end].strip()

            if piece:
                pieces.append(piece)

            # Move forward without overlapping here.
            # Overlap is handled centrally during chunk assembly.
            start = end

            while start < text_length and text[start].isspace():
                start += 1

        return pieces

    # -----------------------------------------------------
    # SEGMENT EXTRACTION
    # -----------------------------------------------------

    def _extract_segments(self, text: str) -> list[tuple[str, str]]:
        """
        Convert text into sentence-level segments.

        Returns:
            Tuples of (segment_text, separator_before).

        Paragraph boundaries are retained where practical.
        """

        segments: list[tuple[str, str]] = []

        paragraphs = re.split(
            r"\n\s*\n+",
            text.strip()
        )

        for paragraph_index, paragraph in enumerate(paragraphs):

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            sentences = self._split_sentences(paragraph)

            for sentence_index, sentence in enumerate(sentences):

                # Preserve paragraph boundaries.
                if paragraph_index > 0 and sentence_index == 0:
                    separator = "\n\n"
                else:
                    separator = " "

                # Split unusually long sentences.
                pieces = self._split_long_text(sentence)

                for piece_index, piece in enumerate(pieces):

                    piece_separator = (
                        separator
                        if piece_index == 0
                        else " "
                    )

                    segments.append((
                        piece,
                        piece_separator
                    ))

        return segments

    # -----------------------------------------------------
    # OVERLAP GENERATION
    # -----------------------------------------------------

    def _get_overlap(self, text: str) -> str:
        """
        Extract a trailing context window from a previous chunk.

        Attempts to preserve complete words at the beginning
        of the overlap.
        """

        if self.chunk_overlap == 0 or not text:
            return ""

        overlap = text[-self.chunk_overlap:]

        # Avoid starting in the middle of a word.
        first_space = overlap.find(" ")

        if first_space != -1:
            overlap = overlap[first_space + 1:]

        return overlap.strip()

    def _fit_overlap(
        self,
        overlap: str,
        segment: str,
        separator: str,
    ) -> str:
        """
        Reduce overlap when necessary to keep the next chunk
        within its configured character limit.
        """

        available = (
            self.chunk_size
            - len(segment)
            - (len(separator) if overlap else 0)
        )

        if available <= 0:
            return ""

        if len(overlap) <= available:
            return overlap

        overlap = overlap[-available:]

        # Align the reduced overlap to a word boundary.
        first_space = overlap.find(" ")

        if first_space != -1:
            overlap = overlap[first_space + 1:]

        return overlap.strip()

    # -----------------------------------------------------
    # CHUNK ASSEMBLY
    # -----------------------------------------------------

    def chunk_text(self, text: str) -> list[str]:
        """
        Convert a text string into overlapping chunks.

        Args:
            text: Processed document text.

        Returns:
            List of text chunks.
        """

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        if not text.strip():
            return []

        segments = self._extract_segments(text)

        if not segments:
            return []

        chunks: list[str] = []
        current_chunk = ""

        for segment, separator in segments:

            if not current_chunk:
                current_chunk = segment
                continue

            candidate = (
                current_chunk
                + separator
                + segment
            )

            # Add segment if it fits.
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
                continue

            # Save the current chunk.
            chunks.append(current_chunk.strip())

            # Carry context into the next chunk.
            overlap = self._get_overlap(current_chunk)

            # Preserve paragraph separation when no overlap exists.
            next_separator = separator if not overlap else " "

            # Ensure overlap + next segment stays within the limit.
            overlap = self._fit_overlap(
                overlap,
                segment,
                next_separator
            )

            if overlap:
                current_chunk = (
                    overlap
                    + next_separator
                    + segment
                )
            else:
                current_chunk = segment

        # Append the final chunk.
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    # -----------------------------------------------------
    # UNIQUE CHUNK IDENTIFIER
    # -----------------------------------------------------

    @staticmethod
    def _generate_chunk_id(
        text: str,
        document_id: str,
        page_number: int,
        chunk_index: int,
    ) -> str:
        """
        Generate a deterministic identifier for a chunk.

        The same document, page, position and content
        generate the same identifier.
        """

        identity = (
            f"{document_id}:"
            f"{page_number}:"
            f"{chunk_index}:"
            f"{text}"
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    # -----------------------------------------------------
    # PAGE-LEVEL CHUNKING
    # -----------------------------------------------------

    def chunk_pages(
        self,
        pages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert preprocessed PDF pages into chunk records.

        Original metadata is retained and additional chunk
        metadata is attached.

        Args:
            pages: Records returned by TextPreprocessor.

        Returns:
            List of chunk records containing text and metadata.
        """

        if not isinstance(pages, list):
            raise TypeError("pages must be a list.")

        all_chunks: list[dict[str, Any]] = []

        for page_index, page in enumerate(pages):

            if not isinstance(page, dict):
                raise ChunkingError(
                    f"Invalid page record at index {page_index}."
                )

            text = page.get("text")
            metadata = page.get("metadata")

            if not isinstance(text, str):
                raise ChunkingError(
                    f"Invalid text at page index {page_index}."
                )

            if not isinstance(metadata, dict):
                raise ChunkingError(
                    f"Invalid metadata at page index {page_index}."
                )

            if not text.strip():
                continue

            page_number = metadata.get("page", page_index + 1)

            try:
                page_number = int(page_number)
            except (TypeError, ValueError) as exc:
                raise ChunkingError(
                    "Page number must be a valid integer."
                ) from exc

            if page_number < 1:
                raise ChunkingError(
                    "Page number must be greater than zero."
                )

            document_id = str(
                metadata.get("document_id")
                or metadata.get("source")
                or "unknown_document"
            )

            text_chunks = self.chunk_text(text)

            total_chunks = len(text_chunks)

            for chunk_index, chunk in enumerate(
                text_chunks,
                start=1
            ):

                chunk_id = self._generate_chunk_id(
                    text=chunk,
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                )

                chunk_metadata = metadata.copy()

                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "total_chunks_in_page": total_chunks,
                    "chunk_char_count": len(chunk),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunking_version": self.VERSION,
                })

                all_chunks.append({
                    "text": chunk,
                    "metadata": chunk_metadata,
                })

        return all_chunks
