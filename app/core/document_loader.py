from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pymupdf

from app.config import settings


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class DocumentLoaderError(Exception):
    """Base exception for document loading errors."""


class DocumentNotFoundError(DocumentLoaderError):
    """Raised when the requested document does not exist."""


class UnsupportedFileTypeError(DocumentLoaderError):
    """Raised when the supplied document is not a PDF."""


class FileSizeLimitError(DocumentLoaderError):
    """Raised when a PDF exceeds the allowed file size."""


class InvalidPDFError(DocumentLoaderError):
    """Raised when a PDF is invalid, empty or corrupted."""


class EncryptedPDFError(DocumentLoaderError):
    """Raised when a PDF requires a password."""


class NoExtractableTextError(DocumentLoaderError):
    """Raised when no readable text is found in a PDF."""


# =========================================================
# PDF DOCUMENT LOADER
# =========================================================

class PDFDocumentLoader:
    """Extract text and metadata from academic PDF documents."""

    def __init__(self, max_upload_size_mb: int | None = None):
        self.max_upload_size_mb = (
            max_upload_size_mb
            if max_upload_size_mb is not None
            else settings.max_upload_size_mb
        )

        if self.max_upload_size_mb <= 0:
            raise ValueError("Maximum upload size must be positive.")

    # -----------------------------------------------------
    # FILE VALIDATION
    # -----------------------------------------------------

    def _validate_file(self, file_path: Path) -> int:
        """Validate the PDF path, extension and file size."""

        if not file_path.exists():
            raise DocumentNotFoundError(
                f"Document not found: {file_path.name}"
            )

        if not file_path.is_file():
            raise InvalidPDFError(
                f"Expected a file, received: {file_path.name}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise UnsupportedFileTypeError(
                "Only PDF documents are supported."
            )

        file_size = file_path.stat().st_size

        if file_size == 0:
            raise InvalidPDFError("The supplied PDF is empty.")

        max_size_bytes = self.max_upload_size_mb * 1024 * 1024

        if file_size > max_size_bytes:
            raise FileSizeLimitError(
                f"PDF exceeds the maximum allowed size of "
                f"{self.max_upload_size_mb} MB."
            )

        return file_size

    # -----------------------------------------------------
    # DOCUMENT FINGERPRINT
    # -----------------------------------------------------

    @staticmethod
    def _calculate_document_id(file_path: Path) -> str:
        """
        Generate a SHA-256 fingerprint for a document.

        This helps identify duplicate document uploads.
        """

        sha256 = hashlib.sha256()

        with file_path.open("rb") as document:
            for block in iter(
                lambda: document.read(1024 * 1024), b""
            ):
                sha256.update(block)

        return sha256.hexdigest()

    # -----------------------------------------------------
    # PDF EXTRACTION
    # -----------------------------------------------------

    def load_pdf(
        self,
        file_path: str | Path
    ) -> list[dict[str, Any]]:
        """
        Extract readable text from each PDF page.

        Returns:
            A list of dictionaries containing:
            - text: Extracted page text.
            - metadata: Document and page information.
        """

        path = Path(file_path).expanduser()

        # Step 1: Validate document.
        file_size = self._validate_file(path)

        # Step 2: Generate unique document fingerprint.
        document_id = self._calculate_document_id(path)

        extracted_pages: list[dict[str, Any]] = []

        try:
            # Step 3: Open the PDF safely.
            with pymupdf.open(str(path)) as pdf:

                # Check password protection.
                if pdf.needs_pass:
                    raise EncryptedPDFError(
                        f"Password-protected PDF: {path.name}"
                    )

                total_pages = pdf.page_count

                if total_pages == 0:
                    raise InvalidPDFError(
                        "The PDF contains no pages."
                    )

                # Read document-level metadata.
                pdf_metadata = pdf.metadata or {}

                document_title = (
                    pdf_metadata.get("title") or path.stem
                )

                # Step 4: Extract text page by page.
                for page in pdf:

                    page_number = page.number + 1

                    text = page.get_text(
                        "text",
                        sort=True
                    )

                    # Remove null characters and whitespace.
                    text = text.replace("\x00", "").strip()

                    # Ignore pages containing no extractable text.
                    if not text:
                        continue

                    # Step 5: Preserve metadata.
                    metadata = {
                        "document_id": document_id,
                        "source": path.name,
                        "title": str(document_title),
                        "page": page_number,
                        "total_pages": total_pages,
                        "file_type": "pdf",
                        "file_size_bytes": file_size,
                    }

                    extracted_pages.append({
                        "text": text,
                        "metadata": metadata,
                    })

        except DocumentLoaderError:
            raise

        except Exception as exc:
            raise InvalidPDFError(
                f"Failed to extract PDF '{path.name}': {exc}"
            ) from exc

        # Step 6: Check whether any readable text was extracted.
        if not extracted_pages:
            raise NoExtractableTextError(
                f"No readable text found in '{path.name}'. "
                "The document may be scanned or image-based. "
                "OCR support is not implemented yet."
            )

        return extracted_pages

    # -----------------------------------------------------
    # DIRECTORY PROCESSING
    # -----------------------------------------------------

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = False
    ) -> list[dict[str, Any]]:
        """
        Load all PDF documents from a directory.

        Args:
            directory: Directory containing PDF documents.
            recursive: Whether to scan subdirectories.

        Returns:
            Combined list of extracted page records.

        Raises:
            DocumentLoaderError if a PDF cannot be processed.
        """

        directory_path = Path(directory).expanduser()

        if not directory_path.exists():
            raise DocumentNotFoundError(
                f"Directory not found: {directory_path}"
            )

        if not directory_path.is_dir():
            raise InvalidPDFError(
                "The supplied path is not a directory."
            )

        search_pattern = (
            directory_path.rglob("*")
            if recursive
            else directory_path.iterdir()
        )

        pdf_files = sorted(
            (
                path for path in search_pattern
                if path.is_file()
                and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: str(path).lower()
        )

        all_pages: list[dict[str, Any]] = []

        for pdf_file in pdf_files:
            pages = self.load_pdf(pdf_file)
            all_pages.extend(pages)

        return all_pages
