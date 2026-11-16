
"""
CampusGuide - Retrieval-Augmented Generation Pipeline

Central orchestration layer connecting:

PDF Loader
    -> Text Preprocessor
    -> Text Chunker
    -> Embedding Generator
    -> ChromaDB Vector Store
    -> Semantic Retriever
    -> Answer Generator

Responsibilities:
1. Ingest academic PDF documents.
2. Process and index document content.
3. Accept natural language questions.
4. Retrieve relevant academic evidence.
5. Apply a configurable relevance threshold.
6. Generate grounded answers with citations.
7. Abstain when sufficient evidence is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import settings
from app.core.document_loader import PDFDocumentLoader
from app.core.preprocessing import TextPreprocessor
from app.core.chunking import TextChunker
from app.core.embeddings import EmbeddingGenerator
from app.core.vector_store import VectorStore
from app.core.retriever import SemanticRetriever
from app.core.generator import AnswerGenerator


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class RAGPipelineError(Exception):
    """Base exception for RAG pipeline operations."""


class RAGIngestionError(RAGPipelineError):
    """Raised when document ingestion fails."""


class RAGQueryError(RAGPipelineError):
    """Raised when question processing fails."""


# =========================================================
# RAG PIPELINE
# =========================================================

class RAGPipeline:
    """
    Complete CampusGuide document ingestion and
    question-answering pipeline.
    """

    def __init__(
        self,
        document_loader: PDFDocumentLoader | None = None,
        preprocessor: TextPreprocessor | None = None,
        chunker: TextChunker | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: VectorStore | None = None,
        retriever: SemanticRetriever | None = None,
        answer_generator: AnswerGenerator | None = None,
        min_similarity: float | None = None,
    ):
        # -------------------------------------------------
        # Initialize core components
        # -------------------------------------------------

        self.document_loader = (
            document_loader or PDFDocumentLoader()
        )

        self.preprocessor = (
            preprocessor or TextPreprocessor()
        )

        self.chunker = (
            chunker or TextChunker()
        )

        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )

        self.vector_store = (
            vector_store or VectorStore()
        )

        # Reuse the same embedding model and vector database.
        self.retriever = (
            retriever
            if retriever is not None
            else SemanticRetriever(
                embedding_generator=self.embedding_generator,
                vector_store=self.vector_store,
            )
        )

        self.answer_generator = (
            answer_generator or AnswerGenerator()
        )

        # -------------------------------------------------
        # Relevance threshold configuration
        # -------------------------------------------------

        self.min_similarity = (
            settings.min_retrieval_similarity
            if min_similarity is None
            else min_similarity
        )

        if (
            isinstance(self.min_similarity, bool)
            or not isinstance(self.min_similarity, (int, float))
            or not -1.0 <= self.min_similarity <= 1.0
        ):
            raise ValueError(
                "min_similarity must be between -1 and 1."
            )

    # =====================================================
    # DOCUMENT INGESTION
    # =====================================================

    def ingest_pdf(
        self,
        file_path: str | Path,
        show_progress_bar: bool = False,
    ) -> dict[str, Any]:
        """
        Load, process, embed and index one PDF document.

        The same PDF content generates deterministic document
        and chunk identifiers, so repeated uploads are
        handled through ChromaDB upserts.

        Returns:
            Document ingestion summary.
        """

        path = Path(file_path).expanduser()

        try:
            # Step 1: Extract PDF text.
            pages = self.document_loader.load_pdf(path)

            if not pages:
                raise RAGIngestionError(
                    "The document contains no extractable pages."
                )

            # Step 2: Preprocess extracted text.
            processed_pages = (
                self.preprocessor.preprocess_pages(pages)
            )

            if not processed_pages:
                raise RAGIngestionError(
                    "No usable text remained after preprocessing."
                )

            # Step 3: Split pages into chunks.
            chunks = self.chunker.chunk_pages(
                processed_pages
            )

            if not chunks:
                raise RAGIngestionError(
                    "Document chunking produced no usable chunks."
                )

            # Step 4: Generate embeddings.
            embedded_chunks = self.embedding_generator.embed_chunks(
                chunks,
                show_progress_bar=show_progress_bar,
            )

            if len(embedded_chunks) != len(chunks):
                raise RAGIngestionError(
                    "Embedding count does not match chunk count."
                )

            # Step 5: Store the vectors in ChromaDB.
            stored_count = self.vector_store.upsert_chunks(
                embedded_chunks
            )

            if stored_count != len(embedded_chunks):
                raise RAGIngestionError(
                    "Not all document chunks were submitted to storage."
                )

            metadata = pages[0]["metadata"]

            return {
                "success": True,
                "document_id": metadata["document_id"],
                "source": metadata["source"],
                "total_pages": metadata["total_pages"],
                "processed_pages": len(processed_pages),
                "total_chunks": len(chunks),
                "stored_chunks": stored_count,
                "status": "indexed",
            }

        except RAGPipelineError:
            raise

        except Exception as exc:
            raise RAGIngestionError(
                f"Failed to ingest '{path.name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # =====================================================
    # DIRECTORY INGESTION
    # =====================================================

    def ingest_directory(
        self,
        directory: str | Path | None = None,
        recursive: bool = False,
        show_progress_bar: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest every PDF document in a directory.

        Individual document failures are recorded without
        preventing other documents from being processed.
        """

        directory_path = Path(
            directory or settings.sample_docs_dir
        ).expanduser()

        if not directory_path.exists():
            raise RAGIngestionError(
                f"Directory not found: {directory_path}"
            )

        if not directory_path.is_dir():
            raise RAGIngestionError(
                f"Not a directory: {directory_path}"
            )

        candidates = (
            directory_path.rglob("*")
            if recursive
            else directory_path.iterdir()
        )

        pdf_files = sorted(
            [
                path
                for path in candidates
                if path.is_file()
                and path.suffix.lower() == ".pdf"
            ],
            key=lambda path: str(path).lower(),
        )

        successful = []
        failed = []

        for pdf_file in pdf_files:

            try:
                result = self.ingest_pdf(
                    pdf_file,
                    show_progress_bar=show_progress_bar,
                )

                successful.append(result)

            except Exception as exc:

                failed.append({
                    "source": pdf_file.name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                })

        return {
            "directory": str(directory_path),
            "total_documents": len(pdf_files),
            "successful_documents": len(successful),
            "failed_documents": len(failed),
            "documents": successful,
            "errors": failed,
            "status": (
                "completed"
                if not failed
                else "completed_with_errors"
            ),
        }

    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        prefer_llm: bool = True,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        """
        Answer a student question using retrieved evidence.

        Pipeline:
            Question
                -> Embedding
                -> Semantic Retrieval
                -> Relevance Filtering
                -> Answer Generation
                -> Source Citations

        Returns:
            Answer, citations, retrieval statistics and
            relevant document chunks.
        """

        if not isinstance(question, str) or not question.strip():
            raise RAGQueryError(
                "A valid non-empty question is required."
            )

        threshold = (
            self.min_similarity
            if min_similarity is None
            else min_similarity
        )

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not -1.0 <= threshold <= 1.0
        ):
            raise RAGQueryError(
                "min_similarity must be between -1 and 1."
            )

        try:
            # Step 1: Retrieve candidate chunks.
            candidates = self.retriever.retrieve(
                query=question.strip(),
                top_k=(
                    top_k
                    if top_k is not None
                    else settings.retrieval_top_k
                ),
                where=where,
            )

            # Step 2: Remove candidates below the threshold.
            relevant_chunks = [
                result
                for result in candidates
                if result["similarity"] >= threshold
            ]

            relevant_chunks.sort(
                key=lambda result: result["similarity"],
                reverse=True,
            )

            # Step 3: Generate grounded answer.
            generated = self.answer_generator.generate_answer(
                question=question,
                retrieved_chunks=relevant_chunks,
                prefer_llm=prefer_llm,
            )

            # Step 4: Prepare response.
            return {
                "success": True,
                "question": question.strip(),
                "answer": generated["answer"],
                "answer_mode": generated["answer_mode"],
                "sources": generated["sources"],
                "retrieved_count": generated["retrieved_count"],
                "retrieval": {
                    "candidate_count": len(candidates),
                    "accepted_count": len(relevant_chunks),
                    "minimum_similarity": threshold,
                    "highest_similarity": (
                        candidates[0]["similarity"]
                        if candidates
                        else None
                    ),
                },
                "retrieved_chunks": [
                    {
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "similarity": item["similarity"],
                    }
                    for item in relevant_chunks
                ],
                "fallback_reason": generated["fallback_reason"],
            }

        except RAGPipelineError:
            raise

        except Exception as exc:
            raise RAGQueryError(
                f"Question processing failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # =====================================================
    # DOCUMENT MANAGEMENT
    # =====================================================

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Remove an indexed document from ChromaDB."""

        try:
            deleted_count = self.vector_store.delete_document(
                document_id
            )

            return {
                "success": True,
                "document_id": document_id,
                "deleted_chunks": deleted_count,
            }

        except Exception as exc:
            raise RAGPipelineError(
                f"Document deletion failed: {exc}"
            ) from exc

    # =====================================================
    # KNOWLEDGE BASE STATISTICS
    # =====================================================

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """Return current knowledge base information."""

        return {
            "application": settings.app_name,
            "embedding_model": (
                self.embedding_generator.model_name
            ),
            "embedding_dimension": (
                self.embedding_generator.embedding_dimension
            ),
            "vector_database": self.vector_store.get_stats(),
            "retrieval_top_k": settings.retrieval_top_k,
            "minimum_similarity": self.min_similarity,
            "llm_provider": self.answer_generator.provider,
        }
