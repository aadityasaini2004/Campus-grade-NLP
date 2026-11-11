
"""
CampusGuide - Semantic Retrieval Engine

Responsibilities:
1. Convert user questions into embeddings.
2. Search the ChromaDB vector database.
3. Rank relevant document chunks.
4. Apply optional metadata filters.
5. Support optional similarity thresholds.
6. Preserve document and page references.
"""

from __future__ import annotations

from typing import Any

from app.core.embeddings import EmbeddingGenerator
from app.core.vector_store import VectorStore


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class RetrievalError(Exception):
    """Base exception for retrieval operations."""


class RetrievalInputError(RetrievalError):
    """Raised when a retrieval query is invalid."""


# =========================================================
# SEMANTIC RETRIEVER
# =========================================================

class SemanticRetriever:
    """
    Retrieve relevant academic document chunks using
    semantic similarity and ChromaDB.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )

        self.vector_store = vector_store or VectorStore()

    # -----------------------------------------------------
    # QUERY VALIDATION
    # -----------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and normalize a user question."""

        if not isinstance(query, str):
            raise RetrievalInputError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise RetrievalInputError(
                "Query cannot be empty."
            )

        return query

    # -----------------------------------------------------
    # SEMANTIC RETRIEVAL
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        min_similarity: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant document chunks.

        Args:
            query:
                Student's natural language question.

            top_k:
                Maximum number of chunks to retrieve.

            where:
                Optional ChromaDB metadata filter.

            min_similarity:
                Optional minimum cosine similarity.
                No threshold is applied by default.

        Returns:
            Ranked list of document chunks with scores
            and original source metadata.
        """

        query = self._validate_query(query)

        if not isinstance(top_k, int) or top_k < 1:
            raise RetrievalInputError(
                "top_k must be a positive integer."
            )

        if (
            min_similarity is not None
            and (
                isinstance(min_similarity, bool)
                or not isinstance(min_similarity, (int, float))
                or not -1.0 <= min_similarity <= 1.0
            )
        ):
            raise RetrievalInputError(
                "min_similarity must be between -1 and 1."
            )

        try:
            # Step 1: Convert the question into an embedding.
            query_embedding = (
                self.embedding_generator.embed_query(query)
            )

            # Step 2: Search the vector database.
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                where=where,
            )

            # Step 3: Apply optional similarity filtering.
            if min_similarity is not None:
                results = [
                    result
                    for result in results
                    if result["similarity"] >= min_similarity
                ]

            # Step 4: Sort by semantic similarity.
            results.sort(
                key=lambda result: result["similarity"],
                reverse=True,
            )

            return results

        except RetrievalError:
            raise

        except Exception as exc:
            raise RetrievalError(
                f"Semantic retrieval failed: {exc}"
            ) from exc

    # -----------------------------------------------------
    # DOCUMENT-SPECIFIC RETRIEVAL
    # -----------------------------------------------------

    def retrieve_from_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve information from a specific document.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise RetrievalInputError(
                "A valid document_id is required."
            )

        return self.retrieve(
            query=query,
            top_k=top_k,
            where={
                "document_id": document_id
            },
        )

    # -----------------------------------------------------
    # RETRIEVAL SUMMARY
    # -----------------------------------------------------

    @staticmethod
    def get_summary(
        results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate a summary of retrieval results."""

        if not results:
            return {
                "total_results": 0,
                "sources": [],
                "average_similarity": 0.0,
            }

        sources = []

        for result in results:

            metadata = result.get("metadata", {})

            source_info = {
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page"),
                "document_id": metadata.get("document_id"),
            }

            if source_info not in sources:
                sources.append(source_info)

        average_similarity = sum(
            result["similarity"] for result in results
        ) / len(results)

        return {
            "total_results": len(results),
            "sources": sources,
            "average_similarity": round(
                average_similarity, 4
            ),
        }
