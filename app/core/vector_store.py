
"""
CampusGuide - ChromaDB Vector Store

Responsibilities:
1. Initialize a persistent ChromaDB database.
2. Create or load the document collection.
3. Store pre-generated embeddings with metadata.
4. Support batch upserts.
5. Retrieve semantically similar document chunks.
6. Support metadata-based filtering.
7. Delete all chunks associated with a document.
8. Provide collection statistics.

Note:
Embeddings are generated externally using EmbeddingGenerator.
ChromaDB is responsible only for storage and retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np

from app.config import settings


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class VectorStoreError(Exception):
    """Base exception for vector database errors."""


class VectorStoreInputError(VectorStoreError):
    """Raised when vector store input is invalid."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a database operation fails."""


# =========================================================
# VECTOR STORE
# =========================================================

class VectorStore:
    """
    Persistent ChromaDB storage for CampusGuide embeddings.

    Uses cosine distance for semantic similarity retrieval.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
        batch_size: int = 100,
    ):
        self.persist_directory = Path(
            persist_directory or settings.chroma_persist_dir
        ).expanduser().resolve()

        self.collection_name = (
            collection_name or settings.chroma_collection_name
        )

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        if not self.collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.batch_size = batch_size

        self.client = None
        self.collection = None

        self._initialize()

    # -----------------------------------------------------
    # DATABASE INITIALIZATION
    # -----------------------------------------------------

    def _initialize(self) -> None:
        """
        Initialize persistent ChromaDB client and collection.

        Existing indexed data is preserved across restarts.
        """

        try:
            self.persist_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "application": "CampusGuide",
                },
            )

        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to initialize ChromaDB: {exc}"
            ) from exc

    # -----------------------------------------------------
    # METADATA VALIDATION
    # -----------------------------------------------------

    @staticmethod
    def _sanitize_metadata(
        metadata: dict[str, Any]
    ) -> dict[str, str | int | float | bool]:
        """
        Convert metadata into ChromaDB-compatible scalar values.

        None values are omitted. Complex nested values are rejected.
        """

        if not isinstance(metadata, dict):
            raise VectorStoreInputError(
                "Chunk metadata must be a dictionary."
            )

        clean_metadata = {}

        for key, value in metadata.items():

            if not isinstance(key, str) or not key.strip():
                raise VectorStoreInputError(
                    "Metadata keys must be non-empty strings."
                )

            if value is None:
                continue

            if isinstance(value, np.generic):
                value = value.item()

            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, float) and not np.isfinite(value):
                    raise VectorStoreInputError(
                        f"Metadata value for '{key}' must be finite."
                    )

                clean_metadata[key] = value

            else:
                raise VectorStoreInputError(
                    f"Unsupported metadata type for '{key}': "
                    f"{type(value).__name__}"
                )

        if not clean_metadata:
            raise VectorStoreInputError(
                "At least one valid metadata field is required."
            )

        return clean_metadata

    # -----------------------------------------------------
    # EMBEDDING VALIDATION
    # -----------------------------------------------------

    @staticmethod
    def _validate_embedding(
        embedding: Sequence[float]
    ) -> list[float]:
        """Validate an embedding and convert it into a float list."""

        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32
            )

        except (TypeError, ValueError) as exc:
            raise VectorStoreInputError(
                "Embedding must contain valid numerical values."
            ) from exc

        if vector.ndim != 1 or vector.size == 0:
            raise VectorStoreInputError(
                "Embedding must be a non-empty one-dimensional vector."
            )

        if not np.isfinite(vector).all():
            raise VectorStoreInputError(
                "Embedding contains invalid numerical values."
            )

        if np.linalg.norm(vector) == 0:
            raise VectorStoreInputError(
                "Zero vectors cannot be indexed."
            )

        return vector.tolist()

    # -----------------------------------------------------
    # CHUNK VALIDATION
    # -----------------------------------------------------

    def _prepare_chunks(
        self,
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Validate all embedded chunks before writing to the database.
        """

        if not isinstance(chunks, list):
            raise VectorStoreInputError(
                "Chunks must be provided as a list."
            )

        prepared = []
        seen_ids = set()

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                raise VectorStoreInputError(
                    f"Invalid chunk at index {index}."
                )

            text = chunk.get("text")
            metadata = chunk.get("metadata")
            embedding = chunk.get("embedding")

            if not isinstance(text, str) or not text.strip():
                raise VectorStoreInputError(
                    f"Chunk at index {index} has invalid text."
                )

            if not isinstance(metadata, dict):
                raise VectorStoreInputError(
                    f"Chunk at index {index} has invalid metadata."
                )

            chunk_id = metadata.get("chunk_id")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise VectorStoreInputError(
                    f"Chunk at index {index} has no valid chunk_id."
                )

            if chunk_id in seen_ids:
                raise VectorStoreInputError(
                    f"Duplicate chunk ID: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            clean_metadata = self._sanitize_metadata(metadata)

            if not isinstance(clean_metadata.get("document_id"), str):
                raise VectorStoreInputError(
                    f"Chunk '{chunk_id}' requires a document_id."
                )

            clean_embedding = self._validate_embedding(embedding)

            prepared.append({
                "id": chunk_id,
                "text": text.strip(),
                "metadata": clean_metadata,
                "embedding": clean_embedding,
            })

        return prepared

    # -----------------------------------------------------
    # STORE / UPSERT EMBEDDINGS
    # -----------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]]
    ) -> int:
        """
        Insert or update embedded chunks in ChromaDB.

        Existing records with identical IDs are updated.

        Returns:
            Number of submitted records.
        """

        prepared = self._prepare_chunks(chunks)

        if not prepared:
            return 0

        try:
            # Process records in smaller batches.
            for start in range(0, len(prepared), self.batch_size):

                batch = prepared[
                    start:start + self.batch_size
                ]

                self.collection.upsert(
                    ids=[item["id"] for item in batch],
                    documents=[item["text"] for item in batch],
                    embeddings=[
                        item["embedding"] for item in batch
                    ],
                    metadatas=[
                        item["metadata"] for item in batch
                    ],
                )

            return len(prepared)

        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to store document chunks: {exc}"
            ) from exc

    # -----------------------------------------------------
    # SEMANTIC SEARCH
    # -----------------------------------------------------

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically similar document chunks.

        Args:
            query_embedding:
                Numerical vector representing the question.

            top_k:
                Number of results to retrieve.

            where:
                Optional ChromaDB metadata filter.

        Returns:
            Ranked records containing text, metadata,
            cosine distance and similarity score.
        """

        if top_k is None:
            top_k = settings.retrieval_top_k

        if not isinstance(top_k, int) or top_k < 1:
            raise VectorStoreInputError(
                "top_k must be a positive integer."
            )

        if where is not None and not isinstance(where, dict):
            raise VectorStoreInputError(
                "Metadata filter must be a dictionary."
            )

        vector = self._validate_embedding(query_embedding)

        try:
            total_records = self.collection.count()

            if total_records == 0:
                return []

            result = self.collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, total_records),
                where=where,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            ids = result.get("ids", [[]])[0]
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            search_results = []

            for chunk_id, text, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
            ):

                if distance is None:
                    continue

                distance = float(distance)

                search_results.append({
                    "id": chunk_id,
                    "text": text or "",
                    "metadata": metadata or {},
                    "distance": distance,
                    "similarity": 1.0 - distance,
                })

            return search_results

        except VectorStoreInputError:
            raise

        except Exception as exc:
            raise VectorStoreOperationError(
                f"Semantic search failed: {exc}"
            ) from exc

    # -----------------------------------------------------
    # DOCUMENT DELETION
    # -----------------------------------------------------

    def delete_document(self, document_id: str) -> int:
        """
        Delete every chunk associated with a document.

        Returns:
            Number of removed records.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise VectorStoreInputError(
                "A valid document ID is required."
            )

        try:
            existing = self.collection.get(
                where={"document_id": document_id},
                include=[],
            )

            ids = existing.get("ids", [])

            if ids:
                self.collection.delete(ids=ids)

            return len(ids)

        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to delete document: {exc}"
            ) from exc

    # -----------------------------------------------------
    # COLLECTION STATISTICS
    # -----------------------------------------------------

    def count(self) -> int:
        """Return the total number of stored chunks."""

        try:
            return self.collection.count()

        except Exception as exc:
            raise VectorStoreOperationError(
                f"Unable to retrieve collection count: {exc}"
            ) from exc

    def get_stats(self) -> dict[str, Any]:
        """Return information about the vector database."""

        return {
            "database": "ChromaDB",
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "total_chunks": self.count(),
            "distance_metric": "cosine",
        }
