
"""
CampusGuide - Embedding Generation Engine

Responsibilities:
1. Load a pretrained Sentence Transformer model.
2. Convert document chunks into numerical vectors.
3. Generate embeddings for user queries.
4. Normalize embeddings for cosine similarity.
5. Validate embedding dimensions and numerical values.
6. Support batch processing for efficient document ingestion.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class EmbeddingError(Exception):
    """Base exception for embedding generation errors."""


class EmbeddingInputError(EmbeddingError):
    """Raised when embedding input is invalid."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when the embedding model cannot be loaded."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""


# =========================================================
# EMBEDDING GENERATOR
# =========================================================

class EmbeddingGenerator:
    """
    Generate semantic embeddings using Sentence Transformers.

    The model is loaded lazily, meaning it is downloaded
    and initialized only when embeddings are first requested.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = (
            model_name or settings.embedding_model_name
        )

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self._model: SentenceTransformer | None = None

    # -----------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """
        Load the Sentence Transformer model once.

        Subsequent calls reuse the same model instance.
        """

        if self._model is not None:
            return self._model

        try:
            print(
                f"Loading embedding model: {self.model_name}"
            )

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            print("Embedding model loaded successfully.")

            return self._model

        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Could not load embedding model "
                f"'{self.model_name}'. Check internet access, "
                f"model availability and installed dependencies. "
                f"Original error: {exc}"
            ) from exc

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of generated embeddings."""

        model = self._load_model()

        dimension = model.get_embedding_dimension()

        if dimension is None:
            raise EmbeddingModelLoadError(
                "The model did not provide an embedding dimension."
            )

        return int(dimension)

    def get_model_info(self) -> dict[str, object]:
        """Return useful information about the loaded model."""

        model = self._load_model()

        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": model.max_seq_length,
            "device": str(model.device),
            "normalized_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
        }

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        """Validate and normalize embedding inputs."""

        if isinstance(texts, str):
            raise EmbeddingInputError(
                "Expected a sequence of strings, not a single string. "
                "Use embed_query() for individual queries."
            )

        if not isinstance(texts, Sequence):
            raise EmbeddingInputError(
                "Embedding input must be a sequence of strings."
            )

        if not texts:
            return []

        validated: list[str] = []

        for index, text in enumerate(texts):

            if not isinstance(text, str):
                raise EmbeddingInputError(
                    f"Input at index {index} is not a string."
                )

            cleaned = text.strip()

            if not cleaned:
                raise EmbeddingInputError(
                    f"Input at index {index} is empty."
                )

            validated.append(cleaned)

        return validated

    # -----------------------------------------------------
    # BATCH EMBEDDING GENERATION
    # -----------------------------------------------------

    def embed_texts(
        self,
        texts: Sequence[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple text inputs.

        Args:
            texts:
                Sequence of sentences, paragraphs or chunks.

            show_progress_bar:
                Display progress during batch processing.

        Returns:
            A float32 NumPy array with shape:
            (number_of_texts, embedding_dimension).
        """

        validated_texts = self._validate_texts(texts)

        if not validated_texts:
            return np.empty(
                (0, 0),
                dtype=np.float32
            )

        model = self._load_model()

        try:
            embeddings = model.encode(
                validated_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            # Validate generated vector dimensions.
            if embeddings.ndim != 2:
                raise EmbeddingGenerationError(
                    "Expected a two-dimensional embedding matrix."
                )

            if embeddings.shape[0] != len(validated_texts):
                raise EmbeddingGenerationError(
                    "Number of embeddings does not match input count."
                )

            if not np.isfinite(embeddings).all():
                raise EmbeddingGenerationError(
                    "Embeddings contain invalid numerical values."
                )

            if embeddings.shape[1] != self.embedding_dimension:
                raise EmbeddingGenerationError(
                    "Unexpected embedding dimension."
                )

            return embeddings

        except EmbeddingError:
            raise

        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings: {exc}"
            ) from exc

    # -----------------------------------------------------
    # QUERY EMBEDDING
    # -----------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate a numerical vector for a single user query.

        Returns:
            One-dimensional NumPy array.
        """

        if not isinstance(query, str) or not query.strip():
            raise EmbeddingInputError(
                "Query must be a non-empty string."
            )

        embeddings = self.embed_texts([query])

        return embeddings[0]

    # -----------------------------------------------------
    # CHUNK EMBEDDING
    # -----------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[dict],
        show_progress_bar: bool = False,
    ) -> list[dict]:
        """
        Generate embeddings for chunk records.

        Original text and metadata are preserved.

        Returns:
            Chunk records enriched with an embedding field.
        """

        if not isinstance(chunks, list):
            raise EmbeddingInputError(
                "Chunks must be provided as a list."
            )

        if not chunks:
            return []

        texts = []

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                raise EmbeddingInputError(
                    f"Invalid chunk at index {index}."
                )

            text = chunk.get("text")

            if not isinstance(text, str) or not text.strip():
                raise EmbeddingInputError(
                    f"Chunk at index {index} has invalid text."
                )

            if not isinstance(chunk.get("metadata"), dict):
                raise EmbeddingInputError(
                    f"Chunk at index {index} has invalid metadata."
                )

            texts.append(text)

        vectors = self.embed_texts(
            texts,
            show_progress_bar=show_progress_bar,
        )

        embedded_chunks = []

        for chunk, vector in zip(chunks, vectors):

            embedded_chunks.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"].copy(),
                "embedding": vector.tolist(),
            })

        return embedded_chunks
