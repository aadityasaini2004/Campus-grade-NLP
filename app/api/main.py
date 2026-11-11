
"""
CampusGuide - FastAPI Backend

Responsibilities:
1. Initialize the RAG pipeline during application startup.
2. Provide health and knowledge base statistics endpoints.
3. Accept PDF uploads and index academic documents.
4. Accept student questions and return grounded answers.
5. Provide document deletion functionality.
6. Validate requests and handle API errors.
"""

from __future__ import annotations

import logging
import re
import shutil
import uuid

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.rag_pipeline import (
    RAGPipeline,
    RAGQueryError,
    RAGIngestionError,
    RAGPipelineError,
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("campusguide.api")


# =========================================================
# API RESPONSE MODELS
# =========================================================

class HealthResponse(BaseModel):
    status: str
    application: str
    environment: str


class SourceCitation(BaseModel):
    citation_id: str
    source: str
    page: int | None = None
    document_id: str
    similarity: float | None = None


class RetrievalMetrics(BaseModel):
    candidate_count: int
    accepted_count: int
    minimum_similarity: float
    highest_similarity: float | None = None


class RetrievedChunk(BaseModel):
    text: str
    metadata: dict[str, Any]
    similarity: float


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Student's academic question.",
    )

    top_k: int = Field(
        default=settings.retrieval_top_k,
        ge=1,
        le=20,
    )

    prefer_llm: bool = True

    min_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )


class ChatResponse(BaseModel):
    success: bool
    question: str
    answer: str
    answer_mode: str
    sources: list[SourceCitation]
    retrieved_count: int
    retrieval: RetrievalMetrics
    retrieved_chunks: list[RetrievedChunk]
    fallback_reason: str | None = None


class DocumentIngestionResponse(BaseModel):
    success: bool
    document_id: str
    source: str
    total_pages: int
    processed_pages: int
    total_chunks: int
    stored_chunks: int
    status: str


class DocumentDeletionResponse(BaseModel):
    success: bool
    document_id: str
    deleted_chunks: int


# =========================================================
# APPLICATION LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared resources once during startup.

    The embedding model itself is loaded lazily on the
    first operation that requires it.
    """

    logger.info("Starting CampusGuide...")

    try:
        settings.create_required_directories()

        app.state.rag_pipeline = RAGPipeline()

        logger.info("CampusGuide RAG pipeline initialized.")

        yield

    finally:
        app.state.rag_pipeline = None

        logger.info("CampusGuide application stopped.")


# =========================================================
# APPLICATION INITIALIZATION
# =========================================================

app = FastAPI(
    title="CampusGuide API",
    description=(
        "A domain-specific NLP assistant for university "
        "academic information retrieval using "
        "semantic search and Retrieval-Augmented Generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =========================================================
# API ROUTER
# =========================================================

router = APIRouter()


# =========================================================
# PIPELINE DEPENDENCY
# =========================================================

def get_pipeline(request: Request) -> RAGPipeline:
    """Retrieve the shared RAG pipeline from app state."""

    pipeline = getattr(
        request.app.state,
        "rag_pipeline",
        None,
    )

    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CampusGuide is not ready.",
        )

    return pipeline


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check():
    """Check whether the application is running."""

    return HealthResponse(
        status="healthy",
        application=settings.app_name,
        environment=settings.app_env,
    )


# =========================================================
# KNOWLEDGE BASE STATISTICS
# =========================================================

@router.get(
    "/stats",
    tags=["System"],
)
def get_statistics(request: Request):
    """Return knowledge base and model information."""

    pipeline = get_pipeline(request)

    try:
        return {
            "success": True,
            "statistics": pipeline.get_knowledge_base_stats(),
        }

    except Exception as exc:

        logger.exception("Failed to retrieve statistics.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve knowledge base statistics.",
        ) from exc


# =========================================================
# PDF DOCUMENT UPLOAD
# =========================================================

@router.post(
    "/documents/upload",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def upload_document(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="Upload an academic PDF document."),
    ],
):
    """
    Upload a PDF and index its content into ChromaDB.

    Original uploaded files are retained in data/uploads.
    """

    pipeline = get_pipeline(request)

    original_filename = file.filename or ""

    # Normalize Windows and Unix path separators.
    safe_filename = Path(
        original_filename.replace("\\", "/")
    ).name

    if not safe_filename or safe_filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid filename is required.",
        )

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    upload_limit = (
        settings.max_upload_size_mb * 1024 * 1024
    )

    # Keep uploads in unique directories while preserving
    # the original filename for source citations.
    upload_directory = (
        settings.data_dir
        / "uploads"
        / uuid.uuid4().hex
    )

    saved_path = upload_directory / safe_filename

    total_bytes = 0
    successful = False

    try:
        upload_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        # Stream data instead of reading the complete upload
        # into memory at once.
        with saved_path.open("wb") as destination:

            while True:

                content = await file.read(1024 * 1024)

                if not content:
                    break

                total_bytes += len(content)

                if total_bytes > upload_limit:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            f"PDF exceeds the maximum upload size "
                            f"of {settings.max_upload_size_mb} MB."
                        ),
                    )

                destination.write(content)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF is empty.",
            )

        logger.info(
            "Processing uploaded document: %s",
            safe_filename,
        )

        # PDF extraction, embeddings and vector operations are
        # synchronous. Execute the complete process in a worker
        # thread instead of blocking the event loop.
        result = await run_in_threadpool(
            pipeline.ingest_pdf,
            saved_path,
        )

        successful = True

        logger.info(
            "Document indexed: %s | Chunks: %s",
            safe_filename,
            result["stored_chunks"],
        )

        return DocumentIngestionResponse(**result)

    except HTTPException:
        raise

    except RAGIngestionError as exc:

        logger.warning(
            "Document ingestion rejected: %s",
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected document upload error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during document upload.",
        ) from exc

    finally:

        await file.close()

        # Retain the original PDF only after successful indexing.
        if not successful:

            if upload_directory.exists():
                shutil.rmtree(
                    upload_directory,
                    ignore_errors=True,
                )


# =========================================================
# ASK ACADEMIC QUESTION
# =========================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Assistant"],
)
def ask_question(
    payload: ChatRequest,
    request: Request,
):
    """
    Submit a question to the CampusGuide RAG assistant.

    Returns:
        Generated answer.
        Answer mode.
        Supporting sources.
        Retrieval metrics.
    """

    pipeline = get_pipeline(request)

    try:

        result = pipeline.ask(
            question=payload.question,
            top_k=payload.top_k,
            prefer_llm=payload.prefer_llm,
            min_similarity=payload.min_similarity,
        )

        return ChatResponse(**result)

    except RAGQueryError as exc:

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Question answering failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process the question.",
        ) from exc


# =========================================================
# DELETE DOCUMENT
# =========================================================

@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeletionResponse,
    tags=["Documents"],
)
def delete_document(
    document_id: str,
    request: Request,
):
    """
    Delete all indexed chunks belonging to a document.

    The document_id is the SHA-256 fingerprint generated
    during PDF ingestion.
    """

    pipeline = get_pipeline(request)

    # Validate the SHA-256 identifier.
    if not re.fullmatch(r"[a-fA-F0-9]{64}", document_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid document identifier.",
        )

    try:

        result = pipeline.delete_document(
            document_id
        )

        return DocumentDeletionResponse(**result)

    except RAGPipelineError as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete the indexed document.",
        ) from exc


# =========================================================
# REGISTER ROUTES
# =========================================================

app.include_router(
    router,
    prefix=settings.api_prefix,
)
