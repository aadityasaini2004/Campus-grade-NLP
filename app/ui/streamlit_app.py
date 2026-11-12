"""
CampusGuide - Streamlit Frontend

Features:
1. Interactive academic chatbot.
2. PDF document upload and indexing.
3. Source citations with page numbers.
4. Retrieval similarity information.
5. Knowledge base statistics.
6. Configurable retrieval settings.
7. Session-based conversation history.
8. FastAPI backend connectivity monitoring.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CampusGuide | Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# APPLICATION CONSTANTS
# =========================================================

DEFAULT_API_URL = os.getenv(
    "CAMPUSGUIDE_API_URL",
    "http://127.0.0.1:8000",
)

MAX_UPLOAD_SIZE_MB = 25


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827, #1f2937);
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #374151;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        color: #d1d5db;
        font-size: 1rem;
    }

    .accent-text {
        color: #4ade80;
        font-weight: 700;
    }

    .source-card {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 3px solid #16a34a;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded_document" not in st.session_state:
    st.session_state.last_uploaded_document = None


# =========================================================
# HTTP CLIENT HELPERS
# =========================================================

def get_timeout() -> httpx.Timeout:
    """Create timeouts suitable for model initialization and PDF upload."""

    return httpx.Timeout(
        connect=5.0,
        read=300.0,
        write=300.0,
        pool=5.0,
    )


@st.cache_data(ttl=10)
def check_backend(api_url: str) -> dict:
    """Check FastAPI availability."""

    response = httpx.get(
        f"{api_url}/health",
        timeout=5.0,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=10)
def fetch_statistics(api_url: str) -> dict:
    """Fetch knowledge base statistics."""

    response = httpx.get(
        f"{api_url}/api/v1/stats",
        timeout=get_timeout(),
    )

    response.raise_for_status()

    return response.json()


def api_error_message(exc: Exception) -> str:
    """Extract a useful error message from HTTPX failures."""

    if isinstance(exc, httpx.HTTPStatusError):

        try:
            body = exc.response.json()
            return str(body.get("detail", exc))

        except ValueError:
            return str(exc)

    return str(exc)


# =========================================================
# RESPONSE RENDERING
# =========================================================

def render_assistant_response(data: dict) -> None:
    """Display an answer and its source information."""

    answer = data.get("answer", "")

    mode = data.get("answer_mode", "unknown")

    if mode == "abstained":
        st.warning("Insufficient supporting evidence found.")

    elif mode == "extractive":
        st.info("Evidence-based extractive answer")

    elif mode == "llm":
        st.success("AI-generated answer with retrieved evidence")

    st.markdown(answer)

    sources = data.get("sources", [])

    if sources:

        with st.expander(
            f"📚 Source References ({len(sources)})",
            expanded=True,
        ):

            for source in sources:

                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>{source.get('citation_id', '')}</strong>
                        &nbsp; {source.get('source', 'Unknown document')}
                        <br/>
                        <small>
                        Page: {source.get('page', 'N/A')}
                        &nbsp; | &nbsp;
                        Similarity: {
                            f"{source['similarity']:.4f}"
                            if source.get("similarity") is not None
                            else "N/A"
                        }
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    retrieval = data.get("retrieval", {})

    if retrieval:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Candidates",
            retrieval.get("candidate_count", 0),
        )

        col2.metric(
            "Accepted",
            retrieval.get("accepted_count", 0),
        )

        highest = retrieval.get("highest_similarity")

        col3.metric(
            "Top Similarity",
            f"{highest:.3f}" if highest is not None else "N/A",
        )

    reason = data.get("fallback_reason")

    if reason and mode == "extractive":
        st.caption(f"Generation mode: {reason}")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎓 CampusGuide")

    st.caption("Domain-Specific NLP Assistant")

    st.divider()

    # ---------------------------------------------
    # Backend Connection
    # ---------------------------------------------

    st.subheader("Backend Connection")

    api_url = st.text_input(
        "API Base URL",
        value=DEFAULT_API_URL,
        help="URL of the running FastAPI server.",
    ).strip().rstrip("/")

    api_available = False
    statistics = {}

    if api_url:

        try:

            health = check_backend(api_url)

            api_available = health.get("status") == "healthy"

            if api_available:
                st.success("Backend connected")

        except httpx.HTTPError as exc:

            st.error("Backend unavailable")
            st.caption(str(exc))

    else:
        st.error("Enter a valid API URL.")

    # ---------------------------------------------
    # Knowledge Base Statistics
    # ---------------------------------------------

    st.divider()

    st.subheader("Knowledge Base")

    if api_available:

        try:

            statistics_response = fetch_statistics(api_url)

            statistics = statistics_response.get(
                "statistics", {}
            )

            database = statistics.get(
                "vector_database", {}
            )

            chunks = database.get("total_chunks", 0)

            st.metric(
                "Indexed Chunks",
                chunks,
            )

            st.caption(
                f"Embedding model: "
                f"{statistics.get('embedding_model', 'Unknown')}"
            )

            st.caption(
                f"Database: "
                f"{database.get('collection_name', 'Unknown')}"
            )

            if chunks > 0:
                st.warning(
                    f"The existing knowledge base contains "
                    f"{chunks} chunks. Verify their source "
                    f"documents before relying on answers."
                )

        except httpx.HTTPError as exc:

            st.error(
                f"Could not load statistics: {exc}"
            )

    else:

        st.caption("Connect the backend to see statistics.")

    if st.button("Refresh backend statistics", use_container_width=True):

        check_backend.clear()
        fetch_statistics.clear()
        st.rerun()

    # ---------------------------------------------
    # Retrieval Settings
    # ---------------------------------------------

    st.divider()

    st.subheader("Retrieval Settings")

    top_k = st.slider(
        "Number of relevant chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    min_similarity = st.slider(
        "Minimum similarity threshold",
        min_value=0.0,
        max_value=0.9,
        value=0.35,
        step=0.05,
        help=(
            "Higher values are stricter. "
            "This threshold must be calibrated using "
            "real academic evaluation data."
        ),
    )

    llm_configured = (
        statistics.get("llm_provider") == "openai"
    )

    prefer_llm = st.checkbox(
        "Use AI answer generation",
        value=llm_configured,
        disabled=not llm_configured,
        help=(
            "Requires a configured OpenAI API key. "
            "Otherwise, extractive answers are used."
        ),
    )

    if not llm_configured:
        st.caption(
            "LLM not configured. Evidence-based "
            "extractive answering is active."
        )

    # ---------------------------------------------
    # PDF Upload
    # ---------------------------------------------

    st.divider()

    st.subheader("📄 Add Academic Documents")

    uploaded_pdf = st.file_uploader(
        "Upload university PDF",
        type=["pdf"],
        max_upload_size=MAX_UPLOAD_SIZE_MB,
        help="Maximum upload size: 25 MB.",
        key="academic_pdf_uploader",
    )

    if st.button(
        "Upload and Index PDF",
        use_container_width=True,
        disabled=not api_available or uploaded_pdf is None,
    ):

        if uploaded_pdf is None:

            st.warning("Please select a PDF first.")

        elif uploaded_pdf.size == 0:

            st.error("The selected PDF is empty.")

        elif uploaded_pdf.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:

            st.error("The PDF exceeds the 25 MB limit.")

        else:

            try:

                with st.spinner(
                    "Extracting text, creating embeddings and indexing..."
                ):

                    response = httpx.post(
                        f"{api_url}/api/v1/documents/upload",
                        files={
                            "file": (
                                uploaded_pdf.name,
                                uploaded_pdf.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=get_timeout(),
                    )

                    response.raise_for_status()

                    result = response.json()

                st.session_state.last_uploaded_document = result

                check_backend.clear()
                fetch_statistics.clear()

                st.success("Document indexed successfully!")

                st.json(result)

                st.rerun()

            except httpx.HTTPError as exc:

                st.error(
                    api_error_message(exc)
                )

    # ---------------------------------------------
    # Upload Confirmation
    # ---------------------------------------------

    last_document = st.session_state.last_uploaded_document

    if last_document:

        st.caption("Last indexed document")

        st.code(
            last_document.get("document_id", ""),
            language="text",
        )

        st.caption(
            last_document.get("source", "Unknown")
        )

    # ---------------------------------------------
    # Conversation Management
    # ---------------------------------------------

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.rerun()

    st.caption("CampusGuide v1.0 · Local Development")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🎓 CampusGuide</div>
        <div class="hero-subtitle">
            Your Academic Knowledge Assistant
        </div>
        <br/>
        <div style="color:#d1d5db;">
            Ask questions about academic regulations,
            examinations, syllabi and university documents.
        </div>
        <br/>
        <div class="accent-text">
            Powered by NLP · Semantic Search · RAG
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Academic Assistant")

st.caption(
    "Ask a question in natural language. "
    "CampusGuide will search its indexed university documents "
    "and show the supporting sources."
)


# =========================================================
# QUICK QUESTION SUGGESTIONS
# =========================================================

st.markdown("**Try asking:**")

suggestions = [
    "What are the attendance requirements?",
    "What are the semester examination rules?",
    "What documents are required for admission?",
]

suggestion_columns = st.columns(3)

selected_suggestion = None

for index, suggestion in enumerate(suggestions):

    with suggestion_columns[index]:

        if st.button(
            suggestion,
            use_container_width=True,
            disabled=not api_available,
            key=f"suggestion_{index}",
        ):

            selected_suggestion = suggestion


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":

            st.markdown(message["content"])

        else:

            if message.get("error"):

                st.error(message["error"])

            elif message.get("data"):

                render_assistant_response(
                    message["data"]
                )


# =========================================================
# QUESTION INPUT
# =========================================================

prompt = st.chat_input(
    "Ask CampusGuide about your academic documents...",
    max_chars=2000,
    disabled=not api_available,
)


# Support the quick question suggestion buttons.
if selected_suggestion:
    prompt = selected_suggestion


if prompt:

    # Record the user question.
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    try:

        with st.spinner(
            "Searching academic documents and preparing your answer..."
        ):

            response = httpx.post(
                f"{api_url}/api/v1/chat",
                json={
                    "question": prompt,
                    "top_k": top_k,
                    "prefer_llm": prefer_llm,
                    "min_similarity": min_similarity,
                },
                timeout=get_timeout(),
            )

            response.raise_for_status()

            result = response.json()

        st.session_state.messages.append({
            "role": "assistant",
            "data": result,
        })

    except httpx.HTTPError as exc:

        st.session_state.messages.append({
            "role": "assistant",
            "error": api_error_message(exc),
        })

    except Exception as exc:

        st.session_state.messages.append({
            "role": "assistant",
            "error": f"Unexpected error: {exc}",
        })

    # Render the latest user and assistant messages.
    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CampusGuide provides information based on its indexed "
    "academic documents. Always verify important academic "
    "decisions using the original university regulations "
    "and official notices."
)