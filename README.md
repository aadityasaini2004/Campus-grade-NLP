# 🎓 CampusGuide – Domain-Specific NLP Assistant

> **Your University's Knowledge, One Conversation Away.**

CampusGuide is a domain-specific Natural Language Processing (NLP) assistant designed to help university students find accurate academic information through natural language conversations.

The system aims to understand student queries and retrieve relevant information from university documents such as academic regulations, examination notices, course syllabi, university guidelines, and FAQs.

Instead of manually searching through multiple PDF documents, students will be able to ask questions in natural language and receive relevant, context-aware answers with supporting document references.


---

## 📌 1. Problem Statement

University students frequently struggle to find accurate academic information because it is distributed across multiple documents, PDF files, notices, and university portals.

Finding a specific piece of information often requires students to manually open several documents, search through multiple pages, and identify the relevant information themselves.

Traditional keyword-based searching may also fail when students phrase their questions differently from the wording used in official documents.

CampusGuide aims to address this problem by building an intelligent academic assistant that understands natural language queries and retrieves relevant information from a controlled collection of university documents.

## 🎯 2. Project Objectives

- Develop a domain-specific NLP assistant for academic information retrieval.
- Extract and process textual information from PDF documents.
- Apply NLP preprocessing techniques to prepare unstructured text.
- Generate semantic embeddings for document representation.
- Retrieve relevant information using semantic similarity.
- Implement a Retrieval-Augmented Generation (RAG) pipeline.
- Generate context-aware answers grounded in retrieved documents.
- Provide source references to improve answer transparency.
- Reduce unsupported answers through retrieval-based grounding and unknown-answer handling.

## ✨ 3. Planned Features

| Feature                       | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| PDF Document Processing       | Extract academic information from PDF files.                 |
| Text Preprocessing            | Clean and normalize extracted text.                          |
| Intelligent Chunking          | Divide documents into meaningful text sections.              |
| Semantic Search               | Retrieve information based on contextual meaning.            |
| AI-Powered Question Answering | Generate answers using retrieved academic information.       |
| Source Citations              | Display document names and page references.                  |
| Hallucination Control         | Handle questions when supporting information is unavailable. |
| Interactive Chat Interface    | Allow users to communicate naturally with the assistant.     |
| Document Management           | Support adding and updating knowledge sources.               |

*These features are planned and will be implemented incrementally.*

## 🧠 4. NLP Concepts Used

The project will explore and implement the following NLP concepts:

- Text Preprocessing
- Tokenization
- Text Normalization
- Stopword Handling
- Text Chunking
- Word and Sentence Embeddings
- Semantic Similarity
- Information Retrieval
- Vector Search
- Retrieval-Augmented Generation (RAG)
- Natural Language Question Answering

## ⚙️ 5. System Architecture

The proposed system follows a Retrieval-Augmented Generation architecture.

### Document Processing Pipeline

1. **Document Ingestion:** Load academic PDF documents.
2. **Text Extraction:** Extract textual content while preserving page information.
3. **Text Preprocessing:** Clean and normalize extracted text.
4. **Text Chunking:** Divide documents into smaller, meaningful sections.
5. **Embedding Generation:** Convert text chunks into numerical vector representations.
6. **Vector Storage:** Store embeddings along with document metadata.
7. **Semantic Retrieval:** Retrieve the most relevant chunks based on the user's question.

### Question Answering Pipeline

1. The student submits a natural language question.
2. The question is converted into an embedding.
3. The retrieval system searches for relevant document chunks.
4. The retrieved context is passed to a Large Language Model (LLM).
5. The model generates an answer grounded in the available context.
6. The system displays the answer along with supporting document references.

### Architecture Overview

```text
           UNIVERSITY DOCUMENTS
          (PDFs, Notices, Syllabus)
                    |
                    v
            DOCUMENT INGESTION
                    |
                    v
              TEXT EXTRACTION
                    |
                    v
            TEXT PREPROCESSING
                    |
                    v
               TEXT CHUNKING
                    |
                    v
            EMBEDDING GENERATION
                    |
                    v
              VECTOR DATABASE
                    |
                    |
          STUDENT ASKS QUESTION
                    |
                    v
            QUERY EMBEDDING
                    |
                    v
             SEMANTIC RETRIEVAL
                    |
                    v
           RELEVANT DOCUMENT CHUNKS
                    |
                    v
             LLM + RAG PIPELINE
                    |
                    v
             GENERATED ANSWER
                    |
                    v
          ANSWER WITH SOURCE REFERENCES
```

## 🛠️ 6. Technology Stack

The following technologies are planned for implementation:

| Technology            | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| Python                | Core programming language                         |
| NLTK / spaCy          | Natural Language Processing                       |
| Sentence Transformers | Text embeddings                                   |
| ChromaDB              | Vector database and similarity search             |
| PyMuPDF               | PDF text extraction                               |
| FastAPI               | Backend API development                           |
| Streamlit             | Interactive user interface                        |
| SQLite                | Lightweight feedback and application data storage |
| Pytest                | Software testing                                  |
| Git & GitHub          | Version control and collaboration                 |

The final model and deployment configuration will be determined during implementation.

## 📁 7. Project Structure

```text
campusguide-nlp-assistant/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── core/
│   │   ├── chunking.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── generator.py
│   │   ├── preprocessing.py
│   │   ├── rag_pipeline.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── ui/
│   │   └── streamlit_app.py
│   │
│   └── config.py
│
├── data/
│   ├── evaluation/
│   │   └── questions.jsonl
│   │
│   └── sample_docs/
│
├── docs/
│   ├── implementation.md
│   ├── methodology.md
│   ├── problem_statement.md
│   ├── requirements.md
│   ├── system_architecture.md
│   ├── testing.md
│   └── user_manual.md
│
├── scripts/
├── tests/
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🚀 8. Installation and Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or later
- Git
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/aadityasaini2004/Campus-grade-NLP.git
```

### Step 2: Navigate to the Project

```bash
cd Campus-grade-NLP
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: Application execution instructions will be added once the core implementation and dependencies are finalized.

## 🧪 9. Testing and Evaluation

The project will be evaluated using a collection of test questions and academic documents.

Planned evaluation areas include:

- Document extraction correctness
- Text preprocessing correctness
- Retrieval relevance
- Recall\@K
- Mean Reciprocal Rank (MRR)
- Answer correctness
- Source citation correctness
- Unsupported question handling
- Response latency

The evaluation process will help identify weaknesses in retrieval quality and answer generation.

## 🗺️ 10. Development Roadmap

- [X] Project planning and requirements analysis
- [X] NLP fundamentals and methodology
- [X] PDF document ingestion
- [X] Text extraction and preprocessing
- [X] Intelligent text chunking
- [X] Embedding generation
- [X] Vector database integration
- [X] Semantic search implementation
- [x] RAG pipeline development
- [X] LLM integration
- [X] FastAPI backend development
- [X] Streamlit frontend development
- [X] Source citation implementation
- [X] Testing and evaluation
- [X] Performance optimization
- [X] Final documentation and deployment

## 🔮 11. Future Enhancements

Potential future improvements include:

- Multilingual query support (English and Hindi).
- Advanced intent classification.
- Hybrid search combining keyword and semantic retrieval.
- Document version tracking.
- Conversation history and context awareness.
- User feedback and answer improvement mechanisms.
- Advanced document analytics.

## 👥 12. Contributors

**Project:** Domain-Specific NLP Assistant (Project 20)

Contributors and individual responsibilities will be documented as the project progresses.
- **Aaditya Saini** – Backend
- **Anirudh** – Frontend

## 📄 13. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
<p align="center">
  <b>CampusGuide – Making Academic Information Accessible Through NLP.</b>
  <br/>
  Built with Python, Natural Language Processing and AI.
</p>
