# AlphaRAG: General Document Assistant

> Document question-answering powered by Parent-Child RAG, automated LLM metadata extraction,
> Groq (LLaMA-3 70B), all-MiniLM-L6-v2 embeddings, and Qdrant.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ALPHARAG SYSTEM ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INGESTION PIPELINE (ingest.py — run once)                             │
│                                                                         │
│   PDF Files ──► PyPDFLoader ──► Page Text                               │
│                                      │                                  │
│   Pydantic + Groq LLM ◄── (Extract Title, Author, Date from Page 1)     │
│                                      │                                  │
│                              ParentDocumentRetriever                    │
│                               ┌──────┴──────┐                           │
│                               ▼             ▼                           │
│                    Parent Splitter      Child Splitter                  │
│                    (configurable)       (configurable)                  │
│                               │             │                           │
│                        LocalFileStore   all-MiniLM-L6-v2                │
│                        (docstore)           │                           │
│                               │           Qdrant                        │
│                               │         (vector DB)                     │
│                                                                         │
│   QUERY PIPELINE (app.py — per user message)                            │
│                                                                         │
│   User Query                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   all-MiniLM-L6-v2 ──► ANN Search in Qdrant ──► Top-K child chunks      │
│                                                   │                     │
│                                          Parent ID lookup               │
│                                                   │                     │
│                                        Parent documents ◄── docstore    │
│                                                   │                     │
│                                     Stuff into prompt {context}         │
│                                                   │                     │
│                                     ChatGroq (LLaMA-3 70B)              │
│                                                   │                     │
│                                       Streamed answer + sources         │
│                                                   │                     │
│                                        Chainlit UI (app.py)             │
│                                     (token streaming + citation sidebar)│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
alpharag/
├── app.py                  # Chainlit UI and streaming orchestration
├── ingest.py               # CLI ingestion pipeline
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── chainlit.md             # Chainlit welcome message
├── src/
│   ├── config.py           # Centralised runtime configuration
│   ├── document_parser.py  # PDF loading, metadata extraction, chunking, indexing
│   └── qa_chain.py         # RAG chain construction and async streaming
└── data/
    ├── raw_pdfs/           # Place source PDF documents here
    ├── qdrant_db/          # Auto-created: local Qdrant vector store
    └── local_docstore/     # Auto-created: parent chunk store

```
---

## Internal Architecture

| Module                   | Responsibility                                                         |
|--------------------------|------------------------------------------------------------------------|
| `src/config.py`          | Centralised runtime configuration; all tunable parameters live here    |
| `src/document_parser.py` | PDF loading, LLM metadata extraction, parent-child chunking, indexing  |
| `src/qa_chain.py`        | RAG chain construction and async token streaming via typed generators  |
| `ingest.py`              | CLI ingestion pipeline with Rich terminal UI and verification support  |
| `app.py`                 | Chainlit UI, session management, streaming orchestration, citation sidebar |

---

## Core Features

### Parent-Child Retrieval

Child chunks (small, dense) are embedded and stored in Qdrant for precise ANN search. When a
match is found, the system performs a parent ID lookup to retrieve the corresponding full-context
parent chunk from the local docstore. The LLM receives the broader parent text, preserving
narrative continuity while benefiting from targeted vector search.

### Automated Metadata Extraction

During ingestion, the first page of each PDF is passed to the Groq LLM via a Pydantic-validated
prompt. The model extracts the document title, author or organisation, document type, and
publication date. No manual tagging is required. This metadata is attached to every chunk and
surfaced in citations at query time.

### Interactive Source Traceability

AlphaRAG attaches expandable sidebar citations to every answer. Each citation includes:

- Document title
- Author or organisation
- Document type
- Publication date
- File name
- Page reference
- Raw source excerpt

This enables transparent, auditable retrieval rather than opaque LLM responses.

### Real-Time Streaming Responses

AlphaRAG streams tokens incrementally to the UI using asynchronous LangChain generators and the
Chainlit streaming API. The async generator in `src/qa_chain.py` yields either string tokens or
a final list of source metadata objects. `app.py` consumes this stream, appending tokens to the
live message via `stream_token` and capturing sources for sidebar rendering once generation
completes. This enables low-latency conversational interaction even during long retrieval
operations.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/HtooTayZa/AlphaRag alpharag
cd alpharag

# Python 3.10+ required
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Optionally add HUGGINGFACE_HUB_TOKEN for gated models
```

Get a free Groq API key at [console.groq.com](https://console.groq.com/keys).

### 3. Add your PDFs

```bash
mkdir -p data/raw_pdfs
# Place any PDF files here: research papers, manuals, contracts, reports, etc.
```

### 4. Run ingestion

```bash
# Basic ingestion
python ingest.py

# Run a verification retrieval query after ingestion
python ingest.py --verify

# Override the PDF source directory
python ingest.py --pdf-dir /path/to/your/documents

# Custom verification query
python ingest.py --verify --query "Summarize the key findings."
```

If no PDFs are found in the target directory, the ingestion script automatically generates a
generic demo PDF using `fpdf2` so first-time users can verify the pipeline immediately.

### 5. Launch the UI

```bash
chainlit run app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Ingestion CLI Features

The `ingest.py` script is a fully featured command-line tool, not a bare ingestion script.

- **Rich terminal UI** — startup banner, config summary table, progress spinner, and formatted
  ingestion summary are rendered using the `rich` library
- **Config inspection** — active parameters (model names, chunk sizes, paths, collection name)
  are printed as a table before any processing begins
- **CLI flags** — `--pdf-dir`, `--verify`, and `--query` allow flexible invocation without
  editing source files
- **Demo PDF auto-generation** — if the PDF directory is empty, a generic demo document is
  created via `fpdf2` to enable immediate pipeline testing
- **Verification retrieval** — `--verify` runs a full end-to-end retrieval after ingestion,
  proving that Qdrant vectors correctly resolve back to parent chunks in the local docstore,
  and prints the retrieved source cards to the terminal

---

## Configuration Reference (`src/config.py`)

All parameters are tunable via environment variables or directly in `src/config.py`. Chunk sizes
are not hardcoded; they are read from config at runtime.

| Parameter               | Default                 | Description                               |
|-------------------------|-------------------------|-------------------------------------------|
| `LLM_MODEL_NAME`        | `llama3-70b-8192`       | Groq model identifier                     |
| `EMBEDDING_MODEL_NAME`  | `all-MiniLM-L6-v2`      | 384-dim embedding model     |
| `EMBEDDING_DEVICE`      | `cpu`                   | Embedding inference device (cpu / cuda)   |
| `PARENT_CHUNK_SIZE`     | configurable            | Character size of parent context chunks   |
| `PARENT_CHUNK_OVERLAP`  | configurable            | Overlap between adjacent parent chunks    |
| `CHILD_CHUNK_SIZE`      | configurable            | Character size of child retrieval targets |
| `CHILD_CHUNK_OVERLAP`   | configurable            | Overlap between adjacent child chunks     |
| `TOP_K_RETRIEVAL`       | `6`                     | Number of child chunks fetched per query  |
| `COLLECTION_NAME`       | `alpharag_general_docs` | Qdrant collection name                    |
| `QDRANT_PATH`           | `data/qdrant_db`        | Local Qdrant persistence path             |
| `LOCAL_DOCSTORE_PATH`   | `data/local_docstore`   | Local parent chunk store path             |
| `RAW_PDFS_DIR`          | `data/raw_pdfs`         | Default PDF source directory              |

---

## Runtime Requirements

- Python 3.10 or higher
- Groq API key (free tier available)
- CUDA is optional but recommended for embedding acceleration; CPU inference is supported
- Approximately 2-4 GB RAM minimum; more is recommended for large document collections
- Disk space for Qdrant vector storage and the local docstore (scales with document volume)
- HuggingFace Hub token only required if accessing gated models

Key runtime dependencies (see `requirements.txt` for pinned versions):

- `torch` — required by `sentence-transformers` for embedding inference
- `sentence-transformers` / `transformers` — embedding runtime
- `langchain`, `langchain-groq`, `langchain-qdrant`, `langchain-huggingface` — RAG stack
- `qdrant-client` — local vector database
- `chainlit` — streaming chat UI
- `rich` — terminal UI for the ingestion CLI
- `fpdf2` — demo PDF generation
- `pydantic` — structured metadata extraction schema
- `tenacity` — retry logic for LLM API calls

---

## Known Limitations

AlphaRAG is an actively developed project. The following constraints apply to the current
implementation:

- **Local Qdrant only** — the vector store runs as a local file-based instance; no distributed
  or cloud Qdrant deployment is configured
- **PDF-only ingestion** — only `.pdf` files are supported; other document formats require
  additional loaders
- **Single-node architecture** — no horizontal scaling or load balancing
- **No authentication or multi-user isolation** — all users of a running Chainlit instance share
  the same vector store and session context
- **No OCR pipeline** — scanned or image-based PDFs without embedded text will not be parsed
  correctly
- **No hybrid reranking** — retrieval relies on dense vector search only; BM25 or cross-encoder
  reranking is not yet implemented
- **Metadata key consistency** — `ingest.py` verification references `author_or_company` while
  `app.py` references `author`; ensure the Pydantic schema in `src/document_parser.py` uses a
  single canonical key across both

---

