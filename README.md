# AlphaRAG: Dynamic Document Assistant

> Document question-answering powered by Parent-Child RAG, automated LLM metadata extraction,
> Groq (LLaMA-3 70B), all-MiniLM-L6-v2 embeddings, and in-memory Qdrant.

---

## Architecture Overview

AlphaRAG utilizes a dynamic, session-scoped architecture. Documents are uploaded directly through the UI, processed in real-time, and stored in isolated in-memory vector databases, ensuring total privacy and multi-user safety.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         ALPHARAG SYSTEM ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   DYNAMIC INGESTION & QUERY PIPELINE (app.py)                           │
│                                                                         │
│   User Uploads PDF (Chainlit UI)                                        │
│       │                                                                 │
│       ▼                                                                 │
│   PyPDFLoader ──► Page Text                                             │
│       │                                                                 │
│   Pydantic + Groq LLM ◄── (Extract Title, Author, Date from Page 1)     │
│       │                                                                 │
│   Parent Splitter & Child Splitter                                      │
│       │                                                                 │
│   Session-Scoped InMemoryStore ◄──► Qdrant (:memory: with unique UUID)  │
│       (Parent docstore)                  (Hybrid child vector DB)       │
│                                                                         │
│   User Query                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   all-MiniLM-L6-v2 + FastEmbed ──► Hybrid Search ──► Top-K child chunks │
│                                                   │                     │
│                                          Parent ID lookup               │
│                                                   │                     │
│                                     Stuff into prompt {context}         │
│                                                   │                     │
│                                     ChatGroq (LLaMA-3 70B)              │
│                                                   │                     │
│                                       Streamed answer + citations       │
└─────────────────────────────────────────────────────────────────────────┘
```


---

## Project Structure

```text
alpharag/
├── app.py                  # Chainlit UI, upload handling, and streaming orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── chainlit.md             # Chainlit welcome message
└── src/
    ├── config.py           # Centralised runtime configuration
    ├── document_parser.py  # PDF loading, metadata extraction, in-memory indexing
    └── qa_chain.py         # RAG chain construction and async streaming

```

---

## Internal Architecture

| Module                   | Responsibility                                                        |
| ------------------------ | --------------------------------------------------------------------- |
| `src/config.py`          | Centralised runtime configuration; tunable parameters live here       |
| `src/document_parser.py` | PDF loading, LLM metadata extraction, chunking, and session isolation |
| `src/qa_chain.py`        | RAG chain construction and async token streaming via typed generators |
| `app.py`                 | Chainlit UI, file upload prompts, and citation sidebar orchestration  |

---

## Core Features

### Dynamic, Multi-User Session Isolation

AlphaRAG spins up a unique, isolated `InMemoryStore` and a `:memory:` Qdrant collection (tagged with a UUID) for every single upload session. Multiple users can access the application simultaneously, upload different documents, and query them without their data ever colliding or persisting to disk.

### Parent-Child Retrieval

Child chunks (small, dense) are embedded and stored in Qdrant for precise ANN search. When a match is found, the system performs a parent ID lookup to retrieve the corresponding full-context parent chunk. The LLM receives the broader parent text, preserving narrative continuity while benefiting from targeted vector search.

### Automated Metadata Extraction

Upon upload, the first page of the PDF is passed to the Groq LLM via a Pydantic-validated prompt. The model extracts the document title, author, document type, and publication date. This metadata is attached to every chunk and surfaced in citations at query time automatically.

### Interactive Source Traceability

AlphaRAG attaches expandable sidebar citations to every answer. Each citation includes:

- Document title
- Author or organisation
- Document type
- Publication date
- File name
- Page reference
- Raw source excerpt

### Real-Time Streaming Responses

Tokens stream incrementally to the UI using asynchronous LangChain generators and the Chainlit API. This enables low-latency conversational interaction even during long retrieval operations.

---

## Quick Start

### 1. Clone and install

```bash
git clone [https://github.com/HtooTayZa/AlphaRag](https://github.com/HtooTayZa/AlphaRag) alpharag
cd alpharag

# Python 3.10+ required
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

```

### 2. Configure environment

```bash
cp .env.example .env

```

Edit `.env` and add your `GROQ_API_KEY`. Get a free API key at [console.groq.com](https://console.groq.com/keys).

### 3. Launch the UI

```bash
chainlit run app.py

```

Open [http://localhost:8000](https://www.google.com/search?q=http://localhost:8000) in your browser. The assistant will prompt you to upload a PDF directly in the chat interface.

---

## Configuration Reference (`src/config.py`)

All parameters are tunable via environment variables or directly in `src/config.py`.

| Parameter              | Default                | Description                               |
| ---------------------- | ---------------------- | ----------------------------------------- |
| `LLM_MODEL_NAME`       | `llama-3.1-8b-instant` | Groq model identifier                     |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2`     | 384-dim dense embedding model             |
| `EMBEDDING_DEVICE`     | `cpu`                  | Embedding inference device (cpu / cuda)   |
| `PARENT_CHUNK_SIZE`    | `1500`                 | Character size of parent context chunks   |
| `PARENT_CHUNK_OVERLAP` | `200`                  | Overlap between adjacent parent chunks    |
| `CHILD_CHUNK_SIZE`     | `200`                  | Character size of child retrieval targets |
| `CHILD_CHUNK_OVERLAP`  | `30`                   | Overlap between adjacent child chunks     |
| `TOP_K_RETRIEVAL`      | `6`                    | Number of child chunks fetched per query  |

---

## Runtime Requirements

- Python 3.10 or higher
- Groq API key (free tier available)
- Approximately 2-4 GB RAM minimum for in-memory vector handling
- HuggingFace Hub token (only required if accessing gated models)

Key runtime dependencies:

- `torch` & `sentence-transformers` — dense embedding runtime
- `fastembed` — sparse vector generation for hybrid search
- `langchain`, `langchain-groq`, `langchain-qdrant` — RAG stack
- `qdrant-client` — in-memory vector database
- `chainlit` — streaming chat UI & file upload handling
- `pydantic` — structured metadata extraction

---

## Known Limitations

- **Volatile Storage** — because the architecture is strictly in-memory for safety and flexibility, uploaded documents are cleared when the server restarts or the session expires.
- **PDF-only ingestion** — only `.pdf` files are currently supported by the loader prompt.
- **No OCR pipeline** — scanned or image-based PDFs without embedded text will not be parsed correctly.

```

```
