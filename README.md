# AlphaRAG: General Document Assistant

> Enterprise-grade document analysis powered by Parent-Child RAG, Automated LLM Metadata Extraction, Groq (LLaMA-3 70B), BAAI/bge-m3 embeddings, and Qdrant.

---

## Architecture Overview

```text
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
│                    (~1500 chars)        (~200 chars)                    │
│                               │             │                           │
│                        LocalFileStore   BGE-M3 Embed                    │
│                        (docstore)           │                           │
│                               │           Qdrant                        │
│                               │         (vector DB)                     │
│                                                                         │
│   QUERY PIPELINE (app.py — per user message)                            │
│                                                                         │
│   User Query                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   BGE-M3 Embed ──► ANN Search in Qdrant ──► Top-K CHILD chunks          │
│                                                   │                     │
│                                          Parent ID lookup               │
│                                                   │                     │
│                                        PARENT Documents ◄── docstore    │
│                                                   │                     │
│                                     Stuff into Prompt {context}         │
│                                                   │                     │
│                                     ChatGroq (LLaMA-3 70B)              │
│                                                   │                     │
│                                             Answer + Sources            │
│                                                   │                     │
│                                        Chainlit UI (app.py)             │
│                                     (Streaming + Citation Sidebar)      │
└─────────────────────────────────────────────────────────────────────────┘

```

## Core Features

| Feature | Description |
| --- | --- |
| **Parent-Child Chunking** | Indexes small **child chunks** (~200 chars) for precise vector search, but feeds broad **parent chunks** (~1500 chars) to the LLM to preserve narrative context. |
| **Automated Metadata** | Uses Pydantic and Groq to read the first page of any ingested PDF and automatically extract the Title, Author, Date, and Document Type. No manual tagging required! |
| **Zero-Hallucination Citations** | The LLM is strictly prompted to cite its sources, and the UI hooks into these citations to generate interactive, clickable sidebar footnotes. |
| **True UI Streaming** | Utilizes LangChain's `astream_events` to deliver real-time token streaming to the Chainlit UI. |

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url> alpharag
cd alpharag

# Python 3.10+ required
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (and optionally HUGGINGFACE_HUB_TOKEN)

```

Get your free Groq API key at [console.groq.com](https://console.groq.com/keys).

### 3. Add your PDFs

```bash
mkdir -p data/raw_pdfs
# Drop ANY pdf files here (Research papers, manuals, legal contracts, etc.)

```

### 4. Run ingestion

```bash
# Basic ingestion (will auto-extract metadata and build the vector DB)
python ingest.py

# With verification query
python ingest.py --verify

# Custom PDF directory
python ingest.py --pdf-dir /path/to/your/documents

```

### 5. Launch the UI

```bash
chainlit run app.py

```

Open [http://localhost:8000](https://www.google.com/search?q=http://localhost:8000) in your browser.

---

## Configuration Reference (`src/config.py`)

All parameters can be easily tuned via environment variables or directly in `src/config.py`.

| Parameter | Default | Description |
| --- | --- | --- |
| `LLM_MODEL_NAME` | `llama3-70b-8192` | Groq model. |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | 1024-dim multilingual embeddings |
| `PARENT_CHUNK_SIZE` | `1500` chars | Context fed to the LLM |
| `CHILD_CHUNK_SIZE` | `200` chars | Dense retrieval targets |
| `TOP_K_RETRIEVAL` | `6` | Child chunks fetched per query |
| `COLLECTION_NAME` | `alpharag_general_docs` | Qdrant collection name |

---

```

### 2. Minor Cosmetic Tweak
In `src/qa_chain.py`, lines 37-38:
```python
    We configure the model with `temperature=0.0` to ensure factual, deterministic
    responses, which is critical for financial data extraction.

```

You can simply change that docstring to: `...which is critical for accurate data extraction.`
