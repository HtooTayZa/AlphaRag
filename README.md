# AlphaRAG: Institutional Knowledge Extractor

> Enterprise-grade financial document analysis powered by Parent-Child RAG, Groq (LLaMA-3 70B), BAAI/bge-m3 embeddings, and Qdrant.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ALPHARAG SYSTEM ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   INGESTION PIPELINE (ingest.py — run once)                             │
│                                                                         │
│   PDF Files ──► PyPDFLoader ──► Raw Documents (with metadata)           │
│                                      │                                  │
│                              ParentDocumentRetriever                    │
│                               ┌──────┴──────┐                           │
│                               ▼             ▼                           │
│                    Parent Splitter      Child Splitter                  │
│                    (~1500 chars)        (~200 chars)                    │
│                               │             │                           │
│                        InMemoryStore    BGE-M3 Embed                   │
│                        (docstore)           │                           │
│                               │           Qdrant                       │
│                               │         (vector DB)                    │
│                                                                         │
│   QUERY PIPELINE (app.py — per user message)                            │
│                                                                         │
│   User Query                                                            │
│       │                                                                 │
│       ▼                                                                 │
│   BGE-M3 Embed ──► ANN Search in Qdrant ──► Top-K CHILD chunks         │
│                                                   │                     │
│                                          Parent ID lookup               │
│                                                   │                     │
│                                        PARENT Documents ◄── docstore   │
│                                                   │                     │
│                                     Stuff into Prompt {context}        │
│                                                   │                     │
│                                     ChatGroq (LLaMA-3 70B)             │
│                                                   │                     │
│                                             Answer + Sources            │
│                                                   │                     │
│                                        Chainlit UI (app.py)            │
│                                     (Streaming + Citation Sidebar)      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why Parent-Child Chunking?

| Problem | Solution |
|---|---|
| Large chunks → embeddings too diluted, poor retrieval precision | Index small **child chunks** (~200 chars) for high-precision ANN search |
| Small chunks → LLM lacks context to formulate a coherent answer | Feed **parent chunks** (~1500 chars) to the LLM as context |
| Result | Best of both worlds: precision retrieval + rich generation context |

## File Structure

```
alpharag/
├── .env.example              ← Copy to .env and fill in your keys
├── .gitignore
├── requirements.txt
├── chainlit.md               ← Chainlit welcome page
├── app.py                    ← Chainlit UI (on_chat_start, on_message)
├── ingest.py                 ← CLI: process PDFs and build vector DB
├── data/
│   ├── raw_pdfs/             ← Drop your PDF filings here
│   └── qdrant_db/            ← Auto-created by ingest.py
└── src/
    ├── __init__.py
    ├── config.py             ← All constants, model names, system prompt
    ├── document_parser.py    ← ParentDocumentRetriever setup + PDF loading
    └── qa_chain.py           ← LangChain RAG chain (Groq LLM + retriever)
```

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
# Copy your financial PDFs here. Suggested naming:
# tsla_2025_10k.pdf, nvda_2025_10k.pdf, msft_2025_10k.pdf …
```

Filenames matching the pattern `{ticker}_{year}_{form}.pdf` are auto-tagged with structured metadata (company name, sector, form type).

### 4. Run ingestion

```bash
# Basic ingestion
python ingest.py

# With verification query
python ingest.py --verify

# Custom PDF directory
python ingest.py --pdf-dir /path/to/your/filings
```

Ingestion only needs to run once per set of documents. The Qdrant DB persists to `data/qdrant_db/`.

### 5. Launch the UI

```bash
chainlit run app.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Configuration Reference (`src/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `llama3-70b-8192` | Groq model. Alt: `mixtral-8x7b-32768` for larger context |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output for factual queries |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | 1024-dim multilingual embeddings |
| `PARENT_CHUNK_SIZE` | `1500` chars | Context fed to the LLM |
| `PARENT_CHUNK_OVERLAP` | `200` chars | Prevents mid-sentence parent splits |
| `CHILD_CHUNK_SIZE` | `200` chars | Dense retrieval targets |
| `CHILD_CHUNK_OVERLAP` | `30` chars | Small overlap for child chunks |
| `TOP_K_RETRIEVAL` | `6` | Child chunks fetched per query |
| `COLLECTION_NAME` | `alpharag_financial_docs` | Qdrant collection name |

All parameters can be overridden via environment variables in `.env`.

---

## Adding New Document Metadata

Edit `DOCUMENT_METADATA_CATALOGUE` in `src/document_parser.py`:

```python
DOCUMENT_METADATA_CATALOGUE = {
    "googl_2025_10k": {
        "company":   "Alphabet Inc.",
        "ticker":    "GOOGL",
        "year":      "2025",
        "form_type": "10-K",
        "source":    "googl_2025_10k.pdf",
        "sector":    "Technology / Advertising",
    },
    # ... add more entries
}
```

---

## Switching to Qdrant Cloud

1. Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Add to `.env`:
   ```
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your_api_key
   ```
3. Re-run `python ingest.py`

---

## Production Considerations

| Limitation | Production Fix |
|---|---|
| `InMemoryStore` lost on restart | Replace with `RedisStore` or `SQLStore` |
| Single Qdrant collection | Separate collections per client/fund |
| No auth on Chainlit | Enable Chainlit authentication (`@cl.password_auth_callback`) |
| Synchronous `run_query()` | Use `astream_query()` for true streaming |
| No query logging | Add LangSmith tracing (`LANGCHAIN_TRACING_V2=true`) |

---

## License

MIT — see `LICENSE` for details.
<<<<<<< HEAD
=======
>>>>>>> b7e2213 (initial commit)
>>>>>>> 5bfa6cc (first commit)
