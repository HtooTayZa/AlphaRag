# src/document_parser.py
# ==============================================================================
# AlphaRAG: Document Parsing & Parent-Child Retriever Construction
# ==============================================================================
# This module is responsible for:
#   1. Loading PDF documents from disk with rich metadata.
#   2. Splitting them into Parent chunks (broad context) and Child chunks
#      (dense retrieval targets) using LangChain's ParentDocumentRetriever.
#   3. Building and returning a ready-to-query ParentDocumentRetriever backed
#      by Qdrant (vector store) and an InMemoryStore (docstore for parents).
# ==============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src import config

logger = logging.getLogger(__name__)


# ==============================================================================
# Metadata Catalogue
# ==============================================================================
# In production, this would be dynamically derived from a database or file
# manifest. Here we use a static catalogue keyed on filename stem to attach
# structured metadata to each ingested document.
# ==============================================================================

DOCUMENT_METADATA_CATALOGUE: dict[str, dict[str, str]] = {
    "tsla_2025_10k": {
        "company":    "Tesla, Inc.",
        "ticker":     "TSLA",
        "year":       "2025",
        "form_type":  "10-K",
        "source":     "tsla_2025_10k.pdf",
        "sector":     "Consumer Discretionary / EV",
    },
    "nvda_2025_10k": {
        "company":    "NVIDIA Corporation",
        "ticker":     "NVDA",
        "year":       "2025",
        "form_type":  "10-K",
        "source":     "nvda_2025_10k.pdf",
        "sector":     "Technology / Semiconductors",
    },
    "msft_2025_10k": {
        "company":    "Microsoft Corporation",
        "ticker":     "MSFT",
        "year":       "2025",
        "form_type":  "10-K",
        "source":     "msft_2025_10k.pdf",
        "sector":     "Technology / Cloud & Software",
    },
    "aapl_2025_10k": {
        "company":    "Apple Inc.",
        "ticker":     "AAPL",
        "year":       "2025",
        "form_type":  "10-K",
        "source":     "aapl_2025_10k.pdf",
        "sector":     "Technology / Consumer Electronics",
    },
    # --- Fallback: applied when filename does not match any catalogue entry ---
    "_default": {
        "company":    "Unknown",
        "ticker":     "N/A",
        "year":       "N/A",
        "form_type":  "Unknown",
        "source":     "unknown.pdf",
        "sector":     "N/A",
    },
}


def _resolve_metadata(pdf_path: Path) -> dict[str, str]:
    """
    Look up structured metadata for a PDF file.

    Priority:
      1. Exact filename stem match in DOCUMENT_METADATA_CATALOGUE.
      2. Partial match (catalogue key is a prefix of the stem).
      3. Default fallback with the actual filename attached.

    Args:
        pdf_path: Resolved path to the PDF file.

    Returns:
        A metadata dict suitable for LangChain Document objects.
    """
    stem = pdf_path.stem.lower()

    # Exact match
    if stem in DOCUMENT_METADATA_CATALOGUE:
        return dict(DOCUMENT_METADATA_CATALOGUE[stem])

    # Prefix / partial match
    for key, meta in DOCUMENT_METADATA_CATALOGUE.items():
        if key == "_default":
            continue
        if key in stem or stem in key:
            resolved = dict(meta)
            resolved["source"] = pdf_path.name  # use the real filename
            return resolved

    # Fallback
    fallback = dict(DOCUMENT_METADATA_CATALOGUE["_default"])
    fallback["source"] = pdf_path.name
    return fallback


# ==============================================================================
# PDF Loading
# ==============================================================================

def load_pdfs_from_directory(pdf_dir: Path) -> list[Document]:
    """
    Load all PDF files from a directory into LangChain Document objects,
    enriching each page with structured financial metadata.

    Each LangChain Document represents one PDF page. Metadata is attached at
    the page level so every chunk derived from it inherits the source info.

    Args:
        pdf_dir: Directory containing raw PDF filings.

    Returns:
        A flat list of LangChain Document objects (one per PDF page).

    Raises:
        FileNotFoundError: If pdf_dir does not exist.
        ValueError: If no PDF files are found in the directory.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"[DocumentParser] PDF directory not found: {pdf_dir}\n"
            f"Create it and place your financial PDFs inside, then re-run ingest.py."
        )

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(
            f"[DocumentParser] No PDF files found in: {pdf_dir}\n"
            f"Place at least one .pdf filing there and re-run ingest.py."
        )

    all_documents: list[Document] = []

    for pdf_path in pdf_files:
        logger.info(f"  📄 Loading: {pdf_path.name}")
        metadata = _resolve_metadata(pdf_path)

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages: list[Document] = loader.load()

            # Enrich every page's metadata with our catalogue data.
            # PyPDFLoader already sets 'page' (int) and 'source' (str).
            for page_doc in pages:
                page_doc.metadata.update(metadata)
                # Keep the PyPDFLoader page number alongside our metadata
                page_doc.metadata["page_number"] = page_doc.metadata.get("page", "N/A")

            all_documents.extend(pages)
            logger.info(f"  ✅ Loaded {len(pages)} pages from '{pdf_path.name}'")

        except Exception as exc:
            # Log and skip broken PDFs rather than crashing the entire pipeline
            logger.error(f"  ❌ Failed to load '{pdf_path.name}': {exc}", exc_info=True)

    logger.info(f"  📚 Total pages loaded: {len(all_documents)}")
    return all_documents


# ==============================================================================
# Text Splitters
# ==============================================================================

def build_text_splitters() -> tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    """
    Construct the parent and child RecursiveCharacterTextSplitter instances.

    RecursiveCharacterTextSplitter tries to split on the most semantic
    boundaries first (paragraphs → sentences → words → chars).

    Returns:
        (parent_splitter, child_splitter) — a tuple of two splitters.
    """
    # Financial documents have dense paragraphs; these separators work well.
    financial_separators: list[str] = [
        "\n\n",   # Paragraph break (strongest signal)
        "\n",     # Line break
        ". ",     # Sentence end
        ", ",     # Clause boundary
        " ",      # Word boundary
        "",       # Character fallback
    ]

    # Parent splitter: produces broad context blocks the LLM will read.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=financial_separators,
        length_function=len,
        add_start_index=True,   # Adds 'start_index' to metadata for auditability
    )

    # Child splitter: produces dense, focused snippets for vector similarity search.
    # Smaller chunks → embeddings capture a single concept → higher retrieval precision.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=financial_separators,
        length_function=len,
        add_start_index=True,
    )

    return parent_splitter, child_splitter


# ==============================================================================
# Embeddings
# ==============================================================================

def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiate the HuggingFace BGE-M3 embedding model.

    BAAI/bge-m3 is a hybrid dense+sparse+colbert model. We use the dense
    output here. The model is downloaded to ~/.cache/huggingface on first run.

    Returns:
        A configured HuggingFaceEmbeddings instance.
    """
    hf_token = config.get_hf_token()
    logger.info(f"  🔢 Loading embedding model: {config.EMBEDDING_MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": config.EMBEDDING_DEVICE,
            # Pass HF token for models that require authentication
            **({"token": hf_token} if hf_token else {}),
        },
        # encode_kwargs controls the sentence-transformers .encode() call
        encode_kwargs={
            "normalize_embeddings": True,    # Cosine similarity requires unit vectors
            "batch_size": 32,
        },
        # BGE models need a query instruction prefix at query time (not index time)
        query_instruction=config.BGE_QUERY_INSTRUCTION,
    )
    return embeddings


# ==============================================================================
# Vector Store (Qdrant)
# ==============================================================================

def build_qdrant_vector_store(embeddings: HuggingFaceEmbeddings) -> QdrantVectorStore:
    """
    Initialize or reconnect to a local Qdrant collection on disk.

    Qdrant is run in embedded/local mode (no separate server process needed).
    Data persists in config.QDRANT_PATH across ingestion runs.

    If QDRANT_URL is set in the environment, connects to Qdrant Cloud instead.

    Args:
        embeddings: The embedding model used to determine vector dimensions.

    Returns:
        A LangChain QdrantVectorStore instance wrapping the collection.
    """
    # Ensure the local storage path exists
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    if config.QDRANT_URL:
        # ── Qdrant Cloud Mode ───────────────────────────────────────────────
        logger.info(f"  ☁️  Connecting to Qdrant Cloud: {config.QDRANT_URL}")
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=60,
        )
    else:
        # ── Local Embedded Mode (default) ───────────────────────────────────
        logger.info(f"  💾 Using local Qdrant at: {config.QDRANT_PATH}")
        client = QdrantClient(path=str(config.QDRANT_PATH))

    # BGE-M3 dense embeddings are 1024-dimensional
    VECTOR_DIM = 1024

    # Create the collection if it doesn't already exist.
    # Recreating on each ingest run would wipe existing data — we check first.
    existing_collections = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME not in existing_collections:
        logger.info(f"  🆕 Creating Qdrant collection: '{config.COLLECTION_NAME}'")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,   # Cosine suits normalized BGE embeddings
            ),
        )
    else:
        logger.info(f"  ♻️  Reusing existing Qdrant collection: '{config.COLLECTION_NAME}'")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store


# ==============================================================================
# ParentDocumentRetriever Assembly
# ==============================================================================

def build_parent_document_retriever(
    vector_store: QdrantVectorStore,
    docstore: InMemoryStore,
) -> ParentDocumentRetriever:
    """
    Assemble the ParentDocumentRetriever from its component parts.

    Data flow at RETRIEVAL time:
      Query (str)
        → embed with BGE-M3
        → ANN search in Qdrant  →  top-K CHILD chunk IDs + scores
        → look up parent_doc_id from child chunk metadata
        → fetch PARENT Documents from InMemoryStore (docstore)
        → return PARENT Documents to the LLM

    Data flow at INGESTION time:
      PDF pages (Documents)
        → parent_splitter  →  PARENT chunks (stored in docstore with UUIDs)
        → child_splitter   →  CHILD chunks (embedded + stored in Qdrant,
                                            with parent_doc_id in metadata)

    Args:
        vector_store: Qdrant-backed vector store for child chunk embeddings.
        docstore:     InMemoryStore that holds the full parent chunk text.

    Returns:
        A configured ParentDocumentRetriever ready for .add_documents() or
        retrieval via .get_relevant_documents().
    """
    parent_splitter, child_splitter = build_text_splitters()

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        # How many child chunks to pull from Qdrant before deduplicating parents
        search_kwargs={"k": config.TOP_K_RETRIEVAL},
    )
    return retriever


# ==============================================================================
# High-Level Ingest Pipeline (called from ingest.py)
# ==============================================================================

def ingest_documents(
    pdf_dir: Path | None = None,
) -> tuple[ParentDocumentRetriever, InMemoryStore]:
    """
    Full ingestion pipeline: load PDFs → chunk → embed → store.

    This is the top-level function called by ingest.py. It orchestrates all
    the steps and returns a fully populated retriever.

    Args:
        pdf_dir: Override the default PDF directory (config.RAW_PDFS_DIR).

    Returns:
        (retriever, docstore) — the populated retriever and the raw parent
        docstore (kept for optional persistence / warm-reloading).
    """
    pdf_dir = pdf_dir or config.RAW_PDFS_DIR

    logger.info("=" * 60)
    logger.info("  AlphaRAG — Document Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1: Load raw PDF pages
    logger.info("\n[Step 1/4] Loading PDFs...")
    documents: list[Document] = load_pdfs_from_directory(pdf_dir)

    # Step 2: Build embedding model
    logger.info("\n[Step 2/4] Initializing embedding model...")
    embeddings = build_embeddings()

    # Step 3: Build / connect to Qdrant
    logger.info("\n[Step 3/4] Connecting to Qdrant vector store...")
    vector_store = build_qdrant_vector_store(embeddings)

    # The docstore is a simple in-memory key-value store.
    # Keys are UUID strings (generated by ParentDocumentRetriever).
    # Values are the full parent Document objects.
    docstore = InMemoryStore()

    # Step 4: Build retriever and ingest documents
    logger.info("\n[Step 4/4] Building ParentDocumentRetriever and ingesting chunks...")
    retriever = build_parent_document_retriever(vector_store, docstore)

    # .add_documents() triggers the full two-pass split:
    #   pass 1 → parent chunks → saved to docstore with UUID keys
    #   pass 2 → child chunks  → embedded and upserted into Qdrant
    #                             with parent UUID stored in payload metadata
    retriever.add_documents(documents, ids=None)

    logger.info("\n✅ Ingestion complete!")
    logger.info(f"   • Parent chunks in docstore : will be reported after ingestion")
    logger.info(f"   • Child chunks in Qdrant    : see collection '{config.COLLECTION_NAME}'")
    logger.info("=" * 60)

    return retriever, docstore


# ==============================================================================
# Retriever Loader (called from app.py at startup — NO re-ingestion)
# ==============================================================================

def load_retriever_for_query() -> ParentDocumentRetriever:
    """
    Build a retriever attached to an EXISTING Qdrant collection for query-time use.

    This function is called by app.py. It does NOT re-ingest any documents.
    It simply reconnects to the already-populated Qdrant collection and
    wraps it in a ParentDocumentRetriever with a fresh InMemoryStore.

    IMPORTANT: Because InMemoryStore is not persisted to disk, the parent
    documents are NOT available in query-only mode via this path. This is a
    known limitation of LangChain's InMemoryStore for ParentDocumentRetriever.

    For production deployments, replace InMemoryStore with a Redis or SQL-backed
    store so that parent chunks survive process restarts.

    The retriever is still useful here: QdrantVectorStore.similarity_search
    returns child chunks that contain the parent text (since child chunks are
    sub-strings of parent chunks, the child text itself provides useful context).

    Returns:
        A configured (but docstore-empty) ParentDocumentRetriever.
    """
    logger.info("  🔌 Connecting to existing Qdrant collection for query mode...")
    embeddings = build_embeddings()
    vector_store = build_qdrant_vector_store(embeddings)
    docstore = InMemoryStore()   # Empty — parent texts won't be restored
    retriever = build_parent_document_retriever(vector_store, docstore)
    return retriever
