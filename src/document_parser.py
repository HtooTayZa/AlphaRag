# src/document_parser.py
# ==============================================================================
# AlphaRAG: Document Parsing & Parent-Child Retriever Construction
# ==============================================================================
"""
Document parsing, chunking, and persistence orchestration.

This module is responsible for loading PDFs, attaching financial metadata,
chunking text using a Parent-Child strategy, and persisting data robustly.
Crucially, it uses an EncoderBackedStore to persist parent chunks to disk,
preventing context loss between ingestion and querying phases.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage.encoder_backed import EncoderBackedStore
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

DOCUMENT_METADATA_CATALOGUE: dict[str, dict[str, str]] = {
    "tsla_2025_10k": {
        "company": "Tesla, Inc.", "ticker": "TSLA", "year": "2025", 
        "form_type": "10-K", "sector": "Consumer Discretionary / EV"
    },
    "nvda_2025_10k": {
        "company": "NVIDIA Corporation", "ticker": "NVDA", "year": "2025", 
        "form_type": "10-K", "sector": "Technology / Semiconductors"
    },
    "msft_2025_10k": {
        "company": "Microsoft Corporation", "ticker": "MSFT", "year": "2025", 
        "form_type": "10-K", "sector": "Technology / Cloud & Software"
    },
    "aapl_2025_10k": {
        "company": "Apple Inc.", "ticker": "AAPL", "year": "2025", 
        "form_type": "10-K", "sector": "Technology / Consumer Electronics"
    },
    "_default": {
        "company": "Unknown", "ticker": "N/A", "year": "N/A", 
        "form_type": "Unknown", "sector": "N/A"
    },
}

def _resolve_metadata(pdf_path: Path) -> dict[str, str]:
    """
    Look up structured metadata for a PDF file based on its filename stem.
    
    Args:
        pdf_path: Resolved path to the PDF file.

    Returns:
        A dictionary containing metadata to be attached to LangChain Documents.
    """
    stem = pdf_path.stem.lower()

    if stem in DOCUMENT_METADATA_CATALOGUE:
        resolved = dict(DOCUMENT_METADATA_CATALOGUE[stem])
        resolved["source"] = pdf_path.name
        return resolved

    fallback = dict(DOCUMENT_METADATA_CATALOGUE["_default"])
    fallback["source"] = pdf_path.name
    return fallback


# ==============================================================================
# PDF Loading & Chunking
# ==============================================================================

def load_pdfs_from_directory(pdf_dir: Path) -> list[Document]:
    """
    Load all PDF files from a directory, enriching pages with catalogued metadata.

    Args:
        pdf_dir: Directory containing raw PDF filings.

    Returns:
        A list of LangChain Document objects (one per PDF page).

    Raises:
        FileNotFoundError: If the directory is missing.
        ValueError: If the directory is empty.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"[DocumentParser] PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"[DocumentParser] No PDF files found in: {pdf_dir}")

    all_documents: list[Document] = []

    for pdf_path in pdf_files:
        logger.info(f"  📄 Loading: {pdf_path.name}")
        metadata = _resolve_metadata(pdf_path)

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            for page_doc in pages:
                page_doc.metadata.update(metadata)
                # Preserve the original page number from PyPDFLoader
                page_doc.metadata["page_number"] = page_doc.metadata.get("page", "N/A")

            all_documents.extend(pages)
            logger.info(f"  ✅ Loaded {len(pages)} pages from '{pdf_path.name}'")

        except Exception as exc:
            logger.error(f"  ❌ Failed to load '{pdf_path.name}': {exc}", exc_info=True)

    logger.info(f"  📚 Total pages loaded: {len(all_documents)}")
    return all_documents


def build_text_splitters() -> tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    """
    Construct the Parent and Child RecursiveCharacterTextSplitters.

    Returns:
        A tuple containing (parent_splitter, child_splitter).
    """
    financial_separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=financial_separators,
        add_start_index=True,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=financial_separators,
        add_start_index=True,
    )

    return parent_splitter, child_splitter


# ==============================================================================
# Model & Storage Initialisation
# ==============================================================================

def build_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate the HuggingFace BGE-M3 embedding model."""
    hf_token = config.get_hf_token()
    logger.info(f"  🔢 Loading embedding model: {config.EMBEDDING_MODEL_NAME}")

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": config.EMBEDDING_DEVICE,
            **({"token": hf_token} if hf_token else {}),
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
        query_instruction=config.BGE_QUERY_INSTRUCTION,
    )


def build_qdrant_vector_store(embeddings: HuggingFaceEmbeddings) -> QdrantVectorStore:
    """
    Initialize or reconnect to the Qdrant collection for child vectors.

    Args:
        embeddings: The active embedding model.

    Returns:
        A LangChain QdrantVectorStore instance.
    """
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    if config.QDRANT_URL:
        logger.info(f"  ☁️  Connecting to Qdrant Cloud: {config.QDRANT_URL}")
        client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)
    else:
        logger.info(f"  💾 Using local Qdrant at: {config.QDRANT_PATH}")
        client = QdrantClient(path=str(config.QDRANT_PATH))

    existing_collections = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME not in existing_collections:
        logger.info(f"  🆕 Creating Qdrant collection: '{config.COLLECTION_NAME}'")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    else:
        logger.info(f"  ♻️  Reusing existing Qdrant collection: '{config.COLLECTION_NAME}'")

    return QdrantVectorStore(
        client=client, 
        collection_name=config.COLLECTION_NAME, 
        embedding=embeddings
    )


def build_local_docstore() -> EncoderBackedStore:
    """
    Build a persistent, JSON-encoded local document store.
    
    Unlike InMemoryStore, this implementation persists parent chunks to disk. 
    This guarantees that when app.py queries Qdrant for a child chunk vector, 
    the system can successfully fetch the associated parent context string.

    Returns:
        An EncoderBackedStore wrapping a LocalFileStore.
    """
    config.LOCAL_DOCSTORE_PATH.mkdir(parents=True, exist_ok=True)
    file_store = LocalFileStore(str(config.LOCAL_DOCSTORE_PATH))
    
    def _doc_to_bytes(doc: Document) -> bytes:
        """Serialize a LangChain Document to a JSON byte string."""
        return json.dumps({
            "page_content": doc.page_content, 
            "metadata": doc.metadata
        }).encode("utf-8")
        
    def _bytes_to_doc(b: bytes) -> Document:
        """Deserialize a JSON byte string back into a LangChain Document."""
        data = json.loads(b.decode("utf-8"))
        return Document(
            page_content=data["page_content"], 
            metadata=data["metadata"]
        )
        
    return EncoderBackedStore(
        store=file_store,
        key_encoder=lambda x: str(x),
        value_serializer=_doc_to_bytes,
        value_deserializer=_bytes_to_doc,
    )


def build_parent_document_retriever(
    vector_store: QdrantVectorStore, 
    docstore: EncoderBackedStore
) -> ParentDocumentRetriever:
    """Assemble the ParentDocumentRetriever."""
    parent_splitter, child_splitter = build_text_splitters()
    
    return ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": config.TOP_K_RETRIEVAL},
    )


# ==============================================================================
# High-Level Orchestration
# ==============================================================================

def ingest_documents(pdf_dir: Path | None = None) -> tuple[ParentDocumentRetriever, EncoderBackedStore]:
    """
    Full ingestion pipeline: load PDFs, split text, embed, and store persistently.

    Args:
        pdf_dir: Override the default PDF directory.

    Returns:
        A tuple of the populated retriever and the local docstore instance.
    """
    pdf_dir = pdf_dir or config.RAW_PDFS_DIR

    logger.info("=" * 60)
    logger.info("  AlphaRAG — Document Ingestion Pipeline")
    logger.info("=" * 60)

    documents = load_pdfs_from_directory(pdf_dir)
    embeddings = build_embeddings()
    vector_store = build_qdrant_vector_store(embeddings)
    docstore = build_local_docstore()

    logger.info("\n[Step 4/4] Building Retriever and indexing chunks...")
    retriever = build_parent_document_retriever(vector_store, docstore)
    
    # Executes text splitting, upserts children to Qdrant, and writes parents to local docstore
    retriever.add_documents(documents, ids=None)

    logger.info("\n✅ Ingestion complete and persisted locally!")
    logger.info("=" * 60)

    return retriever, docstore


def load_retriever_for_query() -> ParentDocumentRetriever:
    """
    Reconstruct the retriever for query mode (called by app.py at startup).
    
    Because we now utilize `build_local_docstore()`, the parent chunks ingested 
    previously will be instantly available for retrieval.

    Returns:
        A fully configured ParentDocumentRetriever connected to persistent storage.
    """
    logger.info("  🔌 Connecting to existing Qdrant and Local DocStore...")
    embeddings = build_embeddings()
    vector_store = build_qdrant_vector_store(embeddings)
    docstore = build_local_docstore()
    
    return build_parent_document_retriever(vector_store, docstore)