# src/document_parser.py
# ==============================================================================
# AlphaRAG: Document Parsing & Parent-Child Retriever Construction
# ==============================================================================
"""
Document parsing, chunking, and persistence orchestration.

This module is responsible for loading PDFs, automatically extracting metadata 
using an LLM, chunking text using a Parent-Child strategy, and persisting data robustly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
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
# Automated Metadata Extraction (Pydantic + Groq)
# ==============================================================================

class DocumentMetadata(BaseModel):
    """Schema for extracting metadata from the first page of any document."""
    title: str = Field(description="The main title or subject of the document")
    author_or_company: str = Field(description="The author, company, or organization that created the document")
    document_type: str = Field(description="The type or category of the document (e.g., Report, Essay, Legal Contract, Manual)")
    date_or_year: str = Field(description="The date or year mentioned, otherwise 'Unknown'")

def extract_metadata_with_llm(page_content: str) -> dict[str, str]:
    """
    Uses the Groq LLM to extract structured metadata from the first page of a document.
    Falls back to generic metadata if the API call fails.
    """
    try:
        llm = ChatGroq(
            api_key=config.get_groq_api_key(), 
            model=config.LLM_MODEL_NAME, 
            temperature=0
        ).with_structured_output(DocumentMetadata)
        
        # Pass only the first 2000 characters to save tokens and speed up extraction
        prompt = f"Extract the following metadata from this document snippet:\n\n{page_content[:2000]}"
        extracted_obj = llm.invoke(prompt)
        
        return extracted_obj.model_dump()
        
    except Exception as e:
        logger.warning(f"  ⚠️ LLM Metadata extraction failed: {e}. Using fallback metadata.")
        return {
            "title": "Unknown Document",
            "author_or_company": "Unknown",
            "document_type": "General",
            "date_or_year": "Unknown"
        }


# ==============================================================================
# PDF Loading & Chunking
# ==============================================================================

def load_pdfs_from_directory(pdf_dir: Path) -> list[Document]:
    """
    Load all PDF files from a directory, enriching pages with automated LLM metadata.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"[DocumentParser] PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"[DocumentParser] No PDF files found in: {pdf_dir}")

    all_documents: list[Document] = []

    for pdf_path in pdf_files:
        logger.info(f"  📄 Loading and analyzing: {pdf_path.name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # AUTOMATED EXTRACTION: Run on the first page only
            first_page_text = pages[0].page_content
            metadata = extract_metadata_with_llm(first_page_text)
            
            # Always append the actual filename
            metadata["source"] = pdf_path.name

            # Attach the extracted metadata to every page of this document
            for page_doc in pages:
                page_doc.metadata.update(metadata)
                page_doc.metadata["page_number"] = page_doc.metadata.get("page", "N/A")

            all_documents.extend(pages)
            logger.info(f"  ✅ Extracted metadata: {metadata['title']} by {metadata['author_or_company']}")

        except Exception as exc:
            logger.error(f"  ❌ Failed to load '{pdf_path.name}': {exc}", exc_info=True)

    logger.info(f"  📚 Total pages loaded: {len(all_documents)}")
    return all_documents


def build_text_splitters() -> tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
    """Construct the Parent and Child RecursiveCharacterTextSplitters."""
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=separators,
        add_start_index=True,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=separators,
        add_start_index=True,
    )

    return parent_splitter, child_splitter


# ==============================================================================
# Model & Storage Initialisation
# ==============================================================================

def build_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate the HuggingFace embedding model."""
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
    """Initialize or reconnect to the Qdrant collection for child vectors."""
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
    """Build a persistent, JSON-encoded local document store."""
    config.LOCAL_DOCSTORE_PATH.mkdir(parents=True, exist_ok=True)
    file_store = LocalFileStore(str(config.LOCAL_DOCSTORE_PATH))
    
    def _doc_to_bytes(doc: Document) -> bytes:
        return json.dumps({
            "page_content": doc.page_content, 
            "metadata": doc.metadata
        }).encode("utf-8")
        
    def _bytes_to_doc(b: bytes) -> Document:
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
    """Full ingestion pipeline: load PDFs, split text, embed, and store persistently."""
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
    
    retriever.add_documents(documents, ids=None)

    logger.info("\n✅ Ingestion complete and persisted locally!")
    logger.info("=" * 60)

    return retriever, docstore


def load_retriever_for_query() -> ParentDocumentRetriever:
    """Reconstruct the retriever for query mode (called by app.py at startup)."""
    logger.info("  🔌 Connecting to existing Qdrant and Local DocStore...")
    embeddings = build_embeddings()
    vector_store = build_qdrant_vector_store(embeddings)
    docstore = build_local_docstore()
    
    return build_parent_document_retriever(vector_store, docstore)