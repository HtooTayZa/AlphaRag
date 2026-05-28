# src/document_parser.py
# ==============================================================================
# AlphaRAG: Dynamic Document Parsing & In-Memory Retrieval
# ==============================================================================
"""
Document parsing, chunking, and session-scoped retrieval orchestration.

This module loads dynamically uploaded PDFs, extracts metadata, 
chunks text, and stores vectors/documents in isolated in-memory stores.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, Modifier

from src import config

logger = logging.getLogger(__name__)


# ==============================================================================
# Automated Metadata Extraction (Pydantic + Groq)
# ==============================================================================

class DocumentMetadata(BaseModel):
    """Schema for extracting metadata from the first page of any document."""
    title: str = Field(description="The main title or subject of the document")
    author_or_company: str = Field(description="The author, company, or organization that created the document")
    document_type: str = Field(description="The type or category of the document")
    date_or_year: str = Field(description="The date or year mentioned, otherwise 'Unknown'")

def extract_metadata_with_llm(page_content: str) -> dict[str, str]:
    """
    Uses the Groq LLM to extract structured metadata from the first page of a document.
    """
    try:
        llm = ChatGroq(
            api_key=config.get_groq_api_key(), 
            model=config.LLM_MODEL_NAME, 
            temperature=0
        ).with_structured_output(DocumentMetadata)
        
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

def load_single_pdf(file_path: str, original_name: str) -> list[Document]:
    """
    Load a single dynamically uploaded PDF and extract metadata.
    """
    logger.info(f"  📄 Loading and analyzing uploaded file: {original_name}")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if pages:
        # AUTOMATED EXTRACTION: Run on the first page only
        first_page_text = pages[0].page_content
        metadata = extract_metadata_with_llm(first_page_text)
        
        # Always append the actual original filename
        metadata["source"] = original_name

        # Attach the extracted metadata to every page of this document
        for page_doc in pages:
            page_doc.metadata.update(metadata)
            page_doc.metadata["page_number"] = page_doc.metadata.get("page", "N/A")
            
        logger.info(f"  ✅ Extracted metadata: {metadata['title']} by {metadata['author_or_company']}")

    return pages

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
# In-Memory Model & Storage Initialisation
# ==============================================================================

def build_memory_qdrant(embeddings: HuggingFaceEmbeddings, collection_name: str) -> QdrantVectorStore:
    """Initialize an isolated in-memory Qdrant instance for the session."""
    logger.info(f"  🧠 Creating In-Memory Qdrant collection: '{collection_name}'")
    
    # Use :memory: for strict session isolation
    client = QdrantClient(location=":memory:")
    
    client.create_collection(
        collection_name=collection_name,
        # FIX: Pass a dictionary mapping the name "dense" to the VectorParams
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse-text": SparseVectorParams(
                index=None,
                modifier=Modifier.IDF
            )
        }
    )

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    return QdrantVectorStore(
        client=client, 
        collection_name=collection_name, 
        embedding=embeddings,
        sparse_embedding=sparse_embeddings, 
        vector_name="dense",            
        sparse_vector_name="sparse-text",
        retrieval_mode=RetrievalMode.HYBRID
    )

def build_parent_document_retriever(
    vector_store: QdrantVectorStore, 
    docstore: InMemoryStore
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

def ingest_uploaded_file(file_path: str, original_name: str) -> ParentDocumentRetriever:
    """Full in-memory ingestion pipeline for a dynamically uploaded file."""
    logger.info("=" * 60)
    logger.info(f"  AlphaRAG — Processing Session File: {original_name}")
    logger.info("=" * 60)

    # 1. Load and parse the PDF
    documents = load_single_pdf(file_path, original_name)
    
    # 2. Setup Embeddings
    logger.info("  ⚙️  Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": config.EMBEDDING_DEVICE}
    )
    
    # 3. Create a unique session ID for Qdrant isolation
    session_id = uuid.uuid4().hex
    unique_collection = f"docs_{session_id}"
    
    # 4. Initialize In-Memory Stores
    vector_store = build_memory_qdrant(embeddings, unique_collection)
    docstore = InMemoryStore() 
    
    # 5. Build Retriever and Index Chunks
    logger.info("  📊 Building Retriever and indexing chunks...")
    retriever = build_parent_document_retriever(vector_store, docstore)
    retriever.add_documents(documents, ids=None)

    logger.info("  ✅ Ingestion complete and ready for query!")
    logger.info("=" * 60)

    return retriever