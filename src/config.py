# ==============================================================================
# AlphaRAG: Centralized Configuration
# ==============================================================================
"""
Central configuration module for AlphaRAG.

This module loads environment variables, defines system-wide constants, file paths, 
model parameters, and core prompts. Centralizing these values ensures consistency 
across ingestion, retrieval, and UI layers.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file values into the environment at import time.
load_dotenv()


# ------------------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------------------

# Root of the alpharag/ project directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Where raw PDF filings are stored prior to ingestion
RAW_PDFS_DIR: Path = PROJECT_ROOT / "data" / "raw_pdfs"

# Vector storage path: Qdrant persists its on-disk index here
QDRANT_PATH: Path = PROJECT_ROOT / "data" / "qdrant_db"

# Document storage path: Local fallback for LangChain's docstore to persist parent chunks
LOCAL_DOCSTORE_PATH: Path = PROJECT_ROOT / "data" / "local_docstore"


# ------------------------------------------------------------------------------
# API Keys & Credentials
# ------------------------------------------------------------------------------

def _require_env(key: str) -> str:
    """
    Fetch a required environment variable or raise a descriptive error.
    
    Args:
        key: The name of the environment variable.
        
    Returns:
        The string value of the environment variable.
        
    Raises:
        EnvironmentError: If the variable is missing or empty.
    """
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"[AlphaRAG] Missing required environment variable: '{key}'. "
            f"Please ensure it is set in your .env file or environment."
        )
    return value

def get_groq_api_key() -> str:
    """Retrieve the Groq API key."""
    return _require_env("GROQ_API_KEY")

def get_hf_token() -> str | None:
    """
    Retrieve the HuggingFace token. 
    Optional for public models, but recommended to avoid rate limits.
    """
    return os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Optional Qdrant Cloud overrides — falls back to local embedded mode if absent
QDRANT_URL: str | None = os.environ.get("QDRANT_URL")
QDRANT_API_KEY: str | None = os.environ.get("QDRANT_API_KEY")


# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------

# --- LLM (Groq) ---
# Optimized for Groq's Free Tier to ensure maximum stability and speed
LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = 0.0          # Keeps output deterministic and strictly grounded
LLM_MAX_TOKENS: int = 2048

# --- Embeddings (HuggingFace) ---
# Default: BAAI/bge-m3 (Multilingual, handles financial jargon well, 1024-dim)
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = os.environ.get("EMBEDDING_DEVICE", "cpu")

# BGE models require a specific query instruction prefix to optimize retrieval
BGE_QUERY_INSTRUCTION: str = ""

# --- Vector Store ---
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "alpharag_general_docs")


# ------------------------------------------------------------------------------
# Parent-Child Chunking Strategy
# ------------------------------------------------------------------------------
# Architecture overview:
# 1. Broad PARENT chunks (~1500 chars) are stored on disk. They provide the LLM with context.
# 2. Dense CHILD chunks (~200 chars) are stored in Qdrant. They optimize vector search accuracy.
# ------------------------------------------------------------------------------

PARENT_CHUNK_SIZE: int = int(os.environ.get("PARENT_CHUNK_SIZE", "1500"))
PARENT_CHUNK_OVERLAP: int = 200

CHILD_CHUNK_SIZE: int = int(os.environ.get("CHILD_CHUNK_SIZE", "200"))
CHILD_CHUNK_OVERLAP: int = 30

# Number of relevant child chunks to retrieve per user query
TOP_K_RETRIEVAL: int = 6


# ------------------------------------------------------------------------------
# System Prompt & Guardrails
# ------------------------------------------------------------------------------

# The guardrail prompt prevents hallucinations and enforces exact citation tracking.
SYSTEM_PROMPT: str = (
    "You are a helpful, precise AI assistant. "
    "Your task is to answer user queries based ONLY on the provided context. "
    "RULES:\n"
    "1. If the answer is not in the context, reply: 'I cannot find the answer in the provided documents.'\n"
    "2. Cite the source file name for every claim you make using the 'Source File' provided in the context blocks.\n"
    "3. Keep answers concise and strictly grounded in the facts provided.\n\n"
    "Context blocks:\n"
    "{context}"
)

# Application Identity
APP_NAME: str = "AlphaRAG"
APP_TAGLINE: str = "General Document Assistant"
APP_VERSION: str = "1.0.0"