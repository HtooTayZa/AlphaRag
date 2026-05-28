# src/config.py
# ==============================================================================
# AlphaRAG: Centralized Configuration
# ==============================================================================
"""
Central configuration module for AlphaRAG.

This module loads environment variables, defines system-wide constants, 
model parameters, and core prompts.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file values into the environment at import time.
load_dotenv()

# Root of the alpharag/ project directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Note: Static storage paths (RAW_PDFS_DIR, QDRANT_PATH, etc.) have been removed
# as the application now relies on session-scoped in-memory storage.

# ------------------------------------------------------------------------------
# API Keys & Credentials
# ------------------------------------------------------------------------------

def _require_env(key: str) -> str:
    """
    Fetch a required environment variable or raise a descriptive error.
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
    """Retrieve the HuggingFace token."""
    return os.environ.get("HUGGINGFACE_HUB_TOKEN")


# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------

# --- LLM (Groq) ---
LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = 0.0          
LLM_MAX_TOKENS: int = 2048

# --- Embeddings (HuggingFace) ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = os.environ.get("EMBEDDING_DEVICE", "cpu")


# ------------------------------------------------------------------------------
# Parent-Child Chunking Strategy
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
APP_TAGLINE: str = "Dynamic Document Assistant"
APP_VERSION: str = "2.0.0"