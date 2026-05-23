# src/config.py
# ==============================================================================
# AlphaRAG: Centralized Configuration
# ==============================================================================
# All tunable parameters, model identifiers, prompt templates, and paths live
# here. This is the single source of truth — no magic strings elsewhere.
# ==============================================================================

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file values into the environment at import time.
# This makes config.py safe to import first in any module.
load_dotenv()


# ------------------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------------------

# Root of the alpharag/ project directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Where raw PDF filings are stored (not committed to git)
RAW_PDFS_DIR: Path = PROJECT_ROOT / "data" / "raw_pdfs"

# Qdrant persists its on-disk index here
QDRANT_PATH: Path = PROJECT_ROOT / "data" / "qdrant_db"

# ------------------------------------------------------------------------------
# API Keys  — read from environment; raise early if missing
# ------------------------------------------------------------------------------

def _require_env(key: str) -> str:
    """Fetch a required environment variable or raise a descriptive error."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"[AlphaRAG] Missing required environment variable: '{key}'. "
            f"Please copy .env.example → .env and fill in your credentials."
        )
    return value


def get_groq_api_key() -> str:
    return _require_env("GROQ_API_KEY")


def get_hf_token() -> str | None:
    """HuggingFace token is optional for public models but avoids rate limits."""
    return os.environ.get("HUGGINGFACE_HUB_TOKEN")


# Optional Qdrant Cloud overrides — falls back to local mode if absent
QDRANT_URL: str | None = os.environ.get("QDRANT_URL")
QDRANT_API_KEY: str | None = os.environ.get("QDRANT_API_KEY")


# ------------------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------------------

# --- LLM (Groq) ---
# llama3-70b-8192  : Best reasoning quality, recommended for production
# mixtral-8x7b-32768: Larger context window (32k), useful for very long docs
LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "llama3-70b-8192")
LLM_TEMPERATURE: float = 0.0          # Zero temperature → deterministic, factual
LLM_MAX_TOKENS: int = 2048            # Max tokens in the generated answer

# --- Embeddings (HuggingFace) ---
# BAAI/bge-m3: State-of-the-art multilingual embeddings, 1024-dim output
# Excellent for financial text — handles abbreviations, numbers, and jargon
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DEVICE: str = os.environ.get("EMBEDDING_DEVICE", "cpu")
# BGE models perform best with this instruction prefix on queries (not docs)
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

# --- Vector Store ---
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "alpharag_financial_docs")

# ------------------------------------------------------------------------------
# Parent-Child Chunking Strategy
# ------------------------------------------------------------------------------
# The core retrieval architecture uses a two-tier chunking scheme:
#
#   ┌─────────────────────────────────────────────────────────┐
#   │  PARENT CHUNK  (~1500 chars)                            │
#   │  Broad semantic block — what the LLM reads to answer.   │
#   │  Stored in an InMemoryStore (docstore).                 │
#   │                                                         │
#   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
#   │  │  CHILD   │ │  CHILD   │ │  CHILD   │ │  CHILD   │  │
#   │  │ (~200c)  │ │ (~200c)  │ │ (~200c)  │ │ (~200c)  │  │
#   │  │ Indexed  │ │ Indexed  │ │ Indexed  │ │ Indexed  │  │
#   │  │ in Qdrant│ │ in Qdrant│ │ in Qdrant│ │ in Qdrant│  │
#   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
#   └─────────────────────────────────────────────────────────┘
#
#  Query → embed query → ANN search finds matching CHILD chunks
#       → look up their PARENT chunk IDs → fetch full PARENT text
#       → stuff PARENT chunks into LLM context → generate answer
#
# Why? Dense vector search works better on short, focused sentences.
# But the LLM needs the wider paragraph context to formulate a coherent answer.
# ------------------------------------------------------------------------------

PARENT_CHUNK_SIZE: int = int(os.environ.get("PARENT_CHUNK_SIZE", "1500"))
PARENT_CHUNK_OVERLAP: int = 200   # Overlap prevents splitting mid-sentence

CHILD_CHUNK_SIZE: int = int(os.environ.get("CHILD_CHUNK_SIZE", "200"))
CHILD_CHUNK_OVERLAP: int = 30     # Small overlap for child chunks

# How many child chunks to retrieve from the vector store per query
TOP_K_RETRIEVAL: int = 6

# ------------------------------------------------------------------------------
# System Prompt
# ------------------------------------------------------------------------------
# This prompt is the guardrail that prevents hallucination. The explicit
# fallback phrase "Insufficient data in the provided filings." is intentional:
# it gives downstream consumers a parseable signal for null-result handling.

SYSTEM_PROMPT: str = (
    "You are a precise financial analyst at an institutional investment firm. "
    "Your role is to extract and synthesize accurate information from SEC filings, "
    "earnings transcripts, and financial reports.\n\n"
    "RULES — follow these absolutely:\n"
    "1. Answer the user's query using ONLY the provided context blocks.\n"
    "2. If the answer cannot be explicitly found in the retrieved context, "
    "you must reply exactly with: 'Insufficient data in the provided filings.'\n"
    "3. Do NOT guess, infer beyond what is stated, or use outside knowledge.\n"
    "4. When quoting figures, always include the unit (e.g., '$4.2B', '18.3%').\n"
    "5. If multiple filings are present in the context, attribute each claim to "
    "its source document explicitly (e.g., 'Per the Tesla 2025 10-K...').\n\n"
    "Context blocks:\n"
    "{context}"
)

# Human-readable application metadata
APP_NAME: str = "AlphaRAG"
APP_TAGLINE: str = "Institutional Knowledge Extractor"
APP_VERSION: str = "1.0.0"
