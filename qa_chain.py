# src/qa_chain.py
# ==============================================================================
# AlphaRAG: LangChain Retrieval-Augmented Generation Chain
# ==============================================================================
# This module builds the end-to-end RAG pipeline:
#
#   User Query (str)
#     ↓
#   [Retriever] ParentDocumentRetriever
#     • Embeds the query with BGE-M3
#     • ANN-searches Qdrant → top-K CHILD chunks
#     • Fetches corresponding PARENT Documents from docstore
#     ↓
#   [Stuffing]  create_stuff_documents_chain
#     • Formats all PARENT chunk texts into the system prompt {context} slot
#     ↓
#   [LLM]       Groq (llama3-70b-8192)
#     • Generates a grounded answer from the context
#     ↓
#   Response dict: {"answer": str, "context": List[Document], "input": str}
# ==============================================================================

from __future__ import annotations

import logging
from typing import Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import config

logger = logging.getLogger(__name__)


# ==============================================================================
# LLM Initialisation
# ==============================================================================

def build_llm() -> ChatGroq:
    """
    Instantiate the Groq-hosted LLM (LLaMA-3 70B by default).

    ChatGroq is a thin wrapper around Groq's OpenAI-compatible API.
    We set temperature=0 for deterministic, factual financial answers.

    Returns:
        A configured ChatGroq instance.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set.
    """
    api_key = config.get_groq_api_key()  # Raises EnvironmentError if absent

    llm = ChatGroq(
        api_key=api_key,
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        # Groq supports streaming — Chainlit will consume the stream token-by-token
        streaming=True,
    )
    logger.info(f"  🤖 LLM initialised: {config.LLM_MODEL_NAME} (temp={config.LLM_TEMPERATURE})")
    return llm


# ==============================================================================
# Prompt Template
# ==============================================================================

def build_prompt() -> ChatPromptTemplate:
    """
    Construct the ChatPromptTemplate for the RAG chain.

    The prompt has two slots injected at runtime by create_stuff_documents_chain:
      • {context}  — the formatted parent chunk texts (joined with separators)
      • {input}    — the raw user query string

    The system message is defined in config.SYSTEM_PROMPT and includes {context}.
    The human message carries the {input}.

    Returns:
        A ChatPromptTemplate ready for use in create_stuff_documents_chain.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            # System message: roles, rules, and context injection point
            ("system", config.SYSTEM_PROMPT),
            # Human message: the analyst's question
            ("human", "{input}"),
        ]
    )
    return prompt


# ==============================================================================
# Document Formatting Helper
# ==============================================================================

def _format_docs_for_display(docs: list[Document]) -> list[dict[str, Any]]:
    """
    Convert retrieved LangChain Documents into JSON-serialisable dicts
    for the Chainlit UI to render as source citations.

    Args:
        docs: List of retrieved Document objects (parent chunks).

    Returns:
        List of dicts with 'content', 'metadata', and 'preview' keys.
    """
    formatted: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        formatted.append(
            {
                "index": i + 1,
                "content": doc.page_content,
                # Preview: first 200 chars for sidebar card display
                "preview": doc.page_content[:200].strip() + ("…" if len(doc.page_content) > 200 else ""),
                "metadata": {
                    "company":    meta.get("company",   "Unknown"),
                    "ticker":     meta.get("ticker",    "N/A"),
                    "year":       meta.get("year",      "N/A"),
                    "form_type":  meta.get("form_type", "N/A"),
                    "source":     meta.get("source",    "N/A"),
                    "sector":     meta.get("sector",    "N/A"),
                    "page":       str(meta.get("page_number", meta.get("page", "N/A"))),
                },
            }
        )
    return formatted


# ==============================================================================
# RAG Chain Builder
# ==============================================================================

def build_rag_chain(retriever: ParentDocumentRetriever) -> Runnable:
    """
    Assemble the full LangChain RAG chain.

    Chain composition:
      create_retrieval_chain
        └─ retriever               (ParentDocumentRetriever → parent Documents)
        └─ create_stuff_documents_chain
              └─ prompt            (ChatPromptTemplate with system + human msgs)
              └─ llm               (ChatGroq — llama3-70b / mixtral)

    The output of chain.invoke({"input": query}) is:
      {
        "input":   str,            # The original query
        "context": List[Document], # The parent chunks used for answering
        "answer":  str,            # The LLM-generated answer (or the fallback phrase)
      }

    Args:
        retriever: A populated ParentDocumentRetriever instance.

    Returns:
        A LangChain Runnable chain ready for .invoke() or .astream().
    """
    llm = build_llm()
    prompt = build_prompt()

    # create_stuff_documents_chain:
    #   • Takes the list of Documents from the retriever
    #   • Formats them by joining doc.page_content with "\n\n---\n\n"
    #   • Inserts the joined text into the {context} slot of the prompt
    #   • Passes the filled prompt to the LLM
    question_answer_chain: Runnable = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        # document_variable_name must match {context} in the system prompt
        document_variable_name="context",
        # Custom separator makes it visually clear where one chunk ends
        document_separator="\n\n--- [Next Source Block] ---\n\n",
    )

    # create_retrieval_chain:
    #   • Runs the retriever on the user "input"
    #   • Passes retrieved docs as "context" to the combine_docs_chain
    #   • Returns the merged output dict
    rag_chain: Runnable = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    logger.info("  ⛓️  RAG chain assembled successfully.")
    return rag_chain


# ==============================================================================
# Query Execution (with Retry Logic)
# ==============================================================================

@retry(
    # Retry on transient network/API errors from Groq
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def run_query(
    chain: Runnable,
    query: str,
) -> dict[str, Any]:
    """
    Execute the RAG chain for a given user query.

    Wraps chain.invoke() with retry logic for transient API failures.
    Structures the response into a consistent dict for app.py to consume.

    Data flow recap:
      1. query → retriever.get_relevant_documents(query)
               = embed query → Qdrant ANN → child chunk IDs
                             → docstore lookup → parent Documents
      2. parent Documents → stuff into {context} slot of the prompt
      3. prompt + {input: query} → ChatGroq → generated answer text
      4. Return: {"input", "context", "answer", "sources"}

    Args:
        chain: The assembled RAG Runnable from build_rag_chain().
        query: The natural-language question from the user.

    Returns:
        A dict containing:
          • "input"   (str)              — the original query
          • "answer"  (str)              — the LLM's answer
          • "context" (List[Document])   — parent chunks used
          • "sources" (List[dict])       — formatted metadata for UI
    """
    logger.info(f"  🔍 Processing query: {query[:80]}{'…' if len(query) > 80 else ''}")

    # The chain returns {"input": str, "context": List[Document], "answer": str}
    raw_output: dict[str, Any] = chain.invoke({"input": query})

    retrieved_docs: list[Document] = raw_output.get("context", [])
    answer: str = raw_output.get("answer", "Insufficient data in the provided filings.")

    # Attach the formatted source list for the Chainlit sidebar
    sources = _format_docs_for_display(retrieved_docs)

    logger.info(f"  ✅ Answer generated. Sources used: {len(retrieved_docs)}")

    return {
        "input":   query,
        "answer":  answer,
        "context": retrieved_docs,  # Raw Document objects (for LangChain internals)
        "sources": sources,          # Formatted dicts (for Chainlit UI)
    }


# ==============================================================================
# Async Streaming Query (for Chainlit token-by-token streaming)
# ==============================================================================

async def astream_query(
    chain: Runnable,
    query: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Stream the RAG chain response asynchronously.

    Used by app.py to stream tokens progressively to the Chainlit UI,
    giving the user real-time feedback as the LLM generates the answer.

    The retrieval step still runs synchronously before streaming begins —
    only the LLM generation phase is streamed.

    Args:
        chain: The assembled RAG Runnable from build_rag_chain().
        query: The user's question.

    Yields:
        Intermediate token strings via the async generator.
        After exhaustion, call .sources on the returned tuple.

    Returns:
        (full_answer: str, sources: List[dict])
    """
    full_answer_parts: list[str] = []
    retrieved_docs: list[Document] = []

    # astream_events gives us fine-grained control over which events to consume
    async for event in chain.astream_events(
        {"input": query},
        version="v2",   # v2 is the stable events API
    ):
        event_name = event.get("name", "")
        event_kind = event.get("event", "")

        # Capture tokens from the LLM generation step
        if event_kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                full_answer_parts.append(chunk.content)
                yield chunk.content   # Yield each token to Chainlit

        # Capture retrieved documents from the retriever step
        elif event_kind == "on_retriever_end":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, list):
                retrieved_docs.extend(output)
            elif isinstance(output, dict):
                retrieved_docs.extend(output.get("documents", []))

    full_answer = "".join(full_answer_parts)
    if not full_answer.strip():
        full_answer = "Insufficient data in the provided filings."

    sources = _format_docs_for_display(retrieved_docs)

    # We return these via a sentinel — app.py collects via the generator protocol
    # (See usage pattern in app.py for how sources are extracted post-stream)
    return full_answer, sources
