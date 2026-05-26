# src/qa_chain.py
# ==============================================================================
# AlphaRAG: LangChain Retrieval-Augmented Generation Chain
# ==============================================================================
"""
Orchestrates the LLM, Prompts, and RAG execution logic.

This module is responsible for building the retrieval chain, injecting metadata 
into the prompt so the LLM can cite its sources, and providing an asynchronous 
generator for real-time UI streaming.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from src import config

logger = logging.getLogger(__name__)


# ==============================================================================
# LLM Initialisation
# ==============================================================================

def build_llm() -> ChatGroq:
    """
    Instantiate the Groq-hosted LLM wrapper.

    We configure the model with `temperature=0.0` to ensure factual, deterministic
    responses, which is critical for financial data extraction.

    Returns:
        A configured ChatGroq instance ready for streaming.
    """
    api_key = config.get_groq_api_key()

    logger.info(f"  🤖 LLM initialised: {config.LLM_MODEL_NAME} (temp={config.LLM_TEMPERATURE})")
    return ChatGroq(
        api_key=api_key,
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        streaming=True,  # Enables token-by-token generation for the UI
    )


# ==============================================================================
# Prompt Engineering & Formatting
# ==============================================================================

def build_prompt() -> ChatPromptTemplate:
    """
    Construct the base ChatPromptTemplate for the RAG chain.

    Returns:
        A ChatPromptTemplate with the system guardrails and user input slot.
    """
    return ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

def _format_docs_for_display(docs: list[Document]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    seen_pages = set()
    
    display_index = 1
    for doc in docs:
        meta = doc.metadata or {}
        
        # 1. Extract identification data
        source_file = meta.get("source", "Unknown")
        page_num = str(meta.get("page_number", meta.get("page", "N/A")))
        
        # 2. Create a unique identifier: (Filename, Page Number)
        identifier = (source_file, page_num)
        
        # 3. Deduplicate UI rendering by Page
        if identifier not in seen_pages:
            seen_pages.add(identifier)
            
            formatted.append({
                "index": display_index,
                "content": doc.page_content,
                "preview": doc.page_content[:200].strip() + ("…" if len(doc.page_content) > 200 else ""),
                "metadata": {
                    # Map to the NEW Pydantic fields
                    "title": meta.get("title", "Unknown Title"),
                    "author": meta.get("author_or_company", "Unknown"),
                    "document_type": meta.get("document_type", "Unknown"),
                    "date": meta.get("date_or_year", "Unknown"),
                    "source": source_file,
                    "page": page_num,
                },
            })
            display_index += 1
            
    return formatted

def build_rag_chain(retriever: ParentDocumentRetriever) -> Runnable:
    llm = build_llm()
    prompt = build_prompt()

    # Update the prompt template variables to match the new schema
    document_prompt = PromptTemplate.from_template(
        "Source File: {source}\n"
        "Title: {title}\n"
        "Author/Entity: {author_or_company}\n"
        "Page: {page_number}\n"
        "Content:\n{page_content}"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt,
        document_variable_name="context",
        document_separator="\n\n--- [Next Source Block] ---\n\n",
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    logger.info("  ⛓️  RAG chain assembled successfully.")
    return rag_chain


# ==============================================================================
# Asynchronous Execution & Streaming
# ==============================================================================

async def astream_query(chain: Runnable, query: str) -> AsyncGenerator[Any, None]:
    """
    Stream the RAG chain response asynchronously.

    This function utilizes LangChain's `astream_events` (v2 API) to tap into 
    the execution graph in real-time. It yields text tokens as they are generated,
    and finally yields the fully formatted source metadata once retrieval completes.

    Args:
        chain: The assembled RAG Runnable.
        query: The user's question.

    Yields:
        str: Tokens from the LLM as they are generated.
        list[dict]: A final list of formatted source dictionaries.
    """
    retrieved_docs: list[Document] = []

    # Stream events from the entire LangChain execution graph
    async for event in chain.astream_events({"input": query}, version="v2"):
        event_kind = event.get("event", "")

        # 1. Capture and yield LLM text tokens
        if event_kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content

        # 2. Capture retrieved documents silently in the background
        elif event_kind == "on_retriever_end":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, list):
                retrieved_docs.extend(output)
            elif isinstance(output, dict):
                retrieved_docs.extend(output.get("documents", []))

    # 3. Format and yield the final sources list so the UI can build the sidebar
    sources = _format_docs_for_display(retrieved_docs)
    yield sources