#!/usr/bin/env python3
# app.py
# ==============================================================================
# AlphaRAG: Chainlit UI Entry Point (Dynamic Ingestion)
# ==============================================================================
"""
User Interface layer for AlphaRAG, powered by Chainlit.

This module handles session initialization and message orchestration. 
It prompts the user to upload a PDF, ingests it in-memory, and streams
responses to the user in real-time with interactive citation sidebars.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import chainlit as cl
from langchain_core.runnables import Runnable

from src import config
from src.document_parser import ingest_uploaded_file
from src.qa_chain import astream_query, build_rag_chain

logger = logging.getLogger(__name__)


# ==============================================================================
# Session Initialisation & File Upload
# ==============================================================================

@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Triggered when a user opens a new chat session.
    Prompts for a PDF upload, then initializes the retrieval engine in-memory.
    """
    # 1. Ask the user for a file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=(
                f"## 🤖 {config.APP_NAME}: {config.APP_TAGLINE}\n\n"
                "Welcome! Please upload a PDF document to begin. I will extract its contents "
                "and allow you to query it instantly."
            ),
            accept=["application/pdf"],
            max_size_mb=50,
            timeout=180,
        ).send()

    # 2. Get the uploaded file
    file = files[0]
    
    # Send a processing message
    msg = cl.Message(
        content=f"⚙️ Processing `{file.name}`... Extracting metadata and building vector space.",
        author=config.APP_NAME,
    )
    await msg.send()

    try:
        # 3. Dynamically ingest the uploaded file into an in-memory store
        retriever = await cl.make_async(ingest_uploaded_file)(file.path, file.name)
        
        # 4. Build the RAG chain with the session-specific retriever
        rag_chain = await cl.make_async(build_rag_chain)(retriever)
        
        # Persist across the user's session
        cl.user_session.set("rag_chain", rag_chain)
        
        msg.content = f"✅ **`{file.name}` successfully ingested!**\n\nAsk me anything about the document."
        await msg.update()
        
    except Exception as e:
        logger.exception("Initialisation failed")
        msg.content = f"❌ **Error processing document:**\n\n```\n{e}\n```"
        await msg.update()


# ==============================================================================
# Message Handling & Streaming
# ==============================================================================

@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Triggered on every user message. 
    Executes the query, streams tokens to the UI, and attaches interactive citations.
    """
    rag_chain: Runnable | None = cl.user_session.get("rag_chain")
    if not rag_chain:
        await cl.Message(content="⚠️ Session not initialised. Please refresh the page and upload a file.").send()
        return

    user_query = message.content.strip()
    if not user_query:
        return

    # 1. Prepare an empty Chainlit message container to stream into
    msg = cl.Message(content="", author=config.APP_NAME)
    await msg.send()

    sources: list[dict[str, Any]] = []
    full_answer = ""

    # 2. Consume the async generator
    try:
        async for output in astream_query(rag_chain, user_query):
            if isinstance(output, str):
                # It's a text token from the LLM -> append and stream to UI
                full_answer += output
                await msg.stream_token(output)
            elif isinstance(output, list):
                # It's the final list of source metadata -> capture for the sidebar
                sources = output
    except Exception as exc:
        logger.exception("Query execution failed")
        await cl.Message(
            content=f"❌ **Error processing query:**\n\n```\n{traceback.format_exc()}\n```"
        ).send()
        return

    # 3. Post-Generation: Build Sidebar Elements & Citation Footer
    elements: list[cl.Text] = []
    source_lines: list[str] = []

    for source in sources:
        idx = source["index"]
        meta = source["metadata"]
        src_file = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        
        # Pull the new metadata variables
        title = meta.get("title", "Unknown Title")
        author = meta.get("author", "Unknown")
        doc_type = meta.get("document_type", "N/A")
        date = meta.get("date", "N/A")

        # Create the visual card for the sidebar
        sidebar_content = (
            f"**{title}**\n"
            f"*{author} · {doc_type} · {date}*\n"
            f"*File:* `{src_file}` | *Page:* {page}\n\n"
            "---\n\n"
            f"{source['content']}"
        )
        
        element_name = f"Source {idx} ({src_file})"
        elements.append(
            cl.Text(name=element_name, content=sidebar_content, display="side")
        )
        
        # Add to the footer summary
        source_lines.append(f"`{element_name}` · **{title}** — (Page {page})")

    # 4. Finalise the message output
    if sources:
        sources_section = (
            "\n\n---\n"
            "**📎 Retrieved Source Blocks** *(click to expand in sidebar)*\n\n"
            + "\n".join(source_lines)
        )
        msg.content = full_answer + sources_section
        msg.elements = elements
    elif full_answer.strip() == "Insufficient data in the provided filings.":
        # Handle the strict fallback scenario cleanly
        msg.content = "⚠️ **Insufficient data in the provided filings.**"
        
    # Flush the final complete message to the UI
    await msg.update()