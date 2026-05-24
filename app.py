#!/usr/bin/env python3
# app.py
# ==============================================================================
# AlphaRAG: Chainlit UI Entry Point
# ==============================================================================
"""
User Interface layer for AlphaRAG, powered by Chainlit.

This module handles session initialization and message orchestration. It consumes
the async token generator from `src/qa_chain.py` to stream responses to the user 
in real-time and constructs interactive citation sidebars.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import chainlit as cl
from langchain_core.runnables import Runnable

from src import config
from src.document_parser import load_retriever_for_query
from src.qa_chain import astream_query, build_rag_chain

logger = logging.getLogger(__name__)


# ==============================================================================
# Session Initialisation
# ==============================================================================

@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Triggered when a user opens a new chat session.
    Initialises the retrieval engine and RAG chain, storing them in the session.
    """
    await cl.Message(
        content=(
            f"## 🤖 {config.APP_NAME}: {config.APP_TAGLINE}\n\n"
            "Welcome. I have access to your indexed documents.\n"
            "Ask me any question, and I will extract and synthesize accurate "
            "information from the provided context.\n\n"
            "---\n"
            "*Initialising retrieval engine…*"
        ),
        author=config.APP_NAME,
    ).send()

    try:
        # Initialise connections to Qdrant and Local DocStore asynchronously
        retriever = await cl.make_async(load_retriever_for_query)()
        rag_chain = await cl.make_async(build_rag_chain)(retriever)
        
        # Persist across the user's session
        cl.user_session.set("rag_chain", rag_chain)
        
        await cl.Message(
            content="**Ready.** Ask me anything about your documents.",
            author=config.APP_NAME,
        ).send()
        
    except FileNotFoundError as e:
        await cl.Message(
            content=f"**Storage not found.** Run `python ingest.py` first.\n\n`{e}`"
        ).send()
    except Exception as e:
        logger.exception("Initialisation failed")
        await cl.Message(
            content=f"❌ **Initialisation error:**\n\n```\n{e}\n```"
        ).send()


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
        await cl.Message(content="⚠️ Session not initialised. Please refresh the page.").send()
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

    # In app.py (inside on_message)

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