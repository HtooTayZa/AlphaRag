#!/usr/bin/env python3
# app.py
# ==============================================================================
# AlphaRAG: Chainlit UI Entry Point
# ==============================================================================
# This is the only file Chainlit needs to know about.
# Run with:  chainlit run app.py
#
# Responsibilities of this file (UI layer ONLY):
#   • @cl.on_chat_start  — initialise session: build retriever & RAG chain
#   • @cl.on_message     — handle user messages: stream answer, render citations
#   • Source Citations   — cl.Text elements create a clickable sidebar with
#                          each retrieved parent chunk's content and metadata
#
# All business logic lives in src/. This file only orchestrates the UI.
# ==============================================================================

from __future__ import annotations

import logging
import traceback
from typing import Any

import chainlit as cl
from langchain_core.runnables import Runnable

# All logic imports come from src/ — app.py has zero business logic.
from src import config
from src.document_parser import load_retriever_for_query
from src.qa_chain import build_rag_chain, run_query

logger = logging.getLogger(__name__)

# ==============================================================================
# Chainlit Session Lifecycle
# ==============================================================================

@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Called once when a user opens a new chat session.

    Initialises:
      1. The Qdrant-backed retriever (connects to existing on-disk collection)
      2. The full RAG chain (LLM + retriever + prompt)

    Both objects are stored in cl.user_session so they persist across
    multiple messages in the same conversation without re-initialising.

    If the Qdrant collection doesn't exist (ingest.py hasn't been run),
    we display a helpful error message rather than crashing.
    """
    # --- Welcome message -------------------------------------------------------
    await cl.Message(
        content=(
            f"## 📊 {config.APP_NAME}: {config.APP_TAGLINE}\n\n"
            "Welcome. I have access to institutional financial filings.\n"
            "Ask me about revenues, risks, guidance, or any specific data "
            "from the indexed documents.\n\n"
            "**Examples:**\n"
            "- *What was Tesla's total revenue and gross margin in 2025?*\n"
            "- *What risk factors related to supply chain are mentioned?*\n"
            "- *How did cash and equivalents change year over year?*\n\n"
            "---\n"
            "*Initialising retrieval engine…*"
        ),
        author=config.APP_NAME,
    ).send()

    # --- Build retriever -------------------------------------------------------
    try:
        retriever = await cl.make_async(load_retriever_for_query)()
    except FileNotFoundError as e:
        await cl.Message(
            content=(
                "⚠️ **Vector database not found.**\n\n"
                f"`{e}`\n\n"
                "Please run the ingestion pipeline first:\n"
                "```bash\npython ingest.py\n```\n"
                "Then restart the app."
            ),
            author=config.APP_NAME,
        ).send()
        return
    except EnvironmentError as e:
        await cl.Message(
            content=(
                "⚠️ **Configuration error:**\n\n"
                f"`{e}`\n\n"
                "Check your `.env` file and ensure all required API keys are set."
            ),
            author=config.APP_NAME,
        ).send()
        return
    except Exception as e:
        await cl.Message(
            content=(
                "❌ **Unexpected initialisation error:**\n\n"
                f"```\n{traceback.format_exc()}\n```"
            ),
            author=config.APP_NAME,
        ).send()
        logger.exception("Failed to initialise retriever")
        return

    # --- Build RAG chain -------------------------------------------------------
    try:
        rag_chain: Runnable = await cl.make_async(build_rag_chain)(retriever)
    except EnvironmentError as e:
        await cl.Message(
            content=(
                "⚠️ **LLM configuration error:**\n\n"
                f"`{e}`\n\n"
                "Please set `GROQ_API_KEY` in your `.env` file."
            ),
            author=config.APP_NAME,
        ).send()
        return

    # Store in session for reuse across messages
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("retriever", retriever)

    await cl.Message(
        content="✅ **Ready.** Ask me anything about the indexed filings.",
        author=config.APP_NAME,
    ).send()


# ==============================================================================
# Message Handler
# ==============================================================================

@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Called on every user message after on_chat_start.

    Execution flow:
      1. Retrieve the RAG chain from the session.
      2. Show a "thinking" step while the chain runs.
      3. Invoke run_query() synchronously (wrapped async).
      4. Stream the answer text into a Chainlit Message.
      5. Build cl.Text source citation elements for the sidebar.
      6. Send the final message with embedded footnotes + sidebar elements.

    Source Citations Design:
      Each retrieved parent chunk becomes a cl.Text element. In Chainlit,
      referencing a cl.Text element's name inside the message content
      (e.g.  "[1]" referencing element name "Source 1") creates a clickable
      footnote. Clicking it opens the full text in the sidebar.
    """
    # --- Guard: session must be initialised -----------------------------------
    rag_chain: Runnable | None = cl.user_session.get("rag_chain")
    if rag_chain is None:
        await cl.Message(
            content=(
                "⚠️ Session not initialised. "
                "Please refresh the page to restart."
            ),
            author=config.APP_NAME,
        ).send()
        return

    user_query: str = message.content.strip()
    if not user_query:
        return

    # --- Thinking step --------------------------------------------------------
    # cl.Step creates a collapsible "thinking" block in the UI.
    # Users can expand it to see which documents were retrieved and why.
    async with cl.Step(name="🔍 Retrieving from knowledge base…", type="retrieval") as step:
        step.input = user_query

        # run_query is synchronous; make_async() runs it in a thread pool
        # to avoid blocking Chainlit's asyncio event loop.
        try:
            result: dict[str, Any] = await cl.make_async(run_query)(rag_chain, user_query)
        except ConnectionError:
            step.output = "❌ Network error connecting to Groq API."
            await cl.Message(
                content=(
                    "❌ **Connection error.** "
                    "Could not reach the Groq API. Please check your network and try again."
                ),
                author=config.APP_NAME,
            ).send()
            return
        except Exception as exc:
            step.output = f"❌ Error: {exc}"
            await cl.Message(
                content=(
                    f"❌ **Error processing your query:**\n\n"
                    f"```\n{traceback.format_exc()}\n```"
                ),
                author=config.APP_NAME,
            ).send()
            logger.exception("Query execution failed")
            return

        sources: list[dict[str, Any]] = result.get("sources", [])
        answer: str                   = result.get("answer", "Insufficient data in the provided filings.")

        # Log what the retrieval step found (visible in Chainlit's step detail)
        step.output = (
            f"Retrieved {len(sources)} source block(s).\n"
            + "\n".join(
                f"  [{s['index']}] {s['metadata']['company']} — "
                f"{s['metadata']['form_type']} {s['metadata']['year']} "
                f"(Page {s['metadata']['page']})"
                for s in sources
            )
        )

    # --- Build source citation elements ----------------------------------------
    # Each cl.Text element is a sidebar card. Its `name` is used as the
    # anchor so we can inline-reference it in the answer message below.
    #
    # Data flow for citations:
    #   CHILD chunk retrieved by Qdrant
    #     → parent lookup in docstore
    #     → parent Document.page_content  (the full ~1500 char block)
    #         → stored in cl.Text.content
    #         → rendered in Chainlit sidebar when user clicks the footnote
    #
    # Note: In query-only mode (no ingest docstore), Qdrant returns the child
    # chunk text itself. We surface that as the citation content — still useful.

    elements: list[cl.Text] = []
    citation_footnotes: list[str] = []   # e.g. ["[1]", "[2]", "[3]"]

    for source in sources:
        idx         = source["index"]
        meta        = source["metadata"]
        content     = source["content"]
        company     = meta.get("company", "Unknown")
        ticker      = meta.get("ticker", "N/A")
        year        = meta.get("year",   "N/A")
        form_type   = meta.get("form_type", "N/A")
        src_file    = meta.get("source", "N/A")
        page        = meta.get("page",   "N/A")
        sector      = meta.get("sector", "N/A")

        # --- Format the sidebar card content ---
        # This is what the user sees when they click a citation footnote.
        sidebar_content = (
            f"**{company} ({ticker})**\n"
            f"*{form_type} · FY{year} · {sector}*\n"
            f"*Source file:* `{src_file}`  |  *Page:* {page}\n\n"
            "---\n\n"
            f"{content}"
        )

        element_name = f"Source {idx}"
        text_element = cl.Text(
            name=element_name,
            content=sidebar_content,
            display="side",           # "side" → opens in the sidebar panel
        )
        elements.append(text_element)

        # Build the inline footnote marker that we'll append to the answer
        footnote = f"[{element_name}]"
        citation_footnotes.append(footnote)

    # --- Compose the final answer message -------------------------------------
    # We append a "Sources" section at the bottom of the answer. Each
    # entry is wrapped in backticks which Chainlit renders as clickable
    # sidebar references when they match a cl.Text element's name exactly.
    #
    # Example rendered output:
    #   Tesla's total revenue for FY2025 was $104.2 billion…
    #
    #   ---
    #   **Retrieved Sources**
    #   `Source 1` · Tesla, Inc. — 10-K 2025 (Page 47)
    #   `Source 2` · Tesla, Inc. — 10-K 2025 (Page 51)

    sources_section = ""
    if sources:
        source_lines = []
        for source in sources:
            idx       = source["index"]
            meta      = source["metadata"]
            company   = meta.get("company",   "Unknown")
            form_type = meta.get("form_type", "N/A")
            year      = meta.get("year",      "N/A")
            page      = meta.get("page",      "N/A")
            preview   = source.get("preview", "")

            source_lines.append(
                f"`Source {idx}` · **{company}** — {form_type} {year} "
                f"(Page {page})\n"
                f"  > {preview}"
            )

        sources_section = (
            "\n\n---\n"
            "**📎 Retrieved Source Blocks** *(click to expand in sidebar)*\n\n"
            + "\n\n".join(source_lines)
        )

    # Handle the "Insufficient data" case with a styled notice
    if answer.strip() == "Insufficient data in the provided filings.":
        final_content = (
            "⚠️ **Insufficient data in the provided filings.**\n\n"
            "The indexed documents do not contain explicit information to answer "
            "this query. Try rephrasing, or ensure the relevant filings have been "
            "ingested via `python ingest.py`."
        )
        if sources_section:
            final_content += sources_section
    else:
        final_content = answer + sources_section

    # --- Send the message with attached source elements -----------------------
    await cl.Message(
        content=final_content,
        elements=elements,       # Registers the sidebar citation cards
        author=config.APP_NAME,
    ).send()


# ==============================================================================
# App Settings (Chainlit config via Python)
# ==============================================================================

@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    """
    Called when the user changes settings in the Chainlit UI panel.
    Currently a no-op — reserved for future model/parameter toggles.
    """
    logger.info(f"Settings updated: {settings}")
