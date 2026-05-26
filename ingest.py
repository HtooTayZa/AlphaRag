#!/usr/bin/env python3
# ingest.py
# ==============================================================================
# AlphaRAG: Document Ingestion Pipeline (CLI Entry Point)
# ==============================================================================
# Run this script ONCE (or whenever new documents are added) to:
#   1. Load all PDFs from data/raw_pdfs/
#   2. Use Groq/Pydantic to automatically extract document metadata (Title, Author, etc.)
#   3. Split into Parent and Child chunks
#   4. Embed child chunks and store in Qdrant (data/qdrant_db/)
#   5. Store parent chunks locally via EncoderBackedStore (data/local_docstore/)
#
# Usage:
#   python ingest.py                   # Process data/raw_pdfs/ (default)
#   python ingest.py --pdf-dir /path   # Override PDF directory
#   python ingest.py --verify          # Also run a test query after ingestion
# ==============================================================================

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Logging — use Rich for beautiful terminal output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("alpharag.ingest")

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ingestion script."""
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="AlphaRAG: Ingest PDFs into the Qdrant vector database and local docstore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py
  python ingest.py --pdf-dir /mnt/data/my_documents
  python ingest.py --verify --query "Summarize the main topic of the documents."
        """,
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing PDF files. Defaults to data/raw_pdfs/.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="After ingestion, run a test retrieval query to verify the pipeline.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Summarize the main topic of the documents.",
        help="Verification query to run (only used with --verify).",
    )
    return parser.parse_args()


def print_banner() -> None:
    """Print a styled startup banner to the terminal."""
    from src import config
    
    banner = (
        f"[bold cyan]{config.APP_NAME}[/bold cyan]: {config.APP_TAGLINE}\n"
        f"[dim]Document Ingestion Pipeline v{config.APP_VERSION}[/dim]"
    )
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))


def print_config_summary() -> None:
    """Display the active configuration as a Rich table."""
    from src import config

    table = Table(title="Active Configuration", border_style="dim", show_header=True)
    table.add_column("Parameter", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("LLM Model",           config.LLM_MODEL_NAME)
    table.add_row("Embedding Model",     config.EMBEDDING_MODEL_NAME)
    table.add_row("Embedding Device",    config.EMBEDDING_DEVICE)
    table.add_row("Qdrant Collection",   config.COLLECTION_NAME)
    table.add_row("Qdrant Path",         str(config.QDRANT_PATH))
    table.add_row("Parent Chunk Size",   f"{config.PARENT_CHUNK_SIZE} chars")
    table.add_row("Parent Chunk Overlap",f"{config.PARENT_CHUNK_OVERLAP} chars")
    table.add_row("Child Chunk Size",    f"{config.CHILD_CHUNK_SIZE} chars")
    table.add_row("Child Chunk Overlap", f"{config.CHILD_CHUNK_OVERLAP} chars")
    table.add_row("Top-K Retrieval",     str(config.TOP_K_RETRIEVAL))

    console.print(table)
    console.print()


def run_verification(retriever: object, query: str) -> None:
    """
    Execute a test retrieval against the freshly populated vector store.
    This proves that vectors in Qdrant successfully link back to parent texts in the local docstore.

    Args:
        retriever: The ParentDocumentRetriever returned by ingest_documents().
        query:     A test query string.
    """
    console.rule("[bold yellow]Verification Query[/bold yellow]")
    console.print(f"[bold]Query:[/bold] {query}\n")

    try:
        # get_relevant_documents performs: embed → ANN → parent lookup
        start = time.perf_counter()
        docs = retriever.get_relevant_documents(query) 
        elapsed = time.perf_counter() - start

        if not docs:
            console.print("[yellow]⚠ No documents retrieved. Check that ingestion succeeded.[/yellow]")
            return

        console.print(f"[green]✅ Retrieved {len(docs)} parent chunk(s) in {elapsed:.2f}s[/green]\n")

        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            
            # Use the new Pydantic metadata keys
            title    = meta.get("title", "Unknown Title")
            author   = meta.get("author_or_company", "Unknown")
            source   = meta.get("source", "N/A")
            page     = meta.get("page_number", meta.get("page", "N/A"))
            preview  = doc.page_content[:300].strip()

            console.print(
                Panel(
                    f"[dim]Title:[/dim] {title}  |  "
                    f"[dim]Author:[/dim] {author}  |  "
                    f"[dim]Source:[/dim] {source}  |  "
                    f"[dim]Page:[/dim] {page}\n\n"
                    f"{preview}…",
                    title=f"[cyan]Source #{i}[/cyan]",
                    border_style="dim",
                )
            )

    except Exception as exc:
        console.print(f"[red]❌ Verification failed: {exc}[/red]")
        logger.exception("Verification error")


def create_demo_pdf(pdf_dir: Path) -> None:
    """
    Create a minimal, generic demo PDF if the raw_pdfs directory is empty.
    This ensures first-time users can test the generalized pipeline without providing their own files.
    """
    try:
        # Try to create a real PDF using fpdf2
        from fpdf import FPDF  # type: ignore

        pdf = FPDF()
        pdf.add_page()
        
        # Explicitly set safe margins to prevent horizontal space errors
        pdf.set_margins(left=15, top=15, right=15)
        pdf.set_font("Helvetica", size=12)
        pdf.set_title("Project Alpha Research Report (Demo)")

        # A generic, domain-agnostic research report
        demo_text = """
PROJECT ALPHA: ANNUAL RESEARCH REPORT
Institute of Advanced Technology
Date: May 24, 2026

1. INTRODUCTION
Project Alpha explores the intersection of artificial intelligence and sustainable workflows.
This document outlines our findings over the past year. We aim to understand how automation
can improve efficiency across diverse environments.

2. METHODOLOGY
Our research team utilized distributed sensor networks to gather data across 14 testing facilities.
The data was processed using next-generation neural architectures to identify performance gaps
and optimize resource allocation.

3. KEY FINDINGS
- Energy consumption in test facilities dropped by 18% over a 6-month period.
- The automated resource allocation model achieved a 99.2% uptime.
- Team productivity increased significantly as repetitive administrative tasks were 
  offloaded to the intelligent assistant.

4. FUTURE OUTLOOK
For the upcoming year, we plan to scale the model to an additional 50 facilities.
Our secondary goal is to open-source the core algorithms for community peer review.
        """.strip()

        # Let fpdf2 handle the line breaks and cursor position automatically
        pdf.multi_cell(w=0, h=8, text=demo_text)

        demo_path = pdf_dir / "project_alpha_demo.pdf"
        pdf.output(str(demo_path))
        console.print(f"[green]📄 Demo PDF created: {demo_path}[/green]")

    except ImportError:
        # fpdf2 not installed — create a text file as a fallback notice
        notice_path = pdf_dir / "README_ADD_PDFS_HERE.txt"
        notice_path.write_text(
            "Place your PDF documents in this directory.\n"
            "The LLM will automatically extract metadata from the first page.\n"
            "Install fpdf2 (pip install fpdf2) to auto-generate a demo PDF.\n"
        )
        console.print(
            f"[yellow]⚠ No PDFs found and fpdf2 not installed.[/yellow]\n"
            f"  Place PDF documents in: {pdf_dir}\n"
            f"  Or install fpdf2:  pip install fpdf2"
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()

    print_banner()

    # --- Import src modules here (after banner) so errors are readable -------
    try:
        from src import config
        from src.document_parser import ingest_documents
    except EnvironmentError as env_err:
        console.print(f"[red bold]Configuration Error:[/red bold] {env_err}")
        console.print("[dim]Hint: Copy .env.example → .env and fill in your API keys.[/dim]")
        sys.exit(1)

    print_config_summary()

    # --- Resolve PDF directory -----------------------------------------------
    pdf_dir: Path = args.pdf_dir or config.RAW_PDFS_DIR
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(
            f"[yellow]📂 No PDFs found in: {pdf_dir}[/yellow]\n"
            f"   Creating a generic demo PDF for you…"
        )
        create_demo_pdf(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))

    console.print(f"[bold]📁 PDF directory:[/bold] {pdf_dir}")
    console.print(f"[bold]📄 Files to ingest:[/bold] {len(pdf_files)}")
    for f in pdf_files:
        console.print(f"   • {f.name}")
    console.print()

    # --- Run Ingestion Pipeline -----------------------------------------------
    console.rule("[bold cyan]Starting Ingestion[/bold cyan]")
    start_time = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Ingesting documents & extracting metadata…", total=None)

        try:
            retriever, docstore = ingest_documents(pdf_dir=pdf_dir)
            progress.update(task, completed=True, description="[green]Ingestion complete!")
        except FileNotFoundError as e:
            console.print(f"[red]❌ {e}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Ingestion failed: {e}[/red]")
            logger.exception("Ingestion pipeline error")
            sys.exit(1)

    elapsed = time.perf_counter() - start_time

    # --- Summary --------------------------------------------------------------
    console.print()
    console.rule("[bold green]Ingestion Summary[/bold green]")

    summary_table = Table(border_style="green", show_header=False)
    summary_table.add_column("Key",   style="bold white")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Status",         "✅ Success")
    summary_table.add_row("Time Elapsed",   f"{elapsed:.1f}s")
    summary_table.add_row("PDFs Processed", str(len(pdf_files)))
    summary_table.add_row("Qdrant Path",    str(config.QDRANT_PATH))
    summary_table.add_row("Local Docstore", str(config.LOCAL_DOCSTORE_PATH))
    summary_table.add_row("Collection",     config.COLLECTION_NAME)

    console.print(summary_table)

    # --- Optional verification -----------------------------------------------
    if args.verify:
        console.print()
        run_verification(retriever, args.query)

    # --- Next Steps -----------------------------------------------------------
    console.print()
    console.print(
        Panel(
            "[bold]Next step:[/bold] Start the Chainlit UI:\n\n"
            "  [cyan]chainlit run app.py[/cyan]\n\n"
            "Then open [link=http://localhost:8000]http://localhost:8000[/link] in your browser.",
            title="[bold white]🚀 Ready![/bold white]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    main()