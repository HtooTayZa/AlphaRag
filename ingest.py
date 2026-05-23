#!/usr/bin/env python3
# ingest.py
# ==============================================================================
# AlphaRAG: Document Ingestion Pipeline (CLI Entry Point)
# ==============================================================================
# Run this script ONCE (or whenever new filings are added) to:
#   1. Load all PDFs from data/raw_pdfs/
#   2. Split into Parent and Child chunks
#   3. Embed child chunks with BAAI/bge-m3
#   4. Persist child embeddings to Qdrant (data/qdrant_db/)
#   5. Store parent chunks in an InMemoryStore (ephemeral demo)
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
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="AlphaRAG: Ingest financial PDFs into the Qdrant vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py
  python ingest.py --pdf-dir /mnt/data/sec_filings
  python ingest.py --verify --query "What was Tesla's revenue in 2025?"
        """,
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing PDF filings. Defaults to data/raw_pdfs/.",
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
        default="What was the total revenue and key risk factors mentioned?",
        help="Verification query to run (only used with --verify).",
    )
    return parser.parse_args()


def print_banner() -> None:
    """Print a styled startup banner to the terminal."""
    banner = (
        "[bold cyan]AlphaRAG[/bold cyan]: Institutional Knowledge Extractor\n"
        "[dim]Document Ingestion Pipeline v1.0.0[/dim]"
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

    Args:
        retriever: The ParentDocumentRetriever returned by ingest_documents().
        query:     A test query string.
    """
    console.rule("[bold yellow]Verification Query[/bold yellow]")
    console.print(f"[bold]Query:[/bold] {query}\n")

    try:
        # get_relevant_documents performs: embed → ANN → parent lookup
        start = time.perf_counter()
        docs = retriever.get_relevant_documents(query)  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - start

        if not docs:
            console.print("[yellow]⚠ No documents retrieved. Check that ingestion succeeded.[/yellow]")
            return

        console.print(f"[green]✅ Retrieved {len(docs)} parent chunk(s) in {elapsed:.2f}s[/green]\n")

        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            company  = meta.get("company", "Unknown")
            source   = meta.get("source", "N/A")
            page     = meta.get("page_number", meta.get("page", "N/A"))
            preview  = doc.page_content[:300].strip()

            console.print(
                Panel(
                    f"[dim]Company:[/dim] {company}  |  "
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
    Create a minimal demo PDF if the raw_pdfs directory is empty.

    This uses the built-in reportlab-free approach via fpdf2 or falls back
    to a plain-text .txt file as a stand-in if no PDF libraries are available.
    This ensures first-time users can test the pipeline without real SEC filings.
    """
    try:
        # Try to create a real PDF using fpdf2 (lightweight, no reportlab dep)
        from fpdf import FPDF  # type: ignore

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.set_title("Tesla 2025 10-K (Demo)")

        demo_text = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K — ANNUAL REPORT

TESLA, INC. — Fiscal Year Ended December 31, 2025

ITEM 1. BUSINESS

Tesla, Inc. ("Tesla," the "Company," "we," "us" or "our") was incorporated in
the State of Delaware on July 1, 2003. We design, develop, manufacture, lease
and sell electric vehicles, energy generation and storage systems, and offer
services related to our sustainable energy products.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Total Revenues
For the year ended December 31, 2025, total revenues were $104.2 billion,
representing a 15% increase from $90.7 billion in fiscal year 2024. Automotive
revenues contributed $91.8 billion, energy generation and storage revenues were
$9.6 billion, and services and other revenues were $2.8 billion.

Gross Profit
Total gross profit for fiscal year 2025 was $18.9 billion, representing a gross
margin of 18.1%, compared to 17.8% in fiscal year 2024.

ITEM 1A. RISK FACTORS

We face risks related to global supply chain disruptions, increased competition
in the EV market from both traditional OEMs and new entrants, potential adverse
changes in government regulations and incentives for electric vehicles, and
foreign currency fluctuation risks given our international manufacturing
footprint in Germany and China.

ITEM 8. FINANCIAL STATEMENTS

Consolidated Statement of Operations (in millions):
  Total revenues:            $104,200
  Cost of revenues:          $85,280
  Gross profit:              $18,920
  Operating expenses:        $11,340
  Income from operations:    $7,580
  Net income:                $6,890

The company held $28.4 billion in cash and cash equivalents as of December 31,
2025, with long-term debt of $5.2 billion.
        """.strip()

        for line in demo_text.split("\n"):
            if line.strip():
                pdf.multi_cell(0, 8, line.strip())
            else:
                pdf.ln(4)

        demo_path = pdf_dir / "tsla_2025_10k.pdf"
        pdf.output(str(demo_path))
        console.print(f"[green]📄 Demo PDF created: {demo_path}[/green]")

    except ImportError:
        # fpdf2 not installed — create a text file as a fallback notice
        notice_path = pdf_dir / "README_ADD_PDFS_HERE.txt"
        notice_path.write_text(
            "Place your financial PDF filings in this directory.\n"
            "Filenames like 'tsla_2025_10k.pdf' are auto-tagged with metadata.\n"
            "Install fpdf2 (pip install fpdf2) to auto-generate a demo PDF.\n"
        )
        console.print(
            f"[yellow]⚠ No PDFs found and fpdf2 not installed.[/yellow]\n"
            f"  Place PDF filings in: {pdf_dir}\n"
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
            f"   Creating a demo PDF for you…"
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
        task = progress.add_task("Ingesting documents…", total=None)

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
