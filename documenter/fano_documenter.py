#!/usr/bin/env python3
"""
Fano Documenter - Autonomous Document Growth System

Usage:
    python fano_documenter.py start       # Start documenter loop
    python fano_documenter.py status      # Show current status
    python fano_documenter.py snapshot    # Create manual snapshot
    python fano_documenter.py help        # Show this help
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file (check both documenter/ and root)
for _env_path in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if _env_path.exists():
        with open(_env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key, value)  # Don't override existing

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# FANO_ROOT for resolving relative paths in config
FANO_ROOT = Path(__file__).resolve().parent.parent

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.logging import get_logger

log = get_logger("documenter", "fano_documenter")
console = Console(force_terminal=True, legacy_windows=False)


def handle_asyncio_exception(loop, context):
    """Handle asyncio exceptions, suppressing Windows connection reset noise."""
    exception = context.get("exception")
    if exception:
        # Suppress Windows connection reset errors (common with proactor event loop)
        if isinstance(exception, ConnectionResetError):
            return
        if "ConnectionResetError" in str(exception):
            return
    # Log other exceptions normally
    msg = context.get("message", "Unhandled exception in event loop")
    log.error("asyncio.error", message=msg)


def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ███████╗ █████╗ ███╗   ██╗ ██████╗                     ║
    ║   ██╔════╝██╔══██╗████╗  ██║██╔═══██╗                    ║
    ║   █████╗  ███████║██╔██╗ ██║██║   ██║                    ║
    ║   ██╔══╝  ██╔══██║██║╚██╗██║██║   ██║                    ║
    ║   ██║     ██║  ██║██║ ╚████║╚██████╔╝                    ║
    ║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝                     ║
    ║                                                           ║
    ║   D O C U M E N T E R                                    ║
    ║                                                           ║
    ║   Growing the Living Mathematical Document               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold magenta")


def cmd_start():
    """Start the documenter loop."""
    from documenter.main import Documenter

    console.print("\n[bold]Starting documenter loop...[/bold]")
    console.print("Press Ctrl+C to stop gracefully.\n")

    documenter = Documenter()

    try:
        # On Windows, set up custom exception handler to suppress connection reset noise
        if sys.platform == "win32":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.set_exception_handler(handle_asyncio_exception)
            loop.run_until_complete(documenter.run())
        else:
            asyncio.run(documenter.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        # Run cleanup in a new event loop since the previous one was interrupted
        try:
            asyncio.run(documenter.cleanup())
        except Exception:
            pass
        console.print("[yellow]Stopped.[/yellow]")
    except RuntimeError as e:
        error_msg = str(e)
        if "backends" in error_msg.lower():
            console.print(Panel(
                f"[red bold]Configuration Error[/red bold]\n\n"
                f"{error_msg}\n\n"
                f"[yellow]To fix this:[/yellow]\n"
                f"  1. Set API keys in .env file:\n"
                f"     ANTHROPIC_API_KEY=sk-...\n"
                f"     OPENROUTER_API_KEY=sk-...\n"
                f"  2. Or start the pool service:\n"
                f"     python pool/src/api.py",
                title="Startup Failed",
                border_style="red",
            ))
        else:
            console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()


def cmd_status():
    """Show current documenter status."""
    from documenter.document import Document
    from documenter.concepts import ConceptTracker
    from documenter.opportunities import OpportunityFinder
    from documenter.snapshots import SnapshotManager

    import yaml

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    doc_config = config.get("documenter", {}).get("document", {})
    doc_path = Path(doc_config.get("path", "document/main.md"))

    # Resolve relative to fano root
    if not doc_path.is_absolute():
        doc_path = FANO_ROOT / doc_path

    # Load document
    document = Document(doc_path)
    if not document.load():
        console.print(Panel("[yellow]Document not found. Run 'start' to create.[/yellow]"))
        return

    # Document info
    console.print(Panel(f"[bold]Document:[/bold] {doc_path}"))

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Sections", str(len(document.sections)))
    table.add_row("Total Length", f"{len(document.content):,} characters")
    table.add_row("Lines", str(len(document.content.split('\\n'))))

    console.print(table)
    console.print()

    # Concepts
    concept_tracker = ConceptTracker(document)
    established = concept_tracker.get_established_concepts()

    console.print(Panel(f"[bold]Established Concepts:[/bold] {len(established)}"))
    if established:
        concepts_list = list(established)[:10]
        for c in concepts_list:
            console.print(f"  - {c}")
        if len(established) > 10:
            console.print(f"  ... and {len(established) - 10} more")
    console.print()

    # Opportunities (blessed insights)
    inputs_config = config.get("documenter", {}).get("inputs", {})
    blessed_dir = Path(inputs_config.get("blessed_insights_dir", "blessed_insights"))
    if not blessed_dir.is_absolute():
        blessed_dir = FANO_ROOT / blessed_dir

    opportunity_finder = OpportunityFinder(
        document,
        concept_tracker,
        blessed_dir,
        max_disputes=3,
    )
    pending = opportunity_finder.get_pending_count()

    console.print(Panel(f"[bold]Pending Insights:[/bold] {pending}"))
    console.print()

    # Snapshots
    archive_dir = Path(doc_config.get("archive_dir", "document/archive"))
    if not archive_dir.is_absolute():
        archive_dir = FANO_ROOT / archive_dir

    from datetime import time as dt_time
    snapshot_time_str = doc_config.get("snapshot_time", "00:00")
    hour, minute = map(int, snapshot_time_str.split(":"))

    snapshot_manager = SnapshotManager(
        document,
        archive_dir,
        dt_time(hour, minute),
    )
    snapshots = snapshot_manager.list_snapshots()

    console.print(Panel(f"[bold]Snapshots:[/bold] {len(snapshots)}"))
    if snapshots:
        latest = snapshots[-1]
        console.print(f"  Latest: {latest.date().isoformat()}")
    console.print()

    # Unresolved comments
    comments = document.find_unresolved_comments()
    if comments:
        console.print(Panel(f"[bold yellow]Unresolved Comments:[/bold yellow] {len(comments)}"))
        for comment_text, line_num in comments[:3]:
            console.print(f"  Line {line_num}: {comment_text[:60]}...")
        if len(comments) > 3:
            console.print(f"  ... and {len(comments) - 3} more")


def cmd_snapshot():
    """Create a manual snapshot."""
    from documenter.document import Document
    from documenter.snapshots import SnapshotManager

    import yaml

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    doc_config = config.get("documenter", {}).get("document", {})
    doc_path = Path(doc_config.get("path", "document/main.md"))
    archive_dir = Path(doc_config.get("archive_dir", "document/archive"))

    # Resolve relative paths
    if not doc_path.is_absolute():
        doc_path = FANO_ROOT / doc_path
    if not archive_dir.is_absolute():
        archive_dir = FANO_ROOT / archive_dir

    # Load document
    document = Document(doc_path)
    if not document.load():
        console.print("[red]Document not found.[/red]")
        return

    from datetime import time as dt_time
    snapshot_manager = SnapshotManager(document, archive_dir, dt_time(0, 0))

    if snapshot_manager.create_snapshot():
        console.print(f"[green]Snapshot created at {archive_dir}[/green]")
    else:
        console.print("[red]Failed to create snapshot.[/red]")


def cmd_help():
    """Show help."""
    console.print(__doc__)


COMMANDS = {
    "start": cmd_start,
    "status": cmd_status,
    "snapshot": cmd_snapshot,
    "help": cmd_help,
    "--help": cmd_help,
    "-h": cmd_help,
}


def main():
    print_banner()

    if len(sys.argv) < 2:
        cmd_help()
        return

    cmd = sys.argv[1].lower()

    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        cmd_help()


if __name__ == "__main__":
    main()
