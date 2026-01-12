#!/usr/bin/env python3
"""
Fano Researcher - External Source Discovery and Analysis

Usage:
    python researcher/main.py start      # Start research loop
    python researcher/main.py status     # Show current status
    python researcher/main.py stats      # Show database statistics
    python researcher/main.py query      # Interactive query mode
    python researcher/main.py cleanup    # Clean expired cache
"""

import sys
import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(force_terminal=True)


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
    ║   R E S E A R C H E R                                    ║
    ║                                                           ║
    ║   Sanskrit Texts · Sacred Numbers · Pattern Discovery    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold magenta")


def cmd_start():
    """Start the research loop."""
    from researcher.src import Orchestrator

    console.print("\n[bold]Starting researcher loop...[/bold]")
    console.print("Press Ctrl+C to stop gracefully.\n")

    orchestrator = Orchestrator()

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        orchestrator.stop()
        console.print("[yellow]Stopped.[/yellow]")


def cmd_status():
    """Show current status."""
    from researcher.src import Orchestrator

    orchestrator = Orchestrator()
    status = orchestrator.get_status()

    console.print(Panel("[bold]Researcher Status[/bold]"))

    # Database stats
    db_stats = status.get("database", {})
    table = Table(title="Database")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Sources", str(db_stats.get("sources", 0)))
    table.add_row("Trusted Sources", str(db_stats.get("trusted_sources", 0)))
    table.add_row("Findings", str(db_stats.get("findings", 0)))
    table.add_row("Concepts", str(db_stats.get("concepts", 0)))
    table.add_row("Number Mentions", str(db_stats.get("number_mentions", 0)))
    table.add_row("Cross-References", str(db_stats.get("cross_references", 0)))

    console.print(table)

    # Top numbers
    top_numbers = db_stats.get("top_numbers", {})
    if top_numbers:
        console.print("\n[bold]Top Numbers Found:[/bold]")
        for num, count in list(top_numbers.items())[:5]:
            console.print(f"  {num}: {count} occurrences")

    # Cache stats
    cache_stats = status.get("cache", {})
    console.print(f"\n[bold]Cache:[/bold] {cache_stats.get('entry_count', 0)} entries, "
                  f"{cache_stats.get('total_size_mb', 0)} MB")


def cmd_stats():
    """Show detailed statistics."""
    from researcher.src import Orchestrator
    from researcher.src.analysis import CrossReferenceDetector

    orchestrator = Orchestrator()
    stats = orchestrator.db.get_statistics()

    console.print(Panel("[bold]Research Statistics[/bold]"))

    # Overview
    console.print(f"[cyan]Total Sources:[/cyan] {stats.get('sources', 0)}")
    console.print(f"[cyan]Total Findings:[/cyan] {stats.get('findings', 0)}")
    console.print(f"[cyan]Cross-References:[/cyan] {stats.get('cross_references', 0)}")

    # Top numbers
    console.print("\n[bold]Most Frequent Numbers:[/bold]")
    top_numbers = stats.get("top_numbers", {})
    for num, count in list(top_numbers.items())[:10]:
        console.print(f"  {num}: {count} occurrences")

    # Cross-domain patterns
    console.print("\n[bold]Cross-Domain Patterns:[/bold]")
    xref = CrossReferenceDetector(orchestrator.db)
    patterns = xref.find_number_patterns()

    for pattern in patterns[:5]:
        domains = list(pattern["domains"].keys())
        console.print(f"  Number {pattern['number']}: found in {', '.join(domains)}")


def cmd_query():
    """Interactive query mode."""
    from researcher.src import ResearcherAPI

    api = ResearcherAPI(Path(__file__).parent / "data")

    console.print(Panel("[bold]Interactive Query Mode[/bold]"))
    console.print("Commands:")
    console.print("  number <N>     - Look up number N")
    console.print("  concept <name> - Look up concept")
    console.print("  search <query> - Search findings")
    console.print("  quit           - Exit")
    console.print()

    while True:
        try:
            cmd = console.input("[bold cyan]researcher>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()

        if action == "quit":
            break

        elif action == "number" and len(parts) > 1:
            try:
                num = int(parts[1])
                info = api.get_number_info(num)
                console.print(f"\n[bold]Number {num}:[/bold]")
                console.print(f"  Occurrences: {info['total_occurrences']}")
                console.print(f"  Domains: {list(info['domains'].keys())}")
                if info.get("pattern_summary"):
                    console.print(f"\n{info['pattern_summary']}")
            except ValueError:
                console.print("[red]Invalid number[/red]")

        elif action == "concept" and len(parts) > 1:
            name = parts[1]
            info = api.get_concept_info(name)
            if info:
                console.print(f"\n[bold]{info['concept']['display_name']}[/bold]")
                console.print(f"  Domain: {info['concept']['domain']}")
                console.print(f"  Occurrences: {info['concept']['occurrence_count']}")
                if info['concept']['aliases']:
                    console.print(f"  Aliases: {', '.join(info['concept']['aliases'])}")
            else:
                console.print("[yellow]Concept not found[/yellow]")

        elif action == "search" and len(parts) > 1:
            query = parts[1]
            # Extract potential concept from query
            findings = api.query(limit=5)
            console.print(f"\n[bold]Findings:[/bold]")
            for f in findings:
                console.print(f"  - {f.summary[:60]}...")

        else:
            console.print("[yellow]Unknown command. Try: number, concept, search, quit[/yellow]")

    console.print("\n[yellow]Goodbye![/yellow]")


def cmd_cleanup():
    """Clean expired cache entries."""
    from researcher.src.sources import ContentCache
    from pathlib import Path

    cache = ContentCache(
        cache_dir=Path(__file__).parent / "data" / "cache",
        config_path=Path(__file__).parent / "config" / "settings.yaml"
    )

    removed = cache.cleanup_expired()
    console.print(f"[green]Removed {removed} expired cache entries[/green]")


def cmd_help():
    """Show help."""
    console.print(__doc__)


COMMANDS = {
    "start": cmd_start,
    "status": cmd_status,
    "stats": cmd_stats,
    "query": cmd_query,
    "cleanup": cmd_cleanup,
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
