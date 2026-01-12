"""
Backlog processing for unextracted exploration threads.

Handles finding and processing threads that haven't had their
insights extracted yet.
"""

from pathlib import Path
from typing import Optional

from shared.logging import get_logger

log = get_logger("explorer", "backlog_processor")


def find_unprocessed_threads(data_dir: Path) -> list:
    """
    Find exploration threads that haven't been processed yet.

    Args:
        data_dir: Path to the explorer data directory

    Returns:
        List of ExplorationThread objects that need processing
    """
    from explorer.src.models.thread import ExplorationThread

    threads_dir = data_dir / "explorations"

    if not threads_dir.exists():
        return []

    unprocessed = []
    for filepath in threads_dir.glob("*.json"):
        try:
            thread = ExplorationThread.load(filepath)
            if thread.status in ["CHUNK_READY", "ARCHIVED"]:
                if not getattr(thread, "chunks_extracted", False):
                    unprocessed.append(thread)
        except Exception as e:
            log.warning(
                "backlog.thread_load_failed",
                path=str(filepath),
                error=str(e),
            )

    log.info("backlog.threads_found", count=len(unprocessed))
    return unprocessed


async def process_backlog(
    data_dir: Path,
    on_progress: Optional[callable] = None,
) -> int:
    """
    Process all unextracted threads in the backlog.

    Args:
        data_dir: Path to the explorer data directory
        on_progress: Optional callback for progress updates

    Returns:
        Number of threads processed
    """
    from explorer.src.orchestrator import Orchestrator

    unprocessed = find_unprocessed_threads(data_dir)

    if not unprocessed:
        log.info("backlog.empty")
        return 0

    log.info("backlog.starting", thread_count=len(unprocessed))

    orchestrator = Orchestrator()

    await orchestrator._connect_models()
    try:
        await orchestrator.process_backlog()
    finally:
        await orchestrator._disconnect_models()

    log.info("backlog.completed", thread_count=len(unprocessed))
    return len(unprocessed)
