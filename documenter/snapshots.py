"""
Snapshot management - daily archiving of the document.
"""

import sys
from datetime import datetime, time
from pathlib import Path
from typing import Optional

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger
from .document import Document

log = get_logger("documenter", "snapshots")


class SnapshotManager:
    """
    Manages daily snapshots of the document.

    Creates append-only archive copies for history and rollback.
    """

    def __init__(
        self,
        document: Document,
        archive_dir: Path,
        snapshot_time: time = time(0, 0),  # Midnight
    ):
        """
        Initialize snapshot manager.

        Args:
            document: The document to snapshot
            archive_dir: Directory for archive files
            snapshot_time: Time of day for snapshots (default midnight)
        """
        self.document = document
        self.archive_dir = Path(archive_dir)
        self.snapshot_time = snapshot_time
        self._last_snapshot_date: Optional[datetime] = None

    def should_create_snapshot(self) -> bool:
        """Check if a snapshot should be created."""
        now = datetime.now()
        today = now.date()

        # Check if we already created a snapshot today
        if self._last_snapshot_date and self._last_snapshot_date.date() == today:
            return False

        # Check if we're past the snapshot time
        current_time = now.time()
        if current_time >= self.snapshot_time:
            # Check if today's snapshot exists
            snapshot_path = self._get_snapshot_path(today)
            if not snapshot_path.exists():
                return True

        return False

    def create_snapshot(self) -> bool:
        """Create a snapshot of the current document."""
        today = datetime.now().date()
        snapshot_path = self._get_snapshot_path(today)

        try:
            # Create archive directory if needed
            self.archive_dir.mkdir(parents=True, exist_ok=True)

            # Copy document to archive
            content = self.document.content
            snapshot_path.write_text(content, encoding="utf-8")

            self._last_snapshot_date = datetime.now()

            log.info(
                "documenter.snapshot.created",
                path=str(snapshot_path),
                content_length=len(content),
            )
            return True

        except Exception as e:
            log.exception(e, "documenter.snapshot.error", {"path": str(snapshot_path)})
            return False

    def _get_snapshot_path(self, date) -> Path:
        """Get the path for a snapshot on a given date."""
        filename = f"{date.isoformat()}.md"
        return self.archive_dir / filename

    def get_snapshot(self, date) -> Optional[str]:
        """Get the content of a snapshot for a given date."""
        snapshot_path = self._get_snapshot_path(date)

        if not snapshot_path.exists():
            return None

        try:
            return snapshot_path.read_text(encoding="utf-8")
        except Exception as e:
            log.warning(
                "documenter.snapshot.read_error",
                path=str(snapshot_path),
                error=str(e),
            )
            return None

    def list_snapshots(self) -> list[datetime]:
        """List all available snapshot dates."""
        if not self.archive_dir.exists():
            return []

        snapshots = []
        for path in self.archive_dir.glob("*.md"):
            try:
                date_str = path.stem  # e.g., "2026-01-10"
                date = datetime.fromisoformat(date_str)
                snapshots.append(date)
            except ValueError:
                continue

        return sorted(snapshots)

    def get_latest_snapshot(self) -> Optional[tuple[datetime, str]]:
        """Get the most recent snapshot."""
        snapshots = self.list_snapshots()

        if not snapshots:
            return None

        latest = snapshots[-1]
        content = self.get_snapshot(latest.date())

        if content:
            return latest, content

        return None

    def check_and_snapshot(self) -> bool:
        """Check if snapshot needed and create if so."""
        if self.should_create_snapshot():
            return self.create_snapshot()
        return False
