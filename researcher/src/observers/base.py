"""
Base observer class for watching module activity.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class BaseObserver(ABC):
    """Base class for observing module activity."""

    def __init__(self, data_path: Path):
        """
        Initialize observer.

        Args:
            data_path: Path to the module's data directory
        """
        self.data_path = data_path
        self.last_check: Optional[datetime] = None
        self._file_mtimes: dict[str, float] = {}

    @abstractmethod
    def get_recent_activity(self) -> list[dict]:
        """
        Get recent activity from the module.

        Returns:
            List of activity records (dicts with type, content, timestamp)
        """
        pass

    @abstractmethod
    def extract_concepts(self, activity: dict) -> list[str]:
        """Extract concept names from an activity record."""
        pass

    @abstractmethod
    def extract_numbers(self, activity: dict) -> list[int]:
        """Extract significant numbers from an activity record."""
        pass

    def _get_changed_files(self, directory: Path, pattern: str = "*.json") -> list[Path]:
        """
        Get files that have changed since last check.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files

        Returns:
            List of changed file paths
        """
        if not directory.exists():
            return []

        changed = []
        for filepath in directory.glob(pattern):
            mtime = filepath.stat().st_mtime
            key = str(filepath)

            if key not in self._file_mtimes or self._file_mtimes[key] < mtime:
                changed.append(filepath)
                self._file_mtimes[key] = mtime

        return changed

    def _load_json_file(self, filepath: Path) -> Optional[dict]:
        """Safely load a JSON file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            return None

    def _extract_numbers_from_text(self, text: str) -> list[int]:
        """
        Extract numbers from text.

        Looks for standalone numbers and common patterns like "7 chakras".
        """
        import re

        numbers = []

        # Find standalone numbers
        for match in re.finditer(r'\b(\d+)\b', text):
            try:
                num = int(match.group(1))
                # Only include numbers that might be significant (not years, etc.)
                if 1 <= num <= 10000 and num not in [19, 20, 21]:  # Skip century numbers
                    numbers.append(num)
            except ValueError:
                pass

        return list(set(numbers))
