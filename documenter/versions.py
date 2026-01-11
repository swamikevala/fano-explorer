"""
Document version management - tracking changes and enabling rollback.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

log = get_logger("documenter", "versions")


@dataclass
class Version:
    """A document version with metadata."""
    version_id: str  # Unique identifier (timestamp-based)
    timestamp: str  # ISO format datetime
    description: str  # What changed
    content_hash: str  # Hash of content for deduplication
    sections_count: int  # Number of sections
    content_length: int  # Character count

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Version":
        return cls(**data)


class VersionManager:
    """
    Manages document version history.

    Stores versions as separate files in a history directory,
    with metadata tracked in a versions.json index.
    """

    def __init__(self, document_path: Path, history_dir: Optional[Path] = None):
        """
        Initialize version manager.

        Args:
            document_path: Path to the main document
            history_dir: Directory for version storage (default: document/history/)
        """
        self.document_path = Path(document_path)
        self.history_dir = history_dir or self.document_path.parent / "history"
        self.index_path = self.history_dir / "versions.json"
        self.versions: list[Version] = []

        # Ensure history directory exists
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Load existing versions
        self._load_index()

    def _load_index(self):
        """Load version index from disk."""
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.versions = [Version.from_dict(v) for v in data.get("versions", [])]
                log.info(
                    "documenter.versions.loaded",
                    count=len(self.versions),
                )
            except Exception as e:
                log.warning("documenter.versions.load_error", error=str(e))
                self.versions = []
        else:
            self.versions = []

    def _save_index(self):
        """Save version index to disk."""
        data = {
            "document": str(self.document_path),
            "versions": [v.to_dict() for v in self.versions],
        }
        self.index_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _get_version_path(self, version_id: str) -> Path:
        """Get path for a version file."""
        return self.history_dir / f"{version_id}.md"

    def save_version(
        self,
        content: str,
        description: str,
        sections_count: int = 0,
    ) -> Optional[Version]:
        """
        Save a new version of the document.

        Args:
            content: The document content
            description: Description of what changed
            sections_count: Number of sections in the document

        Returns:
            The created Version, or None if content unchanged
        """
        content_hash = self._compute_hash(content)

        # Check if content actually changed
        if self.versions and self.versions[-1].content_hash == content_hash:
            log.debug("documenter.versions.unchanged", hash=content_hash)
            return None

        # Create version
        now = datetime.now()
        version_id = now.strftime("%Y%m%d_%H%M%S")

        # Ensure unique version_id
        existing_ids = {v.version_id for v in self.versions}
        counter = 1
        base_id = version_id
        while version_id in existing_ids:
            version_id = f"{base_id}_{counter}"
            counter += 1

        version = Version(
            version_id=version_id,
            timestamp=now.isoformat(),
            description=description,
            content_hash=content_hash,
            sections_count=sections_count,
            content_length=len(content),
        )

        # Save content to version file
        version_path = self._get_version_path(version_id)
        version_path.write_text(content, encoding="utf-8")

        # Add to index
        self.versions.append(version)
        self._save_index()

        log.info(
            "documenter.versions.saved",
            version_id=version_id,
            description=description,
            content_length=len(content),
        )

        return version

    def list_versions(self, limit: int = 50) -> list[Version]:
        """
        List recent versions.

        Args:
            limit: Maximum number of versions to return

        Returns:
            List of versions, most recent first
        """
        return list(reversed(self.versions[-limit:]))

    def get_version(self, version_id: str) -> Optional[tuple[Version, str]]:
        """
        Get a specific version with its content.

        Args:
            version_id: The version identifier

        Returns:
            Tuple of (Version, content) or None if not found
        """
        # Find version in index
        version = None
        for v in self.versions:
            if v.version_id == version_id:
                version = v
                break

        if not version:
            return None

        # Load content
        version_path = self._get_version_path(version_id)
        if not version_path.exists():
            log.warning(
                "documenter.versions.missing_file",
                version_id=version_id,
            )
            return None

        content = version_path.read_text(encoding="utf-8")
        return version, content

    def revert_to(self, version_id: str) -> Optional[str]:
        """
        Revert document to a specific version.

        This saves the current state as a new version first,
        then restores the target version.

        Args:
            version_id: The version to revert to

        Returns:
            The restored content, or None if version not found
        """
        result = self.get_version(version_id)
        if not result:
            return None

        version, content = result

        # Save current state first (if document exists)
        if self.document_path.exists():
            current_content = self.document_path.read_text(encoding="utf-8")
            self.save_version(
                current_content,
                f"Auto-save before revert to {version_id}",
            )

        # Restore the target version
        self.document_path.write_text(content, encoding="utf-8")

        # Record the revert as a new version
        self.save_version(
            content,
            f"Reverted to version {version_id} ({version.description})",
        )

        log.info(
            "documenter.versions.reverted",
            version_id=version_id,
            description=version.description,
        )

        return content

    def get_diff_summary(self, version_id: str) -> Optional[dict]:
        """
        Get a summary of differences between a version and current.

        Returns:
            Dict with diff info, or None if version not found
        """
        result = self.get_version(version_id)
        if not result:
            return None

        version, old_content = result

        if not self.document_path.exists():
            return None

        current_content = self.document_path.read_text(encoding="utf-8")

        old_lines = old_content.splitlines()
        new_lines = current_content.splitlines()

        return {
            "version_id": version_id,
            "version_timestamp": version.timestamp,
            "old_lines": len(old_lines),
            "new_lines": len(new_lines),
            "old_length": len(old_content),
            "new_length": len(current_content),
        }

    def cleanup_old_versions(self, keep_count: int = 100):
        """
        Remove old versions to save disk space.

        Args:
            keep_count: Number of recent versions to keep
        """
        if len(self.versions) <= keep_count:
            return

        to_remove = self.versions[:-keep_count]
        self.versions = self.versions[-keep_count:]

        for version in to_remove:
            version_path = self._get_version_path(version.version_id)
            if version_path.exists():
                version_path.unlink()

        self._save_index()

        log.info(
            "documenter.versions.cleanup",
            removed=len(to_remove),
            kept=len(self.versions),
        )
