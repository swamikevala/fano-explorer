"""
Annotation management - comments and protected regions for document review.

Uses inline markers in the markdown document for positioning.
Markers are HTML comments like: <!-- @ann:c001 -->
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

log = get_logger("documenter", "annotations")


@dataclass
class Annotation:
    """
    A user annotation (comment or protected region).

    The position in the document is determined by a marker in the markdown:
    <!-- @ann:{id} -->
    """
    id: str
    type: str  # "comment" or "protected"
    content: str  # The comment text (empty for protected)
    created: str  # ISO format datetime
    text_preview: str = ""  # First ~50 chars of annotated text for display

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        # Handle missing text_preview for backwards compatibility
        if "text_preview" not in data:
            data["text_preview"] = ""
        return cls(**data)


class AnnotationManager:
    """
    Manages annotations for a document.

    Stores annotation content in a JSON file. Position is determined by
    inline markers in the markdown document itself.
    """

    def __init__(self, document_path: Path):
        """
        Initialize annotation manager.

        Args:
            document_path: Path to the document being annotated
        """
        self.document_path = Path(document_path)
        self.annotations_path = self.document_path.parent / "annotations.json"
        self.annotations: dict[str, Annotation] = {}  # id -> Annotation
        self._counter = 0

        self.load()

    def load(self) -> bool:
        """Load annotations from disk."""
        if not self.annotations_path.exists():
            log.debug("documenter.annotations.no_file", path=str(self.annotations_path))
            return False

        try:
            data = json.loads(self.annotations_path.read_text(encoding="utf-8"))

            # Handle new format (annotations dict)
            if "annotations" in data:
                for ann_id, ann_data in data["annotations"].items():
                    ann_data["id"] = ann_id
                    self.annotations[ann_id] = Annotation.from_dict(ann_data)
            # Handle old format (comments/protected lists) for migration
            elif "comments" in data or "protected" in data:
                self._migrate_old_format(data)

            # Update counter based on existing IDs
            for ann_id in self.annotations:
                if ann_id.startswith("c") or ann_id.startswith("p"):
                    try:
                        num = int(ann_id[1:])
                        self._counter = max(self._counter, num)
                    except ValueError:
                        pass

            log.info(
                "documenter.annotations.loaded",
                count=len(self.annotations),
            )
            return True
        except Exception as e:
            log.warning("documenter.annotations.load_error", error=str(e))
            return False

    def _migrate_old_format(self, data: dict):
        """Migrate from old comments/protected format to new annotations format."""
        # Old comments become unlinked annotations (no marker in document yet)
        for old_comment in data.get("comments", []):
            ann = Annotation(
                id=old_comment["id"],
                type="comment",
                content=old_comment.get("content", ""),
                created=old_comment.get("created", datetime.now().isoformat()),
                text_preview=old_comment.get("selected_text", "")[:50],
            )
            self.annotations[ann.id] = ann

        for old_protected in data.get("protected", []):
            ann = Annotation(
                id=old_protected["id"],
                type="protected",
                content="",
                created=old_protected.get("created", datetime.now().isoformat()),
                text_preview=old_protected.get("selected_text", "")[:50],
            )
            self.annotations[ann.id] = ann

        log.info(
            "documenter.annotations.migrated",
            count=len(self.annotations),
        )

    def save(self) -> bool:
        """Save annotations to disk."""
        try:
            data = {
                "annotations": {
                    ann_id: {
                        "type": ann.type,
                        "content": ann.content,
                        "created": ann.created,
                        "text_preview": ann.text_preview,
                    }
                    for ann_id, ann in self.annotations.items()
                }
            }
            self.annotations_path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8"
            )
            log.debug(
                "documenter.annotations.saved",
                count=len(self.annotations),
            )
            return True
        except Exception as e:
            log.warning("documenter.annotations.save_error", error=str(e))
            return False

    def add(self, ann_type: str, content: str, text_preview: str = "") -> Annotation:
        """
        Add a new annotation.

        Args:
            ann_type: "comment" or "protected"
            content: The comment text (empty for protected)
            text_preview: First ~50 chars of annotated text

        Returns:
            The created Annotation with generated ID
        """
        self._counter += 1
        prefix = "c" if ann_type == "comment" else "p"
        ann_id = f"{prefix}{self._counter:03d}"

        annotation = Annotation(
            id=ann_id,
            type=ann_type,
            content=content,
            created=datetime.now().isoformat(),
            text_preview=text_preview[:50] if text_preview else "",
        )
        self.annotations[ann_id] = annotation
        self.save()

        log.info(
            "documenter.annotations.added",
            id=ann_id,
            type=ann_type,
        )
        return annotation

    def delete(self, ann_id: str) -> bool:
        """
        Delete an annotation by ID.

        Returns:
            True if deleted, False if not found
        """
        if ann_id in self.annotations:
            del self.annotations[ann_id]
            self.save()
            log.info("documenter.annotations.deleted", id=ann_id)
            return True
        return False

    def get(self, ann_id: str) -> Optional[Annotation]:
        """Get an annotation by ID."""
        return self.annotations.get(ann_id)

    def get_all(self) -> dict[str, Annotation]:
        """Get all annotations."""
        return self.annotations.copy()

    def get_all_as_dicts(self) -> dict:
        """Get all annotations as serializable dicts."""
        return {
            ann_id: ann.to_dict()
            for ann_id, ann in self.annotations.items()
        }

    def format_for_llm(self) -> str:
        """
        Format annotations for inclusion in LLM prompts.

        Returns:
            Formatted string for LLM context, or empty string if no annotations
        """
        comments = [a for a in self.annotations.values() if a.type == "comment"]
        protected = [a for a in self.annotations.values() if a.type == "protected"]

        parts = []

        if comments:
            parts.append("\nUSER COMMENTS (address these):")
            for c in comments:
                preview = c.text_preview or "(no preview)"
                parts.append(f'- [{c.id}] On "{preview}": {c.content}')

        if protected:
            parts.append("\nPROTECTED TEXT (preserve exactly, do not modify):")
            for p in protected:
                preview = p.text_preview or "(no preview)"
                parts.append(f'- [{p.id}] "{preview}"')

        return "\n".join(parts) if parts else ""
