"""
Atomic Insight model.

An atomic insight is a single, standalone mathematical observation (1-3 sentences)
extracted from exploration threads. Multiple insights can be extracted per thread.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class InsightStatus(Enum):
    """Status of an atomic insight in the review pipeline."""
    PENDING = "pending"          # Awaiting review
    BLESSED = "blessed"          # ⚡ Approved as axiom
    INTERESTING = "interesting"  # ? Worth exploring more
    REJECTED = "rejected"        # ✗ Discarded


@dataclass
class AtomicInsight:
    """
    A single atomic insight extracted from exploration.

    Each insight is 1-3 sentences capturing ONE clear mathematical
    observation or connection. Multiple insights are extracted per thread.
    """
    # Core identity
    id: str
    insight: str                     # The atomic aphorism (1-3 sentences)
    confidence: str                  # "high" | "medium" | "low"
    tags: list[str]                  # Relevant concepts for searchability

    # Provenance
    source_thread_id: str            # Which exploration thread
    source_exchange_indices: list[int]  # Which exchanges contributed
    extraction_model: str            # Which LLM extracted this
    extracted_at: datetime

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # IDs of blessed insights
    pending_dependencies: list[str] = field(default_factory=list)  # Descriptions not yet matched

    # Review status
    status: InsightStatus = InsightStatus.PENDING
    rating: Optional[str] = None     # "⚡" | "?" | "✗" after review
    is_disputed: bool = False        # True if review panel had split decision
    reviewed_at: Optional[datetime] = None
    review_notes: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "insight": self.insight,
            "confidence": self.confidence,
            "tags": self.tags,
            "source_thread_id": self.source_thread_id,
            "source_exchange_indices": self.source_exchange_indices,
            "extraction_model": self.extraction_model,
            "extracted_at": self.extracted_at.isoformat(),
            "depends_on": self.depends_on,
            "pending_dependencies": self.pending_dependencies,
            "status": self.status.value,
            "rating": self.rating,
            "is_disputed": self.is_disputed,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AtomicInsight":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            insight=data["insight"],
            confidence=data["confidence"],
            tags=data["tags"],
            source_thread_id=data["source_thread_id"],
            source_exchange_indices=data.get("source_exchange_indices", []),
            extraction_model=data["extraction_model"],
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            depends_on=data.get("depends_on", []),
            pending_dependencies=data.get("pending_dependencies", []),
            status=InsightStatus(data.get("status", "pending")),
            rating=data.get("rating"),
            is_disputed=data.get("is_disputed", False),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data.get("reviewed_at") else None,
            review_notes=data.get("review_notes", ""),
        )

    @classmethod
    def create(
        cls,
        insight: str,
        confidence: str,
        tags: list[str],
        source_thread_id: str,
        extraction_model: str,
        depends_on: list[str] = None,
        pending_dependencies: list[str] = None,
        source_exchange_indices: list[int] = None,
    ) -> "AtomicInsight":
        """Factory method to create a new atomic insight."""
        return cls(
            id=str(uuid.uuid4())[:12],
            insight=insight,
            confidence=confidence,
            tags=tags,
            source_thread_id=source_thread_id,
            source_exchange_indices=source_exchange_indices or [],
            extraction_model=extraction_model,
            extracted_at=datetime.now(),
            depends_on=depends_on or [],
            pending_dependencies=pending_dependencies or [],
        )

    def apply_rating(self, rating: str, is_disputed: bool = False, notes: str = ""):
        """Apply review panel rating to this insight."""
        self.rating = rating
        self.is_disputed = is_disputed
        self.review_notes = notes
        self.reviewed_at = datetime.now()

        # Update status based on rating
        status_map = {
            "⚡": InsightStatus.BLESSED,
            "?": InsightStatus.INTERESTING,
            "✗": InsightStatus.REJECTED,
        }
        if rating in status_map:
            self.status = status_map[rating]

    def is_foundation_solid(self, blessed_ids: set[str]) -> bool:
        """Check if all dependencies are blessed."""
        return all(dep_id in blessed_ids for dep_id in self.depends_on)

    def save(self, base_dir: Path):
        """Save insight to appropriate directory based on status."""
        status_dirs = {
            InsightStatus.PENDING: "pending",
            InsightStatus.BLESSED: "blessed",
            InsightStatus.INTERESTING: "interesting",
            InsightStatus.REJECTED: "rejected",
        }

        insight_dir = base_dir / "insights" / status_dirs[self.status]
        insight_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = insight_dir / f"{self.id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # Save markdown for readability
        md_path = insight_dir / f"{self.id}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

    @classmethod
    def load(cls, json_path: Path) -> "AtomicInsight":
        """Load insight from JSON file."""
        with open(json_path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_markdown(self) -> str:
        """Export insight as markdown."""
        rating_display = self.rating or "(pending)"
        disputed_flag = " [DISPUTED]" if self.is_disputed else ""

        lines = [
            f"# {rating_display}{disputed_flag} Insight `{self.id}`",
            "",
            f"> {self.insight}",
            "",
            "---",
            "",
            f"**Confidence:** {self.confidence}",
            f"**Tags:** {', '.join(self.tags)}",
            f"**Status:** {self.status.value}",
            f"**Extracted:** {self.extracted_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Model:** {self.extraction_model}",
            f"**Source Thread:** `{self.source_thread_id}`",
        ]

        if self.depends_on:
            lines.append(f"**Depends On:** {', '.join(self.depends_on)}")

        if self.pending_dependencies:
            lines.append(f"**Pending Dependencies:** {'; '.join(self.pending_dependencies)}")

        if self.reviewed_at:
            lines.append(f"**Reviewed:** {self.reviewed_at.strftime('%Y-%m-%d %H:%M')}")

        if self.review_notes:
            lines.extend(["", "**Review Notes:**", self.review_notes])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AtomicInsight({self.id}, confidence={self.confidence}, status={self.status.value})"
