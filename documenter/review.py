"""
Review mode - reviewing and improving existing sections.
"""

import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger
from .document import Document, Section

log = get_logger("documenter", "review")


@dataclass
class ReviewCandidate:
    """A section candidate for review."""
    section: Section
    priority: str  # "never_reviewed" | "stale" | "recently_reviewed"
    days_since_review: Optional[int] = None


class ReviewManager:
    """
    Manages the review process for existing sections.

    Prioritizes sections that need review and tracks
    review history.
    """

    def __init__(
        self,
        document: Document,
        max_age_days: int = 7,
        work_allocation: int = 30,  # Percentage of work for review
    ):
        """
        Initialize review manager.

        Args:
            document: The document to review
            max_age_days: Days after which a section is considered stale
            work_allocation: Percentage of work cycles for review (0-100)
        """
        self.document = document
        self.max_age_days = max_age_days
        self.work_allocation = work_allocation
        self._review_counter = 0

    def should_do_review(self) -> bool:
        """
        Decide if the next work cycle should be review.

        Based on configured work allocation percentage.
        """
        self._review_counter += 1

        # Every N cycles, do a review (where N = 100/allocation)
        if self.work_allocation <= 0:
            return False

        review_frequency = 100 // self.work_allocation
        should_review = (self._review_counter % review_frequency) == 0

        log.debug(
            "documenter.review.should_do_review",
            counter=self._review_counter,
            should_review=should_review,
        )
        return should_review

    def select_section_for_review(self) -> Optional[Section]:
        """
        Select a section for review.

        Priority: never_reviewed > stale > recently_reviewed
        Within each tier, select randomly.
        """
        candidates = self._categorize_sections()

        # Try each priority tier
        for tier in ["never_reviewed", "stale", "recently_reviewed"]:
            tier_candidates = [c for c in candidates if c.priority == tier]
            if tier_candidates:
                selected = random.choice(tier_candidates)
                log.info(
                    "documenter.review.section_selected",
                    section_id=selected.section.id,
                    priority=tier,
                    days_since_review=selected.days_since_review,
                )
                return selected.section

        log.info("documenter.review.no_sections")
        return None

    def _categorize_sections(self) -> list[ReviewCandidate]:
        """Categorize sections by review priority."""
        candidates = []
        now = datetime.now()

        for section in self.document.sections:
            if section.last_reviewed is None:
                candidates.append(ReviewCandidate(
                    section=section,
                    priority="never_reviewed",
                    days_since_review=None,
                ))
            else:
                days_since = (now - section.last_reviewed).days

                if days_since >= self.max_age_days:
                    priority = "stale"
                else:
                    priority = "recently_reviewed"

                candidates.append(ReviewCandidate(
                    section=section,
                    priority=priority,
                    days_since_review=days_since,
                ))

        return candidates

    def all_sections_recently_reviewed(self) -> bool:
        """Check if all sections have been recently reviewed."""
        if not self.document.sections:
            return True

        candidates = self._categorize_sections()

        # Check if any are not recently reviewed
        for c in candidates:
            if c.priority != "recently_reviewed":
                return False

        return True

    def get_review_stats(self) -> dict:
        """Get statistics about review status."""
        candidates = self._categorize_sections()

        stats = {
            "total_sections": len(self.document.sections),
            "never_reviewed": sum(1 for c in candidates if c.priority == "never_reviewed"),
            "stale": sum(1 for c in candidates if c.priority == "stale"),
            "recently_reviewed": sum(1 for c in candidates if c.priority == "recently_reviewed"),
        }

        return stats

    def get_concepts_since_creation(self, section: Section) -> list[str]:
        """Get concepts established after a section was created."""
        concepts_since = []

        for other in self.document.sections:
            # Check if other section was created after this one
            if other.created > section.created:
                concepts_since.extend(other.establishes)

        return list(set(concepts_since))

    def get_concepts_at_creation(self, section: Section) -> list[str]:
        """Get concepts that existed when a section was created."""
        concepts = []

        for other in self.document.sections:
            # Check if other section was created before this one
            if other.created <= section.created and other.id != section.id:
                concepts.extend(other.establishes)

        return list(set(concepts))
