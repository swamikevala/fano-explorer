"""
Concept tracking - ESTABLISHES/REQUIRES dependency management.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger
from .document import Document, Section

log = get_logger("documenter", "concepts")


@dataclass
class ConceptInfo:
    """Information about a concept."""
    name: str
    established_in: Optional[str] = None  # Section ID
    required_by: list[str] = None  # Section IDs

    def __post_init__(self):
        if self.required_by is None:
            self.required_by = []


class ConceptTracker:
    """
    Tracks concept dependencies across the document.

    Ensures that new content only references concepts that are
    already established, maintaining a valid DAG.
    """

    def __init__(self, document: Document):
        """
        Initialize concept tracker.

        Args:
            document: The document to track concepts for
        """
        self.document = document
        self._concepts: dict[str, ConceptInfo] = {}
        self._refresh()

    def _refresh(self):
        """Refresh concept index from document."""
        self._concepts = {}

        for section in self.document.sections:
            # Track established concepts
            for concept in section.establishes:
                if concept not in self._concepts:
                    self._concepts[concept] = ConceptInfo(name=concept)
                self._concepts[concept].established_in = section.id

            # Track required concepts
            for concept in section.requires:
                if concept not in self._concepts:
                    self._concepts[concept] = ConceptInfo(name=concept)
                self._concepts[concept].required_by.append(section.id)

        log.debug(
            "documenter.concepts.refreshed",
            total_concepts=len(self._concepts),
            established=sum(1 for c in self._concepts.values() if c.established_in),
        )

    def get_established_concepts(self) -> set[str]:
        """Get all concepts that are established in the document."""
        return {
            name for name, info in self._concepts.items()
            if info.established_in is not None
        }

    def get_missing_concepts(self) -> set[str]:
        """Get concepts that are required but not established."""
        established = self.get_established_concepts()
        all_required = set()

        for section in self.document.sections:
            all_required.update(section.requires)

        return all_required - established

    def can_add_content(self, requires: list[str]) -> tuple[bool, list[str]]:
        """
        Check if content with given requirements can be added.

        Args:
            requires: List of concepts the content requires

        Returns:
            (can_add, missing_concepts)
        """
        established = self.get_established_concepts()
        missing = [c for c in requires if c not in established]
        return len(missing) == 0, missing

    def validate_section(self, section: Section) -> tuple[bool, list[str]]:
        """
        Validate that a section's requirements are met.

        Args:
            section: The section to validate

        Returns:
            (valid, missing_concepts)
        """
        return self.can_add_content(section.requires)

    def get_concept_info(self, concept: str) -> Optional[ConceptInfo]:
        """Get information about a concept."""
        return self._concepts.get(concept)

    def get_dependents(self, concept: str) -> list[str]:
        """Get sections that depend on a concept."""
        info = self._concepts.get(concept)
        return info.required_by if info else []

    def extract_concepts_from_text(self, text: str) -> tuple[list[str], list[str]]:
        """
        Extract ESTABLISHES and REQUIRES markers from text.

        Args:
            text: Text to parse

        Returns:
            (establishes_list, requires_list)
        """
        establishes = []
        requires = []

        # Look for explicit markers
        establishes_pattern = re.compile(r'<!-- ESTABLISHES:\s*(\w+)\s*-->')
        requires_pattern = re.compile(r'<!-- REQUIRES:\s*(\w+)\s*-->')

        for match in establishes_pattern.finditer(text):
            establishes.append(match.group(1))

        for match in requires_pattern.finditer(text):
            requires.append(match.group(1))

        return establishes, requires

    def suggest_order(self, items: list[dict]) -> list[dict]:
        """
        Suggest an order for items based on dependencies.

        Args:
            items: List of dicts with 'requires' and 'establishes' keys

        Returns:
            Items sorted in dependency order (independent first)
        """
        established = self.get_established_concepts()

        def can_add(item: dict) -> bool:
            required = item.get('requires', [])
            return all(r in established for r in required)

        result = []
        remaining = items.copy()

        while remaining:
            # Find items that can be added now
            addable = [item for item in remaining if can_add(item)]

            if not addable:
                # No progress possible - circular dependency or unmet requirements
                log.warning(
                    "documenter.concepts.dependency_deadlock",
                    remaining_count=len(remaining),
                )
                result.extend(remaining)
                break

            # Add the first addable item
            item = addable[0]
            result.append(item)
            remaining.remove(item)

            # Update established set
            for concept in item.get('establishes', []):
                established.add(concept)

        return result

    def register_section(self, section: Section):
        """Register a section's concepts after it's added."""
        for concept in section.establishes:
            if concept not in self._concepts:
                self._concepts[concept] = ConceptInfo(name=concept)
            self._concepts[concept].established_in = section.id

        for concept in section.requires:
            if concept not in self._concepts:
                self._concepts[concept] = ConceptInfo(name=concept)
            self._concepts[concept].required_by.append(section.id)

        log.info(
            "documenter.concepts.section_registered",
            section_id=section.id,
            establishes=section.establishes,
            requires=section.requires,
        )
