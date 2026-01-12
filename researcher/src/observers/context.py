"""
Context aggregator that combines observations from explorer and documenter.

Provides unified research context for question generation and relevance scoring.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

from .explorer import ExplorerObserver
from .documenter import DocumenterObserver
from ..store.models import ResearchContext


class ContextAggregator:
    """
    Aggregates context from explorer and documenter observations.

    Maintains a unified view of what the system is currently exploring
    and writing about.
    """

    def __init__(
        self,
        explorer_data_path: Path,
        documenter_path: Path,
        domains_config_path: Path
    ):
        """
        Initialize context aggregator.

        Args:
            explorer_data_path: Path to explorer/data directory
            documenter_path: Path to documenter directory
            domains_config_path: Path to domains.yaml config
        """
        self.explorer_observer = ExplorerObserver(explorer_data_path)
        self.documenter_observer = DocumenterObserver(documenter_path)

        # Load domain config for key numbers
        self.key_numbers = set()
        self._load_key_numbers(domains_config_path)

        # Current context
        self._context = ResearchContext.empty()
        self._last_update: Optional[datetime] = None

    def _load_key_numbers(self, config_path: Path) -> None:
        """Load key numbers from domains config."""
        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Global key numbers
            self.key_numbers.update(config.get("global_key_numbers", []))

            # Domain-specific key numbers
            for domain in config.get("domains", []):
                self.key_numbers.update(domain.get("key_numbers", []))

        except (FileNotFoundError, yaml.YAMLError):
            # Default key numbers if config not found
            self.key_numbers = {3, 5, 7, 9, 12, 27, 28, 36, 64, 72, 84, 108, 112, 1008}

    def update_context(self) -> ResearchContext:
        """
        Update and return current research context.

        Polls observers for new activity and aggregates into context.
        """
        # Get recent activity from both modules
        explorer_activities = self.explorer_observer.get_recent_activity()
        documenter_activities = self.documenter_observer.get_recent_activity()

        # Aggregate concepts
        all_concepts = set(self._context.active_concepts)
        for activity in explorer_activities + documenter_activities:
            if activity["type"] == "insight":
                concepts = self.explorer_observer.extract_concepts(activity)
            elif activity["type"] in ["document_update", "history_entry"]:
                concepts = self.documenter_observer.extract_concepts(activity)
            else:
                concepts = []
            all_concepts.update(concepts)

        # Aggregate numbers (filter to key numbers)
        all_numbers = set(self._context.active_numbers)
        for activity in explorer_activities + documenter_activities:
            if activity["type"] == "insight":
                numbers = self.explorer_observer.extract_numbers(activity)
            elif activity["type"] in ["document_update", "history_entry"]:
                numbers = self.documenter_observer.extract_numbers(activity)
            else:
                numbers = []
            # Only keep numbers that are in our key numbers set
            significant_numbers = [n for n in numbers if n in self.key_numbers]
            all_numbers.update(significant_numbers)

        # Get domains from recent insights
        domains = set(self._context.active_domains)
        for activity in explorer_activities:
            if activity["type"] == "insight":
                tags = activity.get("content", {}).get("tags", [])
                domains.update(self._tags_to_domains(tags))

        # Get recent insights
        recent_insights = []
        for activity in explorer_activities:
            if activity["type"] == "insight":
                recent_insights.append(activity["content"])

        # Get documenter topics
        documenter_topics = self.documenter_observer.get_current_topics()

        # Build updated context
        self._context = ResearchContext(
            active_concepts=list(all_concepts)[:50],  # Limit size
            active_numbers=list(all_numbers)[:20],
            active_domains=list(domains),
            recent_insights=recent_insights[:20],
            documenter_topics=documenter_topics,
            last_updated=datetime.now(),
        )

        self._last_update = datetime.now()
        return self._context

    def get_context(self) -> ResearchContext:
        """Get current research context (without polling)."""
        return self._context

    def _tags_to_domains(self, tags: list[str]) -> list[str]:
        """
        Convert insight tags to research domains.

        Maps common tag patterns to domain names from config.
        """
        domain_mapping = {
            "fano": "projective_geometry",
            "projective": "projective_geometry",
            "chakra": "hatha_yoga",
            "tantra": "tantraloka",
            "yoga": "hatha_yoga",
            "music": "indian_music",
            "raga": "indian_music",
            "sanskrit": "sanskrit_grammar",
            "grammar": "sanskrit_grammar",
            "vedic": "vedas",
            "purana": "bhagavata_purana",
            "shiva": "kashmiri_shaivism",
            "kashmir": "kashmiri_shaivism",
            "yantra": "yantras",
        }

        domains = []
        for tag in tags:
            tag_lower = tag.lower()
            for keyword, domain in domain_mapping.items():
                if keyword in tag_lower:
                    domains.append(domain)
                    break

        return domains

    def get_priority_concepts(self, limit: int = 10) -> list[str]:
        """
        Get concepts that should be prioritized for research.

        Prioritizes concepts from blessed insights and recent activity.
        """
        # Get concepts from blessed insights (high priority)
        blessed_insights = self.explorer_observer.get_blessed_insights(limit=20)
        blessed_concepts = []
        for insight in blessed_insights:
            blessed_concepts.extend(insight.get("tags", []))

        # Count occurrences
        concept_counts = {}
        for concept in blessed_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 2  # Weight blessed higher

        for concept in self._context.active_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1

        # Sort by count
        sorted_concepts = sorted(
            concept_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [c[0] for c in sorted_concepts[:limit]]

    def get_priority_numbers(self, limit: int = 10) -> list[int]:
        """
        Get numbers that should be prioritized for research.

        Returns numbers that appear in current context and are key numbers.
        """
        # Prioritize numbers from blessed insights
        blessed_insights = self.explorer_observer.get_blessed_insights(limit=20)
        number_counts = {}

        for insight in blessed_insights:
            insight_text = insight.get("insight", "")
            numbers = self.explorer_observer._extract_numbers_from_text(insight_text)
            for num in numbers:
                if num in self.key_numbers:
                    number_counts[num] = number_counts.get(num, 0) + 2

        for num in self._context.active_numbers:
            number_counts[num] = number_counts.get(num, 0) + 1

        sorted_numbers = sorted(
            number_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [n[0] for n in sorted_numbers[:limit]]

    def is_idle(self) -> bool:
        """
        Check if the system appears to be idle.

        Returns True if no recent activity detected.
        """
        if self._last_update is None:
            return True

        # Check for any new activity
        explorer_activities = self.explorer_observer.get_recent_activity()
        documenter_activities = self.documenter_observer.get_recent_activity()

        return len(explorer_activities) == 0 and len(documenter_activities) == 0
