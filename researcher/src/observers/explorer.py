"""
Observer for explorer module activity.

Watches explorer's insights and extractions to understand current exploration context.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseObserver


class ExplorerObserver(BaseObserver):
    """Observer for explorer module activity."""

    def __init__(self, explorer_data_path: Path):
        """
        Initialize explorer observer.

        Args:
            explorer_data_path: Path to explorer/data directory
        """
        super().__init__(explorer_data_path)
        self.insights_path = explorer_data_path / "insights"
        self.reviews_path = explorer_data_path / "reviews"

    def get_recent_activity(self) -> list[dict]:
        """
        Get recent explorer activity (insights, reviews).

        Returns:
            List of activity records with insights and review outcomes
        """
        activities = []

        # Check all insight directories
        for status_dir in ["pending", "blessed", "interesting"]:
            insight_dir = self.insights_path / status_dir
            changed_files = self._get_changed_files(insight_dir)

            for filepath in changed_files:
                data = self._load_json_file(filepath)
                if data:
                    activities.append({
                        "type": "insight",
                        "status": status_dir,
                        "content": data,
                        "timestamp": datetime.now(),
                        "filepath": str(filepath),
                    })

        # Check completed reviews for consensus outcomes
        if self.reviews_path.exists():
            completed_dir = self.reviews_path / "completed"
            changed_reviews = self._get_changed_files(completed_dir)

            for filepath in changed_reviews:
                data = self._load_json_file(filepath)
                if data:
                    activities.append({
                        "type": "review",
                        "content": data,
                        "timestamp": datetime.now(),
                        "filepath": str(filepath),
                    })

        self.last_check = datetime.now()
        return activities

    def extract_concepts(self, activity: dict) -> list[str]:
        """
        Extract concept names from an explorer activity.

        Looks at tags, insight text, and review discussions.
        """
        concepts = []
        content = activity.get("content", {})

        # From insight tags
        if "tags" in content:
            concepts.extend(content["tags"])

        # From insight text - extract key terms
        insight_text = content.get("insight", "")
        if insight_text:
            concepts.extend(self._extract_concepts_from_text(insight_text))

        # From review discussions
        if activity["type"] == "review" and "rounds" in content:
            for round_data in content["rounds"]:
                if "responses" in round_data:
                    for response in round_data["responses"].values():
                        if isinstance(response, dict) and "reasoning" in response:
                            concepts.extend(
                                self._extract_concepts_from_text(response["reasoning"])
                            )

        return list(set(concepts))

    def extract_numbers(self, activity: dict) -> list[int]:
        """
        Extract significant numbers from an explorer activity.

        Looks for numbers in insight text and mathematical content.
        """
        content = activity.get("content", {})

        # From insight text
        insight_text = content.get("insight", "")
        numbers = self._extract_numbers_from_text(insight_text)

        # From review discussions
        if activity["type"] == "review" and "rounds" in content:
            for round_data in content["rounds"]:
                if "responses" in round_data:
                    for response in round_data["responses"].values():
                        if isinstance(response, dict) and "reasoning" in response:
                            numbers.extend(
                                self._extract_numbers_from_text(response["reasoning"])
                            )

        return list(set(numbers))

    def _extract_concepts_from_text(self, text: str) -> list[str]:
        """
        Extract potential concept names from text.

        Looks for:
        - Capitalized terms (proper nouns)
        - Sanskrit terms (common patterns)
        - Technical terms in context
        """
        concepts = []

        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        concepts.extend(capitalized)

        # Common Sanskrit/technical terms
        sanskrit_patterns = [
            r'\b(chakra|yantra|mantra|tantra|yoga|nadi|prana|kundalini)\b',
            r'\b(svara|raga|tala|shruti)\b',
            r'\b(tattva|guna|prakriti|purusha)\b',
            r'\b(fano|projective|incidence|duality)\b',
        ]

        for pattern in sanskrit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend([m.lower() for m in matches])

        return list(set(concepts))

    def get_blessed_insights(self, limit: int = 50) -> list[dict]:
        """Get recently blessed insights."""
        insights = []
        blessed_dir = self.insights_path / "blessed"

        if blessed_dir.exists():
            files = sorted(
                blessed_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:limit]

            for filepath in files:
                data = self._load_json_file(filepath)
                if data:
                    insights.append(data)

        return insights

    def get_pending_insights(self) -> list[dict]:
        """Get insights awaiting review."""
        insights = []
        pending_dir = self.insights_path / "pending"

        if pending_dir.exists():
            for filepath in pending_dir.glob("*.json"):
                data = self._load_json_file(filepath)
                if data:
                    insights.append(data)

        return insights

    def get_insight_by_id(self, insight_id: str) -> Optional[dict]:
        """Get a specific insight by ID."""
        for status_dir in ["pending", "blessed", "interesting", "rejected"]:
            filepath = self.insights_path / status_dir / f"{insight_id}.json"
            if filepath.exists():
                return self._load_json_file(filepath)
        return None
