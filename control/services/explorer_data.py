"""
Explorer data access layer for insights and reviews.
"""

import json
from pathlib import Path
from typing import Optional

from .config import EXPLORER_DATA_DIR


def get_explorer_stats() -> dict:
    """Get insight counts by status."""
    stats = {}
    for status in ["pending", "blessed", "interesting", "rejected"]:
        dir_path = EXPLORER_DATA_DIR / "chunks" / "insights" / status
        stats[status] = len(list(dir_path.glob("*.json"))) if dir_path.exists() else 0

    # Count reviewing
    reviewing_dir = EXPLORER_DATA_DIR / "chunks" / "reviewing"
    stats["reviewing"] = len(list(reviewing_dir.glob("*.json"))) if reviewing_dir.exists() else 0

    # Count disputed
    disputed_dir = EXPLORER_DATA_DIR / "reviews" / "disputed"
    stats["disputed"] = len(list(disputed_dir.glob("*.json"))) if disputed_dir.exists() else 0

    return stats


def load_insight_json(json_path: Path) -> Optional[dict]:
    """Load an insight from JSON file."""
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_insights_by_status(status: str) -> list:
    """Get all insights with a given status."""
    insights_dir = EXPLORER_DATA_DIR / "chunks" / "insights" / status
    insights = []

    if not insights_dir.exists():
        return insights

    for json_file in insights_dir.glob("*.json"):
        data = load_insight_json(json_file)
        if data:
            insights.append(data)

    # Sort by most recent
    insights.sort(key=lambda x: x.get("reviewed_at") or x.get("extracted_at", ""), reverse=True)
    return insights


def get_insight_by_id(insight_id: str) -> tuple:
    """Get a specific insight by ID, return (data, status)."""
    for status in ["pending", "blessed", "interesting", "rejected"]:
        json_path = EXPLORER_DATA_DIR / "chunks" / "insights" / status / f"{insight_id}.json"
        if json_path.exists():
            return load_insight_json(json_path), status
    return None, None


def get_review_for_insight(insight_id: str) -> Optional[dict]:
    """Get review data for an insight."""
    for subdir in ["completed", "disputed"]:
        review_path = EXPLORER_DATA_DIR / "reviews" / subdir / f"{insight_id}.json"
        if review_path.exists():
            return load_insight_json(review_path)
    return None
