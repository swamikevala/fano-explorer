"""
Explorer Blueprint - Insights management routes.
"""

import json
from datetime import datetime

from flask import Blueprint, jsonify, request

from ..services import (
    EXPLORER_DATA_DIR,
    get_explorer_stats,
    get_insight_by_id,
    get_insights_by_status,
    get_review_for_insight,
    load_insight_json,
)

bp = Blueprint("explorer", __name__, url_prefix="/api/explorer")


@bp.route("/stats")
def api_explorer_stats():
    """Get explorer insight stats."""
    return jsonify(get_explorer_stats())


@bp.route("/insights/<status>")
def api_explorer_insights(status: str):
    """Get insights by status."""
    valid_statuses = ["pending", "blessed", "interesting", "rejected", "reviewing", "disputed"]
    if status not in valid_statuses:
        return jsonify({"error": f"Invalid status: {status}"}), 400

    if status == "reviewing":
        # Get insights currently being reviewed
        reviewing_dir = EXPLORER_DATA_DIR / "chunks" / "reviewing"
        insights = []
        if reviewing_dir.exists():
            for json_file in reviewing_dir.glob("*.json"):
                data = load_insight_json(json_file)
                if data:
                    review = get_review_for_insight(data.get("id", ""))
                    insights.append({
                        "insight": data,
                        "review": review,
                        "rounds_completed": len(review.get("rounds", [])) if review else 0,
                    })
        return jsonify({"insights": insights, "status": status})

    elif status == "disputed":
        # Get disputed reviews
        disputed_dir = EXPLORER_DATA_DIR / "reviews" / "disputed"
        insights = []
        if disputed_dir.exists():
            for json_file in disputed_dir.glob("*.json"):
                review = load_insight_json(json_file)
                if review:
                    insight_id = review.get("chunk_id", "")
                    insight_data, _ = get_insight_by_id(insight_id)
                    if insight_data:
                        insights.append({
                            "insight": insight_data,
                            "review": review,
                        })
        return jsonify({"insights": insights, "status": status})

    else:
        # Regular status
        raw_insights = get_insights_by_status(status)
        insights = []
        for data in raw_insights:
            review = get_review_for_insight(data.get("id", ""))
            insights.append({
                "insight": data,
                "review": review,
            })
        return jsonify({"insights": insights, "status": status})


@bp.route("/insight/<insight_id>")
def api_explorer_insight(insight_id: str):
    """Get a specific insight."""
    insight, status = get_insight_by_id(insight_id)
    if not insight:
        return jsonify({"error": "Insight not found"}), 404

    review = get_review_for_insight(insight_id)
    return jsonify({
        "insight": insight,
        "review": review,
        "status": status,
    })


@bp.route("/feedback", methods=["POST"])
def api_explorer_feedback():
    """Submit feedback for an insight."""
    data = request.json
    insight_id = data.get("insight_id")
    feedback = data.get("feedback")  # "bless", "interesting", "reject"
    notes = data.get("notes", "")

    if not insight_id or not feedback:
        return jsonify({"error": "Missing insight_id or feedback"}), 400

    # Map feedback to rating and new status
    rating_map = {"bless": "⚡", "interesting": "?", "reject": "✗"}
    status_map = {"bless": "blessed", "interesting": "interesting", "reject": "rejected"}

    if feedback not in rating_map:
        return jsonify({"error": "Invalid feedback value"}), 400

    insight_data, current_status = get_insight_by_id(insight_id)
    if not insight_data:
        return jsonify({"error": "Insight not found"}), 404

    # Update the insight data
    insight_data["rating"] = rating_map[feedback]
    insight_data["status"] = status_map[feedback]
    insight_data["reviewed_at"] = datetime.now().isoformat()
    if notes:
        insight_data["review_notes"] = f"Manual review: {notes}"

    new_status = status_map[feedback]

    # Save to new location
    new_dir = EXPLORER_DATA_DIR / "chunks" / "insights" / new_status
    new_dir.mkdir(parents=True, exist_ok=True)
    new_path = new_dir / f"{insight_id}.json"

    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(insight_data, f, indent=2, default=str)

    # Remove from old location
    old_path = EXPLORER_DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.json"
    if old_path.exists():
        old_path.unlink()
    old_md = EXPLORER_DATA_DIR / "chunks" / "insights" / current_status / f"{insight_id}.md"
    if old_md.exists():
        old_md.unlink()

    return jsonify({"success": True, "new_status": new_status})


@bp.route("/priority", methods=["POST"])
def api_explorer_priority():
    """Update insight priority."""
    data = request.json
    insight_id = data.get("insight_id")
    priority = data.get("priority")

    if not insight_id or priority is None:
        return jsonify({"error": "Missing insight_id or priority"}), 400

    try:
        priority = int(priority)
        if not 1 <= priority <= 10:
            return jsonify({"error": "Priority must be between 1 and 10"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid priority value"}), 400

    insight_data, status = get_insight_by_id(insight_id)
    if not insight_data:
        return jsonify({"error": "Insight not found"}), 404

    # Update priority
    insight_data["priority"] = priority

    # Save in place
    json_path = EXPLORER_DATA_DIR / "chunks" / "insights" / status / f"{insight_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(insight_data, f, indent=2, default=str)

    return jsonify({"success": True, "priority": priority})
