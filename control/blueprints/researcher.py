"""
Researcher Blueprint - Routes for researcher data and control.
"""

from flask import Blueprint, jsonify, request

from ..services import FANO_ROOT

bp = Blueprint("researcher", __name__, url_prefix="/api/researcher")

# Lazy-load API to avoid import overhead at startup
_api = None


def get_api():
    """Get or create the ResearcherAPI instance."""
    global _api
    if _api is None:
        try:
            from researcher.src import ResearcherAPI
            _api = ResearcherAPI(FANO_ROOT / "researcher" / "data")
        except ImportError:
            return None
    return _api


@bp.route("/stats")
def api_stats():
    """Get researcher statistics."""
    api = get_api()
    if api is None:
        return jsonify({
            "error": "Researcher module not available",
            "sources": 0,
            "findings": 0,
            "concepts": 0,
            "number_mentions": 0,
            "cross_references": 0,
            "top_numbers": {},
        })

    try:
        stats = api.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/findings")
def api_findings():
    """Get recent findings."""
    api = get_api()
    if api is None:
        return jsonify({"findings": []})

    limit = request.args.get("limit", 20, type=int)
    domain = request.args.get("domain")
    min_trust = request.args.get("min_trust", 50, type=int)

    try:
        findings = api.query(
            domains=[domain] if domain else None,
            min_trust=min_trust,
            limit=limit
        )
        return jsonify({
            "findings": [f.to_dict() for f in findings]
        })
    except Exception as e:
        return jsonify({"error": str(e), "findings": []}), 500


@bp.route("/finding/<finding_id>")
def api_finding(finding_id):
    """Get a specific finding with cross-references."""
    api = get_api()
    if api is None:
        return jsonify({"error": "Researcher module not available"}), 503

    try:
        finding = api.get_finding(finding_id)
        if not finding:
            return jsonify({"error": "Finding not found"}), 404

        xrefs = api.get_cross_references(finding_id)
        return jsonify({
            "finding": finding.to_dict(),
            "cross_references": xrefs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/patterns")
def api_patterns():
    """Get cross-domain number patterns."""
    api = get_api()
    if api is None:
        return jsonify({"patterns": []})

    min_domains = request.args.get("min_domains", 2, type=int)

    try:
        patterns = api.get_number_patterns(min_domains=min_domains)
        return jsonify({"patterns": patterns})
    except Exception as e:
        return jsonify({"error": str(e), "patterns": []}), 500


@bp.route("/number/<int:number>")
def api_number_info(number):
    """Get info about a specific number."""
    api = get_api()
    if api is None:
        return jsonify({
            "number": number,
            "total_occurrences": 0,
            "domains": {},
            "occurrences": [],
        })

    try:
        info = api.get_number_info(number)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/sources")
def api_sources():
    """Get list of sources."""
    api = get_api()
    if api is None:
        return jsonify({"sources": []})

    limit = request.args.get("limit", 50, type=int)
    min_trust = request.args.get("min_trust", 0, type=int)

    try:
        # Query sources directly from the database
        # The API doesn't have a direct method for this, so we'll use the db
        sources = []
        if hasattr(api, 'db'):
            # Get sources from database
            cursor = api.db.conn.execute(
                """SELECT id, url, domain, title, trust_score, trust_tier,
                          has_sanskrit_citations, has_verse_references,
                          has_bibliography, is_academic_domain, created_at
                   FROM sources
                   WHERE trust_score >= ?
                   ORDER BY trust_score DESC, created_at DESC
                   LIMIT ?""",
                (min_trust, limit)
            )
            for row in cursor.fetchall():
                sources.append({
                    "id": row[0],
                    "url": row[1],
                    "domain": row[2],
                    "title": row[3],
                    "trust_score": row[4],
                    "trust_tier": row[5],
                    "has_sanskrit_citations": bool(row[6]),
                    "has_verse_references": bool(row[7]),
                    "has_bibliography": bool(row[8]),
                    "is_academic_domain": bool(row[9]),
                    "created_at": row[10],
                })
        return jsonify({"sources": sources})
    except Exception as e:
        return jsonify({"error": str(e), "sources": []}), 500
