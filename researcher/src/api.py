"""
Query API for explorer and documenter to access researcher findings.

Provides a clean interface for querying the research database.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from .store.database import ResearcherDatabase
from .store.models import Finding, NumberMention, Concept, Source
from .analysis.cross_ref import CrossReferenceDetector


class ResearcherAPI:
    """
    Public API for querying researcher findings.

    Used by explorer and documenter to access research results.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize API.

        Args:
            data_dir: Path to researcher/data directory
        """
        self.data_dir = data_dir
        self.db = ResearcherDatabase(data_dir / "researcher.db")
        self.cross_ref = CrossReferenceDetector(self.db)

    def query(
        self,
        concepts: list[str] = None,
        numbers: list[int] = None,
        domains: list[str] = None,
        min_trust: int = 50,
        min_confidence: float = 0.5,
        finding_types: list[str] = None,
        limit: int = 20
    ) -> list[Finding]:
        """
        Query findings with filters.

        Args:
            concepts: Filter by concept names
            numbers: Filter by number values
            domains: Filter by research domains
            min_trust: Minimum source trust score (0-100)
            min_confidence: Minimum extraction confidence (0-1)
            finding_types: Filter by type (supporting, challenging, etc.)
            limit: Maximum results to return

        Returns:
            List of matching findings
        """
        return self.db.search_findings(
            concepts=concepts,
            numbers=numbers,
            domains=domains,
            min_trust=min_trust,
            min_confidence=min_confidence,
            finding_types=finding_types,
            limit=limit
        )

    def get_number_info(self, number: int) -> dict:
        """
        Get information about a number across domains.

        Args:
            number: The number to look up

        Returns:
            Dict with occurrences, domains, significances
        """
        summary = self.db.get_number_summary(number)
        occurrences = self.db.get_number_occurrences(number)

        return {
            "number": number,
            "total_occurrences": summary.get("total_occurrences", 0),
            "domains": summary.get("domains", {}),
            "occurrences": [
                {
                    "context": o.context,
                    "significance": o.significance,
                    "mention_type": o.mention_type.value,
                    "domain": o.domain,
                }
                for o in occurrences[:20]  # Limit detail
            ],
            "pattern_summary": self.cross_ref.generate_pattern_summary(number),
        }

    def get_concept_info(self, name: str) -> Optional[dict]:
        """
        Get information about a concept.

        Args:
            name: Concept name or alias

        Returns:
            Dict with concept details and related findings
        """
        concept = self.db.get_concept_by_name(name)
        if not concept:
            return None

        # Get findings mentioning this concept
        findings = self.db.search_findings(concepts=[concept.id], limit=10)

        return {
            "concept": {
                "canonical_name": concept.canonical_name,
                "display_name": concept.display_name,
                "aliases": concept.aliases,
                "domain": concept.domain,
                "description": concept.description,
                "occurrence_count": concept.occurrence_count,
            },
            "related_concepts": concept.related_concepts,
            "sample_findings": [
                {
                    "summary": f.summary,
                    "quote": f.original_quote[:200] if f.original_quote else "",
                    "domain": f.domain,
                    "type": f.finding_type.value,
                }
                for f in findings
            ],
        }

    def find_connections(
        self,
        concept_or_number: str,
        limit: int = 10
    ) -> list[dict]:
        """
        Find cross-domain connections for a concept or number.

        Args:
            concept_or_number: Term to look up (concept name or number)
            limit: Maximum connections to return

        Returns:
            List of connection dicts
        """
        connections = []

        # Try as number
        try:
            number = int(concept_or_number)
            patterns = self.cross_ref.find_number_patterns()
            for pattern in patterns:
                if pattern["number"] == number:
                    connections.append({
                        "type": "number_pattern",
                        "value": number,
                        "domains": list(pattern["domains"].keys()),
                        "details": pattern,
                    })
                    break
        except ValueError:
            # Not a number, try as concept
            concept = self.db.get_concept_by_name(concept_or_number)
            if concept:
                findings = self.db.search_findings(concepts=[concept.id], limit=20)
                domains = list(set(f.domain for f in findings))
                if len(domains) > 1:
                    connections.append({
                        "type": "concept_cross_domain",
                        "value": concept.canonical_name,
                        "domains": domains,
                        "finding_count": len(findings),
                    })

        return connections[:limit]

    def get_supporting_evidence(
        self,
        claim: str,
        domain: Optional[str] = None
    ) -> list[dict]:
        """
        Find findings that might support a claim.

        Args:
            claim: The claim text
            domain: Optional domain to search in

        Returns:
            List of supporting findings with relevance info
        """
        # Extract keywords from claim
        import re
        words = re.findall(r'\b\w+\b', claim.lower())
        keywords = [w for w in words if len(w) > 3]

        # Search for supporting findings
        findings = self.db.search_findings(
            domains=[domain] if domain else None,
            finding_types=["supporting", "contextual"],
            limit=20
        )

        # Score relevance based on keyword overlap
        scored = []
        for finding in findings:
            score = 0
            text = (finding.summary + " " + finding.original_quote).lower()
            for kw in keywords:
                if kw in text:
                    score += 1

            if score > 0:
                scored.append({
                    "finding": {
                        "summary": finding.summary,
                        "quote": finding.original_quote,
                        "source_location": finding.source_location,
                        "domain": finding.domain,
                    },
                    "relevance_score": score / len(keywords) if keywords else 0,
                    "matching_keywords": [kw for kw in keywords if kw in text],
                })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored[:10]

    def get_challenging_evidence(
        self,
        claim: str,
        domain: Optional[str] = None
    ) -> list[dict]:
        """
        Find findings that might challenge a claim.

        Args:
            claim: The claim text
            domain: Optional domain to search in

        Returns:
            List of challenging findings
        """
        findings = self.db.search_findings(
            domains=[domain] if domain else None,
            finding_types=["challenging", "alternative"],
            limit=20
        )

        return [
            {
                "summary": f.summary,
                "quote": f.original_quote,
                "domain": f.domain,
                "type": f.finding_type.value,
            }
            for f in findings
        ]

    def get_statistics(self) -> dict:
        """
        Get research statistics.

        Returns:
            Dict with counts and top items
        """
        stats = self.db.get_statistics()

        # Add pattern statistics
        patterns = self.cross_ref.find_number_patterns()
        stats["cross_domain_numbers"] = len(patterns)
        stats["top_patterns"] = patterns[:5]

        return stats

    def get_recent_findings(self, limit: int = 20) -> list[Finding]:
        """
        Get most recently extracted findings.

        Args:
            limit: Maximum findings to return

        Returns:
            List of recent findings
        """
        # This would need a date-sorted query; for now use generic search
        return self.db.search_findings(limit=limit)

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get source details by ID."""
        return self.db.get_source(source_id)

    def get_source_by_url(self, url: str) -> Optional[Source]:
        """Get source details by URL."""
        return self.db.get_source_by_url(url)

    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """Get finding details by ID."""
        return self.db.get_finding(finding_id)

    def get_cross_references(self, finding_id: str) -> list[dict]:
        """
        Get cross-references for a finding.

        Args:
            finding_id: Finding ID

        Returns:
            List of cross-reference dicts
        """
        xrefs = self.db.get_cross_references(finding_id)
        return [
            {
                "relationship_type": xref.relationship_type,
                "relationship_value": xref.relationship_value,
                "strength": xref.strength,
                "other_finding_id": (
                    xref.finding_id_2 if xref.finding_id_1 == finding_id
                    else xref.finding_id_1
                ),
            }
            for xref in xrefs
        ]
