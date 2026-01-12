"""
Query API for explorer and documenter to access researcher findings.

Provides a clean interface for querying the research database.
Phase 1: Simple queries over the corpus. No semantic matching.
"""

from pathlib import Path
from typing import Optional

from .store.database import ResearcherDatabase
from .store.models import Finding, Source, CrossReference
from .analysis.cross_ref import CrossReferenceDetector


class ResearcherAPI:
    """
    Public API for querying researcher findings.

    Used by explorer and documenter to access research results.

    Phase 1 API - simple queries:
    - query(): Search findings by concepts, numbers, domains
    - get_number_info(): Where does a number appear across domains
    - get_number_patterns(): Numbers that appear in multiple domains
    - get_finding(): Get a specific finding
    - get_cross_references(): Get cross-references for a finding
    - get_statistics(): Database stats
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

        Example:
            # Find findings about chakras with the number 7
            findings = api.query(concepts=["chakra"], numbers=[7])

            # Find high-trust findings from yoga domain
            findings = api.query(domains=["hatha_yoga"], min_trust=80)
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

        This is the core "cross-domain pattern" query - where does
        this number appear and what significance does it have?

        Args:
            number: The number to look up (e.g., 7, 28, 108)

        Returns:
            Dict with:
            - number: The queried number
            - total_occurrences: How many times it appears
            - domains: Dict of domain -> {count, significances}
            - occurrences: List of individual occurrences with context
            - pattern_summary: Human-readable summary

        Example:
            info = api.get_number_info(28)
            # Returns:
            # {
            #     "number": 28,
            #     "total_occurrences": 5,
            #     "domains": {
            #         "astronomy": {"count": 2, "significances": ["nakshatras"]},
            #         "music": {"count": 1, "significances": ["tala cycle"]},
            #     },
            #     "occurrences": [...],
            #     "pattern_summary": "Number 28 appears in 2 domains: ..."
            # }
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

    def get_number_patterns(self, min_domains: int = 2) -> list[dict]:
        """
        Get numbers that appear across multiple domains.

        This surfaces the interesting cross-domain patterns -
        numbers that might encode structural relationships.

        Args:
            min_domains: Minimum number of domains (default 2)

        Returns:
            List of pattern dicts, sorted by domain count

        Example:
            patterns = api.get_number_patterns()
            # Returns:
            # [
            #     {"number": 7, "domain_count": 4, "domains": {...}},
            #     {"number": 28, "domain_count": 3, "domains": {...}},
            #     ...
            # ]
        """
        patterns = self.cross_ref.find_number_patterns()
        return [p for p in patterns if p.get("domain_count", 0) >= min_domains]

    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """
        Get a specific finding by ID.

        Args:
            finding_id: Finding ID

        Returns:
            Finding object or None
        """
        return self.db.get_finding(finding_id)

    def get_cross_references(self, finding_id: str) -> list[dict]:
        """
        Get cross-references for a finding.

        Cross-references link findings that share numbers or concepts
        across different domains.

        Args:
            finding_id: Finding ID

        Returns:
            List of cross-reference dicts with:
            - relationship_type: "same_number", "same_concept", etc.
            - relationship_value: The shared number or concept
            - strength: How strong the connection is (0-1)
            - other_finding_id: The linked finding

        Example:
            xrefs = api.get_cross_references("finding-123")
            for xref in xrefs:
                if xref["relationship_type"] == "same_number":
                    print(f"Shares number {xref['relationship_value']} with {xref['other_finding_id']}")
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

    def get_statistics(self) -> dict:
        """
        Get research database statistics.

        Returns:
            Dict with:
            - sources: Total source count
            - trusted_sources: Sources with trust >= 70
            - findings: Total finding count
            - concepts: Unique concept count
            - number_mentions: Total number mentions
            - cross_references: Total cross-reference count
            - top_numbers: Most mentioned numbers
            - cross_domain_numbers: Numbers appearing in multiple domains
            - top_patterns: Top cross-domain patterns
        """
        stats = self.db.get_statistics()

        # Add pattern statistics
        patterns = self.cross_ref.find_number_patterns()
        stats["cross_domain_numbers"] = len(patterns)
        stats["top_patterns"] = patterns[:5]

        return stats

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get source details by ID."""
        return self.db.get_source(source_id)
