"""
Cross-reference detection between findings.

Finds patterns and connections across different sources and domains.
"""

from datetime import datetime
from typing import Optional
import uuid

from shared.logging import get_logger
from ..store.models import CrossReference, Finding, NumberMention
from ..store.database import ResearcherDatabase

log = get_logger("researcher", "cross_ref")


class CrossReferenceDetector:
    """
    Detects cross-references between findings.

    Finds:
    - Same numbers appearing in different domains
    - Same concepts across sources
    - Supporting/contradicting findings
    """

    def __init__(self, database: ResearcherDatabase):
        """
        Initialize detector.

        Args:
            database: Researcher database
        """
        self.db = database

    def find_cross_references(
        self,
        finding: Finding,
        numbers: list[NumberMention]
    ) -> list[CrossReference]:
        """
        Find cross-references for a new finding.

        Args:
            finding: New finding to cross-reference
            numbers: Number mentions in this finding

        Returns:
            List of detected cross-references
        """
        cross_refs = []

        # Cross-reference by numbers
        for number in numbers:
            existing = self.db.get_number_occurrences(number.value)
            for other in existing:
                if other.finding_id != finding.id and other.domain != finding.domain:
                    # Same number in different domain - interesting!
                    xref = CrossReference(
                        id=str(uuid.uuid4())[:12],
                        finding_id_1=finding.id,
                        finding_id_2=other.finding_id,
                        relationship_type="same_number",
                        relationship_value=str(number.value),
                        strength=self._calculate_number_strength(number, other),
                    )
                    cross_refs.append(xref)
                    log.debug(
                        "cross_ref.number_match",
                        number=number.value,
                        domain1=finding.domain,
                        domain2=other.domain
                    )

        # Cross-reference by concepts
        for concept_id in finding.concepts:
            # Find other findings with same concept
            concept_findings = self.db.search_findings(
                concepts=[concept_id],
                limit=50
            )
            for other in concept_findings:
                if other.id != finding.id:
                    xref = CrossReference(
                        id=str(uuid.uuid4())[:12],
                        finding_id_1=finding.id,
                        finding_id_2=other.id,
                        relationship_type="same_concept",
                        relationship_value=concept_id,
                        strength=0.6,
                    )
                    cross_refs.append(xref)

        return cross_refs

    def _calculate_number_strength(
        self,
        num1: NumberMention,
        num2: NumberMention
    ) -> float:
        """Calculate strength of number cross-reference."""
        base_strength = 0.5

        # Boost if both are structural mentions
        from ..store.models import MentionType
        if num1.mention_type == MentionType.STRUCTURAL and num2.mention_type == MentionType.STRUCTURAL:
            base_strength += 0.3

        # Boost if both have same significance
        if num1.significance and num2.significance:
            if num1.significance.lower() == num2.significance.lower():
                base_strength += 0.2
            # Partial match
            elif any(w in num2.significance.lower() for w in num1.significance.lower().split()):
                base_strength += 0.1

        return min(1.0, base_strength)

    def find_number_patterns(self) -> list[dict]:
        """
        Find interesting patterns in number occurrences.

        Returns list of patterns with numbers that appear across multiple domains.
        """
        patterns = []

        # Get summary for all key numbers
        # Note: This would need optimization for large datasets
        from ..store.models import MentionType

        # Query database for numbers with multiple domain occurrences
        stats = self.db.get_statistics()
        top_numbers = stats.get("top_numbers", {})

        for number, count in top_numbers.items():
            if count >= 2:
                summary = self.db.get_number_summary(number)
                domains = summary.get("domains", {})

                if len(domains) >= 2:
                    patterns.append({
                        "number": number,
                        "domain_count": len(domains),
                        "total_occurrences": summary.get("total_occurrences", 0),
                        "domains": domains,
                        "pattern_type": "cross_domain",
                    })

        # Sort by domain count then occurrence count
        patterns.sort(key=lambda p: (p["domain_count"], p["total_occurrences"]), reverse=True)

        log.info("cross_ref.patterns_found", pattern_count=len(patterns))
        return patterns

    def generate_pattern_summary(self, number: int) -> str:
        """
        Generate a human-readable summary of number patterns.

        Args:
            number: The number to summarize

        Returns:
            Summary string
        """
        summary_data = self.db.get_number_summary(number)
        domains = summary_data.get("domains", {})

        if not domains:
            return f"Number {number}: No occurrences found"

        parts = [f"Number {number} appears in {len(domains)} domains:"]

        for domain, info in domains.items():
            significances = info.get("significances", [])
            sig_str = ", ".join(significances[:3]) if significances else "unspecified"
            parts.append(f"  - {domain}: {info['count']}x ({sig_str})")

        return "\n".join(parts)

    def find_supporting_findings(self, finding: Finding) -> list[Finding]:
        """
        Find findings that might support this one.

        Looks for same domain, similar concepts, supporting type.
        """
        from ..store.models import FindingType

        # Search for supporting findings in same domain
        supporting = self.db.search_findings(
            domains=[finding.domain],
            finding_types=["supporting"],
            limit=10
        )

        # Filter to those with overlapping concepts
        if finding.concepts:
            supporting = [
                f for f in supporting
                if any(c in f.concepts for c in finding.concepts)
            ]

        return [f for f in supporting if f.id != finding.id]

    def find_challenging_findings(self, finding: Finding) -> list[Finding]:
        """
        Find findings that might challenge this one.

        Looks for challenging type with similar concepts.
        """
        challenging = self.db.search_findings(
            finding_types=["challenging", "alternative"],
            limit=20
        )

        # Filter to those with overlapping concepts
        if finding.concepts:
            challenging = [
                f for f in challenging
                if any(c in f.concepts for c in finding.concepts)
            ]

        return [f for f in challenging if f.id != finding.id]
