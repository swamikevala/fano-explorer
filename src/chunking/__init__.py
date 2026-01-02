"""
Atomic chunking module.

Extracts multiple atomic insights (1-3 sentences each) from exploration threads,
with dependency tracking to blessed axioms.
"""

from .models import (
    AtomicInsight,
    InsightStatus,
    InsightVersion,
    Refinement,
    VersionedInsight,
)
from .extractor import AtomicExtractor
from .dependencies import resolve_dependencies, find_keyword_matches
from .prompts import (
    build_refinement_prompt,
    build_post_refinement_review_prompt,
    parse_refinement_response,
    parse_post_refinement_review,
)

__all__ = [
    # Models
    "AtomicInsight",
    "InsightStatus",
    "InsightVersion",
    "Refinement",
    "VersionedInsight",
    # Extractor
    "AtomicExtractor",
    # Dependencies
    "resolve_dependencies",
    "find_keyword_matches",
    # Refinement prompts
    "build_refinement_prompt",
    "build_post_refinement_review_prompt",
    "parse_refinement_response",
    "parse_post_refinement_review",
]
