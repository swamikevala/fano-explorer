"""
Atomic chunking module.

Extracts multiple atomic insights (1-3 sentences each) from exploration threads,
with dependency tracking to blessed axioms.
"""

from .models import AtomicInsight, InsightStatus
from .extractor import AtomicExtractor
from .dependencies import resolve_dependencies, find_keyword_matches

__all__ = [
    "AtomicInsight",
    "InsightStatus",
    "AtomicExtractor",
    "resolve_dependencies",
    "find_keyword_matches",
]
