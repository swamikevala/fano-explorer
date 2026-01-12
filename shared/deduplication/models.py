"""
Data models for deduplication.

Contains:
- ContentType enum
- ContentItem dataclass
- SimilarityScore dataclass
- DuplicateResult dataclass
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ContentType(Enum):
    """Types of content that can be deduplicated."""
    INSIGHT = "insight"
    SECTION = "section"
    PREREQUISITE = "prerequisite"
    COMMENT = "comment"
    UNKNOWN = "unknown"


@dataclass
class ContentItem:
    """
    A piece of content to check for duplicates.

    This is the unified representation used across all modules.
    """
    id: str
    text: str
    content_type: ContentType = ContentType.UNKNOWN
    created_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    # Computed on first access (lazy evaluation)
    _signature: Optional[str] = field(default=None, repr=False)
    _keywords: Optional[set] = field(default=None, repr=False)
    _concepts: Optional[set] = field(default=None, repr=False)
    _ngrams: Optional[set] = field(default=None, repr=False)

    def __post_init__(self):
        # Normalize content_type if passed as string
        if isinstance(self.content_type, str):
            try:
                self.content_type = ContentType(self.content_type)
            except ValueError:
                self.content_type = ContentType.UNKNOWN

    @property
    def signature(self) -> str:
        """Get content signature (MD5 of normalized text)."""
        if self._signature is None:
            from .text_processing import compute_signature
            self._signature = compute_signature(self.text)
        return self._signature

    @property
    def keywords(self) -> set[str]:
        """Get extracted keywords."""
        if self._keywords is None:
            from .text_processing import extract_keywords
            self._keywords = extract_keywords(self.text)
        return self._keywords

    @property
    def concepts(self) -> set[str]:
        """Get extracted domain concepts."""
        if self._concepts is None:
            from .text_processing import extract_concepts
            self._concepts = extract_concepts(self.text)
        return self._concepts

    @property
    def ngrams(self) -> set[str]:
        """Get character n-grams for fuzzy matching."""
        if self._ngrams is None:
            from .text_processing import extract_ngrams
            self._ngrams = extract_ngrams(self.text, n=4)
        return self._ngrams


@dataclass
class SimilarityScore:
    """Detailed similarity scores between two content items."""
    signature_match: bool = False  # Exact normalized match
    keyword_similarity: float = 0.0  # Jaccard on keywords
    concept_similarity: float = 0.0  # Jaccard on domain concepts
    ngram_similarity: float = 0.0  # Jaccard on character n-grams
    combined_score: float = 0.0  # Weighted combination
    llm_confirmed: Optional[bool] = None  # LLM semantic check result
    llm_explanation: str = ""

    @property
    def passed_heuristics(self) -> bool:
        """Whether the heuristic checks suggest a duplicate."""
        return (
            self.signature_match or
            self.keyword_similarity >= 0.50 or
            self.concept_similarity >= 0.55 or
            self.combined_score >= 0.55
        )


@dataclass
class DuplicateResult:
    """Result of a duplicate check."""
    is_duplicate: bool
    checked_item_id: str
    duplicate_of: Optional[str] = None  # ID of the matching content
    similarity: Optional[SimilarityScore] = None
    check_method: str = "none"  # "signature", "heuristic", "llm", "batch_llm"
    reason: str = ""
    check_time_ms: float = 0.0
