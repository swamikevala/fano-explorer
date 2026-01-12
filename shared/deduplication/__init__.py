"""
Robust Deduplication Module for Fano.

Provides multi-layer duplicate detection that works across modules:
- Explorer: Check insights against each other
- Documenter: Check insights against document sections

Detection layers (in order of speed):
1. Content signatures (MD5 hash of normalized text) - instant
2. Keyword/concept overlap (Jaccard similarity) - fast
3. N-gram similarity (for partial matches) - fast
4. LLM semantic confirmation - thorough but slower

Usage:
    from shared.deduplication import DeduplicationChecker, ContentItem

    # Create checker with LLM callback
    async def llm_check(prompt: str) -> str:
        response = await llm_client.send("claude", prompt)
        return response.text

    checker = DeduplicationChecker(llm_callback=llm_check)

    # Add known content
    checker.add_content(ContentItem(id="section_1", text="...", content_type="section"))

    # Check for duplicates
    result = await checker.check_duplicate("new insight text...")
    if result.is_duplicate:
        print(f"Duplicates: {result.duplicate_of}")
"""

# Models
from .models import (
    ContentType,
    ContentItem,
    SimilarityScore,
    DuplicateResult,
)

# Main checker class
from .checker import (
    DeduplicationChecker,
    LLMCallback,
)

# Text processing functions
from .text_processing import (
    normalize_text,
    compute_signature,
    stem_word,
    extract_keywords,
    extract_concepts,
    extract_ngrams,
    jaccard_similarity,
    calculate_similarity,
    calculate_keyword_similarity,
)

# LLM prompts
from .llm_prompts import (
    build_pairwise_llm_prompt,
    build_batch_llm_prompt,
    parse_llm_duplicate_response,
)

# Utility functions
from .utils import (
    load_dedup_config,
    get_dedup_checker,
    create_content_item,
    quick_duplicate_check,
    is_similar_heuristic,
)

__all__ = [
    # Models
    "ContentType",
    "ContentItem",
    "SimilarityScore",
    "DuplicateResult",
    # Checker
    "DeduplicationChecker",
    "LLMCallback",
    # Text processing
    "normalize_text",
    "compute_signature",
    "stem_word",
    "extract_keywords",
    "extract_concepts",
    "extract_ngrams",
    "jaccard_similarity",
    "calculate_similarity",
    "calculate_keyword_similarity",
    # LLM prompts
    "build_pairwise_llm_prompt",
    "build_batch_llm_prompt",
    "parse_llm_duplicate_response",
    # Utils
    "load_dedup_config",
    "get_dedup_checker",
    "create_content_item",
    "quick_duplicate_check",
    "is_similar_heuristic",
]
