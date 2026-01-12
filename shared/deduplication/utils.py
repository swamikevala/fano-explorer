"""
Utility functions for deduplication.

Contains:
- Configuration loading
- Factory functions
- Convenience helpers
"""

from pathlib import Path
from typing import Callable, Optional, Awaitable

from shared.logging import get_logger

from .models import ContentItem, ContentType
from .text_processing import (
    extract_keywords,
    jaccard_similarity,
    calculate_similarity,
)
from .checker import DeduplicationChecker, LLMCallback

log = get_logger("shared", "deduplication.utils")


def load_dedup_config(config_path: Optional[str] = None) -> dict:
    """
    Load deduplication config from YAML file.

    Args:
        config_path: Path to config file. If None, tries to find fano/config.yaml

    Returns:
        Deduplication config dict
    """
    import yaml

    if config_path is None:
        # Try to find config.yaml in the fano project root
        fano_root = Path(__file__).resolve().parent.parent.parent
        config_path = fano_root / "config.yaml"

    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f) or {}
        return full_config.get("deduplication", {})
    except (FileNotFoundError, yaml.YAMLError) as e:
        log.warning(
            "deduplication.config.load_failed",
            config_path=str(config_path),
            error=str(e),
        )
        return {}


def get_dedup_checker(
    claude_reviewer=None,
    config: dict = None,
    llm_callback: Optional[LLMCallback] = None,
) -> DeduplicationChecker:
    """
    Factory function to create a DeduplicationChecker.

    This provides compatibility with the explorer's existing interface while
    using the new shared implementation. Configuration is read from the
    centralized config.yaml if not provided.

    Args:
        claude_reviewer: Object with send_message method (ClaudeReviewer interface).
                        Should be a cheap/fast model to preserve expensive quota.
        config: Configuration dict (optional). If None, loads from config.yaml
        llm_callback: Direct LLM callback (alternative to claude_reviewer)

    Returns:
        Configured DeduplicationChecker with LLM-first approach
    """
    # Load config from file if not provided
    if config is None:
        dedup_config = load_dedup_config()
    else:
        dedup_config = config.get("deduplication", config)

    # Create LLM callback if claude_reviewer is provided
    if llm_callback is None and claude_reviewer is not None:
        async def callback(prompt: str) -> str:
            return await claude_reviewer.send_message(prompt, extended_thinking=False)
        llm_callback = callback

    return DeduplicationChecker(
        llm_callback=llm_callback,
        use_signature_check=dedup_config.get("use_signature_check", True),
        use_heuristic_check=dedup_config.get("use_heuristic_check", False),
        use_llm_check=dedup_config.get("use_llm_check", True),
        use_batch_llm=dedup_config.get("use_batch_llm", True),
        batch_size=dedup_config.get("batch_size", 20),
        keyword_threshold=dedup_config.get("keyword_threshold", 0.40),
        concept_threshold=dedup_config.get("concept_threshold", 0.45),
        combined_threshold=dedup_config.get("combined_threshold", 0.50),
        require_high_confidence=dedup_config.get("require_high_confidence", False),
        stats_log_interval=dedup_config.get("stats_log_interval", 50),
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_content_item(
    id: str,
    text: str,
    content_type: str = "unknown",
    **metadata,
) -> ContentItem:
    """
    Convenience function to create a ContentItem.

    Args:
        id: Unique identifier
        text: The text content
        content_type: Type of content ("insight", "section", etc.)
        **metadata: Additional metadata

    Returns:
        ContentItem instance
    """
    try:
        ct = ContentType(content_type)
    except ValueError:
        ct = ContentType.UNKNOWN

    return ContentItem(
        id=id,
        text=text,
        content_type=ct,
        metadata=metadata,
    )


async def quick_duplicate_check(
    new_text: str,
    existing_texts: list[dict[str, str]],
    llm_callback: Optional[LLMCallback] = None,
) -> tuple[bool, Optional[str], str]:
    """
    Quick one-shot duplicate check.

    Args:
        new_text: Text to check
        existing_texts: List of {"id": str, "text": str} dicts
        llm_callback: Optional LLM callback for semantic check

    Returns:
        (is_duplicate, duplicate_of_id, reason)
    """
    checker = DeduplicationChecker(llm_callback=llm_callback)
    checker.load_from_dicts(existing_texts)

    result = await checker.check_duplicate(new_text)
    return result.is_duplicate, result.duplicate_of, result.reason


def is_similar_heuristic(
    text1: str,
    text2: str,
    threshold: float = 0.50,
) -> tuple[bool, float]:
    """
    Quick heuristic similarity check between two texts.

    Args:
        text1: First text
        text2: Second text
        threshold: Combined score threshold

    Returns:
        (is_similar, combined_score)
    """
    item1 = ContentItem(id="1", text=text1)
    item2 = ContentItem(id="2", text=text2)

    score = calculate_similarity(item1, item2)
    return score.combined_score >= threshold, score.combined_score


def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity based on keyword overlap.

    Backward-compatible function for explorer module.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity between 0.0 and 1.0
    """
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    return jaccard_similarity(keywords1, keywords2)
