"""
Dependency resolution for atomic insights.

Uses keyword-based matching to link new insights to existing blessed axioms.
"""

import re
from typing import Tuple


# Domain-specific terms that are meaningful even if short
DOMAIN_TERMS = {
    # Mathematical
    "fano", "plane", "line", "point", "incidence", "dual", "automorphism",
    "symmetry", "group", "order", "cycle", "permutation", "collineation",
    "projective", "affine", "geometry", "klein", "heawood",
    # Sanskrit/grammar
    "sanskrit", "panini", "sutra", "pratyahara", "sandhi", "vibhakti",
    "dhatu", "prakriti", "pratyaya", "samasa", "krit", "taddhita",
    "maheshvara", "shiva", "sutras",
    # Music
    "shruti", "swara", "raga", "tala", "saptak", "octave", "interval",
    "consonance", "dissonance", "mela", "thaat", "vadi", "samvadi",
    "22", "7", "12", "3",  # Key numbers
    # General
    "structure", "pattern", "relation", "connection", "mapping",
    "isomorphism", "correspondence", "natural", "inevitable",
}


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords from text.

    Args:
        text: Input text to extract keywords from

    Returns:
        Set of lowercase keywords
    """
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # Keep words that are:
    # - In domain terms, OR
    # - Longer than 4 characters (likely meaningful)
    keywords = set()
    for word in words:
        if word in DOMAIN_TERMS or len(word) > 4:
            keywords.add(word)

    return keywords


def calculate_keyword_overlap(keywords1: set[str], keywords2: set[str]) -> float:
    """
    Calculate overlap score between two keyword sets.

    Args:
        keywords1: First set of keywords
        keywords2: Second set of keywords

    Returns:
        Overlap score between 0.0 and 1.0
    """
    if not keywords1 or not keywords2:
        return 0.0

    intersection = keywords1 & keywords2
    # Use Jaccard-like similarity but weighted toward smaller set
    min_size = min(len(keywords1), len(keywords2))
    return len(intersection) / min_size if min_size > 0 else 0.0


def find_keyword_matches(
    description: str,
    blessed_insights: list,
    threshold: float = 0.5,
) -> list[str]:
    """
    Find blessed insights that match a dependency description.

    Args:
        description: Text description of the dependency
        blessed_insights: List of blessed insight objects
        threshold: Minimum overlap score to consider a match

    Returns:
        List of matching insight IDs
    """
    desc_keywords = extract_keywords(description)
    if not desc_keywords:
        return []

    matches = []
    for insight in blessed_insights:
        # Get text content from insight
        if hasattr(insight, 'insight'):
            text = insight.insight
        elif hasattr(insight, 'content'):
            text = insight.content
        else:
            text = str(insight)

        # Also include title if available
        if hasattr(insight, 'title'):
            text = f"{insight.title} {text}"

        insight_keywords = extract_keywords(text)
        overlap = calculate_keyword_overlap(desc_keywords, insight_keywords)

        if overlap >= threshold:
            insight_id = getattr(insight, 'id', None)
            if insight_id:
                matches.append((insight_id, overlap))

    # Sort by overlap score (highest first) and return just IDs
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]


def resolve_dependencies(
    pending_descriptions: list[str],
    blessed_insights: list,
    threshold: float = 0.5,
) -> Tuple[list[str], list[str]]:
    """
    Resolve pending dependency descriptions to blessed insight IDs.

    Args:
        pending_descriptions: List of dependency descriptions to resolve
        blessed_insights: List of available blessed insights
        threshold: Minimum match threshold

    Returns:
        Tuple of (resolved_ids, still_pending_descriptions)
    """
    resolved = []
    still_pending = []

    for desc in pending_descriptions:
        matches = find_keyword_matches(desc, blessed_insights, threshold)
        if matches:
            # Take the best match
            resolved.append(matches[0])
        else:
            still_pending.append(desc)

    return resolved, still_pending


def check_foundation_validity(
    insight_depends_on: list[str],
    blessed_ids: set[str],
) -> Tuple[bool, list[str]]:
    """
    Check if an insight's dependencies are all blessed.

    Args:
        insight_depends_on: List of dependency IDs
        blessed_ids: Set of currently blessed insight IDs

    Returns:
        Tuple of (all_valid, invalid_ids)
    """
    invalid = [dep_id for dep_id in insight_depends_on if dep_id not in blessed_ids]
    return len(invalid) == 0, invalid


def handle_demotion(
    demoted_id: str,
    demoted_content: str,
    all_insights: list,
) -> list:
    """
    Handle when a blessed insight is demoted - update dependent insights.

    When a blessed chunk is demoted, move its ID from depends_on to
    pending_dependencies for all insights that referenced it.

    Args:
        demoted_id: ID of the demoted insight
        demoted_content: Content of the demoted insight (for pending desc)
        all_insights: List of all insights to check

    Returns:
        List of affected insight IDs
    """
    affected = []

    for insight in all_insights:
        if demoted_id in insight.depends_on:
            # Remove from resolved dependencies
            insight.depends_on.remove(demoted_id)

            # Add to pending dependencies with description
            desc = f"Previously: {demoted_content[:100]}..."
            insight.pending_dependencies.append(desc)

            affected.append(insight.id)

    return affected
