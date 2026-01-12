"""
Convergence Detection - Determines when consensus has been reached.

Handles multiple convergence strategies:
- Decision-based (explicit DECISION: field)
- Structural (similar response patterns)
- Rating-based (for review tasks)
"""

from typing import Optional

from .response_parser import parse_decision, text_similarity


# Configuration constants
CONFIDENCE_THRESHOLD = 0.8
TEXT_SIMILARITY_THRESHOLD = 0.5
TEXT_CHUNK_LENGTH = 100


def check_convergence(
    responses: dict[str, str],
    response_format: Optional[str] = None,
) -> tuple[bool, float, Optional[str]]:
    """
    Check if responses have converged.

    Tries multiple strategies:
    1. Decision-based convergence (if DECISION: format expected)
    2. Structural similarity convergence (fallback)

    Args:
        responses: Dict mapping backend to response text
        response_format: Optional format hint from task

    Returns:
        Tuple of (converged, confidence, extracted_decision)
    """
    # Try decision-based convergence first
    if response_format and "DECISION:" in response_format.upper():
        result = check_decision_convergence(responses)
        if result[0] or result[1] > 0.5:  # If converged or has meaningful confidence
            return result

    # Fallback to structural similarity
    return check_structural_convergence(responses)


def check_decision_convergence(
    responses: dict[str, str],
) -> tuple[bool, float, Optional[str]]:
    """
    Check convergence based on explicit DECISION: fields.

    Args:
        responses: Dict mapping backend to response text

    Returns:
        Tuple of (converged, confidence, majority_decision)
    """
    decisions = {}
    for backend, text in responses.items():
        decision = parse_decision(text)
        if decision:
            decisions[backend] = decision

    if not decisions:
        return False, 0.5, None

    # Count votes
    vote_counts: dict[str, int] = {}
    for decision in decisions.values():
        vote_counts[decision] = vote_counts.get(decision, 0) + 1

    total = len(decisions)
    if total == 0:
        return False, 0.5, None

    max_votes = max(vote_counts.values())
    confidence = max_votes / total
    majority_decision = max(vote_counts, key=vote_counts.get)

    # Converged if all agree
    converged = max_votes == total
    return converged, confidence, majority_decision


def check_structural_convergence(
    responses: dict[str, str],
) -> tuple[bool, float, Optional[str]]:
    """
    Check convergence based on structural similarity of responses.

    Uses simple text similarity heuristic on response beginnings.

    Args:
        responses: Dict mapping backend to response text

    Returns:
        Tuple of (converged, confidence, None)
    """
    if len(responses) < 2:
        return False, 0.5, None

    # Filter out error responses
    valid_texts = [
        text for text in responses.values()
        if not text.startswith("[Error")
    ]

    if len(valid_texts) < 2:
        return False, 0.5, None

    # Check first chunks for similarity
    first_chunks = [t[:TEXT_CHUNK_LENGTH].lower() for t in valid_texts]

    # Compare all pairs
    all_similar = all(
        text_similarity(first_chunks[0], chunk) > TEXT_SIMILARITY_THRESHOLD
        for chunk in first_chunks[1:]
    )

    if all_similar:
        return True, 0.7, None

    return False, 0.5, None


def check_rating_convergence(
    ratings: list[str],
) -> tuple[bool, str]:
    """
    Check if review ratings have converged.

    Args:
        ratings: List of rating strings

    Returns:
        Tuple of (is_unanimous, majority_rating)
    """
    if not ratings:
        return False, "uncertain"

    unique_ratings = set(ratings)
    is_unanimous = len(unique_ratings) == 1

    if is_unanimous:
        return True, ratings[0]

    # Find majority (2 out of 3)
    rating_counts = {}
    for r in ratings:
        rating_counts[r] = rating_counts.get(r, 0) + 1

    for rating, count in rating_counts.items():
        if count >= 2:
            return False, rating

    # No clear majority
    return False, "uncertain"


def find_dissent(
    responses: dict[str, str],
    response_format: Optional[str] = None,
) -> Optional[str]:
    """
    Find dissenting view if not converged.

    Args:
        responses: Dict mapping backend to response text
        response_format: Optional format hint

    Returns:
        String describing minority position, or None.
    """
    if not response_format or "DECISION:" not in response_format.upper():
        return None

    # Extract decisions
    decisions = {}
    for backend, text in responses.items():
        decision = parse_decision(text)
        if decision:
            decisions[backend] = decision

    if len(set(decisions.values())) <= 1:
        return None

    # Group by decision
    vote_counts: dict[str, list[str]] = {}
    for backend, decision in decisions.items():
        if decision not in vote_counts:
            vote_counts[decision] = []
        vote_counts[decision].append(backend)

    # Get the minority
    minority_decision = min(vote_counts, key=lambda d: len(vote_counts[d]))
    minority_backends = vote_counts[minority_decision]

    # Return the minority view
    if minority_backends:
        backend = minority_backends[0]
        return f"[{backend}] voted {minority_decision}: {responses[backend][:500]}"

    return None


def synthesize_outcome(responses: dict[str, str]) -> str:
    """
    Synthesize outcome from multiple responses (fallback strategy).

    Returns the longest valid response, which is often the most complete.

    Args:
        responses: Dict mapping backend to response text

    Returns:
        Selected response text.
    """
    valid_responses = [
        (backend, text) for backend, text in responses.items()
        if not text.startswith("[Error")
    ]

    if not valid_responses:
        return "No valid responses received."

    # Return the longest response
    return max(valid_responses, key=lambda x: len(x[1]))[1]
