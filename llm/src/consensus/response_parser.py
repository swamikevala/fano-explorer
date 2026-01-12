"""
Response Parser - Extracts structured data from LLM responses.

Centralizes all response parsing logic to eliminate duplication
and provide consistent extraction across the consensus system.
"""

import re
from typing import Optional

from ..models import ReviewResponse


# Compiled regex patterns for better performance
RATING_PATTERN = re.compile(r'RATING:\s*(\w+)', re.IGNORECASE)
REASONING_PATTERN = re.compile(r'REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', re.IGNORECASE | re.DOTALL)
CONFIDENCE_PATTERN = re.compile(r'CONFIDENCE:\s*(\w+)', re.IGNORECASE)
MATH_VERIFICATION_PATTERN = re.compile(
    r'MATHEMATICAL_VERIFICATION:\s*(.+?)(?=\n[A-Z_]+:|$)', re.IGNORECASE | re.DOTALL
)
STRUCTURAL_ANALYSIS_PATTERN = re.compile(
    r'STRUCTURAL_ANALYSIS:\s*(.+?)(?=\n[A-Z_]+:|$)', re.IGNORECASE | re.DOTALL
)
NATURALNESS_PATTERN = re.compile(
    r'NATURALNESS:\s*(.+?)(?=\n[A-Z_]+:|$)', re.IGNORECASE | re.DOTALL
)
DECISION_PATTERN = re.compile(r'DECISION:\s*(\w+)', re.IGNORECASE)
REASON_PATTERN = re.compile(r'REASON:\s*(.+)', re.IGNORECASE)
VOTE_PATTERN = re.compile(r'VOTE:\s*\[?([A-Z])\]?', re.IGNORECASE)


def parse_review_response(
    llm: str,
    text: str,
    mode: str,
) -> ReviewResponse:
    """
    Parse LLM response into ReviewResponse.

    Args:
        llm: Backend name (e.g., 'claude', 'gemini')
        text: Raw LLM response text
        mode: Mode used ('standard' or 'deep')

    Returns:
        Parsed ReviewResponse with extracted fields.
    """
    # Default values
    rating = "uncertain"
    reasoning = text[:500]
    confidence = "medium"
    math_verification = ""
    structural_analysis = ""
    naturalness = ""

    # Extract rating
    rating_match = RATING_PATTERN.search(text)
    if rating_match:
        r = rating_match.group(1).lower()
        if "bless" in r or "⚡" in r:
            rating = "bless"
        elif "reject" in r or "✗" in r:
            rating = "reject"
        else:
            rating = "uncertain"

    # Extract reasoning
    reasoning_match = REASONING_PATTERN.search(text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()[:500]

    # Extract confidence
    confidence_match = CONFIDENCE_PATTERN.search(text)
    if confidence_match:
        confidence = confidence_match.group(1).lower()

    # Extract mathematical verification
    math_match = MATH_VERIFICATION_PATTERN.search(text)
    if math_match:
        math_verification = math_match.group(1).strip()[:300]

    # Extract structural analysis
    struct_match = STRUCTURAL_ANALYSIS_PATTERN.search(text)
    if struct_match:
        structural_analysis = struct_match.group(1).strip()[:300]

    # Extract naturalness assessment
    natural_match = NATURALNESS_PATTERN.search(text)
    if natural_match:
        naturalness = natural_match.group(1).strip()[:300]

    return ReviewResponse(
        llm=llm,
        mode=mode,
        rating=rating,
        reasoning=reasoning,
        confidence=confidence,
        mathematical_verification=math_verification,
        structural_analysis=structural_analysis,
        naturalness_assessment=naturalness,
    )


def parse_decision(text: str) -> Optional[str]:
    """
    Extract DECISION field from response.

    Args:
        text: Raw response text

    Returns:
        Extracted decision (uppercase) or None if not found.
    """
    match = DECISION_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


def parse_quick_rating(text: str) -> str:
    """
    Parse rating from quick check response.

    Args:
        text: Raw response text

    Returns:
        Rating string ('bless', 'reject', or 'uncertain').
    """
    text_lower = text.lower()
    if "bless" in text_lower or "valid" in text_lower:
        return "bless"
    elif "reject" in text_lower or "flawed" in text_lower:
        return "reject"
    return "uncertain"


def parse_quick_reason(text: str) -> str:
    """
    Extract reason from quick check response.

    Args:
        text: Raw response text

    Returns:
        Extracted reason or truncated response.
    """
    match = REASON_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text[:200]


def parse_vote(text: str) -> Optional[str]:
    """
    Extract vote letter from selection response.

    Args:
        text: Raw response text

    Returns:
        Vote letter (uppercase) or None if not found.
    """
    match = VOTE_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


def text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word overlap similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity (0.0 to 1.0).
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) if union else 0.0


def create_error_response(
    llm: str,
    mode: str,
    error: str,
    message: str,
) -> ReviewResponse:
    """
    Create a ReviewResponse for error cases.

    Args:
        llm: Backend name
        mode: Mode attempted
        error: Error type
        message: Error message

    Returns:
        ReviewResponse with uncertain rating and error info.
    """
    return ReviewResponse(
        llm=llm,
        mode=mode,
        rating="uncertain",
        reasoning=f"Error: {error} - {message}",
        confidence="low",
    )
