"""
Mathematical verification trigger detection.

Determines when insights should be sent to DeepSeek for rigorous
mathematical verification based on:
1. Tags indicating mathematical content
2. Content patterns suggesting verifiable claims
3. Reviewer concerns flagging math issues
"""

import re
from typing import Optional


# Tags that require mathematical verification
MATH_REQUIRED_TAGS = {
    'proof', 'theorem', 'lemma', 'corollary',
    'formula', 'equation', 'derivation',
    'group', 'isomorphism', 'automorphism',
    'geometry', 'projective', 'incidence',
    'arithmetic', 'number_theory',
}

# Content patterns suggesting mathematical claims
MATH_CONTENT_PATTERNS = [
    r'\b\d+\s*[+\-*/=]\s*\d+',           # Arithmetic: "72 + 12 = 84"
    r'\bexactly\s+\d+\b',                 # "exactly 7 points"
    r'\bprove[sd]?\b',                    # "proves", "proved"
    r'\btheorem\b',                       # "theorem"
    r'\bif and only if\b',                # "if and only if"
    r'\bisomorphic\s+to\b',               # "isomorphic to"
    r'\bgroup\s+of\s+order\s+\d+',        # "group of order 168"
    r'\bPSL|PGL|GL|SL|SO|SU\b',           # Group notation
    r'\b(fano|klein|heawood)\s+plane\b',  # Known structures
    r'\|\w+\|\s*=\s*\d+',                 # Cardinality: "|G| = 168"
    r'\bcyclic\s+group\b',                # "cyclic group"
    r'\bpermutation\b',                   # "permutation"
    r'\bcollineation\b',                  # "collineation"
    r'\bdual\b.*\bstructure\b',           # "dual structure"
]

# Patterns in reviewer responses suggesting math concerns
REVIEWER_MATH_CONCERN_PATTERNS = [
    r'verify\s+(the\s+)?math',
    r'check\s+(the\s+)?(arithmetic|calculation)',
    r'mathematical\s+(claim|error|mistake)',
    r'numerically\s+(correct|incorrect)',
    r'proof\s+(needed|required|missing)',
    r'not\s+sure\s+about\s+the\s+(numbers?|math)',
]


def needs_math_verification(
    insight: str,
    tags: list[str],
    reviewer_responses: Optional[list[str]] = None,
) -> tuple[bool, str]:
    """
    Determine if insight should be sent to DeepSeek for verification.

    Args:
        insight: The insight text
        tags: Tags assigned to the insight
        reviewer_responses: Optional list of Round 1 reviewer reasoning

    Returns:
        Tuple of (should_verify, reason)
    """
    # Check required tags
    tag_set = set(t.lower() for t in tags)
    matching_tags = MATH_REQUIRED_TAGS & tag_set
    if matching_tags:
        return True, f"tags: {', '.join(matching_tags)}"

    # Check content patterns
    insight_text = insight.lower()
    for pattern in MATH_CONTENT_PATTERNS:
        if re.search(pattern, insight_text, re.IGNORECASE):
            return True, f"content pattern: {pattern}"

    # Check reviewer concerns
    if reviewer_responses:
        all_responses = " ".join(reviewer_responses).lower()
        for pattern in REVIEWER_MATH_CONCERN_PATTERNS:
            if re.search(pattern, all_responses, re.IGNORECASE):
                return True, f"reviewer concern: {pattern}"

    return False, "no mathematical claims detected"


def extract_mathematical_claims(insight: str) -> list[str]:
    """
    Extract potential mathematical claims from insight text.

    This is a simple heuristic extraction - DeepSeek will do the
    rigorous claim extraction. This is for quick pre-filtering.

    Args:
        insight: The insight text

    Returns:
        List of potential claim strings
    """
    claims = []

    # Sentences with numbers and mathematical keywords
    sentences = re.split(r'[.!?]', insight)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if sentence has numerical content
        has_numbers = bool(re.search(r'\d+', sentence))

        # Check if sentence has mathematical keywords
        math_keywords = [
            'exactly', 'equals', 'is equal to', 'corresponds to',
            'maps to', 'isomorphic', 'has order', 'contains',
            'consists of', 'there are', 'there exist'
        ]
        has_math_keyword = any(kw in sentence.lower() for kw in math_keywords)

        if has_numbers and has_math_keyword:
            claims.append(sentence)

    return claims


def get_verification_priority(
    insight: str,
    tags: list[str],
    reviewer_responses: Optional[list[str]] = None,
) -> str:
    """
    Determine priority level for mathematical verification.

    Args:
        insight: The insight text
        tags: Tags assigned to the insight
        reviewer_responses: Optional list of Round 1 reviewer reasoning

    Returns:
        Priority level: "high", "medium", "low", or "skip"
    """
    should_verify, reason = needs_math_verification(
        insight, tags, reviewer_responses
    )

    if not should_verify:
        return "skip"

    # High priority: explicit proof/theorem claims
    high_priority_tags = {'proof', 'theorem', 'lemma', 'corollary'}
    if high_priority_tags & set(t.lower() for t in tags):
        return "high"

    # High priority: reviewer flagged math concern
    if "reviewer concern" in reason:
        return "high"

    # Medium priority: numerical/structural claims
    if "exactly" in insight.lower() or "isomorphic" in insight.lower():
        return "medium"

    # Low priority: other mathematical content
    return "low"
