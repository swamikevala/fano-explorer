"""
Prompt Builder - Centralizes prompt construction for consensus tasks.

All prompt templates and building logic is consolidated here,
making it easier to modify prompts and A/B test variations.
"""

from typing import Optional

from ..models import ReviewResponse


def build_review_round1_prompt(
    text: str,
    tags: list[str],
    context: str,
    confidence: str,
    dependencies: list[str],
) -> str:
    """
    Build prompt for Round 1 independent review.

    Args:
        text: The insight/claim to review
        tags: Tags for context
        context: Additional context
        confidence: Confidence level from extraction
        dependencies: Dependencies on other insights

    Returns:
        Formatted prompt string.
    """
    tags_str = ", ".join(tags) if tags else "none"
    deps_str = ", ".join(dependencies) if dependencies else "none"

    return f"""You are reviewing a mathematical/philosophical insight for validity.

INSIGHT TO REVIEW:
{text}

METADATA:
- Tags: {tags_str}
- Confidence: {confidence}
- Dependencies: {deps_str}

CONTEXT:
{context}

REVIEW CRITERIA:
1. Mathematical Verification: Are any numerical claims correct?
2. Structural Analysis: Is this a deep connection or superficial pattern?
3. Naturalness: Does this feel DISCOVERED (inevitable) or INVENTED (forced)?

RATE THIS INSIGHT:
- "bless" (⚡): Profound, verified, inevitable - should become an axiom
- "uncertain" (?): Interesting but needs more development
- "reject" (✗): Flawed, superficial, or unfalsifiable

FORMAT YOUR RESPONSE AS:
RATING: [bless/uncertain/reject]
MATHEMATICAL_VERIFICATION: [your analysis]
STRUCTURAL_ANALYSIS: [your analysis]
NATURALNESS: [your assessment]
REASONING: [2-4 sentences justifying your rating]
CONFIDENCE: [high/medium/low]
"""


def build_review_round2_prompt(
    text: str,
    round1_responses: dict[str, ReviewResponse],
    context: str,
) -> str:
    """
    Build prompt for Round 2 deep analysis.

    Args:
        text: The insight being reviewed
        round1_responses: Responses from Round 1
        context: Additional context

    Returns:
        Formatted prompt string.
    """
    # Summarize Round 1 responses
    r1_summary = []
    for llm, resp in round1_responses.items():
        r1_summary.append(f"{llm.upper()} rated {resp.rating}:")
        r1_summary.append(f"  Reasoning: {resp.reasoning}")
        if resp.mathematical_verification:
            r1_summary.append(f"  Math: {resp.mathematical_verification}")

    r1_text = "\n".join(r1_summary)

    return f"""DEEP ANALYSIS - Round 2

You previously reviewed this insight. Now consider the other reviewers' perspectives.

INSIGHT:
{text}

ROUND 1 REVIEWS:
{r1_text}

CONTEXT:
{context}

INSTRUCTIONS:
1. Consider what the other reviewers pointed out
2. Did they notice something you missed?
3. Has your assessment changed?

Provide your updated rating and reasoning.

FORMAT YOUR RESPONSE AS:
RATING: [bless/uncertain/reject]
NEW_INFORMATION: [what did others point out that you missed?]
CHANGED_MIND: [yes/no]
REASONING: [updated justification]
CONFIDENCE: [high/medium/low]
"""


def build_quick_check_prompt(text: str, context: str) -> str:
    """
    Build prompt for quick single-LLM check.

    Args:
        text: The text to check
        context: Optional context

    Returns:
        Formatted prompt string.
    """
    return f"""Quick review of this insight:

{text}

Context: {context}

Rate as: bless (valid), uncertain (needs work), or reject (flawed)
Give a one-sentence reason.

RATING:
REASON:"""


def build_consensus_initial_prompt(
    context: str,
    task: str,
    response_format: Optional[str] = None,
) -> str:
    """
    Build initial prompt for general consensus task.

    Args:
        context: Background information
        task: What the LLMs should do
        response_format: Optional format hint

    Returns:
        Formatted prompt string.
    """
    format_hint = f"\n\nRespond in this format:\n{response_format}" if response_format else ""

    return f"""CONTEXT:
{context}

TASK:
{task}{format_hint}"""


def build_consensus_deliberation_prompt(
    context: str,
    task: str,
    prev_responses: dict[str, str],
    response_format: Optional[str] = None,
) -> str:
    """
    Build deliberation prompt for subsequent consensus rounds.

    Args:
        context: Background information
        task: What the LLMs should do
        prev_responses: Responses from previous round
        response_format: Optional format hint

    Returns:
        Formatted prompt string.
    """
    others_text = format_responses(prev_responses)
    format_hint = f"\n\nRespond in this format:\n{response_format}" if response_format else ""

    return f"""CONTEXT:
{context}

TASK:
{task}

PREVIOUS RESPONSES FROM OTHER REVIEWERS:
{others_text}

Consider what others said. Continue the discussion or confirm your position.
If you've changed your mind, explain why.{format_hint}"""


def build_selection_prompt(
    context: str,
    task: str,
    candidates: list[dict],
) -> str:
    """
    Build prompt for selecting best response from candidates.

    Args:
        context: Background information
        task: Original task description
        candidates: List of candidate dicts with 'round', 'backend', 'text'

    Returns:
        Formatted prompt string.
    """
    candidate_texts = []
    for i, c in enumerate(candidates):
        letter = chr(65 + i)  # A, B, C, ...
        candidate_texts.append(f"[{letter}] (Round {c['round']}, {c['backend']}):\n{c['text']}\n")

    return f"""CONTEXT:
{context}

ORIGINAL TASK:
{task}

CANDIDATE RESPONSES:
{"".join(candidate_texts)}

SELECTION TASK:
Review all candidate responses above. Vote for the BEST one based on:
1. Accuracy and correctness
2. Clarity and completeness
3. How well it fulfills the original task

VOTE: [A/B/C/etc - single letter only]
REASONING: [Brief explanation of why this is best]
"""


def format_responses(responses: dict[str, str]) -> str:
    """
    Format responses for inclusion in prompts.

    Args:
        responses: Dict mapping backend name to response text

    Returns:
        Formatted string with labeled responses.
    """
    parts = []
    for backend, text in responses.items():
        parts.append(f"[{backend.upper()}]:\n{text}\n")
    return "\n".join(parts)
