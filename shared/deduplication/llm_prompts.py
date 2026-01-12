"""
LLM prompt building and response parsing for deduplication.

Contains:
- Pairwise LLM prompt building
- Batch LLM prompt building
- Response parsing
"""

import re
from typing import Optional


def build_pairwise_llm_prompt(text1: str, text2: str) -> str:
    """Build prompt for pairwise LLM duplicate confirmation."""
    return f"""Compare these two pieces of mathematical content and determine if they express the SAME core idea.

CONTENT A:
{text1[:1000]}

CONTENT B:
{text2[:1000]}

A duplicate means they make the SAME fundamental claim, even if:
- Different words are used
- One has more detail than the other
- The structure/ordering differs

NOT a duplicate if they make distinct claims, even if related.

Respond EXACTLY in this format:
IS_DUPLICATE: [yes/no]
CONFIDENCE: [high/medium/low]
REASON: [one sentence explanation]
BETTER_VERSION: [A/B/equal] (if duplicate, which is clearer)"""


def build_batch_llm_prompt(
    new_text: str,
    existing_items: list[dict[str, str]],
) -> str:
    """
    Build prompt for batch LLM duplicate check.

    Args:
        new_text: The new content to check
        existing_items: List of dicts with 'id' and 'text' keys
    """
    items_text = "\n\n".join([
        f"[{i+1}] (ID: {item['id']})\n{item['text'][:300]}{'...' if len(item['text']) > 300 else ''}"
        for i, item in enumerate(existing_items)
    ])

    return f"""You are checking if NEW content is a semantic duplicate of any EXISTING content.

NEW CONTENT:
{new_text[:800]}

EXISTING CONTENT ({len(existing_items)} items):

{items_text}

A duplicate means the NEW content expresses the SAME core claim as an existing item.
Even if worded differently or with different detail level, if the core idea is the same, it's a duplicate.

If it makes a DISTINCT claim (even if related/adjacent), it is NOT a duplicate.

Respond EXACTLY in this format:
IS_DUPLICATE: [yes/no]
DUPLICATE_OF: [number 1-{len(existing_items)} or 'none']
CONFIDENCE: [high/medium/low]
REASON: [one sentence explanation]"""


def parse_llm_duplicate_response(response: str) -> tuple[bool, Optional[int], str, str]:
    """
    Parse LLM response for duplicate check.

    Returns: (is_duplicate, duplicate_index, confidence, reason)
    """
    is_duplicate = False
    duplicate_idx = None
    confidence = "low"
    reason = ""

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("IS_DUPLICATE:"):
            value = line.replace("IS_DUPLICATE:", "").strip().lower()
            is_duplicate = value in ("yes", "true", "y")
        elif line.startswith("DUPLICATE_OF:"):
            value = line.replace("DUPLICATE_OF:", "").strip().lower()
            if value not in ("none", "n/a", "-"):
                try:
                    duplicate_idx = int(re.search(r'\d+', value).group()) - 1
                except (ValueError, AttributeError):
                    pass
        elif line.startswith("CONFIDENCE:"):
            confidence = line.replace("CONFIDENCE:", "").strip().lower()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return is_duplicate, duplicate_idx, confidence, reason
