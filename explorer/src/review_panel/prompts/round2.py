"""
Round 2 prompt building and response parsing.

Round 2 is the deep analysis phase where each LLM sees all Round 1
responses and re-evaluates with extended reasoning.
"""

from typing import Optional, TYPE_CHECKING

from shared.prompts import MATH_FORMATTING_INSTRUCTION

if TYPE_CHECKING:
    from ..models import VerificationResult

from .round1 import parse_round1_response


def build_round2_prompt(
    chunk_insight: str,
    blessed_axioms_summary: str,
    gemini_response: dict,
    chatgpt_response: dict,
    claude_response: dict,
    this_llm: str,
    this_llm_round1_rating: str,
    math_verification: Optional["VerificationResult"] = None,
    accepted_modification: str = "",
    modification_source: str = "",
) -> str:
    """
    Build the Round 2 deep analysis prompt.

    Each LLM sees all Round 1 responses and re-evaluates with extended reasoning.

    Args:
        chunk_insight: The insight being reviewed (may be modified from Round 1)
        blessed_axioms_summary: Summary of blessed axioms
        gemini_response: Gemini's Round 1 response dict
        chatgpt_response: ChatGPT's Round 1 response dict
        claude_response: Claude's Round 1 response dict
        this_llm: Which LLM is receiving this prompt
        this_llm_round1_rating: This LLM's Round 1 rating
        math_verification: Optional DeepSeek verification result
        accepted_modification: If a modification was accepted after Round 1
        modification_source: Which LLM proposed the accepted modification

    Returns:
        The formatted Round 2 prompt
    """
    def format_response(resp: dict) -> str:
        base = f"""- Mathematical verification: {resp.get('mathematical_verification', 'N/A')}
- Structural analysis: {resp.get('structural_analysis', 'N/A')}
- Naturalness: {resp.get('naturalness_assessment', 'N/A')}
- Reasoning: {resp.get('reasoning', 'N/A')}"""
        # Include modification proposal if present
        if resp.get('proposed_modification'):
            base += f"\n- PROPOSED MODIFICATION: {resp.get('proposed_modification', '')[:200]}..."
            base += f"\n- Modification rationale: {resp.get('modification_rationale', 'N/A')}"
        return base

    # Build math verification section if available
    math_section = ""
    if math_verification:
        math_section = f"""

MATHEMATICAL VERIFICATION (DeepSeek V2 Math Prover):
{math_verification.summary_for_reviewers()}
"""

    # Build modification section if one was accepted
    modification_section = ""
    if accepted_modification:
        modification_section = f"""

NOTE: A MODIFICATION WAS ACCEPTED AFTER ROUND 1
Proposed by: {modification_source}
You are now reviewing the MODIFIED insight below (not the original).
"""

    return f"""You are a rigorous mathematician engaged in collaborative truth-seeking.
A proposed insight has received reviews. Your task is to deeply analyze and
determine the correct assessment.
{modification_section}
DELIBERATION PRINCIPLES:
- OPEN-MINDED about conclusions - you may have missed something
- UNCOMPROMISING about method - criteria never soften
- The goal is TRUTH, not consensus
- Changing your mind when presented with good arguments is intellectual honesty
- Refusing to change despite good arguments is stubbornness

SADHGURU'S CORE AXIOMS:
{blessed_axioms_summary if blessed_axioms_summary else "(none yet established)"}

CHUNK UNDER REVIEW:
{chunk_insight}

ROUND 1 ASSESSMENTS:

GEMINI ({gemini_response.get('rating', '?')}):
{format_response(gemini_response)}

CHATGPT ({chatgpt_response.get('rating', '?')}):
{format_response(chatgpt_response)}

CLAUDE ({claude_response.get('rating', '?')}):
{format_response(claude_response)}

YOUR ROUND 1 ASSESSMENT WAS: {this_llm_round1_rating}
{math_section}
RATING OPTIONS:
- ⚡ (Profound) - Mathematically sound, structurally deep, feels inevitable
- ? (Interesting) - Has merit but needs development or has minor issues
- ✗ (Reject) - Flawed, superficial, or incorrect
- ABANDON - Too vague, confused, or unsalvageable to be worth further review

TASK:
Consider the other perspectives seriously. They may have seen something
you missed. But do not lower your standards - only change if genuinely persuaded.

If you see a way to FIX the insight (different wording, corrected definition, etc.)
that would make it valid, you may propose a modification.

{MATH_FORMATTING_INSTRUCTION}

RESPOND IN THIS EXACT FORMAT:

NEW_INFORMATION: [What, if anything, did the other reviewers point out
                  that you hadn't fully considered?]

REASSESSMENT:
- Mathematical claims: [Any errors found by others? Any corrections?]
- Structural depth: [Did others reveal deeper/shallower connections?]
- Naturalness: [Did others' framing change how inevitable this feels?]

DOES_THIS_CHANGE_THINGS: [yes/no - and why]

UPDATED_RATING: [⚡ or ? or ✗ or ABANDON]

UPDATED_REASONING: [If changed: what convinced you.
                    If unchanged: why the arguments don't meet the bar
                    despite serious consideration.]

CONFIDENCE: [high / medium / low]

PROPOSED_MODIFICATION: [If you see a way to fix/improve the insight, provide the
COMPLETE rewritten insight here. If no modification needed, write "NONE"]

MODIFICATION_RATIONALE: [If proposing a modification, explain what you fixed and why
it makes the insight valid/better. If no modification, write "N/A"]
"""


def parse_round2_response(response: str) -> dict:
    """
    Parse a Round 2 review response.

    Args:
        response: Raw LLM response

    Returns:
        Dictionary with parsed fields including mind change info
    """
    result = parse_round1_response(response)  # Get base fields (includes modification fields)

    # Add Round 2 specific fields
    result["new_information"] = ""
    result["changed_mind"] = False

    lines = response.split("\n")
    current_field = None
    current_value = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("NEW_INFORMATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "new_information"
            current_value = [line_stripped[16:].strip()]
        elif line_stripped.startswith("DOES_THIS_CHANGE_THINGS:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            change_text = line_stripped[24:].strip().lower()
            result["changed_mind"] = "yes" in change_text
            current_field = None
            current_value = []
        elif line_stripped.startswith("UPDATED_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[15:].strip().upper()
            if "⚡" in rating_text or "PROFOUND" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text or "REJECT" in rating_text:
                result["rating"] = "✗"
            elif "ABANDON" in rating_text:
                result["rating"] = "ABANDON"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("UPDATED_REASONING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[18:].strip()]
        elif line_stripped.startswith("PROPOSED_MODIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "proposed_modification"
            current_value = [line_stripped[22:].strip()]
        elif line_stripped.startswith("MODIFICATION_RATIONALE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "modification_rationale"
            current_value = [line_stripped[23:].strip()]
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    # Clean up proposed_modification - check if it's "NONE" or "N/A"
    mod = result.get("proposed_modification", "").strip().upper()
    if mod in ["NONE", "N/A", ""]:
        result["proposed_modification"] = ""

    return result
