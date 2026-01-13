"""
Round 4 prompt building and response parsing.

Round 4 is the modification-focused phase for disputed insights that
couldn't reach consensus after 3 rounds. Shows full deliberation history
and asks for modification proposals.
"""

from shared.prompts import MATH_FORMATTING_INSTRUCTION


def build_round4_modification_prompt(
    chunk_insight: str,
    round1_summary: str,
    round2_summary: str,
    round3_summary: str,
    final_ratings: dict[str, str],
) -> str:
    """
    Build the Round 4 prompt for modification-focused review.

    This is used for disputed insights that went through all 3 rounds
    but couldn't reach consensus. Shows full deliberation history and
    asks for modification proposals.

    Args:
        chunk_insight: The insight being reviewed
        round1_summary: Summary of Round 1 responses
        round2_summary: Summary of Round 2 deep analysis
        round3_summary: Summary of Round 3 deliberation
        final_ratings: Dict of LLM name to their final rating

    Returns:
        The formatted Round 4 modification prompt
    """
    ratings_str = ", ".join(f"{llm}: {rating}" for llm, rating in final_ratings.items())

    return f"""ROUND 4: MODIFICATION FOCUS

=== CRITICAL PHILOSOPHY ===

This is a DISCOVERY system, not a proof verification system. The original insight
text was a rough approximation - an attempt to articulate something glimpsed but
not yet precisely understood. Our goal is to UNCOVER PROFOUND TRUTHS, not to
defend or prove pre-written statements.

EVOLVING DEFINITIONS IS THE GOAL, NOT A FLAW.

When a modification changes "what X means" to make the mathematics work, that IS
the discovery. Finding the RIGHT definition that reveals elegant structure is
exactly what we're looking for. The original wording was just a starting point.

Do NOT criticize modifications for "changing the original definition." Instead ask:
- Does the MODIFIED insight reveal something profound and true?
- Is the new definition mathematically natural and elegant?
- Does this feel DISCOVERED rather than forced?

=== THE DISPUTED INSIGHT ===

{chunk_insight}

FINAL RATINGS AFTER 3 ROUNDS: {ratings_str}

=== FULL DELIBERATION HISTORY ===

ROUND 1 (Independent Review):
{round1_summary}

ROUND 2 (Deep Analysis):
{round2_summary}

ROUND 3 (Structured Deliberation):
{round3_summary}

=== YOUR TASK ===

Propose a modification that transforms this into a PROFOUND insight. You have
complete freedom to redefine terms, change the mathematical framing, or even
substantially rewrite the claim - as long as the result is TRUE and INTERESTING.

The question is NOT "does this prove the original statement?"
The question IS "can we find a TRUE and PROFOUND statement in this territory?"

Consider:
- What mathematical structure is actually present here?
- What definition SHOULD the terms have to make something elegant emerge?
- Is there a genuine discovery hiding behind the imprecise original wording?
- What would make this feel INEVITABLE rather than arbitrary?

{MATH_FORMATTING_INSTRUCTION}

RESPOND IN THIS EXACT FORMAT:

CAN_BE_FIXED: [YES / NO - Can we find a profound truth in this territory?]

DIAGNOSIS: [What was imprecise or wrong in the original? 2-3 sentences]

PROPOSED_MODIFICATION: [If CAN_BE_FIXED is YES, provide the COMPLETE rewritten
insight here. This should be the full text, not a diff. Freely redefine terms
if needed - just make sure the result is mathematically sound and interesting.
If NO, write "NONE"]

MODIFICATION_RATIONALE: [Explain the key changes. If you redefined terms, explain
why your definitions are more natural. Show how the modification addresses the
mathematical concerns while revealing something genuinely interesting.]

EXPECTED_RATING: [What rating would you give the MODIFIED insight? ⚡ or ? or ✗]
"""


def build_round4_final_vote_prompt(
    original_insight: str,
    modified_insight: str,
    modification_source: str,
    modification_rationale: str,
    round3_summary: str,
) -> str:
    """
    Build the Round 4 final vote prompt after modification is proposed.

    Args:
        original_insight: The original disputed insight
        modified_insight: The proposed modified insight
        modification_source: Which LLM proposed the modification
        modification_rationale: Why the modification was made
        round3_summary: Summary of original Round 3 deliberation

    Returns:
        The formatted Round 4 final vote prompt
    """
    return f"""ROUND 4: FINAL VOTE ON MODIFIED INSIGHT

=== CRITICAL PHILOSOPHY ===

This is a DISCOVERY system. The original insight was a rough approximation.
The modification may have CHANGED DEFINITIONS - this is not a flaw, it's the
entire point. Finding the RIGHT definition that makes elegant mathematics
emerge IS the discovery.

DO NOT penalize the modification for "not proving the original statement."
The original statement was just a starting point for exploration.

JUDGE THE MODIFIED INSIGHT ON ITS OWN MERITS:
- Is it mathematically TRUE?
- Is it INTERESTING and ELEGANT?
- Does it reveal genuine structure?
- Does it feel DISCOVERED rather than forced?

If the modified insight is profound and true, rate it ⚡ - even if it's
quite different from the original. We're hunting for truth, not defending
initial guesses.

=== ORIGINAL INSIGHT (for context only) ===
{original_insight}

=== MODIFIED INSIGHT (proposed by {modification_source}) ===
{modified_insight}

=== MODIFICATION RATIONALE ===
{modification_rationale}

=== CONTEXT FROM ROUND 3 DELIBERATION ===
{round3_summary}

=== YOUR TASK ===

Vote on the MODIFIED insight based on its own truth and profundity.
Forget about the original - does THIS statement reveal something genuine?

RESPOND IN THIS EXACT FORMAT:

RATING: [⚡ or ? or ✗]

IS_MODIFIED_INSIGHT_PROFOUND: [YES / NO / PARTIALLY - Judge the NEW insight
on its own merits, not on fidelity to the original]

REASONING: [2-3 sentences explaining your rating. Focus on whether the
MODIFIED insight is true and interesting, not on whether it "proves"
the original statement.]

CONFIDENCE: [high / medium / low]
"""


def parse_round4_modification_response(response: str) -> dict:
    """Parse a Round 4 modification proposal response."""
    result = {
        "can_be_fixed": False,
        "diagnosis": "",
        "proposed_modification": "",
        "modification_rationale": "",
        "expected_rating": "?",
    }

    current_field = None
    current_value = []

    for line in response.split("\n"):
        line_stripped = line.strip()

        if line_stripped.startswith("CAN_BE_FIXED:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            answer = line_stripped[13:].strip().upper()
            result["can_be_fixed"] = "YES" in answer
            current_field = None
            current_value = []
        elif line_stripped.startswith("DIAGNOSIS:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "diagnosis"
            current_value = [line_stripped[10:].strip()]
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
        elif line_stripped.startswith("EXPECTED_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[16:].strip()
            if "⚡" in rating_text:
                result["expected_rating"] = "⚡"
            elif "✗" in rating_text:
                result["expected_rating"] = "✗"
            else:
                result["expected_rating"] = "?"
            current_field = None
            current_value = []
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    # Clean up proposed_modification
    mod = result.get("proposed_modification", "").strip().upper()
    if mod in ["NONE", "N/A", ""]:
        result["proposed_modification"] = ""
        result["can_be_fixed"] = False

    return result


def parse_round4_final_vote_response(response: str) -> dict:
    """Parse a Round 4 final vote response."""
    result = {
        "rating": "?",
        "resolves_dispute": "",
        "reasoning": "",
        "confidence": "medium",
    }

    current_field = None
    current_value = []

    for line in response.split("\n"):
        line_stripped = line.strip()

        if line_stripped.startswith("RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[7:].strip()
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("DOES_MODIFICATION_RESOLVE_DISPUTE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            result["resolves_dispute"] = line_stripped[34:].strip()
            current_field = None
            current_value = []
        elif line_stripped.startswith("IS_MODIFIED_INSIGHT_PROFOUND:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            result["resolves_dispute"] = line_stripped[29:].strip()
            current_field = None
            current_value = []
        elif line_stripped.startswith("REASONING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[10:].strip()]
        elif line_stripped.startswith("CONFIDENCE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            conf = line_stripped[11:].strip().lower()
            if "high" in conf:
                result["confidence"] = "high"
            elif "low" in conf:
                result["confidence"] = "low"
            else:
                result["confidence"] = "medium"
            current_field = None
            current_value = []
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    return result
