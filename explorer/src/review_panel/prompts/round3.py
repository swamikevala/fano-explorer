"""
Round 3 prompt building and response parsing.

Round 3 is the structured deliberation phase where minority and majority
positions engage in direct argument and counter-argument.
"""

from shared.prompts import MATH_FORMATTING_INSTRUCTION


def build_round3_minority_prompt(
    chunk_insight: str,
    majority_rating: str,
    majority_count: int,
    majority_reasoning_summary: str,
    minority_rating: str,
    minority_count: int,
    minority_reasoning_summary: str,
) -> str:
    """
    Build the Round 3 prompt for minority position to state their case.

    Args:
        chunk_insight: The insight being reviewed
        majority_rating: The majority's rating
        majority_count: Number of reviewers in majority
        majority_reasoning_summary: Summary of majority reasoning
        minority_rating: The minority's rating
        minority_count: Number of reviewers in minority
        minority_reasoning_summary: Summary of minority reasoning

    Returns:
        The formatted Round 3 minority prompt
    """
    return f"""There remains disagreement after deep analysis.

THE CONTESTED INSIGHT:
{chunk_insight}

CURRENT POSITIONS:
- MAJORITY ({majority_rating}): {majority_count} reviewers
  {majority_reasoning_summary}

- MINORITY ({minority_rating}): {minority_count} reviewer(s)
  {minority_reasoning_summary}

YOU HOLD THE MINORITY POSITION.

TASK:
State your single strongest argument. Be maximally specific.
What exact criterion does this chunk meet or fail that the majority is
mis-evaluating? Point to specific mathematical facts, structural features,
or axiom connections.

IMPORTANT: If you believe the insight would be valid/profound with a
MODIFICATION to its wording, definitions, or claims, you may propose one.
Sometimes a small adjustment to a definition or scope reveals a genuine
mathematical truth that the current wording obscures.

{MATH_FORMATTING_INSTRUCTION}

RESPOND IN THIS EXACT FORMAT:

STRONGEST_ARGUMENT: [Your single most compelling point]

PROPOSED_MODIFICATION: [If you believe the insight should be reworded, provide
the COMPLETE rewritten insight here. If no modification needed, write "NONE"]

MODIFICATION_RATIONALE: [If proposing a modification, explain why this change
makes the insight more valid/profound. If no modification, write "N/A"]
"""


def build_round3_majority_response_prompt(
    chunk_insight: str,
    majority_rating: str,
    minority_strongest_argument: str,
    proposed_modification: str = "",
    modification_rationale: str = "",
) -> str:
    """
    Build the Round 3 prompt for majority to respond to minority's argument.

    Args:
        chunk_insight: The insight being reviewed
        majority_rating: The majority's rating
        minority_strongest_argument: The minority's strongest argument
        proposed_modification: Optional modified insight proposed by minority
        modification_rationale: Rationale for the proposed modification

    Returns:
        The formatted Round 3 majority response prompt
    """
    # Build modification section if one was proposed
    modification_section = ""
    if proposed_modification and proposed_modification.upper() not in ["NONE", "N/A", ""]:
        modification_section = f"""

THE MINORITY HAS ALSO PROPOSED A MODIFICATION:
{proposed_modification}

RATIONALE FOR MODIFICATION:
{modification_rationale}
"""

    return f"""THE CONTESTED INSIGHT:
{chunk_insight}

YOU HOLD THE MAJORITY POSITION ({majority_rating}).

THE MINORITY'S STRONGEST ARGUMENT:
{minority_strongest_argument}
{modification_section}
TASK:
Respond directly to this specific argument. Do not restate your general case.
Either:
- Refute it with specific counter-evidence
- Acknowledge it changes your assessment

If a modification was proposed, evaluate it carefully. Sometimes the minority
sees a valid mathematical truth that the current wording obscures. If the
modification reveals a genuine insight that the original wording missed,
accept it.

RESPOND IN THIS EXACT FORMAT:

RESPONSE: [Direct engagement with their specific point]

ACCEPT_MODIFICATION: [yes/no - if a modification was proposed, would adopting
it resolve your concerns and make the insight valid/profound?]

MODIFICATION_ASSESSMENT: [If a modification was proposed, explain why you
accept or reject it. What does it fix or fail to fix?]

DOES_THIS_CHANGE_YOUR_RATING: [yes/no]

UPDATED_RATING: [⚡ or ? or ✗ - only if changed]
"""


def build_round3_final_prompt(
    chunk_insight: str,
    minority_strongest_argument: str,
    majority_response: str,
    is_minority: bool,
    modified_insight: str = "",
    modification_accepted: bool = False,
) -> str:
    """
    Build the Round 3 final resolution prompt.

    Args:
        chunk_insight: The insight being reviewed (original or modified)
        minority_strongest_argument: The minority's strongest argument
        majority_response: The majority's response
        is_minority: Whether this prompt is for minority or majority
        modified_insight: If modification was accepted, the new wording
        modification_accepted: Whether a modification was accepted

    Returns:
        The formatted Round 3 final prompt
    """
    # Build modification context if one was accepted
    modification_context = ""
    insight_to_review = chunk_insight
    if modification_accepted and modified_insight:
        modification_context = f"""
NOTE: A MODIFICATION WAS PROPOSED AND ACCEPTED BY MAJORITY.
You are now voting on the MODIFIED insight, not the original.

ORIGINAL INSIGHT:
{chunk_insight}

MODIFIED INSIGHT (now under review):
{modified_insight}

"""
        insight_to_review = modified_insight

    if is_minority:
        return f"""FINAL RESOLUTION ROUND
{modification_context}
THE {'MODIFIED ' if modification_accepted else ''}INSIGHT UNDER REVIEW:
{insight_to_review}

MINORITY'S ARGUMENT:
{minority_strongest_argument}

MAJORITY'S RESPONSE:
{majority_response}

You made the minority argument. The majority has responded.
{f'Your proposed modification was ACCEPTED.' if modification_accepted else ''}

TASK: Cast your final vote on the {'modified ' if modification_accepted else ''}insight.
{'Since your modification was accepted, reconsider if this now meets the bar for blessing.' if modification_accepted else 'Either concede or maintain.'}

RESPOND IN THIS EXACT FORMAT:

FINAL_STANCE: [CONCEDE or MAINTAIN]

REASON: [If conceding: what convinced you. If maintaining: why their response fails to address your point]

FINAL_RATING: [⚡ or ? or ✗]
"""
    else:
        return f"""FINAL RESOLUTION ROUND
{modification_context}
THE {'MODIFIED ' if modification_accepted else ''}INSIGHT UNDER REVIEW:
{insight_to_review}

MINORITY'S ARGUMENT:
{minority_strongest_argument}

MAJORITY'S RESPONSE:
{majority_response}

You held the majority position. You have heard the full exchange.
{f'A modification was accepted. You are now rating the MODIFIED version.' if modification_accepted else ''}

RESPOND IN THIS EXACT FORMAT:

FINAL_RATING: [⚡ or ? or ✗]

ONE_SENTENCE_JUSTIFICATION: [final reasoning{' for the modified insight' if modification_accepted else ''}]
"""


def parse_round3_response(response: str, is_minority: bool) -> dict:
    """
    Parse a Round 3 review response.

    Args:
        response: Raw LLM response
        is_minority: Whether this is from minority position

    Returns:
        Dictionary with parsed fields
    """
    result = {
        "rating": "?",
        "strongest_argument": "",
        "response_to_argument": "",
        "final_stance": "",
        "reasoning": "",
        # Modification fields
        "proposed_modification": "",
        "modification_rationale": "",
        "accept_modification": False,
        "modification_assessment": "",
    }

    lines = response.split("\n")
    current_field = None
    current_value = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("STRONGEST_ARGUMENT:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "strongest_argument"
            current_value = [line_stripped[19:].strip()]
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
        elif line_stripped.startswith("ACCEPT_MODIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            accept_text = line_stripped[20:].strip().lower()
            result["accept_modification"] = "yes" in accept_text
            current_field = None
            current_value = []
        elif line_stripped.startswith("MODIFICATION_ASSESSMENT:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "modification_assessment"
            current_value = [line_stripped[24:].strip()]
        elif line_stripped.startswith("RESPONSE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "response_to_argument"
            current_value = [line_stripped[9:].strip()]
        elif line_stripped.startswith("FINAL_STANCE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            stance = line_stripped[13:].strip().upper()
            result["final_stance"] = "concede" if "CONCEDE" in stance else "maintain"
            current_field = None
            current_value = []
        elif line_stripped.startswith("FINAL_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[13:].strip()
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("DOES_THIS_CHANGE_YOUR_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            change_text = line_stripped[29:].strip().lower()
            result["changed_mind"] = "yes" in change_text
            current_field = None
            current_value = []
        elif line_stripped.startswith("UPDATED_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[15:].strip()
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("REASON:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[7:].strip()]
        elif line_stripped.startswith("ONE_SENTENCE_JUSTIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[27:].strip()]
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    # Clean up proposed_modification - check if it's "NONE" or "N/A"
    mod = result.get("proposed_modification", "").strip().upper()
    if mod in ["NONE", "N/A", ""]:
        result["proposed_modification"] = ""

    return result
