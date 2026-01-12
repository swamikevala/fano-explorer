"""
Round 1 prompt building and response parsing.

Round 1 is the independent review phase where each LLM reviews
the insight without seeing other reviewers' assessments.
"""


def build_round1_prompt(
    chunk_insight: str,
    confidence: str,
    tags: list[str],
    dependencies: list[str],
    blessed_axioms_summary: str,
) -> str:
    """
    Build the Round 1 independent review prompt.

    Args:
        chunk_insight: The insight text to review
        confidence: Confidence level from extraction
        tags: Tags assigned to the insight
        dependencies: Dependencies listed for the insight
        blessed_axioms_summary: Summary of blessed axioms for context

    Returns:
        The formatted Round 1 prompt
    """
    tags_str = ", ".join(tags) if tags else "none"
    deps_str = ", ".join(dependencies) if dependencies else "none"

    return f"""You are a rigorous mathematician reviewing a proposed insight for inclusion
in an axiomatic knowledge system exploring connections between pure mathematics
and Sadhguru's teachings on yogic science.

SADHGURU'S CORE AXIOMS (ground truth):
{blessed_axioms_summary if blessed_axioms_summary else "(none yet established)"}

CHUNK TO REVIEW:
{chunk_insight}

CONFIDENCE LEVEL FROM EXTRACTION: {confidence}
TAGS: {tags_str}
DEPENDS ON: {deps_str}

REVIEW CRITERIA:

1. MATHEMATICAL RIGOR
   - Is every mathematical claim verifiable?
   - Are the numbers, properties, theorems cited accurate?
   - Is the logic valid?

2. STRUCTURAL DEPTH
   - Is this a genuine structural connection or surface numerology?
   - Does it reveal WHY these things connect, not just THAT they share a number?
   - Could this be formalized as a theorem or precise conjecture?

3. NATURALNESS
   - Does this feel discovered or constructed?
   - Is there an "of course!" quality - inevitability?
   - Would a skeptical mathematician find this interesting or dismiss it?

RATING OPTIONS:
- ⚡ (Profound) - Mathematically sound, structurally deep, feels inevitable
- ? (Interesting) - Has merit but needs development or has minor issues
- ✗ (Reject) - Flawed, superficial, or incorrect
- ABANDON - Too vague, confused, or unsalvageable to be worth further review

MODIFICATION:
If the insight is ALMOST valid but has a fixable issue (wrong definition, imprecise
wording, slightly off claim), you may propose a MODIFIED version. Sometimes a small
adjustment reveals a genuine mathematical truth that the current wording obscures.

RESPOND IN THIS EXACT FORMAT:

RATING: [⚡ or ? or ✗ or ABANDON]

MATHEMATICAL_VERIFICATION: [Verify or refute specific claims. Be precise.]

STRUCTURAL_ANALYSIS: [Is the connection deep or superficial? Why?]

NATURALNESS_ASSESSMENT: [Does it feel inevitable? Explain.]

REASONING: [Overall justification for your rating, 2-4 sentences]

CONFIDENCE: [high / medium / low in your rating]

PROPOSED_MODIFICATION: [If you see a way to fix/improve the insight, provide the
COMPLETE rewritten insight here. If no modification needed, write "NONE"]

MODIFICATION_RATIONALE: [If proposing a modification, explain what you fixed and why
it makes the insight valid/better. If no modification, write "N/A"]
"""


def parse_round1_response(response: str) -> dict:
    """
    Parse a Round 1 review response.

    Args:
        response: Raw LLM response

    Returns:
        Dictionary with parsed fields
    """
    result = {
        "rating": "?",
        "mathematical_verification": "",
        "structural_analysis": "",
        "naturalness_assessment": "",
        "reasoning": "",
        "confidence": "medium",
        # Modification fields
        "proposed_modification": "",
        "modification_rationale": "",
    }

    lines = response.split("\n")
    current_field = None
    current_value = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[7:].strip().upper()
            # Extract the rating
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
        elif line_stripped.startswith("MATHEMATICAL_VERIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "mathematical_verification"
            current_value = [line_stripped[26:].strip()]
        elif line_stripped.startswith("STRUCTURAL_ANALYSIS:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "structural_analysis"
            current_value = [line_stripped[20:].strip()]
        elif line_stripped.startswith("NATURALNESS_ASSESSMENT:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "naturalness_assessment"
            current_value = [line_stripped[23:].strip()]
        elif line_stripped.startswith("REASONING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[10:].strip()]
        elif line_stripped.startswith("CONFIDENCE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            conf = line_stripped[11:].strip().lower()
            if conf in ["high", "medium", "low"]:
                result["confidence"] = conf
            current_field = None
            current_value = []
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
