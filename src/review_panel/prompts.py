"""
Review prompts for the automated review panel.

Prompts for each round of the three-round review process.
"""

from datetime import datetime


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

RESPOND IN THIS EXACT FORMAT:

RATING: [⚡ or ? or ✗]

MATHEMATICAL_VERIFICATION: [Verify or refute specific claims. Be precise.]

STRUCTURAL_ANALYSIS: [Is the connection deep or superficial? Why?]

NATURALNESS_ASSESSMENT: [Does it feel inevitable? Explain.]

REASONING: [Overall justification for your rating, 2-4 sentences]

CONFIDENCE: [high / medium / low in your rating]
"""


def build_round2_prompt(
    chunk_insight: str,
    blessed_axioms_summary: str,
    gemini_response: dict,
    chatgpt_response: dict,
    claude_response: dict,
    this_llm: str,
    this_llm_round1_rating: str,
) -> str:
    """
    Build the Round 2 deep analysis prompt.

    Each LLM sees all Round 1 responses and re-evaluates with extended reasoning.

    Args:
        chunk_insight: The insight being reviewed
        blessed_axioms_summary: Summary of blessed axioms
        gemini_response: Gemini's Round 1 response dict
        chatgpt_response: ChatGPT's Round 1 response dict
        claude_response: Claude's Round 1 response dict
        this_llm: Which LLM is receiving this prompt
        this_llm_round1_rating: This LLM's Round 1 rating

    Returns:
        The formatted Round 2 prompt
    """
    def format_response(resp: dict) -> str:
        return f"""- Mathematical verification: {resp.get('mathematical_verification', 'N/A')}
- Structural analysis: {resp.get('structural_analysis', 'N/A')}
- Naturalness: {resp.get('naturalness_assessment', 'N/A')}
- Reasoning: {resp.get('reasoning', 'N/A')}"""

    return f"""You are a rigorous mathematician engaged in collaborative truth-seeking.
A proposed insight has received conflicting reviews. Your task is to
deeply analyze and determine the correct assessment.

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

TASK:
Consider the other perspectives seriously. They may have seen something
you missed. But do not lower your standards - only change if genuinely persuaded.

RESPOND IN THIS EXACT FORMAT:

NEW_INFORMATION: [What, if anything, did the other reviewers point out
                  that you hadn't fully considered?]

REASSESSMENT:
- Mathematical claims: [Any errors found by others? Any corrections?]
- Structural depth: [Did others reveal deeper/shallower connections?]
- Naturalness: [Did others' framing change how inevitable this feels?]

DOES_THIS_CHANGE_THINGS: [yes/no - and why]

UPDATED_RATING: [⚡ or ? or ✗]

UPDATED_REASONING: [If changed: what convinced you.
                    If unchanged: why the arguments don't meet the bar
                    despite serious consideration.]

CONFIDENCE: [high / medium / low]
"""


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

Do not restate your general position. Make your best case in one focused argument.

RESPOND IN THIS EXACT FORMAT:

STRONGEST_ARGUMENT: [Your single most compelling point]
"""


def build_round3_majority_response_prompt(
    chunk_insight: str,
    majority_rating: str,
    minority_strongest_argument: str,
) -> str:
    """
    Build the Round 3 prompt for majority to respond to minority's argument.

    Args:
        chunk_insight: The insight being reviewed
        majority_rating: The majority's rating
        minority_strongest_argument: The minority's strongest argument

    Returns:
        The formatted Round 3 majority response prompt
    """
    return f"""THE CONTESTED INSIGHT:
{chunk_insight}

YOU HOLD THE MAJORITY POSITION ({majority_rating}).

THE MINORITY'S STRONGEST ARGUMENT:
{minority_strongest_argument}

TASK:
Respond directly to this specific argument. Do not restate your general case.
Either:
- Refute it with specific counter-evidence
- Acknowledge it changes your assessment

RESPOND IN THIS EXACT FORMAT:

RESPONSE: [Direct engagement with their specific point]

DOES_THIS_CHANGE_YOUR_RATING: [yes/no]

UPDATED_RATING: [⚡ or ? or ✗ - only if changed]
"""


def build_round3_final_prompt(
    chunk_insight: str,
    minority_strongest_argument: str,
    majority_response: str,
    is_minority: bool,
) -> str:
    """
    Build the Round 3 final resolution prompt.

    Args:
        chunk_insight: The insight being reviewed
        minority_strongest_argument: The minority's strongest argument
        majority_response: The majority's response
        is_minority: Whether this prompt is for minority or majority

    Returns:
        The formatted Round 3 final prompt
    """
    if is_minority:
        return f"""FINAL RESOLUTION ROUND

THE CONTESTED INSIGHT:
{chunk_insight}

MINORITY'S ARGUMENT:
{minority_strongest_argument}

MAJORITY'S RESPONSE:
{majority_response}

You made the minority argument. The majority has responded.

TASK: Either concede or maintain.

RESPOND IN THIS EXACT FORMAT:

FINAL_STANCE: [CONCEDE or MAINTAIN]

REASON: [If conceding: what convinced you. If maintaining: why their response fails to address your point]

FINAL_RATING: [⚡ or ? or ✗]
"""
    else:
        return f"""FINAL RESOLUTION ROUND

THE CONTESTED INSIGHT:
{chunk_insight}

MINORITY'S ARGUMENT:
{minority_strongest_argument}

MAJORITY'S RESPONSE:
{majority_response}

You held the majority position. You have heard the full exchange.

RESPOND IN THIS EXACT FORMAT:

FINAL_RATING: [⚡ or ? or ✗]

ONE_SENTENCE_JUSTIFICATION: [final reasoning]
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
    }

    lines = response.split("\n")
    current_field = None
    current_value = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[7:].strip()
            # Extract just the rating symbol
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
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
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    return result


def parse_round2_response(response: str) -> dict:
    """
    Parse a Round 2 review response.

    Args:
        response: Raw LLM response

    Returns:
        Dictionary with parsed fields including mind change info
    """
    result = parse_round1_response(response)  # Get base fields

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
            rating_text = line_stripped[15:].strip()
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("UPDATED_REASONING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[18:].strip()]
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    return result


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
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    return result
