"""
Extraction prompts for atomic chunking.

Prompts designed to extract individual atomic insights (1-3 sentences each)
from exploration threads, with dependency tracking.
"""

from datetime import datetime


def build_extraction_prompt(
    thread_context: str,
    blessed_chunks_summary: str,
    max_insights: int = 10,
) -> str:
    """
    Build the extraction prompt for atomic insights.

    Args:
        thread_context: The exploration thread content to extract from
        blessed_chunks_summary: Summary of existing blessed axioms
        max_insights: Maximum number of insights to extract

    Returns:
        The formatted extraction prompt
    """
    date_prefix = f"[FANO {datetime.now().strftime('%m-%d')}]"

    return f"""{date_prefix} Extract INDIVIDUAL insights from this exploration as separate aphorisms.

RULES:
- Each insight must be ONE standalone claim or connection
- Maximum 1-3 sentences per insight
- Must be understandable on its own OR explicitly state what it depends on
- Only extract claims that are SPECIFIC and TESTABLE or PROFOUND and PRECISE
- Skip anything vague, hedging, or speculative ("might be", "could possibly", "perhaps")
- Do NOT bundle multiple ideas together
- Do NOT repeat the same insight with different wording
- Extract at most {max_insights} insights

DEPENDENCIES:
- If an insight REQUIRES a prior concept to make sense, note it in DEPENDS_ON
- Only list dependencies that are themselves worthy of being standalone insights
- Reference existing blessed axioms when possible (provided below)

BLESSED AXIOMS (available as foundations):
{blessed_chunks_summary if blessed_chunks_summary else "(none yet)"}

=== EXPLORATION CONTENT ===
{thread_context}

=== OUTPUT FORMAT ===
Format your response EXACTLY as:

===
INSIGHT: [single atomic aphorism here]
CONFIDENCE: [high/medium/low]
TAGS: [comma-separated concepts]
DEPENDS_ON: [comma-separated IDs of blessed axioms, or "none"]
PENDING_DEPENDS: [description of required prior insight not yet blessed, or "none"]
===
INSIGHT: [next aphorism]
CONFIDENCE: [high/medium/low]
TAGS: [concepts]
DEPENDS_ON: [IDs or "none"]
PENDING_DEPENDS: [description or "none"]
===
(continue for each distinct insight worth extracting)

If no insights meet the quality bar, respond with:
===
NO_INSIGHTS: [brief explanation why]
===
"""


def format_blessed_summary(blessed_insights: list) -> str:
    """
    Format blessed insights for inclusion in extraction prompt.

    Args:
        blessed_insights: List of BlessedInsight objects

    Returns:
        Formatted string summary
    """
    if not blessed_insights:
        return ""

    lines = []
    for insight in blessed_insights:
        # Handle both BlessedInsight and AtomicInsight
        if hasattr(insight, 'insight'):
            # AtomicInsight
            text = insight.insight
        elif hasattr(insight, 'content'):
            # BlessedInsight
            text = insight.content
        else:
            text = str(insight)

        insight_id = getattr(insight, 'id', 'unknown')
        tags = getattr(insight, 'tags', [])
        tags_str = f" [{', '.join(tags)}]" if tags else ""

        lines.append(f"- [{insight_id}]{tags_str}: {text[:200]}...")

    return "\n".join(lines)


def parse_extraction_response(response: str) -> list[dict]:
    """
    Parse the LLM extraction response into structured insights.

    Args:
        response: Raw LLM response following the extraction format

    Returns:
        List of dictionaries with insight data
    """
    insights = []

    # Check for no insights case
    if "NO_INSIGHTS:" in response:
        return []

    # Split by === delimiter
    blocks = response.split("===")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        insight_data = {}

        # Parse each field
        lines = block.split("\n")
        current_field = None
        current_value = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for field markers
            if line.startswith("INSIGHT:"):
                if current_field:
                    insight_data[current_field] = " ".join(current_value).strip()
                current_field = "insight"
                current_value = [line[8:].strip()]
            elif line.startswith("CONFIDENCE:"):
                if current_field:
                    insight_data[current_field] = " ".join(current_value).strip()
                current_field = "confidence"
                current_value = [line[11:].strip().lower()]
            elif line.startswith("TAGS:"):
                if current_field:
                    insight_data[current_field] = " ".join(current_value).strip()
                current_field = "tags"
                current_value = [line[5:].strip()]
            elif line.startswith("DEPENDS_ON:"):
                if current_field:
                    insight_data[current_field] = " ".join(current_value).strip()
                current_field = "depends_on"
                current_value = [line[11:].strip()]
            elif line.startswith("PENDING_DEPENDS:"):
                if current_field:
                    insight_data[current_field] = " ".join(current_value).strip()
                current_field = "pending_depends"
                current_value = [line[16:].strip()]
            elif current_field:
                current_value.append(line)

        # Save final field
        if current_field:
            insight_data[current_field] = " ".join(current_value).strip()

        # Validate and add insight
        if insight_data.get("insight"):
            # Parse tags into list
            tags_str = insight_data.get("tags", "")
            insight_data["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]

            # Parse depends_on into list
            depends_str = insight_data.get("depends_on", "none")
            if depends_str.lower() == "none":
                insight_data["depends_on"] = []
            else:
                insight_data["depends_on"] = [d.strip() for d in depends_str.split(",") if d.strip()]

            # Parse pending_depends
            pending_str = insight_data.get("pending_depends", "none")
            if pending_str.lower() == "none":
                insight_data["pending_dependencies"] = []
            else:
                insight_data["pending_dependencies"] = [pending_str]

            # Validate confidence
            if insight_data.get("confidence") not in ["high", "medium", "low"]:
                insight_data["confidence"] = "medium"

            insights.append(insight_data)

    return insights


def build_refinement_prompt(
    original_insight: str,
    confidence: str,
    tags: list[str],
    dependencies: list[str],
    gemini_rating: str,
    gemini_reasoning: str,
    chatgpt_rating: str,
    chatgpt_reasoning: str,
    claude_rating: str,
    claude_reasoning: str,
) -> str:
    """
    Build prompt for Claude Opus to refine an insight based on review critiques.

    This is used when reviews are mixed but critiques indicate fixable issues
    (articulation, precision, framing) rather than fundamental flaws.

    Args:
        original_insight: The original insight text
        confidence: Original extraction confidence
        tags: Original tags
        dependencies: Original dependency IDs
        gemini_rating: Gemini's rating (⚡/?/✗)
        gemini_reasoning: Gemini's critique reasoning
        chatgpt_rating: ChatGPT's rating (⚡/?/✗)
        chatgpt_reasoning: ChatGPT's critique reasoning
        claude_rating: Claude's rating (⚡/?/✗)
        claude_reasoning: Claude's critique reasoning

    Returns:
        The refinement prompt
    """
    tags_str = ", ".join(tags) if tags else "none"
    deps_str = ", ".join(dependencies) if dependencies else "none"

    return f"""An insight was extracted but received mixed reviews. Your task is to
REFINE the articulation based on the critiques, not to change the
underlying claim.

ORIGINAL CHUNK:
{original_insight}

EXTRACTION CONFIDENCE: {confidence}
TAGS: {tags_str}
DEPENDS ON: {deps_str}

REVIEWER CRITIQUES:

GEMINI ({gemini_rating}):
{gemini_reasoning}

CHATGPT ({chatgpt_rating}):
{chatgpt_reasoning}

CLAUDE ({claude_rating}):
{claude_reasoning}

TASK:
Rewrite the insight to address valid critiques while preserving
what is genuinely valuable.

YOU MAY:
- Sharpen vague language
- Correct minor errors noted by reviewers
- Reframe to make the structure clearer
- Add precision that reviewers noted was missing
- Remove hedging if the claim is actually solid
- Strengthen the mathematical grounding

YOU MAY NOT:
- Change the fundamental claim
- Add new claims not present in the original
- Weaken the insight just to avoid criticism
- Ignore valid critiques

RESPOND:

REFINED_INSIGHT: [1-3 sentences, precise, standalone]

CHANGES_MADE: [List each change and why]

ADDRESSED_CRITIQUES: [Which specific reviewer concerns this resolves]

UNRESOLVED_ISSUES: [Any critiques that couldn't be addressed
                    without changing the core claim — these may
                    need deliberation]

REFINEMENT_CONFIDENCE: [high/medium/low — how much better is this?]
"""


def build_post_refinement_review_prompt(
    original_insight: str,
    refined_insight: str,
    changes_made: str,
    summary_of_critiques: str,
    reviewer_original_rating: str,
    reviewer_original_critique: str,
) -> str:
    """
    Build prompt for reviewing a refined chunk.

    All 3 reviewers see both original and refined versions to evaluate
    whether the refinement addressed the issues.

    Args:
        original_insight: Original insight text
        refined_insight: Refined insight text
        changes_made: Description of changes made
        summary_of_critiques: Summary of critiques that prompted refinement
        reviewer_original_rating: This reviewer's Round 1 rating
        reviewer_original_critique: This reviewer's Round 1 critique

    Returns:
        The post-refinement review prompt
    """
    return f"""A chunk has been refined based on initial review feedback.
Evaluate the refined version.

ORIGINAL CHUNK:
{original_insight}

REFINED CHUNK:
{refined_insight}

CHANGES MADE:
{changes_made}

CRITIQUES THAT PROMPTED REFINEMENT:
{summary_of_critiques}

YOUR ORIGINAL RATING: {reviewer_original_rating}
YOUR ORIGINAL CRITIQUE: {reviewer_original_critique}

TASK:
Evaluate the refined version. Has the refinement addressed the issues?

RESPOND:

ISSUES_ADDRESSED: [yes/partially/no — which specific concerns were fixed?]

NEW_ISSUES: [Did refinement introduce any problems?]

RATING: ⚡ / ? / ✗

REASONING: [Justify rating for the refined version]

PREFER_ORIGINAL: [yes/no — in rare cases the original might be better]
"""


def parse_refinement_response(response: str) -> dict:
    """
    Parse Claude's refinement response.

    Args:
        response: Raw refinement response

    Returns:
        Dictionary with refinement data
    """
    result = {
        "refined_insight": "",
        "changes_made": [],
        "addressed_critiques": [],
        "unresolved_issues": [],
        "refinement_confidence": "medium",
    }

    # Parse REFINED_INSIGHT
    if "REFINED_INSIGHT:" in response:
        start = response.find("REFINED_INSIGHT:") + len("REFINED_INSIGHT:")
        end = response.find("\n\nCHANGES_MADE:", start)
        if end == -1:
            end = response.find("CHANGES_MADE:", start)
        if end != -1:
            result["refined_insight"] = response[start:end].strip()
        else:
            # Take until next section or end
            for marker in ["CHANGES_MADE:", "ADDRESSED_CRITIQUES:", "UNRESOLVED_ISSUES:", "REFINEMENT_CONFIDENCE:"]:
                pos = response.find(marker, start)
                if pos != -1:
                    result["refined_insight"] = response[start:pos].strip()
                    break
            else:
                result["refined_insight"] = response[start:].strip()

    # Parse CHANGES_MADE as list
    if "CHANGES_MADE:" in response:
        start = response.find("CHANGES_MADE:") + len("CHANGES_MADE:")
        end = response.find("ADDRESSED_CRITIQUES:", start)
        if end == -1:
            end = response.find("UNRESOLVED_ISSUES:", start)
        if end != -1:
            changes_text = response[start:end].strip()
        else:
            changes_text = response[start:response.find("REFINEMENT_CONFIDENCE:", start)].strip()

        # Parse bullet points or numbered list
        result["changes_made"] = _parse_list(changes_text)

    # Parse ADDRESSED_CRITIQUES as list
    if "ADDRESSED_CRITIQUES:" in response:
        start = response.find("ADDRESSED_CRITIQUES:") + len("ADDRESSED_CRITIQUES:")
        end = response.find("UNRESOLVED_ISSUES:", start)
        if end == -1:
            end = response.find("REFINEMENT_CONFIDENCE:", start)
        if end != -1:
            critiques_text = response[start:end].strip()
        else:
            critiques_text = response[start:].strip()

        result["addressed_critiques"] = _parse_list(critiques_text)

    # Parse UNRESOLVED_ISSUES as list
    if "UNRESOLVED_ISSUES:" in response:
        start = response.find("UNRESOLVED_ISSUES:") + len("UNRESOLVED_ISSUES:")
        end = response.find("REFINEMENT_CONFIDENCE:", start)
        if end != -1:
            issues_text = response[start:end].strip()
        else:
            issues_text = response[start:].strip()

        result["unresolved_issues"] = _parse_list(issues_text)

    # Parse REFINEMENT_CONFIDENCE
    if "REFINEMENT_CONFIDENCE:" in response:
        start = response.find("REFINEMENT_CONFIDENCE:") + len("REFINEMENT_CONFIDENCE:")
        conf_text = response[start:].strip().split()[0].lower()
        if conf_text in ["high", "medium", "low"]:
            result["refinement_confidence"] = conf_text

    return result


def parse_post_refinement_review(response: str) -> dict:
    """
    Parse a post-refinement review response.

    Args:
        response: Raw review response

    Returns:
        Dictionary with review data
    """
    result = {
        "issues_addressed": "partially",
        "new_issues": "",
        "rating": "?",
        "reasoning": "",
        "prefer_original": False,
    }

    # Parse ISSUES_ADDRESSED
    if "ISSUES_ADDRESSED:" in response:
        start = response.find("ISSUES_ADDRESSED:") + len("ISSUES_ADDRESSED:")
        end = response.find("NEW_ISSUES:", start)
        if end != -1:
            text = response[start:end].strip()
        else:
            text = response[start:response.find("RATING:", start)].strip()

        text_lower = text.lower()
        if text_lower.startswith("yes"):
            result["issues_addressed"] = "yes"
        elif text_lower.startswith("no"):
            result["issues_addressed"] = "no"
        else:
            result["issues_addressed"] = "partially"

    # Parse NEW_ISSUES
    if "NEW_ISSUES:" in response:
        start = response.find("NEW_ISSUES:") + len("NEW_ISSUES:")
        end = response.find("RATING:", start)
        if end != -1:
            result["new_issues"] = response[start:end].strip()

    # Parse RATING
    if "RATING:" in response:
        start = response.find("RATING:") + len("RATING:")
        end = response.find("REASONING:", start)
        if end != -1:
            rating_text = response[start:end].strip()
        else:
            rating_text = response[start:response.find("PREFER_ORIGINAL:", start)].strip()

        # Extract the rating symbol
        if "⚡" in rating_text:
            result["rating"] = "⚡"
        elif "✗" in rating_text:
            result["rating"] = "✗"
        else:
            result["rating"] = "?"

    # Parse REASONING
    if "REASONING:" in response:
        start = response.find("REASONING:") + len("REASONING:")
        end = response.find("PREFER_ORIGINAL:", start)
        if end != -1:
            result["reasoning"] = response[start:end].strip()
        else:
            result["reasoning"] = response[start:].strip()

    # Parse PREFER_ORIGINAL
    if "PREFER_ORIGINAL:" in response:
        start = response.find("PREFER_ORIGINAL:") + len("PREFER_ORIGINAL:")
        prefer_text = response[start:].strip().lower()
        result["prefer_original"] = prefer_text.startswith("yes")

    return result


def _parse_list(text: str) -> list[str]:
    """Parse a bullet or numbered list from text."""
    items = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove bullet points and numbers
        if line.startswith("-") or line.startswith("•"):
            line = line[1:].strip()
        elif line[0].isdigit() and (line[1:2] in [".", ")", ":"]):
            line = line[2:].strip()
        elif line[:2].isdigit() and (line[2:3] in [".", ")", ":"]):
            line = line[3:].strip()

        if line:
            items.append(line)

    return items
