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
