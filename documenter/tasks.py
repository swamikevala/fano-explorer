"""
Consensus task builders - constructing prompts for the LLM consensus.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

log = get_logger("documenter", "tasks")


# Response format templates
FORMAT_DECISION = """DECISION: [INCLUDE or EXCLUDE]
REASONING: [your justification]"""

FORMAT_MATH_EVALUATION = """DECISION: [INCLUDE or EXCLUDE]
NATURAL: [assessment - does it arise on its own?]
ELEGANT: [assessment - is this the simplest form?]
BEAUTIFUL: [assessment - does it have aesthetic quality?]
INTERESTING: [assessment - does it connect and illuminate?]
INEVITABLE: [assessment - given the axioms, must this exist?]
REASONING: [overall justification]"""

FORMAT_EXPOSITION = """DECISION: [APPROVE, REVISE, or REJECT]
CLARITY: [assessment]
CORRECTNESS: [assessment]
FLOW: [assessment]
DEPENDENCIES: [any missing concepts?]
REVISION_NOTES: [if REVISE, what to fix]"""

FORMAT_DRAFT = """DRAFT:
[your drafted section text]

ESTABLISHES: [comma-separated list of concepts this establishes]
REQUIRES: [comma-separated list of concepts this requires]
DIAGRAM_NEEDED: [yes/no, and if yes, description]"""

# Instruction to include in all content-generation tasks
FINAL_DOCUMENT_INSTRUCTION = """
IMPORTANT: This is for a FINAL PUBLISHED DOCUMENT, not a conversation.
- Do NOT include conversational phrases like "Let me explain...", "Next step:", "Would you like..."
- Do NOT ask questions or offer options to the reader
- Do NOT include meta-commentary about the writing process
- Write in a clear, authoritative expository style suitable for publication
"""

FORMAT_REVIEW = """DECISION: [CONFIRMED or REVISE]
ACCURACY: [still accurate?]
CLARITY: [still clear?]
CONNECTIONS: [any new connections to add?]
REVISION_NOTES: [if REVISE, specific changes]"""

FORMAT_BRIDGE = """DECISION: [INCLUDE bridge_name, WAIT, or NO_BRIDGE]
REASONING: [justification]
PLACEMENT: [if INCLUDE, suggested placement]"""

FORMAT_COMMENT = """UNDERSTOOD: [restate what the author wants]
RESOLUTION: [how to address it]
REVISED_SECTION: [the fixed section, or UNCHANGED if no fix needed]"""

FORMAT_RECOMMENDATION = """RECOMMENDATION: [item number, or "none ready"]
REASONING: [why this is the best next step]"""


class TaskBuilder:
    """Builds consensus tasks for the documenter."""

    def __init__(self, max_context_tokens: int = 8000):
        """
        Initialize task builder.

        Args:
            max_context_tokens: Maximum tokens for context (approximate)
        """
        self.max_context_tokens = max_context_tokens

    def build_what_to_work_on(
        self,
        document_summary: str,
        established_concepts: list[str],
        unincorporated_items: list[dict],
    ) -> tuple[str, str, str]:
        """
        Build task for deciding what to work on next.

        Returns: (context, task, response_format)
        """
        concepts_str = ", ".join(established_concepts) if established_concepts else "none yet"

        items_str = ""
        for i, item in enumerate(unincorporated_items[:10], 1):  # Limit to 10
            requires = item.get('requires', [])
            requires_str = f" (requires: {', '.join(requires)})" if requires else ""
            items_str += f"  {i}. {item.get('text', '')[:200]}...{requires_str}\n"

        if not items_str:
            items_str = "  (no unincorporated items)"

        context = f"""Document summary:
{document_summary}

Established concepts: {concepts_str}

Unincorporated blessed items:
{items_str}"""

        task = """What should we add next? Consider:
- Which items have all their required concepts already established?
- What would flow most naturally from the current document state?
- What would be most valuable to add?

If no items are ready (unmet dependencies), say "none ready"."""

        return context, task, FORMAT_RECOMMENDATION

    def build_math_evaluation(
        self,
        document_summary: str,
        proposed_math: str,
    ) -> tuple[str, str, str]:
        """
        Build task for evaluating mathematics for inclusion.

        Returns: (context, task, response_format)
        """
        context = f"""Document summary:
{document_summary}

Proposed mathematical content:
{proposed_math}"""

        task = """Should this mathematics be included in the document?

Evaluate in TWO contexts:

1. THE OVERALL PROJECT - The document aims to be beautiful, inevitable mathematics:
   - Beautiful: Aesthetic quality, rightness, elegance
   - Inevitable: Given the axioms, this mathematics HAS to exist
   - Natural: Arises on its own, not artificially constructed

2. THIS SPECIFIC CHUNK - Does it serve the larger vision?
   - It may be a profound insight that stands on its own
   - OR it may be foundational material that connects/supports profound parts
   - OR it may be necessary scaffolding that makes the beautiful parts possible

A chunk does NOT need to be individually profound. Accept content that contributes
to the overall beauty and inevitability of the project, even if this particular
piece is just foundational glue between more profound sections."""

        return context, task, FORMAT_MATH_EVALUATION

    def build_exposition_evaluation(
        self,
        document_summary: str,
        preceding_section: str,
        established_concepts: list[str],
        proposed_addition: str,
    ) -> tuple[str, str, str]:
        """
        Build task for evaluating exposition quality.

        Returns: (context, task, response_format)
        """
        concepts_str = ", ".join(established_concepts) if established_concepts else "none"

        context = f"""Document summary:
{document_summary}

Preceding section:
{preceding_section[:500]}

Established concepts: {concepts_str}

Proposed addition:
{proposed_addition}"""

        task = """Evaluate this exposition:
- Is it clear and understandable?
- Is it correct?
- Does it flow from what precedes?
- Does it only reference established concepts (or properly introduce new ones)?

Be strict about flow and dependencies."""

        return context, task, FORMAT_EXPOSITION

    def build_draft_content(
        self,
        document_summary: str,
        preceding_section: str,
        established_concepts: list[str],
        topic: str,
        source_insight: str,
    ) -> tuple[str, str, str]:
        """
        Build task for drafting new content.

        Returns: (context, task, response_format)
        """
        concepts_str = ", ".join(established_concepts) if established_concepts else "none"

        context = f"""Document summary:
{document_summary}

Preceding section:
{preceding_section[:500]}

Established concepts: {concepts_str}

Source insight to incorporate:
{source_insight}"""

        task = f"""Draft a section about: {topic}
{FINAL_DOCUMENT_INSTRUCTION}
The section should:
- Introduce the topic clearly
- Connect to established concepts
- Maintain a neutral expository voice
- Include proof if this is a theorem
- Specify what concepts it ESTABLISHES and REQUIRES

Use <!-- ESTABLISHES: concept_name --> and <!-- REQUIRES: concept_name --> markers."""

        return context, task, FORMAT_DRAFT

    def build_bridge_decision(
        self,
        document_summary: str,
        established_numbers: list[str],
        recent_additions: list[str],
        bridge_candidates: list[dict],
    ) -> tuple[str, str, str]:
        """
        Build task for deciding on a cross-domain bridge.

        Returns: (context, task, response_format)
        """
        numbers_str = ", ".join(established_numbers) if established_numbers else "none"
        recent_str = "\n".join(f"- {a[:100]}..." for a in recent_additions[:3])

        bridges_str = ""
        for bridge in bridge_candidates[:5]:
            bridges_str += f"- {bridge.get('number', '?')}: {bridge.get('description', '')}\n"

        context = f"""Document summary:
{document_summary}

Numbers established so far: {numbers_str}

Recent additions:
{recent_str}

Available bridge candidates:
{bridges_str}"""

        task = """Should we include a cross-domain bridge at this point?
- Is there a number we've established that a tradition validates?
- Would it reinforce the mathematical development or distract?
- Have we reached a natural pause point?

Only include bridges that are precise correspondences, not vague associations."""

        return context, task, FORMAT_BRIDGE

    def build_review_section(
        self,
        section_text: str,
        section_created: str,
        concepts_at_creation: list[str],
        concepts_since: list[str],
    ) -> tuple[str, str, str]:
        """
        Build task for reviewing existing content.

        Returns: (context, task, response_format)
        """
        at_creation_str = ", ".join(concepts_at_creation) if concepts_at_creation else "none"
        since_str = ", ".join(concepts_since) if concepts_since else "none"

        context = f"""Section to review:
{section_text}

Written: {section_created}
Concepts known at writing: {at_creation_str}
New concepts since then: {since_str}"""

        task = """Re-evaluate this section:
- Is it still accurate given current knowledge?
- Is it still clear in current context?
- Are there connections to later material we should add?
- Is it optimally placed in the document?"""

        return context, task, FORMAT_REVIEW

    def build_address_comment(
        self,
        section_text: str,
        comment_text: str,
    ) -> tuple[str, str, str]:
        """
        Build task for addressing an author comment.

        Returns: (context, task, response_format)
        """
        context = f"""Section with comment:
{section_text}

Author's comment: "{comment_text}" """

        task = f"""Address the author's concern:
- If it's a correction, verify and fix
- If it's a question, elaborate or clarify
- If it's a style concern, revise for clarity
{FINAL_DOCUMENT_INSTRUCTION}
Provide the revised section text."""

        return context, task, FORMAT_COMMENT

    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of tokens (1 token ≈ 4 chars)."""
        return len(text) // 4

    def truncate_context(self, context: str) -> str:
        """Truncate context to fit within token budget."""
        estimated = self.estimate_tokens(context)
        if estimated <= self.max_context_tokens:
            return context

        # Truncate proportionally
        target_chars = self.max_context_tokens * 4
        if len(context) > target_chars:
            context = context[:target_chars] + "\n\n[...truncated for length...]"

        return context
