"""
Utility functions for review panel prompts.

Contains helper functions shared across rounds.
"""


def build_round_summary(round_responses: dict, round_number: int) -> str:
    """
    Build a FULL summary of a round's responses for Round 4 context.

    For Round 4 to make good decisions, it needs the complete deliberation
    history - no truncation. Modern LLMs have large context windows.

    Args:
        round_responses: Dict of LLM name to ReviewResponse
        round_number: The round number (1, 2, or 3)

    Returns:
        Formatted summary string with FULL content (no truncation)
    """
    summaries = []
    for llm_name, response in round_responses.items():
        rating = response.rating if hasattr(response, 'rating') else response.get('rating', '?')
        reasoning = response.reasoning if hasattr(response, 'reasoning') else response.get('reasoning', '')

        # Build full summary without truncation
        summary_parts = [f"--- {llm_name.upper()} [Rating: {rating}] ---"]

        # Get key fields based on round
        if round_number == 1:
            math = response.mathematical_verification if hasattr(response, 'mathematical_verification') else response.get('mathematical_verification', '')
            struct = response.structural_analysis if hasattr(response, 'structural_analysis') else response.get('structural_analysis', '')
            natural = response.naturalness_assessment if hasattr(response, 'naturalness_assessment') else response.get('naturalness_assessment', '')

            if math:
                summary_parts.append(f"Mathematical Verification: {math}")
            if struct:
                summary_parts.append(f"Structural Analysis: {struct}")
            if natural:
                summary_parts.append(f"Naturalness Assessment: {natural}")
            if reasoning:
                summary_parts.append(f"Reasoning: {reasoning}")

        elif round_number == 2:
            math = response.mathematical_verification if hasattr(response, 'mathematical_verification') else response.get('mathematical_verification', '')
            struct = response.structural_analysis if hasattr(response, 'structural_analysis') else response.get('structural_analysis', '')
            new_info = response.new_information if hasattr(response, 'new_information') else response.get('new_information', '')
            changed = response.changed_mind if hasattr(response, 'changed_mind') else response.get('changed_mind', '')

            if math:
                summary_parts.append(f"Mathematical Verification: {math}")
            if struct:
                summary_parts.append(f"Structural Analysis: {struct}")
            if reasoning:
                summary_parts.append(f"Reasoning: {reasoning}")
            if new_info:
                summary_parts.append(f"New Information Considered: {new_info}")
            if changed:
                summary_parts.append(f"Mind Changed: {changed}")

        else:  # Round 3 - deliberation
            if reasoning:
                summary_parts.append(f"Position: {reasoning}")

        # Add modification if proposed (full text)
        mod = response.proposed_modification if hasattr(response, 'proposed_modification') else response.get('proposed_modification', '')
        mod_rationale = response.modification_rationale if hasattr(response, 'modification_rationale') else response.get('modification_rationale', '')
        if mod:
            summary_parts.append(f"PROPOSED MODIFICATION:\n{mod}")
            if mod_rationale:
                summary_parts.append(f"Modification Rationale: {mod_rationale}")

        # Add confidence
        conf = response.confidence if hasattr(response, 'confidence') else response.get('confidence', '')
        if conf:
            summary_parts.append(f"Confidence: {conf}")

        summaries.append("\n".join(summary_parts))

    return "\n\n".join(summaries)
