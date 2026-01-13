"""
Shared prompt constants and utilities.

This module provides common prompt instructions that should be included
across all LLM interactions to ensure consistent formatting.
"""

# Math formatting instruction - ensures LLMs output raw LaTeX instead of rendered symbols
# This is critical for Gemini which tends to render math as visual symbols that
# get lost during text extraction.
MATH_FORMATTING_INSTRUCTION = """
=== MATH FORMATTING ===
When writing mathematical formulas or expressions, provide the raw LaTeX
code inside a Markdown code block (```latex ... ```). Do not allow math
to render into visual symbols. Use double dollar signs ($$...$$) for
display equations and single dollar signs ($...$) for inline math.
Ensure no whitespace immediately inside dollar-sign delimiters.
""".strip()


def get_math_formatting_section() -> str:
    """Return the math formatting instruction as a prompt section."""
    return MATH_FORMATTING_INSTRUCTION
