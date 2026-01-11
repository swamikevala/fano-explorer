"""
Math formatting detection and fixing.

Detects LaTeX math that isn't properly delimited and fixes it using an LLM.
"""

import re
import sys
from pathlib import Path
from typing import Optional

# Add shared module to path
SHARED_PATH = Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH.parent))

from shared.logging import get_logger

log = get_logger("documenter", "formatting")

# Common LaTeX commands that indicate math content
LATEX_COMMANDS = [
    r"\\frac\{",
    r"\\sum",
    r"\\prod",
    r"\\int",
    r"\\sqrt",
    r"\\oplus",
    r"\\otimes",
    r"\\times",
    r"\\cdot",
    r"\\in",
    r"\\notin",
    r"\\subset",
    r"\\subseteq",
    r"\\cup",
    r"\\cap",
    r"\\mathbb\{",
    r"\\mathcal\{",
    r"\\text\{",
    r"\\left",
    r"\\right",
    r"\\vec\{",
    r"\\hat\{",
    r"\\bar\{",
    r"\\overline\{",
    r"\\underline\{",
    r"\\alpha",
    r"\\beta",
    r"\\gamma",
    r"\\delta",
    r"\\epsilon",
    r"\\theta",
    r"\\lambda",
    r"\\mu",
    r"\\pi",
    r"\\sigma",
    r"\\phi",
    r"\\psi",
    r"\\omega",
    r"\\infty",
    r"\\partial",
    r"\\nabla",
    r"\\forall",
    r"\\exists",
    r"\\neg",
    r"\\land",
    r"\\lor",
    r"\\implies",
    r"\\iff",
    r"\\equiv",
    r"\\approx",
    r"\\neq",
    r"\\leq",
    r"\\geq",
]

# Pattern to match $...$ or $$...$$ delimited regions
MATH_DELIMITER_PATTERN = re.compile(
    r"\$\$.*?\$\$|\$(?!\$).*?(?<!\$)\$",
    re.DOTALL
)


def is_inside_math_delimiters(text_before_pos: str) -> bool:
    """
    Check if a position is inside math delimiters.

    Counts opening and closing $ and $$ to determine if we're inside math mode.
    """
    # Remove all properly closed math regions
    cleaned = MATH_DELIMITER_PATTERN.sub("", text_before_pos)

    # Count remaining $ signs - odd number means we're inside math
    dollar_count = cleaned.count("$")
    return dollar_count % 2 == 1


def detect_undelimited_math(content: str) -> list[tuple[int, str]]:
    """
    Find LaTeX commands that appear outside of math delimiters.

    Returns:
        List of (position, command) tuples for each issue found
    """
    issues = []

    for cmd_pattern in LATEX_COMMANDS:
        # Use actual backslash in search (the pattern has double backslash for regex)
        search_pattern = cmd_pattern.replace(r"\\", "\\")

        for match in re.finditer(cmd_pattern, content):
            pos = match.start()
            before = content[:pos]

            if not is_inside_math_delimiters(before):
                issues.append((pos, match.group()))

    # Sort by position
    issues.sort(key=lambda x: x[0])

    if issues:
        log.debug(
            "documenter.formatting.issues_found",
            count=len(issues),
            first_few=[cmd for _, cmd in issues[:3]],
        )

    return issues


def detect_damaged_math(content: str) -> list[tuple[int, str]]:
    """
    Find math delimiters that only wrap a single command (damaged/fragmented).

    This detects patterns like $\oplus$ that should be part of a larger expression
    like $A \oplus B$.

    Returns:
        List of (position, matched_text) tuples for each issue found
    """
    issues = []

    # Pattern: $\command$ with nothing else inside (single LaTeX command wrapped alone)
    # Matches $\oplus$, $\alpha$, $\times$, etc.
    single_cmd_pattern = r'\$\\[a-zA-Z]+\$'

    for match in re.finditer(single_cmd_pattern, content):
        issues.append((match.start(), match.group()))

    if issues:
        log.debug(
            "documenter.formatting.damaged_math_found",
            count=len(issues),
            first_few=[text for _, text in issues[:3]],
        )

    return issues


def detect_markdown_issues(content: str) -> list[tuple[int, str]]:
    """
    Find markdown formatting issues.

    Detects:
    - Headers with trailing hashes: ### Title ### (should be ### Title)
    - Numbered lines that should be headers: "1. Title" at start of line (not in a list context)
    - Standalone title lines without header prefix

    Returns:
        List of (position, matched_text) tuples for each issue found
    """
    issues = []

    # Pattern: Headers with trailing hashes (### Title ###)
    trailing_hash_pattern = r'^(#{1,6})\s+.+?\s+(#{1,6})\s*$'
    for match in re.finditer(trailing_hash_pattern, content, re.MULTILINE):
        issues.append((match.start(), match.group()))

    # Pattern: Numbered section headers without # prefix
    # Matches "1. Title" or "1.1 Title" at start of line after blank line
    # These are likely meant to be headers, not list items
    numbered_header_pattern = r'(?:^|\n\n)(\d+\.[\d.]*\s+[A-Z][^\n]{5,50})$'
    for match in re.finditer(numbered_header_pattern, content, re.MULTILINE):
        issues.append((match.start(), match.group(1)))

    # Pattern: Title-like lines (capitalized, no punctuation at end, preceded by blank line)
    # These might be missing header prefixes
    title_line_pattern = r'\n\n([A-Z][A-Za-z\s:]{10,60})\n(?=[A-Z]|\d)'
    for match in re.finditer(title_line_pattern, content):
        line = match.group(1).strip()
        # Skip if it already looks like a header or is clearly a sentence
        if not line.startswith('#') and not line.endswith('.') and not line.endswith(','):
            issues.append((match.start(), line))

    if issues:
        log.debug(
            "documenter.formatting.markdown_issues_found",
            count=len(issues),
            first_few=[text[:50] for _, text in issues[:3]],
        )

    return issues


def detect_formatting_issues(content: str) -> list[tuple[int, str]]:
    """
    Find all formatting issues (math + markdown).

    This is the main detection entry point that combines all detection methods.

    Returns:
        List of (position, matched_text) tuples sorted by position
    """
    issues = []
    issues.extend(detect_undelimited_math(content))
    issues.extend(detect_damaged_math(content))
    issues.extend(detect_markdown_issues(content))
    return sorted(issues, key=lambda x: x[0])


# Keep old name for backward compatibility
detect_math_issues = detect_formatting_issues


def build_formatting_fix_prompt(content: str) -> str:
    """Build prompt for LLM to fix math and markdown formatting."""
    return f"""Fix the formatting issues in this markdown content.

MATH RULES:
1. Inline math (within a sentence) must use single dollar signs: $...$
2. Display math (standalone equations on their own line) must use double dollar signs: $$...$$
3. If you see isolated math like $\\oplus$ between operands (e.g., "A $\\oplus$ B"), merge them into a single expression: $A \\oplus B$
4. Adjacent small math blocks that form one expression should be merged (e.g., "$A$ $\\oplus$ $B$" becomes "$A \\oplus B$")

MARKDOWN RULES:
5. Headers should NOT have trailing hashes. "### Title ###" becomes "### Title"
6. Numbered section titles MUST be proper headers:
   - "1. The Title" alone on a line should become "## 1. The Title" or "### 1. The Title"
   - "1.1 Subsection" alone on a line should become "### 1.1 Subsection"
   - Use ## for main sections, ### for subsections, #### for sub-subsections
7. Title-like lines (capitalized phrases alone on a line) need header prefixes:
   - "The Generative Grammar" alone should become "## The Generative Grammar"
8. List items should use proper markdown: "- item" or "* item", not just indented text

IMPORTANT:
- PRESERVE ALL CONTENT EXACTLY - only fix formatting issues
- Do not change any mathematical expressions or text content
- Do not add any explanations - just return the fixed content

CONTENT TO FIX:
{content}

FIXED CONTENT:"""


# Keep old name for backward compatibility
build_math_fix_prompt = build_formatting_fix_prompt


async def fix_math_formatting(
    content: str,
    llm_client,
    backend: str = "claude",
) -> str:
    """
    Fix math delimiter issues in content using LLM.

    Args:
        content: The content with potential math formatting issues
        llm_client: LLMClient instance
        backend: Which LLM backend to use for fixing

    Returns:
        Content with fixed math delimiters
    """
    prompt = build_math_fix_prompt(content)

    try:
        response = await llm_client.send(
            backend,
            prompt,
            timeout_seconds=60,
        )

        if response.success and response.response:
            fixed = response.response.strip()

            # Basic validation - content shouldn't be drastically different
            original_word_count = len(content.split())
            fixed_word_count = len(fixed.split())

            # Allow some variance (math delimiters add tokens)
            if abs(original_word_count - fixed_word_count) > original_word_count * 0.2:
                log.warning(
                    "documenter.formatting.suspicious_fix",
                    original_words=original_word_count,
                    fixed_words=fixed_word_count,
                )
                # Return original if fix seems wrong
                return content

            log.info("documenter.formatting.fixed")
            return fixed
        else:
            log.warning(
                "documenter.formatting.llm_error",
                error=response.error or "No response",
            )
            return content

    except Exception as e:
        log.warning("documenter.formatting.exception", error=str(e))
        return content


async def fix_math_if_needed(
    content: str,
    llm_client,
    backend: str = "claude",
) -> str:
    """
    Check for math formatting issues and fix if needed.

    This is the main entry point - detects issues first, only calls LLM if needed.

    Args:
        content: Content to check and potentially fix
        llm_client: LLMClient instance
        backend: Which LLM backend to use

    Returns:
        Fixed content (or original if no issues found)
    """
    issues = detect_undelimited_math(content)

    if not issues:
        return content

    log.info(
        "documenter.formatting.fixing_math",
        issue_count=len(issues),
    )

    return await fix_math_formatting(content, llm_client, backend)
