"""
Quick Check - Single-LLM fast validation.

Provides a fast path for validation when full consensus isn't needed.
"""

from typing import TYPE_CHECKING

from .prompts import build_quick_check_prompt
from .response_parser import parse_quick_rating, parse_quick_reason

if TYPE_CHECKING:
    from ..client import LLMClient


class QuickChecker:
    """
    Single-LLM quick validation.

    Useful for fast checks when full consensus isn't required.
    Prefers Claude (API) for speed.
    """

    def __init__(self, client: "LLMClient"):
        """
        Initialize quick checker.

        Args:
            client: LLMClient for sending requests
        """
        self.client = client

    async def check(
        self,
        text: str,
        *,
        context: str = "",
    ) -> tuple[str, str]:
        """
        Quick single-LLM check (no consensus, just one opinion).

        Args:
            text: The text to check
            context: Optional context

        Returns:
            Tuple of (rating, reasoning)
        """
        available = await self.client.get_available_backends()
        if not available:
            return "uncertain", "No backends available"

        # Prefer Claude for quick checks (API is faster)
        backend = "claude" if "claude" in available else available[0]

        prompt = build_quick_check_prompt(text, context)
        response = await self.client.send(backend, prompt, timeout_seconds=60)

        if not response.success:
            return "uncertain", f"Error: {response.message}"

        rating = parse_quick_rating(response.text)
        reason = parse_quick_reason(response.text)

        return rating, reason
