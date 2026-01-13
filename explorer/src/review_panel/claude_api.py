"""
Claude browser client for the review panel.

Provides Claude access via the browser pool for review panel operations.
"""

import asyncio
from typing import Optional

import aiohttp

from shared.logging import get_logger

log = get_logger("explorer", "review_panel.claude_api")

# Pool service URL
POOL_URL = "http://127.0.0.1:9000"


class ClaudeReviewer:
    """
    Claude browser client for review panel.

    Uses the browser pool service to communicate with Claude via playwright.
    Supports extended thinking for deep analysis rounds.
    """

    def __init__(self, pool_url: str = POOL_URL):
        """
        Initialize the Claude reviewer.

        Args:
            pool_url: URL of the pool service
        """
        self.pool_url = pool_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send_message(
        self,
        prompt: str,
        extended_thinking: bool = False,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a message to Claude via the browser pool.

        Args:
            prompt: The prompt to send
            extended_thinking: Whether to use extended thinking mode
            max_tokens: Maximum tokens in response (ignored for browser mode)

        Returns:
            Claude's response text
        """
        log.info(f"[claude] Sending message ({len(prompt)} chars, extended_thinking={extended_thinking})")

        request_data = {
            "backend": "claude",
            "prompt": prompt,
            "options": {
                "deep_mode": extended_thinking,
                "new_chat": True,
                "timeout_seconds": 3600,  # Long timeout for extended thinking
                "priority": "normal",
            },
        }

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.pool_url}/send",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=3630),  # Slightly longer than request timeout
                ) as resp:
                    data = await resp.json()

                    if not data.get("success"):
                        error = data.get("error", "unknown")
                        message = data.get("message", "No message")
                        log.error(f"[claude] Request failed: {error}: {message}")
                        raise RuntimeError(f"{error}: {message}")

                    response_text = data.get("response", "")
                    log.info(f"[claude] Got response ({len(response_text)} chars)")
                    return response_text

            except aiohttp.ClientError as e:
                last_error = e
                error_str = str(e).lower()

                # Retry on connection errors
                if "connection" in error_str or attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    log.warning(f"[claude] Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    log.error(f"[claude] Error: {type(e).__name__}: {e}")
                    raise

            except asyncio.TimeoutError:
                log.error("[claude] Request timed out")
                raise RuntimeError("Request timed out")

        # All retries exhausted
        log.error(f"[claude] All {max_retries} attempts failed: {last_error}")
        raise last_error

    async def review(
        self,
        prompt: str,
        extended_thinking: bool = False,
    ) -> str:
        """
        Alias for send_message for review operations.

        Args:
            prompt: Review prompt
            extended_thinking: Whether to use extended thinking

        Returns:
            Review response text
        """
        return await self.send_message(prompt, extended_thinking=extended_thinking)

    def is_available(self) -> bool:
        """
        Check if Claude is available.

        Returns True since pool availability is checked at request time.
        For a proper async check, use check_available().
        """
        return True

    async def check_available(self) -> bool:
        """Async check if Claude is available via the pool."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.pool_url}/status",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                backends = data.get("backends", {})
                claude_status = backends.get("claude", {})
                return claude_status.get("available", False) and claude_status.get("authenticated", False)
        except Exception as e:
            log.warning(f"[claude] check_available failed: {e}")
            return False


def get_claude_reviewer(config: dict = None) -> Optional[ClaudeReviewer]:
    """
    Factory function to create a ClaudeReviewer from config.

    Args:
        config: Review panel configuration (unused, kept for compatibility)

    Returns:
        ClaudeReviewer instance
    """
    return ClaudeReviewer()
