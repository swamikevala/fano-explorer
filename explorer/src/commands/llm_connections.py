"""
Shared LLM connection management for Explorer commands.

Provides a unified interface for connecting to Gemini, ChatGPT, and Claude
with consistent error handling and status reporting.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable

from shared.logging import get_logger

log = get_logger("explorer", "llm_connections")


@dataclass
class ConnectionStatus:
    """Status of an LLM connection attempt."""
    name: str
    connected: bool
    message: str


@dataclass
class LLMConnections:
    """
    Manages connections to LLM providers.

    Provides a unified interface for connecting/disconnecting
    and checking availability of Gemini, ChatGPT, and Claude.
    """
    gemini: Optional[object] = None
    chatgpt: Optional[object] = None
    claude: Optional[object] = None
    statuses: list[ConnectionStatus] = field(default_factory=list)

    def has_any(self) -> bool:
        """Check if any LLM is available."""
        return bool(self.gemini or self.chatgpt or self.claude)

    def available_names(self) -> list[str]:
        """Get names of available LLMs."""
        names = []
        if self.gemini:
            names.append("Gemini")
        if self.chatgpt:
            names.append("ChatGPT")
        if self.claude:
            names.append("Claude")
        return names


async def connect_llms(
    on_status: Optional[Callable[[ConnectionStatus], None]] = None
) -> LLMConnections:
    """
    Connect to all available LLM providers.

    Args:
        on_status: Optional callback for status updates during connection.
                   Called with ConnectionStatus for each provider.

    Returns:
        LLMConnections with connected providers (None for failed ones)
    """
    from explorer.src.browser.gemini import GeminiInterface
    from explorer.src.browser.chatgpt import ChatGPTInterface
    from explorer.src.review_panel.claude_api import ClaudeReviewer

    connections = LLMConnections()

    def report(status: ConnectionStatus):
        connections.statuses.append(status)
        if on_status:
            on_status(status)

    # Connect Gemini
    gemini = GeminiInterface()
    try:
        await gemini.connect()
        logged_in = await gemini._check_login_status()
        if not logged_in:
            report(ConnectionStatus(
                "Gemini", False,
                "Not logged in - run 'python fano_explorer.py auth' first"
            ))
            log.warning("llm.connection.not_logged_in", provider="gemini")
        else:
            connections.gemini = gemini
            report(ConnectionStatus("Gemini", True, "Connected"))
            log.info("llm.connection.success", provider="gemini")
    except Exception as e:
        report(ConnectionStatus("Gemini", False, f"Failed: {e}"))
        log.error("llm.connection.failed", provider="gemini", error=str(e))

    # Connect ChatGPT
    chatgpt = ChatGPTInterface()
    try:
        await chatgpt.connect()
        page_text = await chatgpt.page.inner_text("body")
        if "log in" in page_text.lower() or "sign up" in page_text.lower():
            report(ConnectionStatus(
                "ChatGPT", False,
                "Not logged in - run 'python fano_explorer.py auth' first"
            ))
            log.warning("llm.connection.not_logged_in", provider="chatgpt")
        else:
            connections.chatgpt = chatgpt
            report(ConnectionStatus("ChatGPT", True, "Connected"))
            log.info("llm.connection.success", provider="chatgpt")
    except Exception as e:
        report(ConnectionStatus("ChatGPT", False, f"Failed: {e}"))
        log.error("llm.connection.failed", provider="chatgpt", error=str(e))

    # Setup Claude API
    claude = ClaudeReviewer()
    if claude.api_key:
        connections.claude = claude
        report(ConnectionStatus("Claude", True, "API ready"))
        log.info("llm.connection.success", provider="claude")
    else:
        report(ConnectionStatus("Claude", False, "API key not found"))
        log.warning("llm.connection.no_api_key", provider="claude")

    return connections


async def disconnect_llms(connections: LLMConnections):
    """
    Disconnect all LLM providers.

    Args:
        connections: The LLMConnections to disconnect
    """
    if connections.gemini:
        try:
            await connections.gemini.disconnect()
            log.info("llm.disconnected", provider="gemini")
        except Exception as e:
            log.error("llm.disconnect.failed", provider="gemini", error=str(e))

    if connections.chatgpt:
        try:
            await connections.chatgpt.disconnect()
            log.info("llm.disconnected", provider="chatgpt")
        except Exception as e:
            log.error("llm.disconnect.failed", provider="chatgpt", error=str(e))

    # Claude API doesn't need disconnection
