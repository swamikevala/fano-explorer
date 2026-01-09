"""Shared fixtures for LLM library tests."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

import pytest


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_pool_response():
    """Create a mock successful pool response."""
    return {
        "success": True,
        "response": "Hello from pool!",
        "error": None,
        "message": None,
        "retry_after_seconds": None,
        "metadata": {
            "backend": "gemini",
            "deep_mode_used": False,
            "response_time_seconds": 1.5,
            "session_id": "session-123",
        },
    }


@pytest.fixture
def mock_pool_error_response():
    """Create a mock error pool response."""
    return {
        "success": False,
        "response": None,
        "error": "rate_limited",
        "message": "Too many requests",
        "retry_after_seconds": 3600,
        "metadata": None,
    }


@pytest.fixture
def mock_http_session():
    """Create a mock aiohttp ClientSession."""
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="Claude response")]
    client.messages.create = MagicMock(return_value=response)
    return client
