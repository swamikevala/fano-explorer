"""Shared fixtures for Browser Pool tests."""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_state_file(tmp_path):
    """Create a temporary state file."""
    return tmp_path / "pool_state.json"


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "service": {
            "host": "127.0.0.1",
            "port": 9000,
        },
        "queue": {
            "max_depth_per_backend": 10,
        },
        "backends": {
            "gemini": {
                "enabled": True,
                "deep_mode": {
                    "daily_limit": 20,
                    "reset_hour": 0,
                },
            },
            "chatgpt": {
                "enabled": True,
                "pro_mode": {
                    "daily_limit": 100,
                    "reset_hour": 0,
                },
            },
            "claude": {
                "enabled": True,
                "model": "claude-sonnet-4-20250514",
            },
        },
        "timeouts": {
            "request_default_seconds": 300,
        },
    }


@pytest.fixture
def mock_browser():
    """Create a mock browser interface."""
    browser = MagicMock()
    browser.connect = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.start_new_chat = AsyncMock()
    browser.send_message = AsyncMock(return_value="Test response")
    browser.enable_deep_think = AsyncMock()
    browser.enable_pro_mode = AsyncMock()
    browser._check_rate_limit = MagicMock(return_value=False)
    browser.chat_logger = MagicMock()
    browser.chat_logger.get_session_id = MagicMock(return_value="session-123")
    return browser


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="Claude response")]
    client.messages.create = MagicMock(return_value=response)
    return client
