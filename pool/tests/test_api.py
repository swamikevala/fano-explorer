"""Tests for pool FastAPI HTTP API."""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from pool.src.api import BrowserPool, create_app
from pool.src.models import (
    SendRequest, SendResponse, SendOptions,
    Backend, Priority, PoolStatus, BackendStatus,
)
from pool.src.queue import QueueFullError


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "service": {"host": "127.0.0.1", "port": 9000},
        "queue": {"max_depth_per_backend": 10},
        "backends": {
            "gemini": {
                "enabled": True,
                "deep_mode": {"daily_limit": 20},
            },
            "chatgpt": {
                "enabled": True,
                "pro_mode": {"daily_limit": 100},
            },
            "claude": {
                "enabled": True,
                "model": "claude-sonnet-4-20250514",
            },
        },
    }


@pytest.fixture
def minimal_config():
    """Minimal configuration with only Claude enabled."""
    return {
        "backends": {
            "gemini": {"enabled": False},
            "chatgpt": {"enabled": False},
            "claude": {"enabled": True},
        },
    }


class TestBrowserPoolInit:
    """Tests for BrowserPool initialization."""

    def test_init_creates_state_manager(self, sample_config, tmp_path):
        """BrowserPool creates a StateManager."""
        with patch("pool.src.api.Path") as mock_path:
            mock_path.return_value.parent.parent = tmp_path
            pool = BrowserPool(sample_config)

            assert pool.state is not None

    def test_init_creates_queue_manager(self, sample_config):
        """BrowserPool creates a QueueManager."""
        with patch("pool.src.api.StateManager"):
            pool = BrowserPool(sample_config)

            assert pool.queues is not None

    def test_init_creates_enabled_workers(self, sample_config):
        """BrowserPool creates workers for enabled backends."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.GeminiWorker"):
                with patch("pool.src.api.ChatGPTWorker"):
                    with patch("pool.src.api.ClaudeWorker"):
                        pool = BrowserPool(sample_config)

                        assert "gemini" in pool.workers
                        assert "chatgpt" in pool.workers
                        assert "claude" in pool.workers

    def test_init_skips_disabled_workers(self, minimal_config):
        """BrowserPool skips workers for disabled backends."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.ClaudeWorker"):
                pool = BrowserPool(minimal_config)

                assert "gemini" not in pool.workers
                assert "chatgpt" not in pool.workers
                assert "claude" in pool.workers


class TestBrowserPoolSend:
    """Tests for BrowserPool.send() method."""

    @pytest.mark.asyncio
    async def test_send_returns_error_for_disabled_backend(self, minimal_config):
        """send() returns error for disabled backend."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.ClaudeWorker"):
                pool = BrowserPool(minimal_config)

                request = SendRequest(
                    backend=Backend.GEMINI,
                    prompt="Test",
                )

                response = await pool.send(request)

                assert response.success is False
                assert response.error == "unavailable"
                assert "not enabled" in response.message.lower()

    @pytest.mark.asyncio
    async def test_send_returns_auth_required_when_not_authenticated(self, sample_config):
        """send() returns auth_required when backend not authenticated."""
        with patch("pool.src.api.StateManager") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.is_available.return_value = False
            mock_state.get_backend_state.return_value = {"authenticated": False, "rate_limited": False}
            mock_state_cls.return_value = mock_state

            with patch("pool.src.api.GeminiWorker"):
                with patch("pool.src.api.ChatGPTWorker"):
                    with patch("pool.src.api.ClaudeWorker"):
                        pool = BrowserPool(sample_config)

                        request = SendRequest(
                            backend=Backend.GEMINI,
                            prompt="Test",
                        )

                        response = await pool.send(request)

                        assert response.success is False
                        assert response.error == "auth_required"

    @pytest.mark.asyncio
    async def test_send_returns_rate_limited(self, sample_config):
        """send() returns rate_limited when backend is rate limited."""
        with patch("pool.src.api.StateManager") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.is_available.return_value = False
            mock_state.get_backend_state.return_value = {"authenticated": True, "rate_limited": True}
            mock_state_cls.return_value = mock_state

            with patch("pool.src.api.GeminiWorker"):
                with patch("pool.src.api.ChatGPTWorker"):
                    with patch("pool.src.api.ClaudeWorker"):
                        pool = BrowserPool(sample_config)

                        request = SendRequest(
                            backend=Backend.GEMINI,
                            prompt="Test",
                        )

                        response = await pool.send(request)

                        assert response.success is False
                        assert response.error == "rate_limited"
                        assert response.retry_after_seconds == 3600

    @pytest.mark.asyncio
    async def test_send_returns_queue_full_error(self, sample_config):
        """send() returns queue_full when queue is at capacity."""
        with patch("pool.src.api.StateManager") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.is_available.return_value = True
            mock_state_cls.return_value = mock_state

            with patch("pool.src.api.QueueManager") as mock_queue_cls:
                mock_queue = MagicMock()
                mock_queue.enqueue = AsyncMock(side_effect=QueueFullError("Queue full"))
                mock_queue_manager = MagicMock()
                mock_queue_manager.get_queue.return_value = mock_queue
                mock_queue_cls.return_value = mock_queue_manager

                with patch("pool.src.api.GeminiWorker"):
                    with patch("pool.src.api.ChatGPTWorker"):
                        with patch("pool.src.api.ClaudeWorker"):
                            pool = BrowserPool(sample_config)
                            pool.queues = mock_queue_manager

                            request = SendRequest(
                                backend=Backend.GEMINI,
                                prompt="Test",
                            )

                            response = await pool.send(request)

                            assert response.success is False
                            assert response.error == "queue_full"

    @pytest.mark.asyncio
    async def test_send_returns_timeout_error(self, sample_config):
        """send() returns timeout when request times out."""
        with patch("pool.src.api.StateManager") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.is_available.return_value = True
            mock_state_cls.return_value = mock_state

            with patch("pool.src.api.QueueManager") as mock_queue_cls:
                # Create a future that never resolves
                never_resolving_future = asyncio.Future()
                mock_queue = MagicMock()
                mock_queue.enqueue = AsyncMock(return_value=never_resolving_future)
                mock_queue_manager = MagicMock()
                mock_queue_manager.get_queue.return_value = mock_queue
                mock_queue_cls.return_value = mock_queue_manager

                with patch("pool.src.api.GeminiWorker"):
                    with patch("pool.src.api.ChatGPTWorker"):
                        with patch("pool.src.api.ClaudeWorker"):
                            pool = BrowserPool(sample_config)
                            pool.queues = mock_queue_manager

                            request = SendRequest(
                                backend=Backend.GEMINI,
                                prompt="Test",
                                options=SendOptions(timeout_seconds=1),
                            )

                            response = await pool.send(request)

                            assert response.success is False
                            assert response.error == "timeout"

    @pytest.mark.asyncio
    async def test_send_returns_success(self, sample_config):
        """send() returns successful response."""
        with patch("pool.src.api.StateManager") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.is_available.return_value = True
            mock_state_cls.return_value = mock_state

            with patch("pool.src.api.QueueManager") as mock_queue_cls:
                # Create a future that resolves immediately
                future = asyncio.Future()
                expected_response = SendResponse(success=True, response="Hello!")
                future.set_result(expected_response)

                mock_queue = MagicMock()
                mock_queue.enqueue = AsyncMock(return_value=future)
                mock_queue_manager = MagicMock()
                mock_queue_manager.get_queue.return_value = mock_queue
                mock_queue_cls.return_value = mock_queue_manager

                with patch("pool.src.api.GeminiWorker"):
                    with patch("pool.src.api.ChatGPTWorker"):
                        with patch("pool.src.api.ClaudeWorker"):
                            pool = BrowserPool(sample_config)
                            pool.queues = mock_queue_manager

                            request = SendRequest(
                                backend=Backend.GEMINI,
                                prompt="Hello",
                            )

                            response = await pool.send(request)

                            assert response.success is True
                            assert response.response == "Hello!"


class TestBrowserPoolGetStatus:
    """Tests for BrowserPool.get_status() method."""

    def test_get_status_returns_pool_status(self, sample_config):
        """get_status() returns PoolStatus object."""
        with patch("pool.src.api.StateManager") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.get_backend_state.return_value = {
                "authenticated": True,
                "rate_limited": False,
                "deep_mode_uses_today": 5,
            }
            mock_state.is_available.return_value = True
            mock_state_cls.return_value = mock_state

            with patch("pool.src.api.QueueManager") as mock_queue_cls:
                mock_queue_manager = MagicMock()
                mock_queue_manager.get_depths.return_value = {
                    "gemini": 2,
                    "chatgpt": 1,
                    "claude": 0,
                }
                mock_queue_cls.return_value = mock_queue_manager

                with patch("pool.src.api.GeminiWorker"):
                    with patch("pool.src.api.ChatGPTWorker"):
                        with patch("pool.src.api.ClaudeWorker"):
                            pool = BrowserPool(sample_config)

                            status = pool.get_status()

                            assert isinstance(status, PoolStatus)
                            assert status.gemini is not None
                            assert status.chatgpt is not None
                            assert status.claude is not None

    def test_get_status_includes_queue_depths(self, sample_config):
        """get_status() includes queue depths."""
        with patch("pool.src.api.StateManager") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.get_backend_state.return_value = {"authenticated": True, "rate_limited": False}
            mock_state.is_available.return_value = True
            mock_state_cls.return_value = mock_state

            with patch("pool.src.api.QueueManager") as mock_queue_cls:
                mock_queue_manager = MagicMock()
                mock_queue_manager.get_depths.return_value = {
                    "gemini": 5,
                    "chatgpt": 3,
                    "claude": 1,
                }
                mock_queue_cls.return_value = mock_queue_manager

                with patch("pool.src.api.GeminiWorker"):
                    with patch("pool.src.api.ChatGPTWorker"):
                        with patch("pool.src.api.ClaudeWorker"):
                            pool = BrowserPool(sample_config)

                            status = pool.get_status()

                            assert status.gemini.queue_depth == 5
                            assert status.chatgpt.queue_depth == 3
                            assert status.claude.queue_depth == 1


class TestBrowserPoolAuthenticate:
    """Tests for BrowserPool.authenticate() method."""

    @pytest.mark.asyncio
    async def test_authenticate_returns_false_for_unknown_backend(self, sample_config):
        """authenticate() returns False for unknown backend."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.GeminiWorker"):
                with patch("pool.src.api.ChatGPTWorker"):
                    with patch("pool.src.api.ClaudeWorker"):
                        pool = BrowserPool(sample_config)

                        result = await pool.authenticate("unknown")

                        assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_calls_worker_authenticate(self, sample_config):
        """authenticate() delegates to worker."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.GeminiWorker") as mock_worker_cls:
                mock_worker = MagicMock()
                mock_worker.authenticate = AsyncMock(return_value=True)
                mock_worker_cls.return_value = mock_worker

                with patch("pool.src.api.ChatGPTWorker"):
                    with patch("pool.src.api.ClaudeWorker"):
                        pool = BrowserPool(sample_config)

                        result = await pool.authenticate("gemini")

                        mock_worker.authenticate.assert_called_once()
                        assert result is True


class TestBrowserPoolLifecycle:
    """Tests for BrowserPool startup/shutdown."""

    @pytest.mark.asyncio
    async def test_startup_starts_workers(self, sample_config):
        """startup() starts all workers."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.GeminiWorker") as mock_gemini:
                mock_worker_gemini = MagicMock()
                mock_worker_gemini.connect = AsyncMock()
                mock_worker_gemini.start = AsyncMock()
                mock_gemini.return_value = mock_worker_gemini

                with patch("pool.src.api.ChatGPTWorker") as mock_chatgpt:
                    mock_worker_chatgpt = MagicMock()
                    mock_worker_chatgpt.connect = AsyncMock()
                    mock_worker_chatgpt.start = AsyncMock()
                    mock_chatgpt.return_value = mock_worker_chatgpt

                    with patch("pool.src.api.ClaudeWorker") as mock_claude:
                        mock_worker_claude = MagicMock()
                        mock_worker_claude.connect = AsyncMock()
                        mock_worker_claude.start = AsyncMock()
                        mock_claude.return_value = mock_worker_claude

                        pool = BrowserPool(sample_config)
                        await pool.startup()

                        mock_worker_gemini.connect.assert_called_once()
                        mock_worker_gemini.start.assert_called_once()
                        mock_worker_chatgpt.connect.assert_called_once()
                        mock_worker_chatgpt.start.assert_called_once()
                        mock_worker_claude.connect.assert_called_once()
                        mock_worker_claude.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_handles_connection_failure(self, sample_config):
        """startup() handles connection failures gracefully."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.GeminiWorker") as mock_gemini:
                mock_worker = MagicMock()
                mock_worker.connect = AsyncMock(side_effect=Exception("Connection failed"))
                mock_worker.start = AsyncMock()
                mock_gemini.return_value = mock_worker

                with patch("pool.src.api.ChatGPTWorker"):
                    with patch("pool.src.api.ClaudeWorker"):
                        pool = BrowserPool(sample_config)

                        # Should not raise
                        await pool.startup()

                        # Worker should still be started
                        mock_worker.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_stops_workers(self, sample_config):
        """shutdown() stops all workers."""
        with patch("pool.src.api.StateManager"):
            with patch("pool.src.api.GeminiWorker") as mock_gemini:
                mock_worker_gemini = MagicMock()
                mock_worker_gemini.stop = AsyncMock()
                mock_worker_gemini.disconnect = AsyncMock()
                mock_gemini.return_value = mock_worker_gemini

                with patch("pool.src.api.ChatGPTWorker") as mock_chatgpt:
                    mock_worker_chatgpt = MagicMock()
                    mock_worker_chatgpt.stop = AsyncMock()
                    mock_worker_chatgpt.disconnect = AsyncMock()
                    mock_chatgpt.return_value = mock_worker_chatgpt

                    with patch("pool.src.api.ClaudeWorker") as mock_claude:
                        mock_worker_claude = MagicMock()
                        mock_worker_claude.stop = AsyncMock()
                        mock_worker_claude.disconnect = AsyncMock()
                        mock_claude.return_value = mock_worker_claude

                        pool = BrowserPool(sample_config)
                        await pool.shutdown()

                        mock_worker_gemini.stop.assert_called_once()
                        mock_worker_gemini.disconnect.assert_called_once()


class TestCreateApp:
    """Tests for create_app() factory function."""

    def test_creates_fastapi_app(self, sample_config):
        """create_app() creates a FastAPI application."""
        with patch("pool.src.api.BrowserPool"):
            from fastapi import FastAPI

            app = create_app(sample_config)

            assert isinstance(app, FastAPI)
            assert app.title == "Browser Pool Service"
            assert app.version == "0.1.0"

    def test_app_has_endpoints(self, sample_config):
        """App has all required endpoints."""
        with patch("pool.src.api.BrowserPool"):
            app = create_app(sample_config)

            routes = [route.path for route in app.routes]

            assert "/send" in routes
            assert "/status" in routes
            assert "/auth/{backend}" in routes
            assert "/health" in routes
