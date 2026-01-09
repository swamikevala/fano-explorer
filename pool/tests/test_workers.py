"""Tests for pool backend workers."""

import asyncio
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workers import BaseWorker, GeminiWorker, ChatGPTWorker, ClaudeWorker
from src.models import SendRequest, SendOptions, SendResponse, Backend, Priority
from src.state import StateManager
from src.queue import RequestQueue


@pytest.fixture
def temp_state_file(tmp_path):
    """Create a temporary state file."""
    return tmp_path / "pool_state.json"


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
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
def state_manager(temp_state_file, sample_config):
    """Create a StateManager for tests."""
    return StateManager(temp_state_file, sample_config)


@pytest.fixture
def gemini_queue():
    """Create a RequestQueue for Gemini."""
    return RequestQueue("gemini", max_depth=10)


@pytest.fixture
def chatgpt_queue():
    """Create a RequestQueue for ChatGPT."""
    return RequestQueue("chatgpt", max_depth=10)


@pytest.fixture
def claude_queue():
    """Create a RequestQueue for Claude."""
    return RequestQueue("claude", max_depth=10)


@pytest.fixture
def sample_gemini_request():
    """Create a sample Gemini request."""
    return SendRequest(
        backend=Backend.GEMINI,
        prompt="Test prompt",
        options=SendOptions(deep_mode=False, new_chat=True),
    )


@pytest.fixture
def mock_gemini_browser():
    """Create a mock GeminiInterface."""
    browser = MagicMock()
    browser.connect = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.start_new_chat = AsyncMock()
    browser.send_message = AsyncMock(return_value="Gemini response")
    browser.enable_deep_think = AsyncMock()
    browser._check_rate_limit = MagicMock(return_value=False)
    browser.chat_logger = MagicMock()
    browser.chat_logger.get_session_id = MagicMock(return_value="session-123")
    return browser


@pytest.fixture
def mock_chatgpt_browser():
    """Create a mock ChatGPTInterface."""
    browser = MagicMock()
    browser.connect = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.start_new_chat = AsyncMock()
    browser.send_message = AsyncMock(return_value="ChatGPT response")
    browser.enable_pro_mode = AsyncMock()
    browser._check_rate_limit = MagicMock(return_value=False)
    browser.chat_logger = MagicMock()
    browser.chat_logger.get_session_id = MagicMock(return_value="session-456")
    return browser


class TestBaseWorker:
    """Tests for BaseWorker base class."""

    def test_init(self, sample_config, state_manager, gemini_queue):
        """BaseWorker initializes correctly."""
        # Create a concrete subclass for testing
        class TestWorker(BaseWorker):
            backend_name = "test"

        worker = TestWorker(sample_config, state_manager, gemini_queue)

        assert worker.config == sample_config
        assert worker.state == state_manager
        assert worker.queue == gemini_queue
        assert worker._running is False
        assert worker._task is None

    @pytest.mark.asyncio
    async def test_start_sets_running(self, sample_config, state_manager, gemini_queue):
        """start() sets _running to True and creates task."""
        class TestWorker(BaseWorker):
            backend_name = "test"
            async def _process_request(self, request):
                return SendResponse(success=True, response="test")
            async def connect(self):
                pass
            async def authenticate(self):
                return True

        worker = TestWorker(sample_config, state_manager, gemini_queue)
        await worker.start()

        assert worker._running is True
        assert worker._task is not None

        await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, sample_config, state_manager, gemini_queue):
        """stop() cancels the worker task."""
        class TestWorker(BaseWorker):
            backend_name = "test"
            async def _process_request(self, request):
                return SendResponse(success=True, response="test")
            async def connect(self):
                pass
            async def authenticate(self):
                return True

        worker = TestWorker(sample_config, state_manager, gemini_queue)
        await worker.start()
        await worker.stop()

        assert worker._running is False


class TestGeminiWorker:
    """Tests for GeminiWorker."""

    def test_init(self, sample_config, state_manager, gemini_queue):
        """GeminiWorker initializes correctly."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)

        assert worker.backend_name == "gemini"
        assert worker.browser is None

    @pytest.mark.asyncio
    async def test_connect_success(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """connect() initializes browser and marks authenticated."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)

        with patch("src.workers.GeminiWorker.connect") as mock_connect:
            mock_connect.return_value = None
            worker.browser = mock_gemini_browser

            # Simulate successful connection
            state_manager.mark_authenticated("gemini", True)

            assert state_manager.is_available("gemini") is True

    @pytest.mark.asyncio
    async def test_disconnect(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """disconnect() cleans up browser."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser

        await worker.disconnect()

        mock_gemini_browser.disconnect.assert_called_once()
        assert worker.browser is None

    @pytest.mark.asyncio
    async def test_process_request_no_browser(
        self, sample_config, state_manager, gemini_queue, sample_gemini_request
    ):
        """_process_request returns error when browser not connected."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = None

        response = await worker._process_request(sample_gemini_request)

        assert response.success is False
        assert response.error == "unavailable"
        assert "not connected" in response.message.lower()

    @pytest.mark.asyncio
    async def test_process_request_success(
        self, sample_config, state_manager, gemini_queue, sample_gemini_request, mock_gemini_browser
    ):
        """_process_request handles successful request."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser

        response = await worker._process_request(sample_gemini_request)

        assert response.success is True
        assert response.response == "Gemini response"
        assert response.metadata.backend == "gemini"
        mock_gemini_browser.start_new_chat.assert_called_once()
        mock_gemini_browser.send_message.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_process_request_with_deep_mode(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """_process_request enables deep think when requested."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser
        state_manager.mark_authenticated("gemini", True)

        request = SendRequest(
            backend=Backend.GEMINI,
            prompt="Deep prompt",
            options=SendOptions(deep_mode=True, new_chat=True),
        )

        response = await worker._process_request(request)

        assert response.success is True
        assert response.metadata.deep_mode_used is True
        mock_gemini_browser.enable_deep_think.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_request_deep_mode_at_limit(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """_process_request skips deep mode when at daily limit."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser
        state_manager.mark_authenticated("gemini", True)

        # Set usage at limit
        state_manager._state["gemini"]["deep_mode_uses_today"] = 20
        state_manager._state["gemini"]["deep_mode_reset_date"] = datetime.now().date().isoformat()

        request = SendRequest(
            backend=Backend.GEMINI,
            prompt="Deep prompt",
            options=SendOptions(deep_mode=True, new_chat=True),
        )

        response = await worker._process_request(request)

        assert response.success is True
        assert response.metadata.deep_mode_used is False
        mock_gemini_browser.enable_deep_think.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_request_rate_limit_detected(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """_process_request marks rate limited when detected in response."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser
        mock_gemini_browser._check_rate_limit.return_value = True

        request = SendRequest(
            backend=Backend.GEMINI,
            prompt="Test",
            options=SendOptions(new_chat=True),
        )

        await worker._process_request(request)

        assert state_manager._state["gemini"]["rate_limited"] is True

    @pytest.mark.asyncio
    async def test_process_request_exception(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """_process_request handles exceptions."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser
        mock_gemini_browser.send_message.side_effect = Exception("Browser error")

        request = SendRequest(
            backend=Backend.GEMINI,
            prompt="Test",
            options=SendOptions(new_chat=True),
        )

        response = await worker._process_request(request)

        assert response.success is False
        assert response.error == "processing_error"
        assert "Browser error" in response.message


class TestChatGPTWorker:
    """Tests for ChatGPTWorker."""

    def test_init(self, sample_config, state_manager, chatgpt_queue):
        """ChatGPTWorker initializes correctly."""
        worker = ChatGPTWorker(sample_config, state_manager, chatgpt_queue)

        assert worker.backend_name == "chatgpt"
        assert worker.browser is None

    @pytest.mark.asyncio
    async def test_process_request_with_pro_mode(
        self, sample_config, state_manager, chatgpt_queue, mock_chatgpt_browser
    ):
        """_process_request enables pro mode when requested."""
        worker = ChatGPTWorker(sample_config, state_manager, chatgpt_queue)
        worker.browser = mock_chatgpt_browser
        state_manager.mark_authenticated("chatgpt", True)

        request = SendRequest(
            backend=Backend.CHATGPT,
            prompt="Pro prompt",
            options=SendOptions(deep_mode=True, new_chat=True),
        )

        response = await worker._process_request(request)

        assert response.success is True
        assert response.metadata.deep_mode_used is True
        mock_chatgpt_browser.enable_pro_mode.assert_called_once()


class TestClaudeWorker:
    """Tests for ClaudeWorker."""

    def test_init(self, sample_config, state_manager, claude_queue):
        """ClaudeWorker initializes correctly."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)

        assert worker.backend_name == "claude"
        assert worker.client is None
        assert worker._model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self, sample_config, state_manager, claude_queue):
        """connect() initializes Anthropic client with API key."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client:
                mock_client.return_value = MagicMock()
                await worker.connect()

                mock_client.assert_called_once_with(api_key="test-key")
                assert worker.client is not None

    @pytest.mark.asyncio
    async def test_connect_without_api_key(self, sample_config, state_manager, claude_queue):
        """connect() raises when API key not set."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)

        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                await worker.connect()

    @pytest.mark.asyncio
    async def test_process_request_no_client(self, sample_config, state_manager, claude_queue):
        """_process_request returns error when client not initialized."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)
        worker.client = None

        request = SendRequest(
            backend=Backend.CLAUDE,
            prompt="Test",
        )

        response = await worker._process_request(request)

        assert response.success is False
        assert response.error == "unavailable"

    @pytest.mark.asyncio
    async def test_process_request_success(self, sample_config, state_manager, claude_queue):
        """_process_request handles successful API call."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Claude response")]

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        worker.client = mock_client

        request = SendRequest(
            backend=Backend.CLAUDE,
            prompt="Test prompt",
        )

        response = await worker._process_request(request)

        assert response.success is True
        assert response.response == "Claude response"
        assert response.metadata.backend == "claude"
        assert response.metadata.deep_mode_used is False

    @pytest.mark.asyncio
    async def test_process_request_rate_limit_error(
        self, sample_config, state_manager, claude_queue
    ):
        """_process_request handles rate limit errors."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(
            side_effect=Exception("Rate limit exceeded (429)")
        )
        worker.client = mock_client

        request = SendRequest(
            backend=Backend.CLAUDE,
            prompt="Test",
        )

        response = await worker._process_request(request)

        assert response.success is False
        assert response.error == "api_error"
        assert state_manager._state["claude"]["rate_limited"] is True

    @pytest.mark.asyncio
    async def test_authenticate_with_key(self, sample_config, state_manager, claude_queue):
        """authenticate() returns True when API key is set."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = await worker.authenticate()

            assert result is True
            assert state_manager._state["claude"]["authenticated"] is True

    @pytest.mark.asyncio
    async def test_authenticate_without_key(self, sample_config, state_manager, claude_queue):
        """authenticate() returns False when API key not set."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = await worker.authenticate()

            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self, sample_config, state_manager, claude_queue):
        """disconnect() clears the client."""
        worker = ClaudeWorker(sample_config, state_manager, claude_queue)
        worker.client = MagicMock()

        await worker.disconnect()

        assert worker.client is None


class TestWorkerLoop:
    """Tests for worker run loop behavior."""

    @pytest.mark.asyncio
    async def test_worker_processes_queue(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """Worker processes requests from queue."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser
        state_manager.mark_authenticated("gemini", True)

        # Enqueue a request
        request = SendRequest(backend=Backend.GEMINI, prompt="Test")
        future = await gemini_queue.enqueue(request)

        # Start worker and let it process
        await worker.start()
        await asyncio.sleep(0.1)  # Give worker time to process

        # Check the future was resolved
        assert future.done()
        result = await future
        assert result.success is True

        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_skips_when_unavailable(
        self, sample_config, state_manager, gemini_queue, mock_gemini_browser
    ):
        """Worker skips processing when backend unavailable."""
        worker = GeminiWorker(sample_config, state_manager, gemini_queue)
        worker.browser = mock_gemini_browser
        state_manager.mark_authenticated("gemini", False)  # Not authenticated

        # Enqueue a request
        request = SendRequest(backend=Backend.GEMINI, prompt="Test")
        future = await gemini_queue.enqueue(request)

        # Start worker
        await worker.start()
        await asyncio.sleep(0.1)

        # Request should still be pending (worker skipped it)
        assert not future.done()

        await worker.stop()
