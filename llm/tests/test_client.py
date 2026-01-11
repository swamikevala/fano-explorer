"""Tests for LLM client."""

import asyncio
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from llm.src.client import LLMClient, PoolUnavailableError, BROWSER_BACKENDS, API_BACKENDS
from llm.src.models import LLMResponse, PoolStatus, BackendStatus


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_default_pool_url(self):
        client = LLMClient()
        assert client.pool_url == "http://127.0.0.1:9000"

    def test_custom_pool_url(self):
        client = LLMClient(pool_url="http://custom:8080/")
        assert client.pool_url == "http://custom:8080"  # Trailing slash stripped

    def test_anthropic_key_from_arg(self):
        client = LLMClient(anthropic_api_key="test-key")
        assert client._anthropic_key == "test-key"

    def test_anthropic_key_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            client = LLMClient()
            assert client._anthropic_key == "env-key"

    def test_openrouter_key_from_arg(self):
        client = LLMClient(openrouter_api_key="or-key")
        assert client._openrouter_key == "or-key"

    def test_openrouter_key_from_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-env-key"}):
            client = LLMClient()
            assert client._openrouter_key == "or-env-key"

    def test_lazy_http_session(self):
        client = LLMClient()
        assert client._http_session is None

    def test_lazy_anthropic_client(self):
        client = LLMClient()
        assert client._anthropic_client is None


class TestLLMClientHttpSession:
    """Tests for HTTP session management."""

    @pytest.mark.asyncio
    async def test_get_http_session_creates_session(self):
        client = LLMClient()

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            session = await client._get_http_session()

            mock_session_cls.assert_called_once()
            assert session == mock_session

            await client.close()

    @pytest.mark.asyncio
    async def test_get_http_session_reuses_session(self):
        client = LLMClient()

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            session1 = await client._get_http_session()
            session2 = await client._get_http_session()

            assert mock_session_cls.call_count == 1
            assert session1 == session2

            await client.close()

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        client = LLMClient()

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._http_session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client._http_session is None


class TestLLMClientAnthropicClient:
    """Tests for Anthropic client management."""

    def test_get_anthropic_client_creates_client(self):
        client = LLMClient(anthropic_api_key="test-key")

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            result = client._get_anthropic_client()

            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert result == mock_client

    def test_get_anthropic_client_returns_none_without_key(self):
        client = LLMClient()
        client._anthropic_key = None

        result = client._get_anthropic_client()

        assert result is None


class TestLLMClientPoolMethods:
    """Tests for pool service methods."""

    @pytest.mark.asyncio
    async def test_is_pool_available_true(self):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(client, "_get_http_session", return_value=mock_session):
            result = await client.is_pool_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_pool_available_false_on_error(self):
        client = LLMClient()

        with patch.object(client, "_get_http_session", side_effect=Exception("Connection failed")):
            result = await client.is_pool_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_pool_status_success(self):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "gemini": {"available": True, "authenticated": True, "rate_limited": False},
            "chatgpt": {"available": True, "authenticated": True, "rate_limited": False},
            "claude": {"available": True, "authenticated": True, "rate_limited": False},
        })

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            status = await client.get_pool_status()

        assert isinstance(status, PoolStatus)
        assert status.gemini.available is True

    @pytest.mark.asyncio
    async def test_get_pool_status_error(self):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.status = 500

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            with pytest.raises(PoolUnavailableError):
                await client.get_pool_status()


class TestLLMClientSend:
    """Tests for the unified send method."""

    @pytest.mark.asyncio
    async def test_send_routes_gemini_to_pool(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_via_pool", return_value=expected_response) as mock_pool:
            response = await client.send("gemini", "Test prompt")

            mock_pool.assert_called_once_with(
                "gemini", "Test prompt", False, True, 300, "normal"
            )
            assert response == expected_response

    @pytest.mark.asyncio
    async def test_send_routes_chatgpt_to_pool(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_via_pool", return_value=expected_response) as mock_pool:
            response = await client.send("chatgpt", "Test prompt", deep_mode=True)

            mock_pool.assert_called_once_with(
                "chatgpt", "Test prompt", True, True, 300, "normal"
            )

    @pytest.mark.asyncio
    async def test_send_routes_claude_to_api(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello from Claude")

        with patch.object(client, "_send_to_claude", return_value=expected_response) as mock_api:
            response = await client.send("claude", "Test prompt")

            mock_api.assert_called_once_with(
                "Test prompt",
                model="claude-sonnet-4-20250514",
                timeout_seconds=300,
            )
            assert response == expected_response

    @pytest.mark.asyncio
    async def test_send_routes_openrouter_to_api(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello from OpenRouter")

        with patch.object(client, "_send_to_openrouter", return_value=expected_response) as mock_api:
            response = await client.send("openrouter", "Test prompt")

            mock_api.assert_called_once_with(
                "Test prompt",
                model="deepseek/deepseek-r1",
                timeout_seconds=300,
            )

    @pytest.mark.asyncio
    async def test_send_unknown_backend(self):
        client = LLMClient()

        response = await client.send("unknown", "Test")

        assert response.success is False
        assert response.error == "unknown_backend"

    @pytest.mark.asyncio
    async def test_send_with_custom_model(self):
        client = LLMClient()

        expected_response = LLMResponse(success=True, text="Hello")

        with patch.object(client, "_send_to_claude", return_value=expected_response) as mock_api:
            await client.send("claude", "Test", model="claude-opus-4-20250514")

            mock_api.assert_called_once_with(
                "Test",
                model="claude-opus-4-20250514",
                timeout_seconds=300,
            )


class TestLLMClientSendViaPool:
    """Tests for _send_via_pool method."""

    @pytest.mark.asyncio
    async def test_send_via_pool_success(self, mock_pool_response):
        client = LLMClient()

        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_pool_response)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            response = await client._send_via_pool(
                "gemini", "Test", False, True, 300, "normal"
            )

        assert response.success is True
        assert response.text == "Hello from pool!"
        assert response.backend == "gemini"


class TestLLMClientSendToClaude:
    """Tests for _send_to_claude method."""

    @pytest.mark.asyncio
    async def test_send_to_claude_success(self, mock_anthropic_client):
        client = LLMClient(anthropic_api_key="test-key")

        with patch.object(client, "_get_anthropic_client", return_value=mock_anthropic_client):
            response = await client._send_to_claude("Test prompt")

        assert response.success is True
        assert response.text == "Claude response"
        assert response.backend == "claude"

    @pytest.mark.asyncio
    async def test_send_to_claude_no_api_key(self):
        client = LLMClient()
        client._anthropic_key = None

        response = await client._send_to_claude("Test prompt")

        assert response.success is False
        assert response.error == "auth_required"

    @pytest.mark.asyncio
    async def test_send_to_claude_rate_limit(self):
        client = LLMClient(anthropic_api_key="test-key")

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(
            side_effect=Exception("Rate limit exceeded (429)")
        )

        with patch.object(client, "_get_anthropic_client", return_value=mock_client):
            response = await client._send_to_claude("Test prompt")

        assert response.success is False
        assert response.error == "rate_limited"
        assert response.retry_after_seconds == 60


class TestLLMClientSendToOpenRouter:
    """Tests for _send_to_openrouter method."""

    @pytest.mark.asyncio
    async def test_send_to_openrouter_no_api_key(self):
        client = LLMClient()
        client._openrouter_key = None

        response = await client._send_to_openrouter("Test prompt")

        assert response.success is False
        assert response.error == "auth_required"

    @pytest.mark.asyncio
    async def test_send_to_openrouter_success(self):
        client = LLMClient(openrouter_api_key="or-key")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "DeepSeek response"}}]
        })

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_context)

        with patch.object(client, "_get_http_session", return_value=mock_session):
            response = await client._send_to_openrouter("Test prompt")

        assert response.success is True
        assert response.text == "DeepSeek response"
        assert response.backend == "openrouter"


class TestLLMClientSendParallel:
    """Tests for send_parallel method."""

    @pytest.mark.asyncio
    async def test_send_parallel_all_success(self):
        client = LLMClient()

        responses = {
            "gemini": LLMResponse(success=True, text="Gemini says hi"),
            "claude": LLMResponse(success=True, text="Claude says hi"),
        }

        async def mock_send(backend, prompt, **kwargs):
            return responses[backend]

        with patch.object(client, "send", side_effect=mock_send):
            results = await client.send_parallel({
                "gemini": "Test",
                "claude": "Test",
            })

        assert results["gemini"].text == "Gemini says hi"
        assert results["claude"].text == "Claude says hi"

    @pytest.mark.asyncio
    async def test_send_parallel_handles_exceptions(self):
        client = LLMClient()

        async def mock_send(backend, prompt, **kwargs):
            if backend == "gemini":
                raise Exception("Gemini failed")
            return LLMResponse(success=True, text="Claude says hi")

        with patch.object(client, "send", side_effect=mock_send):
            results = await client.send_parallel({
                "gemini": "Test",
                "claude": "Test",
            })

        assert results["gemini"].success is False
        assert results["gemini"].error == "exception"
        assert "Gemini failed" in results["gemini"].message
        assert results["claude"].success is True


class TestLLMClientGetAvailableBackends:
    """Tests for get_available_backends method."""

    @pytest.mark.asyncio
    async def test_includes_pool_backends(self):
        client = LLMClient()

        mock_status = PoolStatus(
            gemini=BackendStatus(available=True, authenticated=True, rate_limited=False),
            chatgpt=BackendStatus(available=False, authenticated=True, rate_limited=True),
            claude=BackendStatus(available=True, authenticated=True, rate_limited=False),
        )

        with patch.object(client, "get_pool_status", return_value=mock_status):
            available = await client.get_available_backends()

        assert "gemini" in available
        assert "chatgpt" not in available

    @pytest.mark.asyncio
    async def test_includes_api_backends_with_keys(self):
        client = LLMClient(anthropic_api_key="key", openrouter_api_key="key")

        with patch.object(client, "get_pool_status", side_effect=PoolUnavailableError()):
            available = await client.get_available_backends()

        assert "claude" in available
        assert "openrouter" in available

    @pytest.mark.asyncio
    async def test_handles_pool_unavailable(self):
        client = LLMClient(anthropic_api_key="key")

        with patch.object(client, "get_pool_status", side_effect=PoolUnavailableError()):
            available = await client.get_available_backends()

        # Pool backends not available, but Claude is
        assert "gemini" not in available
        assert "claude" in available


class TestLLMClientConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_gemini_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.gemini("Test", deep_think=True)

            mock_send.assert_called_once_with(
                "gemini", "Test",
                deep_mode=True,
                new_chat=True,
                timeout_seconds=300,
            )
            assert result == expected

    @pytest.mark.asyncio
    async def test_chatgpt_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.chatgpt("Test", pro_mode=True)

            mock_send.assert_called_once_with(
                "chatgpt", "Test",
                deep_mode=True,
                new_chat=True,
                timeout_seconds=300,
            )

    @pytest.mark.asyncio
    async def test_claude_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.claude("Test", model="claude-opus-4-20250514")

            mock_send.assert_called_once_with(
                "claude", "Test",
                model="claude-opus-4-20250514",
                timeout_seconds=300,
            )

    @pytest.mark.asyncio
    async def test_deepseek_method(self):
        client = LLMClient()
        expected = LLMResponse(success=True, text="Hello")

        with patch.object(client, "send", return_value=expected) as mock_send:
            result = await client.deepseek("Test")

            mock_send.assert_called_once_with(
                "openrouter", "Test",
                model="deepseek/deepseek-r1",
                timeout_seconds=300,
            )


class TestConstants:
    """Tests for module constants."""

    def test_browser_backends(self):
        assert "gemini" in BROWSER_BACKENDS
        assert "chatgpt" in BROWSER_BACKENDS
        assert "claude" not in BROWSER_BACKENDS

    def test_api_backends(self):
        assert "claude" in API_BACKENDS
        assert "openrouter" in API_BACKENDS
        assert "gemini" not in API_BACKENDS
