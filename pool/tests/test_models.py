"""Tests for pool models."""

import pytest
from datetime import datetime

from pool.src.models import (
    Priority,
    Backend,
    SendOptions,
    SendRequest,
    ResponseMetadata,
    SendResponse,
    BackendStatus,
    PoolStatus,
    HealthResponse,
    AuthResponse,
)


class TestPriorityEnum:
    """Tests for Priority enum."""

    def test_priority_values(self):
        assert Priority.LOW.value == "low"
        assert Priority.NORMAL.value == "normal"
        assert Priority.HIGH.value == "high"

    def test_priority_is_string_enum(self):
        assert isinstance(Priority.LOW, str)
        assert Priority.LOW == "low"


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_backend_values(self):
        assert Backend.GEMINI.value == "gemini"
        assert Backend.CHATGPT.value == "chatgpt"
        assert Backend.CLAUDE.value == "claude"

    def test_backend_is_string_enum(self):
        assert isinstance(Backend.GEMINI, str)
        assert Backend.GEMINI == "gemini"


class TestSendOptions:
    """Tests for SendOptions model."""

    def test_default_values(self):
        options = SendOptions()
        assert options.deep_mode is False
        assert options.timeout_seconds == 300
        assert options.priority == Priority.NORMAL
        assert options.new_chat is True

    def test_custom_values(self):
        options = SendOptions(
            deep_mode=True,
            timeout_seconds=600,
            priority=Priority.HIGH,
            new_chat=False,
        )
        assert options.deep_mode is True
        assert options.timeout_seconds == 600
        assert options.priority == Priority.HIGH
        assert options.new_chat is False


class TestSendRequest:
    """Tests for SendRequest model."""

    def test_minimal_request(self):
        request = SendRequest(backend=Backend.GEMINI, prompt="Hello")
        assert request.backend == Backend.GEMINI
        assert request.prompt == "Hello"
        assert request.options.deep_mode is False

    def test_request_with_options(self):
        options = SendOptions(deep_mode=True, priority=Priority.HIGH)
        request = SendRequest(
            backend=Backend.CLAUDE,
            prompt="Test prompt",
            options=options,
        )
        assert request.backend == Backend.CLAUDE
        assert request.prompt == "Test prompt"
        assert request.options.deep_mode is True
        assert request.options.priority == Priority.HIGH

    def test_request_serialization(self):
        request = SendRequest(backend=Backend.CHATGPT, prompt="Test")
        data = request.model_dump()
        assert data["backend"] == "chatgpt"
        assert data["prompt"] == "Test"


class TestResponseMetadata:
    """Tests for ResponseMetadata model."""

    def test_minimal_metadata(self):
        meta = ResponseMetadata(backend="gemini", response_time_seconds=1.5)
        assert meta.backend == "gemini"
        assert meta.response_time_seconds == 1.5
        assert meta.deep_mode_used is False
        assert meta.session_id is None

    def test_full_metadata(self):
        meta = ResponseMetadata(
            backend="chatgpt",
            deep_mode_used=True,
            response_time_seconds=2.5,
            session_id="session-abc",
        )
        assert meta.deep_mode_used is True
        assert meta.session_id == "session-abc"


class TestSendResponse:
    """Tests for SendResponse model."""

    def test_success_response(self):
        meta = ResponseMetadata(backend="gemini", response_time_seconds=1.0)
        response = SendResponse(
            success=True,
            response="Hello world",
            metadata=meta,
        )
        assert response.success is True
        assert response.response == "Hello world"
        assert response.error is None
        assert response.metadata.backend == "gemini"

    def test_error_response(self):
        response = SendResponse(
            success=False,
            error="rate_limited",
            message="Too many requests",
            retry_after_seconds=3600,
        )
        assert response.success is False
        assert response.error == "rate_limited"
        assert response.message == "Too many requests"
        assert response.retry_after_seconds == 3600


class TestBackendStatus:
    """Tests for BackendStatus model."""

    def test_available_backend(self):
        status = BackendStatus(
            available=True,
            authenticated=True,
            rate_limited=False,
            queue_depth=5,
        )
        assert status.available is True
        assert status.authenticated is True
        assert status.rate_limited is False
        assert status.queue_depth == 5

    def test_rate_limited_backend(self):
        reset_time = datetime.now()
        status = BackendStatus(
            available=False,
            authenticated=True,
            rate_limited=True,
            rate_limit_resets_at=reset_time,
        )
        assert status.available is False
        assert status.rate_limited is True
        assert status.rate_limit_resets_at == reset_time

    def test_gemini_deep_mode_tracking(self):
        status = BackendStatus(
            available=True,
            authenticated=True,
            rate_limited=False,
            deep_mode_uses_today=15,
            deep_mode_limit=20,
        )
        assert status.deep_mode_uses_today == 15
        assert status.deep_mode_limit == 20

    def test_chatgpt_pro_mode_tracking(self):
        status = BackendStatus(
            available=True,
            authenticated=True,
            rate_limited=False,
            pro_mode_uses_today=50,
            pro_mode_limit=100,
        )
        assert status.pro_mode_uses_today == 50
        assert status.pro_mode_limit == 100


class TestPoolStatus:
    """Tests for PoolStatus model."""

    def test_empty_pool_status(self):
        status = PoolStatus()
        assert status.gemini is None
        assert status.chatgpt is None
        assert status.claude is None

    def test_full_pool_status(self):
        gemini = BackendStatus(available=True, authenticated=True, rate_limited=False)
        chatgpt = BackendStatus(available=False, authenticated=True, rate_limited=True)
        claude = BackendStatus(available=True, authenticated=True, rate_limited=False)

        status = PoolStatus(gemini=gemini, chatgpt=chatgpt, claude=claude)
        assert status.gemini.available is True
        assert status.chatgpt.rate_limited is True
        assert status.claude.available is True


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self):
        health = HealthResponse(
            status="ok",
            uptime_seconds=3600.5,
            version="0.1.0",
        )
        assert health.status == "ok"
        assert health.uptime_seconds == 3600.5
        assert health.version == "0.1.0"


class TestAuthResponse:
    """Tests for AuthResponse model."""

    def test_success_auth_response(self):
        response = AuthResponse(success=True, message="Authenticated successfully")
        assert response.success is True
        assert response.message == "Authenticated successfully"

    def test_failure_auth_response(self):
        response = AuthResponse(success=False, message="Authentication failed")
        assert response.success is False
        assert response.message == "Authentication failed"
