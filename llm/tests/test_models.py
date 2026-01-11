"""Tests for LLM library models."""

import pytest
from datetime import datetime

from llm.src.models import (
    Backend,
    Priority,
    LLMResponse,
    BackendStatus,
    PoolStatus,
    ReviewResponse,
    ConsensusResult,
)


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_backend_values(self):
        assert Backend.GEMINI.value == "gemini"
        assert Backend.CHATGPT.value == "chatgpt"
        assert Backend.CLAUDE.value == "claude"

    def test_backend_is_string_enum(self):
        assert isinstance(Backend.GEMINI, str)
        assert Backend.GEMINI == "gemini"


class TestPriorityEnum:
    """Tests for Priority enum."""

    def test_priority_values(self):
        assert Priority.LOW.value == "low"
        assert Priority.NORMAL.value == "normal"
        assert Priority.HIGH.value == "high"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_success_response(self):
        response = LLMResponse(
            success=True,
            text="Hello world",
            backend="gemini",
            response_time_seconds=1.5,
        )
        assert response.success is True
        assert response.text == "Hello world"
        assert response.backend == "gemini"
        assert response.response_time_seconds == 1.5

    def test_error_response(self):
        response = LLMResponse(
            success=False,
            error="rate_limited",
            message="Too many requests",
            retry_after_seconds=3600,
        )
        assert response.success is False
        assert response.error == "rate_limited"
        assert response.retry_after_seconds == 3600

    def test_from_pool_response_success(self, mock_pool_response):
        response = LLMResponse.from_pool_response(mock_pool_response)

        assert response.success is True
        assert response.text == "Hello from pool!"
        assert response.backend == "gemini"
        assert response.deep_mode_used is False
        assert response.response_time_seconds == 1.5
        assert response.session_id == "session-123"

    def test_from_pool_response_error(self, mock_pool_error_response):
        response = LLMResponse.from_pool_response(mock_pool_error_response)

        assert response.success is False
        assert response.error == "rate_limited"
        assert response.message == "Too many requests"
        assert response.retry_after_seconds == 3600

    def test_from_pool_response_no_metadata(self):
        data = {
            "success": True,
            "response": "Hello",
            "metadata": None,
        }
        response = LLMResponse.from_pool_response(data)

        assert response.success is True
        assert response.text == "Hello"
        assert response.backend is None
        assert response.deep_mode_used is False

    def test_default_values(self):
        response = LLMResponse(success=True)
        assert response.text is None
        assert response.error is None
        assert response.deep_mode_used is False
        assert response.response_time_seconds == 0.0


class TestBackendStatus:
    """Tests for BackendStatus dataclass."""

    def test_from_dict_complete(self):
        data = {
            "available": True,
            "authenticated": True,
            "rate_limited": False,
            "rate_limit_resets_at": "2024-01-01T12:00:00",
            "queue_depth": 5,
            "deep_mode_uses_today": 10,
            "deep_mode_limit": 20,
        }
        status = BackendStatus.from_dict(data)

        assert status.available is True
        assert status.authenticated is True
        assert status.rate_limited is False
        assert status.rate_limit_resets_at == datetime.fromisoformat("2024-01-01T12:00:00")
        assert status.queue_depth == 5
        assert status.deep_mode_uses_today == 10
        assert status.deep_mode_limit == 20

    def test_from_dict_minimal(self):
        data = {}
        status = BackendStatus.from_dict(data)

        assert status.available is False
        assert status.authenticated is False
        assert status.rate_limited is False
        assert status.rate_limit_resets_at is None
        assert status.queue_depth == 0

    def test_from_dict_invalid_datetime(self):
        data = {
            "available": True,
            "rate_limit_resets_at": "not-a-date",
        }
        status = BackendStatus.from_dict(data)

        assert status.rate_limit_resets_at is None

    def test_from_dict_pro_mode(self):
        data = {
            "available": True,
            "authenticated": True,
            "rate_limited": False,
            "pro_mode_uses_today": 50,
            "pro_mode_limit": 100,
        }
        status = BackendStatus.from_dict(data)

        assert status.pro_mode_uses_today == 50
        assert status.pro_mode_limit == 100


class TestPoolStatus:
    """Tests for PoolStatus dataclass."""

    def test_from_dict_complete(self):
        data = {
            "gemini": {
                "available": True,
                "authenticated": True,
                "rate_limited": False,
            },
            "chatgpt": {
                "available": False,
                "authenticated": True,
                "rate_limited": True,
            },
            "claude": {
                "available": True,
                "authenticated": True,
                "rate_limited": False,
            },
        }
        status = PoolStatus.from_dict(data)

        assert status.gemini is not None
        assert status.gemini.available is True
        assert status.chatgpt is not None
        assert status.chatgpt.rate_limited is True
        assert status.claude is not None

    def test_from_dict_partial(self):
        data = {
            "gemini": {
                "available": True,
                "authenticated": True,
                "rate_limited": False,
            },
        }
        status = PoolStatus.from_dict(data)

        assert status.gemini is not None
        assert status.chatgpt is None
        assert status.claude is None

    def test_get_available_backends(self):
        data = {
            "gemini": {"available": True, "authenticated": True, "rate_limited": False},
            "chatgpt": {"available": False, "authenticated": True, "rate_limited": True},
            "claude": {"available": True, "authenticated": True, "rate_limited": False},
        }
        status = PoolStatus.from_dict(data)

        available = status.get_available_backends()

        assert "gemini" in available
        assert "chatgpt" not in available
        assert "claude" in available

    def test_get_available_backends_empty(self):
        status = PoolStatus()
        available = status.get_available_backends()
        assert available == []


class TestReviewResponse:
    """Tests for ReviewResponse dataclass."""

    def test_basic_creation(self):
        response = ReviewResponse(
            llm="gemini",
            mode="standard",
            rating="bless",
            reasoning="This is valid",
            confidence="high",
        )
        assert response.llm == "gemini"
        assert response.mode == "standard"
        assert response.rating == "bless"
        assert response.reasoning == "This is valid"
        assert response.confidence == "high"

    def test_with_analysis_fields(self):
        response = ReviewResponse(
            llm="claude",
            mode="deep",
            rating="uncertain",
            reasoning="Needs more work",
            confidence="medium",
            mathematical_verification="Math checks out",
            structural_analysis="Deep connection",
            naturalness_assessment="Feels discovered",
        )
        assert response.mathematical_verification == "Math checks out"
        assert response.structural_analysis == "Deep connection"
        assert response.naturalness_assessment == "Feels discovered"

    def test_with_modification(self):
        response = ReviewResponse(
            llm="chatgpt",
            mode="pro",
            rating="bless",
            reasoning="Valid",
            confidence="high",
            proposed_modification="Updated text",
            modification_rationale="Minor clarification",
        )
        assert response.proposed_modification == "Updated text"
        assert response.modification_rationale == "Minor clarification"

    def test_to_dict(self):
        response = ReviewResponse(
            llm="gemini",
            mode="standard",
            rating="bless",
            reasoning="Valid",
            confidence="high",
            mathematical_verification="Correct",
        )
        data = response.to_dict()

        assert data["llm"] == "gemini"
        assert data["mode"] == "standard"
        assert data["rating"] == "bless"
        assert data["reasoning"] == "Valid"
        assert data["confidence"] == "high"
        assert data["mathematical_verification"] == "Correct"

    def test_from_dict(self):
        data = {
            "llm": "claude",
            "mode": "deep",
            "rating": "uncertain",
            "reasoning": "Needs review",
            "confidence": "low",
            "mathematical_verification": "Some issues",
            "structural_analysis": "Shallow",
            "naturalness_assessment": "Feels forced",
        }
        response = ReviewResponse.from_dict(data)

        assert response.llm == "claude"
        assert response.mode == "deep"
        assert response.rating == "uncertain"
        assert response.mathematical_verification == "Some issues"

    def test_roundtrip(self):
        original = ReviewResponse(
            llm="gemini",
            mode="standard",
            rating="bless",
            reasoning="Valid insight",
            confidence="high",
            mathematical_verification="2+2=4",
            structural_analysis="Deep",
            naturalness_assessment="Natural",
            proposed_modification=None,
            modification_rationale=None,
        )

        data = original.to_dict()
        restored = ReviewResponse.from_dict(data)

        assert restored.llm == original.llm
        assert restored.mode == original.mode
        assert restored.rating == original.rating
        assert restored.reasoning == original.reasoning


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""

    def test_basic_creation(self):
        result = ConsensusResult(
            success=True,
            final_rating="bless",
            is_unanimous=True,
            is_disputed=False,
        )
        assert result.success is True
        assert result.final_rating == "bless"
        assert result.is_unanimous is True
        assert result.is_disputed is False

    def test_with_rounds(self):
        result = ConsensusResult(
            success=True,
            final_rating="uncertain",
            is_unanimous=False,
            is_disputed=True,
            rounds=[
                {"round": 1, "responses": {"gemini": {}, "claude": {}}},
                {"round": 2, "responses": {"gemini": {}, "claude": {}}},
            ],
        )
        assert len(result.rounds) == 2
        assert result.rounds[0]["round"] == 1

    def test_with_mind_changes(self):
        result = ConsensusResult(
            success=True,
            final_rating="bless",
            is_unanimous=True,
            is_disputed=False,
            mind_changes=[
                {"llm": "gemini", "from": "uncertain", "to": "bless"},
            ],
        )
        assert len(result.mind_changes) == 1
        assert result.mind_changes[0]["llm"] == "gemini"

    def test_to_dict(self):
        result = ConsensusResult(
            success=True,
            final_rating="bless",
            is_unanimous=True,
            is_disputed=False,
            rounds=[{"round": 1}],
            final_text="Updated insight",
            mind_changes=[],
            review_duration_seconds=5.5,
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["final_rating"] == "bless"
        assert data["is_unanimous"] is True
        assert data["is_disputed"] is False
        assert data["rounds"] == [{"round": 1}]
        assert data["final_text"] == "Updated insight"
        assert data["review_duration_seconds"] == 5.5

    def test_default_values(self):
        result = ConsensusResult(
            success=True,
            final_rating="bless",
            is_unanimous=True,
            is_disputed=False,
        )
        assert result.rounds == []
        assert result.final_text is None
        assert result.mind_changes == []
        assert result.review_duration_seconds == 0.0
