"""Tests for pool state management."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from pool.src.state import StateManager


class TestStateManagerInit:
    """Tests for StateManager initialization."""

    def test_creates_default_state_when_no_file(self, temp_state_file, sample_config):
        """StateManager creates default state when file doesn't exist."""
        manager = StateManager(temp_state_file, sample_config)

        state = manager._state
        assert "gemini" in state
        assert "chatgpt" in state
        assert "claude" in state

        # Gemini defaults
        assert state["gemini"]["authenticated"] is False
        assert state["gemini"]["rate_limited"] is False
        assert state["gemini"]["deep_mode_uses_today"] == 0

        # ChatGPT defaults
        assert state["chatgpt"]["authenticated"] is False
        assert state["chatgpt"]["pro_mode_uses_today"] == 0

        # Claude defaults (API-based, always "authenticated")
        assert state["claude"]["authenticated"] is True

    def test_loads_existing_state(self, temp_state_file, sample_config):
        """StateManager loads existing state from file."""
        existing_state = {
            "gemini": {
                "authenticated": True,
                "rate_limited": False,
                "deep_mode_uses_today": 5,
                "deep_mode_reset_date": "2024-01-01",
            },
            "chatgpt": {
                "authenticated": True,
                "rate_limited": True,
                "rate_limit_resets_at": "2024-01-01T12:00:00",
            },
            "claude": {
                "authenticated": True,
                "rate_limited": False,
            },
        }
        temp_state_file.write_text(json.dumps(existing_state))

        manager = StateManager(temp_state_file, sample_config)

        assert manager._state["gemini"]["authenticated"] is True
        assert manager._state["gemini"]["deep_mode_uses_today"] == 5
        assert manager._state["chatgpt"]["rate_limited"] is True

    def test_handles_corrupt_state_file(self, temp_state_file, sample_config):
        """StateManager handles corrupt state file gracefully."""
        temp_state_file.write_text("not valid json {{{")

        manager = StateManager(temp_state_file, sample_config)

        # Should fall back to defaults
        assert manager._state["gemini"]["authenticated"] is False
        assert manager._state["claude"]["authenticated"] is True


class TestStateManagerSave:
    """Tests for state persistence."""

    def test_save_creates_file(self, temp_state_file, sample_config):
        """StateManager saves state to file."""
        manager = StateManager(temp_state_file, sample_config)
        manager._save()

        assert temp_state_file.exists()
        saved_state = json.loads(temp_state_file.read_text())
        assert "gemini" in saved_state

    def test_save_creates_parent_directories(self, tmp_path, sample_config):
        """StateManager creates parent directories when saving."""
        nested_path = tmp_path / "nested" / "dir" / "state.json"
        manager = StateManager(nested_path, sample_config)
        manager._save()

        assert nested_path.exists()


class TestGetBackendState:
    """Tests for getting backend state."""

    def test_returns_copy_of_state(self, temp_state_file, sample_config):
        """get_backend_state returns a copy, not the original."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["authenticated"] = True

        state = manager.get_backend_state("gemini")
        state["authenticated"] = False  # Modify the copy

        # Original should be unchanged
        assert manager._state["gemini"]["authenticated"] is True

    def test_returns_empty_dict_for_unknown_backend(self, temp_state_file, sample_config):
        """get_backend_state returns empty dict for unknown backend."""
        manager = StateManager(temp_state_file, sample_config)
        state = manager.get_backend_state("unknown")
        assert state == {}


class TestRateLimitExpiry:
    """Tests for rate limit auto-expiry."""

    def test_clears_expired_rate_limit(self, temp_state_file, sample_config):
        """get_backend_state clears expired rate limits."""
        manager = StateManager(temp_state_file, sample_config)

        # Set rate limit that expired an hour ago
        past_time = datetime.now() - timedelta(hours=1)
        manager._state["gemini"]["rate_limited"] = True
        manager._state["gemini"]["rate_limit_resets_at"] = past_time.isoformat()

        state = manager.get_backend_state("gemini")

        assert state["rate_limited"] is False
        assert manager._state["gemini"]["rate_limited"] is False

    def test_keeps_active_rate_limit(self, temp_state_file, sample_config):
        """get_backend_state keeps active rate limits."""
        manager = StateManager(temp_state_file, sample_config)

        # Set rate limit that expires in an hour
        future_time = datetime.now() + timedelta(hours=1)
        manager._state["gemini"]["rate_limited"] = True
        manager._state["gemini"]["rate_limit_resets_at"] = future_time.isoformat()

        state = manager.get_backend_state("gemini")

        assert state["rate_limited"] is True


class TestDailyReset:
    """Tests for daily counter reset."""

    def test_resets_deep_mode_counter_on_new_day(self, temp_state_file, sample_config):
        """Deep mode counter resets on new day."""
        manager = StateManager(temp_state_file, sample_config)

        # Set uses from yesterday
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        manager._state["gemini"]["deep_mode_uses_today"] = 15
        manager._state["gemini"]["deep_mode_reset_date"] = yesterday

        state = manager.get_backend_state("gemini")

        assert state["deep_mode_uses_today"] == 0
        assert state["deep_mode_reset_date"] == datetime.now().date().isoformat()

    def test_resets_pro_mode_counter_on_new_day(self, temp_state_file, sample_config):
        """Pro mode counter resets on new day."""
        manager = StateManager(temp_state_file, sample_config)

        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        manager._state["chatgpt"]["pro_mode_uses_today"] = 50
        manager._state["chatgpt"]["pro_mode_reset_date"] = yesterday

        state = manager.get_backend_state("chatgpt")

        assert state["pro_mode_uses_today"] == 0

    def test_keeps_counter_on_same_day(self, temp_state_file, sample_config):
        """Counter is not reset on same day."""
        manager = StateManager(temp_state_file, sample_config)

        today = datetime.now().date().isoformat()
        manager._state["gemini"]["deep_mode_uses_today"] = 10
        manager._state["gemini"]["deep_mode_reset_date"] = today

        state = manager.get_backend_state("gemini")

        assert state["deep_mode_uses_today"] == 10


class TestMarkAuthenticated:
    """Tests for authentication status management."""

    def test_marks_backend_authenticated(self, temp_state_file, sample_config):
        """mark_authenticated sets authentication status."""
        manager = StateManager(temp_state_file, sample_config)

        manager.mark_authenticated("gemini", True)

        assert manager._state["gemini"]["authenticated"] is True
        # Also check it was saved
        saved = json.loads(temp_state_file.read_text())
        assert saved["gemini"]["authenticated"] is True

    def test_marks_backend_unauthenticated(self, temp_state_file, sample_config):
        """mark_authenticated can set False."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["authenticated"] = True

        manager.mark_authenticated("gemini", False)

        assert manager._state["gemini"]["authenticated"] is False


class TestMarkRateLimited:
    """Tests for rate limit management."""

    def test_marks_backend_rate_limited(self, temp_state_file, sample_config):
        """mark_rate_limited sets rate limit with expiry."""
        manager = StateManager(temp_state_file, sample_config)

        before = datetime.now()
        manager.mark_rate_limited("gemini", retry_after_seconds=3600)
        after = datetime.now()

        assert manager._state["gemini"]["rate_limited"] is True
        reset_at = datetime.fromisoformat(manager._state["gemini"]["rate_limit_resets_at"])
        assert reset_at >= before + timedelta(seconds=3600)
        assert reset_at <= after + timedelta(seconds=3600)

    def test_default_retry_after(self, temp_state_file, sample_config):
        """mark_rate_limited uses default of 3600 seconds."""
        manager = StateManager(temp_state_file, sample_config)

        manager.mark_rate_limited("chatgpt")

        reset_at = datetime.fromisoformat(manager._state["chatgpt"]["rate_limit_resets_at"])
        expected = datetime.now() + timedelta(seconds=3600)
        # Allow 1 second tolerance
        assert abs((reset_at - expected).total_seconds()) < 1


class TestClearRateLimit:
    """Tests for clearing rate limits."""

    def test_clears_rate_limit(self, temp_state_file, sample_config):
        """clear_rate_limit removes rate limit status."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["rate_limited"] = True
        manager._state["gemini"]["rate_limit_resets_at"] = datetime.now().isoformat()

        manager.clear_rate_limit("gemini")

        assert manager._state["gemini"]["rate_limited"] is False
        assert manager._state["gemini"]["rate_limit_resets_at"] is None


class TestIncrementDeepModeUsage:
    """Tests for deep mode usage tracking."""

    def test_increments_gemini_deep_mode(self, temp_state_file, sample_config):
        """increment_deep_mode_usage increments Gemini counter."""
        manager = StateManager(temp_state_file, sample_config)
        today = datetime.now().date().isoformat()
        manager._state["gemini"]["deep_mode_uses_today"] = 5
        manager._state["gemini"]["deep_mode_reset_date"] = today

        manager.increment_deep_mode_usage("gemini")

        assert manager._state["gemini"]["deep_mode_uses_today"] == 6

    def test_increments_chatgpt_pro_mode(self, temp_state_file, sample_config):
        """increment_deep_mode_usage increments ChatGPT pro mode counter."""
        manager = StateManager(temp_state_file, sample_config)
        today = datetime.now().date().isoformat()
        manager._state["chatgpt"]["pro_mode_uses_today"] = 10
        manager._state["chatgpt"]["pro_mode_reset_date"] = today

        manager.increment_deep_mode_usage("chatgpt")

        assert manager._state["chatgpt"]["pro_mode_uses_today"] == 11

    def test_saves_after_increment(self, temp_state_file, sample_config):
        """Counter increment is persisted."""
        manager = StateManager(temp_state_file, sample_config)
        manager.increment_deep_mode_usage("gemini")

        saved = json.loads(temp_state_file.read_text())
        assert saved["gemini"]["deep_mode_uses_today"] == 1


class TestCanUseDeepMode:
    """Tests for deep mode availability check."""

    def test_returns_true_under_limit(self, temp_state_file, sample_config):
        """can_use_deep_mode returns True when under limit."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["deep_mode_uses_today"] = 5
        manager._state["gemini"]["deep_mode_reset_date"] = datetime.now().date().isoformat()

        assert manager.can_use_deep_mode("gemini") is True

    def test_returns_false_at_limit(self, temp_state_file, sample_config):
        """can_use_deep_mode returns False at limit."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["deep_mode_uses_today"] = 20  # Default limit
        manager._state["gemini"]["deep_mode_reset_date"] = datetime.now().date().isoformat()

        assert manager.can_use_deep_mode("gemini") is False

    def test_returns_true_for_chatgpt_under_limit(self, temp_state_file, sample_config):
        """can_use_deep_mode works for ChatGPT pro mode."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["chatgpt"]["pro_mode_uses_today"] = 50
        manager._state["chatgpt"]["pro_mode_reset_date"] = datetime.now().date().isoformat()

        assert manager.can_use_deep_mode("chatgpt") is True

    def test_returns_false_for_chatgpt_at_limit(self, temp_state_file, sample_config):
        """can_use_deep_mode returns False for ChatGPT at limit."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["chatgpt"]["pro_mode_uses_today"] = 100  # Default limit
        manager._state["chatgpt"]["pro_mode_reset_date"] = datetime.now().date().isoformat()

        assert manager.can_use_deep_mode("chatgpt") is False

    def test_returns_true_for_claude(self, temp_state_file, sample_config):
        """can_use_deep_mode always returns True for Claude."""
        manager = StateManager(temp_state_file, sample_config)
        assert manager.can_use_deep_mode("claude") is True


class TestIsAvailable:
    """Tests for backend availability check."""

    def test_available_when_authenticated_and_not_rate_limited(self, temp_state_file, sample_config):
        """is_available returns True when authenticated and not rate limited."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["authenticated"] = True
        manager._state["gemini"]["rate_limited"] = False

        assert manager.is_available("gemini") is True

    def test_unavailable_when_not_authenticated(self, temp_state_file, sample_config):
        """is_available returns False when not authenticated."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["authenticated"] = False
        manager._state["gemini"]["rate_limited"] = False

        assert manager.is_available("gemini") is False

    def test_unavailable_when_rate_limited(self, temp_state_file, sample_config):
        """is_available returns False when rate limited."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["authenticated"] = True
        manager._state["gemini"]["rate_limited"] = True
        future = datetime.now() + timedelta(hours=1)
        manager._state["gemini"]["rate_limit_resets_at"] = future.isoformat()

        assert manager.is_available("gemini") is False

    def test_available_after_rate_limit_expires(self, temp_state_file, sample_config):
        """is_available returns True after rate limit expires."""
        manager = StateManager(temp_state_file, sample_config)
        manager._state["gemini"]["authenticated"] = True
        manager._state["gemini"]["rate_limited"] = True
        past = datetime.now() - timedelta(hours=1)
        manager._state["gemini"]["rate_limit_resets_at"] = past.isoformat()

        assert manager.is_available("gemini") is True


class TestThreadSafety:
    """Tests for thread safety."""

    def test_has_lock(self, temp_state_file, sample_config):
        """StateManager has a lock for thread safety."""
        manager = StateManager(temp_state_file, sample_config)
        assert hasattr(manager, "_lock")
        assert manager._lock is not None
