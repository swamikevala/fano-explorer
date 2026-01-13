"""Tests for async job store."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from pool.src.jobs import JobStore, Job, JobStatus


class TestJobDataclass:
    """Tests for Job dataclass."""

    def test_creates_job_with_required_fields(self):
        """Job can be created with required fields."""
        job = Job(job_id="test-123", backend="gemini", prompt="Hello")

        assert job.job_id == "test-123"
        assert job.backend == "gemini"
        assert job.prompt == "Hello"
        assert job.status == JobStatus.QUEUED
        assert job.thread_id is None

    def test_creates_job_with_all_fields(self):
        """Job can be created with all optional fields."""
        job = Job(
            job_id="test-456",
            backend="chatgpt",
            prompt="Test prompt",
            thread_id="thread-abc",
            deep_mode=True,
            new_chat=False,
            priority="high",
        )

        assert job.thread_id == "thread-abc"
        assert job.deep_mode is True
        assert job.new_chat is False
        assert job.priority == "high"

    def test_to_dict_serializes_job(self):
        """Job.to_dict() serializes job correctly."""
        job = Job(
            job_id="test-789",
            backend="gemini",
            prompt="Test",
            status=JobStatus.PROCESSING,
        )

        d = job.to_dict()

        assert d["job_id"] == "test-789"
        assert d["backend"] == "gemini"
        assert d["status"] == "processing"  # Enum converted to string

    def test_from_dict_deserializes_job(self):
        """Job.from_dict() deserializes job correctly."""
        data = {
            "job_id": "test-abc",
            "backend": "claude",
            "prompt": "Hello",
            "status": "complete",
            "result": "Response text",
            "thread_id": None,
            "deep_mode": False,
            "new_chat": True,
            "priority": "normal",
            "queue_position": 0,
            "error": None,
            "deep_mode_used": False,
            "chat_url": None,
            "created_at": 1234567890.0,
            "started_at": None,
            "completed_at": None,
            "content_hash": "abc123",
        }

        job = Job.from_dict(data)

        assert job.job_id == "test-abc"
        assert job.status == JobStatus.COMPLETE
        assert job.result == "Response text"


class TestJobStoreInit:
    """Tests for JobStore initialization."""

    def test_creates_empty_store(self):
        """JobStore initializes with empty state."""
        store = JobStore()

        assert len(store._jobs) == 0
        assert len(store._backend_queues) == 0

    def test_creates_store_with_persist_path(self, tmp_path):
        """JobStore initializes with persistence path."""
        persist_file = tmp_path / "jobs.json"
        store = JobStore(persist_path=persist_file)

        assert store.persist_path == persist_file

    def test_loads_existing_jobs_from_disk(self, tmp_path):
        """JobStore loads existing jobs from disk."""
        persist_file = tmp_path / "jobs.json"
        existing_data = {
            "jobs": {
                "job-1": {
                    "job_id": "job-1",
                    "backend": "gemini",
                    "prompt": "Test",
                    "status": "complete",
                    "result": "Result",
                    "thread_id": None,
                    "deep_mode": False,
                    "new_chat": True,
                    "priority": "normal",
                    "queue_position": 0,
                    "error": None,
                    "deep_mode_used": False,
                    "chat_url": None,
                    "created_at": 1234567890.0,
                    "started_at": None,
                    "completed_at": None,
                    "content_hash": "abc",
                },
            },
            "queues": {"gemini": []},
        }
        persist_file.write_text(json.dumps(existing_data))

        store = JobStore(persist_path=persist_file)

        assert "job-1" in store._jobs
        assert store._jobs["job-1"].status == JobStatus.COMPLETE


class TestJobStoreSubmit:
    """Tests for job submission."""

    def test_submits_new_job(self):
        """submit() creates a new job."""
        store = JobStore()

        result = store.submit(
            job_id="new-job",
            backend="gemini",
            prompt="Test prompt",
        )

        assert result["status"] == "queued"
        assert result["job_id"] == "new-job"
        assert "new-job" in store._jobs
        assert store._jobs["new-job"].status == JobStatus.QUEUED

    def test_submits_job_with_options(self):
        """submit() creates job with all options."""
        store = JobStore()

        result = store.submit(
            job_id="opt-job",
            backend="chatgpt",
            prompt="Test",
            thread_id="thread-123",
            deep_mode=True,
            new_chat=False,
            priority="high",
        )

        job = store._jobs["opt-job"]
        assert job.thread_id == "thread-123"
        assert job.deep_mode is True
        assert job.new_chat is False
        assert job.priority == "high"

    def test_returns_exists_for_duplicate_job_id(self):
        """submit() returns 'exists' for duplicate job_id."""
        store = JobStore()
        store.submit(job_id="dup-job", backend="gemini", prompt="First")

        result = store.submit(job_id="dup-job", backend="gemini", prompt="Second")

        assert result["status"] == "exists"
        assert result["job_id"] == "dup-job"
        # Original job unchanged
        assert store._jobs["dup-job"].prompt == "First"

    def test_returns_cached_for_same_content(self):
        """submit() returns 'cached' for same backend+prompt content."""
        store = JobStore()
        store.submit(job_id="orig-job", backend="gemini", prompt="Same prompt")
        # Complete the original job
        store._jobs["orig-job"].status = JobStatus.COMPLETE
        store._jobs["orig-job"].result = "Cached result"

        result = store.submit(job_id="new-job", backend="gemini", prompt="Same prompt")

        assert result["status"] == "cached"
        assert result["cached_job_id"] == "orig-job"

    def test_no_cache_hit_for_incomplete_job(self):
        """submit() doesn't cache incomplete jobs."""
        store = JobStore()
        store.submit(job_id="orig-job", backend="gemini", prompt="Same prompt")
        # Job still processing (not complete)

        result = store.submit(job_id="new-job", backend="gemini", prompt="Same prompt")

        assert result["status"] == "queued"
        assert result["job_id"] == "new-job"

    def test_assigns_queue_position(self):
        """submit() assigns incrementing queue positions."""
        store = JobStore()

        store.submit(job_id="job-1", backend="gemini", prompt="First")
        store.submit(job_id="job-2", backend="gemini", prompt="Second")
        store.submit(job_id="job-3", backend="gemini", prompt="Third")

        assert store._jobs["job-1"].queue_position == 1
        assert store._jobs["job-2"].queue_position == 2
        assert store._jobs["job-3"].queue_position == 3

    def test_separate_queues_per_backend(self):
        """submit() maintains separate queues per backend."""
        store = JobStore()

        store.submit(job_id="gem-1", backend="gemini", prompt="G1")
        store.submit(job_id="gpt-1", backend="chatgpt", prompt="C1")
        store.submit(job_id="gem-2", backend="gemini", prompt="G2")

        assert store._jobs["gem-1"].queue_position == 1
        assert store._jobs["gpt-1"].queue_position == 1  # Separate queue
        assert store._jobs["gem-2"].queue_position == 2


class TestJobStoreGetNextJob:
    """Tests for getting next job from queue."""

    def test_returns_next_queued_job(self):
        """get_next_job() returns the next queued job."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="First")
        store.submit(job_id="job-2", backend="gemini", prompt="Second")

        job = store.get_next_job("gemini")

        assert job.job_id == "job-1"
        assert job.status == JobStatus.PROCESSING
        assert job.started_at is not None

    def test_returns_none_for_empty_queue(self):
        """get_next_job() returns None for empty queue."""
        store = JobStore()

        job = store.get_next_job("gemini")

        assert job is None

    def test_returns_none_for_unknown_backend(self):
        """get_next_job() returns None for unknown backend."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")

        job = store.get_next_job("unknown")

        assert job is None

    def test_skips_already_processing_jobs(self):
        """get_next_job() skips jobs not in QUEUED state."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="First")
        store.submit(job_id="job-2", backend="gemini", prompt="Second")
        # Mark first as processing
        store._jobs["job-1"].status = JobStatus.PROCESSING

        job = store.get_next_job("gemini")

        assert job.job_id == "job-2"

    def test_updates_queue_positions(self):
        """get_next_job() updates positions of remaining jobs."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="First")
        store.submit(job_id="job-2", backend="gemini", prompt="Second")
        store.submit(job_id="job-3", backend="gemini", prompt="Third")

        store.get_next_job("gemini")  # Gets job-1

        # Remaining jobs should have updated positions
        assert store._jobs["job-2"].queue_position == 1
        assert store._jobs["job-3"].queue_position == 2


class TestJobStoreComplete:
    """Tests for completing jobs."""

    def test_marks_job_complete(self):
        """complete() marks job as complete with result."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        store._jobs["job-1"].status = JobStatus.PROCESSING

        store.complete("job-1", result="Success!", deep_mode_used=True)

        job = store._jobs["job-1"]
        assert job.status == JobStatus.COMPLETE
        assert job.result == "Success!"
        assert job.deep_mode_used is True
        assert job.completed_at is not None

    def test_complete_unknown_job_does_nothing(self):
        """complete() handles unknown job gracefully."""
        store = JobStore()

        # Should not raise
        store.complete("unknown-job", result="Test")


class TestJobStoreFail:
    """Tests for failing jobs."""

    def test_marks_job_failed(self):
        """fail() marks job as failed with error."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        store._jobs["job-1"].status = JobStatus.PROCESSING

        store.fail("job-1", error="Something went wrong")

        job = store._jobs["job-1"]
        assert job.status == JobStatus.FAILED
        assert job.error == "Something went wrong"
        assert job.completed_at is not None

    def test_fail_unknown_job_does_nothing(self):
        """fail() handles unknown job gracefully."""
        store = JobStore()

        # Should not raise
        store.fail("unknown-job", error="Error")


class TestJobStoreSetChatUrl:
    """Tests for setting chat URL."""

    def test_sets_chat_url(self):
        """set_chat_url() updates job's chat URL."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")

        store.set_chat_url("job-1", "https://gemini.google.com/chat/123")

        assert store._jobs["job-1"].chat_url == "https://gemini.google.com/chat/123"

    def test_set_chat_url_unknown_job_does_nothing(self):
        """set_chat_url() handles unknown job gracefully."""
        store = JobStore()

        # Should not raise
        store.set_chat_url("unknown-job", "https://example.com")


class TestJobStoreGetStatus:
    """Tests for getting job status."""

    def test_returns_status_dict(self):
        """get_status() returns status dictionary."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")

        status = store.get_status("job-1")

        assert status["job_id"] == "job-1"
        assert status["status"] == "queued"
        assert status["queue_position"] == 1
        assert status["backend"] == "gemini"
        assert "created_at" in status

    def test_returns_none_for_unknown_job(self):
        """get_status() returns None for unknown job."""
        store = JobStore()

        status = store.get_status("unknown-job")

        assert status is None

    def test_queue_position_zero_when_not_queued(self):
        """get_status() returns queue_position=0 when not queued."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        store._jobs["job-1"].status = JobStatus.PROCESSING

        status = store.get_status("job-1")

        assert status["queue_position"] == 0


class TestJobStoreGetResult:
    """Tests for getting job result."""

    def test_returns_complete_result(self):
        """get_result() returns result for complete job."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test", thread_id="t-1")
        store._jobs["job-1"].status = JobStatus.COMPLETE
        store._jobs["job-1"].result = "The answer"
        store._jobs["job-1"].deep_mode_used = True

        result = store.get_result("job-1")

        assert result["job_id"] == "job-1"
        assert result["status"] == "complete"
        assert result["result"] == "The answer"
        assert result["deep_mode_used"] is True
        assert result["thread_id"] == "t-1"

    def test_returns_failed_result(self):
        """get_result() returns error for failed job."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        store._jobs["job-1"].status = JobStatus.FAILED
        store._jobs["job-1"].error = "Error message"

        result = store.get_result("job-1")

        assert result["status"] == "failed"
        assert result["error"] == "Error message"

    def test_returns_status_for_pending_job(self):
        """get_result() returns status for non-complete job."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")

        result = store.get_result("job-1")

        assert result["status"] == "queued"
        assert "result" not in result

    def test_returns_none_for_unknown_job(self):
        """get_result() returns None for unknown job."""
        store = JobStore()

        result = store.get_result("unknown-job")

        assert result is None


class TestJobStoreGetProcessingJob:
    """Tests for getting currently processing job."""

    def test_returns_processing_job(self):
        """get_processing_job() returns job being processed."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        store._jobs["job-1"].status = JobStatus.PROCESSING

        job = store.get_processing_job("gemini")

        assert job.job_id == "job-1"

    def test_returns_none_when_no_processing(self):
        """get_processing_job() returns None when nothing processing."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        # Still queued, not processing

        job = store.get_processing_job("gemini")

        assert job is None

    def test_returns_none_for_different_backend(self):
        """get_processing_job() only returns jobs for specified backend."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        store._jobs["job-1"].status = JobStatus.PROCESSING

        job = store.get_processing_job("chatgpt")

        assert job is None


class TestJobStoreQueueDepth:
    """Tests for queue depth tracking."""

    def test_returns_queue_depth(self):
        """get_queue_depth() returns count of queued jobs."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test 1")
        store.submit(job_id="job-2", backend="gemini", prompt="Test 2")
        store.submit(job_id="job-3", backend="gemini", prompt="Test 3")

        depth = store.get_queue_depth("gemini")

        assert depth == 3

    def test_excludes_processing_jobs(self):
        """get_queue_depth() excludes processing jobs."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test 1")
        store.submit(job_id="job-2", backend="gemini", prompt="Test 2")
        store._jobs["job-1"].status = JobStatus.PROCESSING

        depth = store.get_queue_depth("gemini")

        assert depth == 1

    def test_returns_zero_for_empty_queue(self):
        """get_queue_depth() returns 0 for empty queue."""
        store = JobStore()

        depth = store.get_queue_depth("gemini")

        assert depth == 0

    def test_get_all_queues(self):
        """get_all_queues() returns depths for all backends."""
        store = JobStore()
        store.submit(job_id="gem-1", backend="gemini", prompt="G1")
        store.submit(job_id="gem-2", backend="gemini", prompt="G2")
        store.submit(job_id="gpt-1", backend="chatgpt", prompt="C1")

        queues = store.get_all_queues()

        assert queues["gemini"] == 2
        assert queues["chatgpt"] == 1


class TestJobStoreCleanup:
    """Tests for job cleanup."""

    def test_removes_old_completed_jobs(self):
        """cleanup_old_jobs() removes old completed jobs."""
        store = JobStore()
        store.submit(job_id="old-job", backend="gemini", prompt="Old")
        store._jobs["old-job"].status = JobStatus.COMPLETE
        store._jobs["old-job"].completed_at = time.time() - 10000  # Very old

        store.cleanup_old_jobs(max_age=1000)

        assert "old-job" not in store._jobs

    def test_keeps_recent_completed_jobs(self):
        """cleanup_old_jobs() keeps recent completed jobs."""
        store = JobStore()
        store.submit(job_id="recent-job", backend="gemini", prompt="Recent")
        store._jobs["recent-job"].status = JobStatus.COMPLETE
        store._jobs["recent-job"].completed_at = time.time() - 100  # Recent

        store.cleanup_old_jobs(max_age=1000)

        assert "recent-job" in store._jobs

    def test_keeps_queued_jobs(self):
        """cleanup_old_jobs() keeps queued jobs regardless of age."""
        store = JobStore()
        store.submit(job_id="queued-job", backend="gemini", prompt="Queued")
        # Job is still queued (not complete)

        store.cleanup_old_jobs(max_age=0)  # Would remove everything if it could

        assert "queued-job" in store._jobs

    def test_removes_content_hash_on_cleanup(self):
        """cleanup_old_jobs() removes content hash from cache."""
        store = JobStore()
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        content_hash = store._jobs["job-1"].content_hash
        store._jobs["job-1"].status = JobStatus.COMPLETE
        store._jobs["job-1"].completed_at = time.time() - 10000

        store.cleanup_old_jobs(max_age=1000)

        assert content_hash not in store._content_cache


class TestJobStorePersistence:
    """Tests for disk persistence."""

    def test_persists_on_submit(self, tmp_path):
        """submit() persists to disk."""
        persist_file = tmp_path / "jobs.json"
        store = JobStore(persist_path=persist_file)

        store.submit(job_id="job-1", backend="gemini", prompt="Test")

        assert persist_file.exists()
        data = json.loads(persist_file.read_text())
        assert "job-1" in data["jobs"]

    def test_persists_on_complete(self, tmp_path):
        """complete() persists to disk."""
        persist_file = tmp_path / "jobs.json"
        store = JobStore(persist_path=persist_file)
        store.submit(job_id="job-1", backend="gemini", prompt="Test")

        store.complete("job-1", result="Done")

        data = json.loads(persist_file.read_text())
        assert data["jobs"]["job-1"]["status"] == "complete"

    def test_persists_on_fail(self, tmp_path):
        """fail() persists to disk."""
        persist_file = tmp_path / "jobs.json"
        store = JobStore(persist_path=persist_file)
        store.submit(job_id="job-1", backend="gemini", prompt="Test")

        store.fail("job-1", error="Failed")

        data = json.loads(persist_file.read_text())
        assert data["jobs"]["job-1"]["status"] == "failed"

    def test_survives_restart(self, tmp_path):
        """Jobs survive store restart."""
        persist_file = tmp_path / "jobs.json"

        # First store instance
        store1 = JobStore(persist_path=persist_file)
        store1.submit(job_id="job-1", backend="gemini", prompt="Test")
        store1._jobs["job-1"].status = JobStatus.PROCESSING
        store1._persist()

        # Second store instance (simulating restart)
        store2 = JobStore(persist_path=persist_file)

        assert "job-1" in store2._jobs
        assert store2._jobs["job-1"].status == JobStatus.PROCESSING


class TestJobStoreCacheTTL:
    """Tests for content cache TTL."""

    def test_cache_expires_after_ttl(self):
        """Content cache entries expire after TTL."""
        store = JobStore(cache_ttl=1)  # 1 second TTL
        store.submit(job_id="job-1", backend="gemini", prompt="Test")
        store._jobs["job-1"].status = JobStatus.COMPLETE
        store._jobs["job-1"].result = "Cached"

        # Wait for cache to expire
        time.sleep(1.5)

        # New submission with same content should be queued, not cached
        result = store.submit(job_id="job-2", backend="gemini", prompt="Test")

        assert result["status"] == "queued"


class TestJobStoreThreadSafety:
    """Tests for thread safety."""

    def test_has_lock(self):
        """JobStore has a lock for thread safety."""
        store = JobStore()
        assert hasattr(store, "_lock")
        assert store._lock is not None
