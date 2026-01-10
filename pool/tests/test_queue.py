"""Tests for pool request queue management."""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.queue import QueuedRequest, RequestQueue, QueueManager, QueueFullError
from src.models import SendRequest, SendOptions, Backend, Priority


@pytest.fixture
def sample_request():
    """Create a sample SendRequest."""
    return SendRequest(
        backend=Backend.GEMINI,
        prompt="Test prompt",
        options=SendOptions(priority=Priority.NORMAL),
    )


@pytest.fixture
def high_priority_request():
    """Create a high priority SendRequest."""
    return SendRequest(
        backend=Backend.GEMINI,
        prompt="Urgent prompt",
        options=SendOptions(priority=Priority.HIGH),
    )


@pytest.fixture
def low_priority_request():
    """Create a low priority SendRequest."""
    return SendRequest(
        backend=Backend.GEMINI,
        prompt="Background prompt",
        options=SendOptions(priority=Priority.LOW),
    )


class TestQueuedRequest:
    """Tests for QueuedRequest dataclass."""

    def test_create_with_normal_priority(self, sample_request):
        """QueuedRequest.create sets correct priority value."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        queued = QueuedRequest.create(sample_request, future, "test-1")

        assert queued.priority_value == 1  # NORMAL
        assert queued.request == sample_request
        assert queued.request_id == "test-1"
        assert queued.future == future

        loop.close()

    def test_create_with_high_priority(self, high_priority_request):
        """High priority gets priority_value 0."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        queued = QueuedRequest.create(high_priority_request, future, "test-2")

        assert queued.priority_value == 0  # HIGH

        loop.close()

    def test_create_with_low_priority(self, low_priority_request):
        """Low priority gets priority_value 2."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()

        queued = QueuedRequest.create(low_priority_request, future, "test-3")

        assert queued.priority_value == 2  # LOW

        loop.close()

    def test_ordering_by_priority(self, sample_request, high_priority_request):
        """QueuedRequest orders by priority value."""
        loop = asyncio.new_event_loop()
        future1 = loop.create_future()
        future2 = loop.create_future()

        normal = QueuedRequest.create(sample_request, future1, "normal")
        high = QueuedRequest.create(high_priority_request, future2, "high")

        # High priority (0) should be less than normal (1)
        assert high < normal

        loop.close()


class TestRequestQueue:
    """Tests for RequestQueue."""

    @pytest.mark.asyncio
    async def test_enqueue_returns_future(self, sample_request):
        """enqueue returns a Future that can be resolved."""
        queue = RequestQueue("gemini", max_depth=10)

        future = await queue.enqueue(sample_request)

        assert isinstance(future, asyncio.Future)
        assert queue.depth == 1

    @pytest.mark.asyncio
    async def test_enqueue_increments_depth(self, sample_request):
        """Each enqueue increments the queue depth."""
        queue = RequestQueue("gemini", max_depth=10)

        await queue.enqueue(sample_request)
        await queue.enqueue(sample_request)
        await queue.enqueue(sample_request)

        assert queue.depth == 3

    @pytest.mark.asyncio
    async def test_enqueue_raises_when_full(self, sample_request):
        """enqueue raises QueueFullError when at max depth."""
        queue = RequestQueue("gemini", max_depth=2)

        await queue.enqueue(sample_request)
        await queue.enqueue(sample_request)

        with pytest.raises(QueueFullError) as exc_info:
            await queue.enqueue(sample_request)

        assert "gemini" in str(exc_info.value)
        assert "full" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_dequeue_returns_highest_priority(
        self, sample_request, high_priority_request, low_priority_request
    ):
        """dequeue returns highest priority first."""
        queue = RequestQueue("gemini", max_depth=10)

        # Enqueue in order: normal, low, high
        await queue.enqueue(sample_request)
        await queue.enqueue(low_priority_request)
        await queue.enqueue(high_priority_request)

        # Should dequeue: high, normal, low
        first = await queue.dequeue()
        assert first.request.options.priority == Priority.HIGH

        second = await queue.dequeue()
        assert second.request.options.priority == Priority.NORMAL

        third = await queue.dequeue()
        assert third.request.options.priority == Priority.LOW

    @pytest.mark.asyncio
    async def test_dequeue_fifo_within_same_priority(self, sample_request):
        """dequeue uses FIFO for same priority requests."""
        queue = RequestQueue("gemini", max_depth=10)

        # Create requests with different prompts to distinguish them
        req1 = SendRequest(
            backend=Backend.GEMINI,
            prompt="First",
            options=SendOptions(priority=Priority.NORMAL),
        )
        req2 = SendRequest(
            backend=Backend.GEMINI,
            prompt="Second",
            options=SendOptions(priority=Priority.NORMAL),
        )
        req3 = SendRequest(
            backend=Backend.GEMINI,
            prompt="Third",
            options=SendOptions(priority=Priority.NORMAL),
        )

        await queue.enqueue(req1)
        await asyncio.sleep(0.001)  # Ensure different timestamps
        await queue.enqueue(req2)
        await asyncio.sleep(0.001)
        await queue.enqueue(req3)

        first = await queue.dequeue()
        second = await queue.dequeue()
        third = await queue.dequeue()

        assert first.request.prompt == "First"
        assert second.request.prompt == "Second"
        assert third.request.prompt == "Third"

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_when_empty(self):
        """dequeue returns None when queue is empty."""
        queue = RequestQueue("gemini", max_depth=10)

        result = await queue.dequeue()

        assert result is None

    @pytest.mark.asyncio
    async def test_is_empty_property(self, sample_request):
        """is_empty reflects queue state."""
        queue = RequestQueue("gemini", max_depth=10)

        assert queue.is_empty is True

        await queue.enqueue(sample_request)
        assert queue.is_empty is False

        await queue.dequeue()
        assert queue.is_empty is True

    @pytest.mark.asyncio
    async def test_request_id_is_unique(self, sample_request):
        """Each request gets a unique ID."""
        queue = RequestQueue("gemini", max_depth=10)

        await queue.enqueue(sample_request)
        await queue.enqueue(sample_request)
        await queue.enqueue(sample_request)

        req1 = await queue.dequeue()
        req2 = await queue.dequeue()
        req3 = await queue.dequeue()

        ids = {req1.request_id, req2.request_id, req3.request_id}
        assert len(ids) == 3  # All unique

    @pytest.mark.asyncio
    async def test_request_id_includes_backend_name(self, sample_request):
        """Request IDs include the backend name."""
        queue = RequestQueue("gemini", max_depth=10)

        await queue.enqueue(sample_request)
        req = await queue.dequeue()

        assert req.request_id.startswith("gemini-")


class TestQueueManager:
    """Tests for QueueManager."""

    def test_creates_queues_for_all_backends(self, sample_config):
        """QueueManager creates queues for all three backends."""
        manager = QueueManager(sample_config)

        assert manager.get_queue("gemini") is not None
        assert manager.get_queue("chatgpt") is not None
        assert manager.get_queue("claude") is not None

    def test_uses_config_max_depth(self):
        """QueueManager uses max_depth from config."""
        config = {"queue": {"max_depth_per_backend": 5}}
        manager = QueueManager(config)

        queue = manager.get_queue("gemini")
        assert queue.max_depth == 5

    def test_uses_default_max_depth(self):
        """QueueManager uses default max_depth of 10."""
        manager = QueueManager({})

        queue = manager.get_queue("gemini")
        assert queue.max_depth == 10

    def test_get_queue_returns_none_for_unknown(self, sample_config):
        """get_queue returns None for unknown backend."""
        manager = QueueManager(sample_config)

        assert manager.get_queue("unknown") is None

    @pytest.mark.asyncio
    async def test_get_depths_returns_all_depths(self, sample_config):
        """get_depths returns queue depths for all backends."""
        manager = QueueManager(sample_config)

        # Enqueue some requests
        gemini_queue = manager.get_queue("gemini")
        request = SendRequest(backend=Backend.GEMINI, prompt="test")
        await gemini_queue.enqueue(request)
        await gemini_queue.enqueue(request)

        chatgpt_queue = manager.get_queue("chatgpt")
        chatgpt_request = SendRequest(backend=Backend.CHATGPT, prompt="test")
        await chatgpt_queue.enqueue(chatgpt_request)

        depths = manager.get_depths()

        assert depths["gemini"] == 2
        assert depths["chatgpt"] == 1
        assert depths["claude"] == 0


class TestQueueFullError:
    """Tests for QueueFullError exception."""

    def test_is_exception(self):
        """QueueFullError is an Exception."""
        error = QueueFullError("Queue full")
        assert isinstance(error, Exception)

    def test_message(self):
        """QueueFullError stores message."""
        error = QueueFullError("gemini queue is full (10 items)")
        assert "gemini" in str(error)
        assert "10" in str(error)


class TestConcurrentAccess:
    """Tests for concurrent queue access."""

    @pytest.mark.asyncio
    async def test_concurrent_enqueue(self, sample_request):
        """Queue handles concurrent enqueue safely."""
        queue = RequestQueue("gemini", max_depth=100)

        async def enqueue_task():
            await queue.enqueue(sample_request)

        # Enqueue 50 items concurrently
        await asyncio.gather(*[enqueue_task() for _ in range(50)])

        assert queue.depth == 50

    @pytest.mark.asyncio
    async def test_concurrent_dequeue(self, sample_request):
        """Queue handles concurrent dequeue safely."""
        queue = RequestQueue("gemini", max_depth=100)

        # Pre-fill queue
        for _ in range(50):
            await queue.enqueue(sample_request)

        dequeued = []

        async def dequeue_task():
            result = await queue.dequeue()
            if result:
                dequeued.append(result)

        # Dequeue all concurrently
        await asyncio.gather(*[dequeue_task() for _ in range(60)])

        assert len(dequeued) == 50
        assert queue.is_empty

    @pytest.mark.asyncio
    async def test_future_can_be_resolved(self, sample_request):
        """Future returned by enqueue can be resolved."""
        queue = RequestQueue("gemini", max_depth=10)

        future = await queue.enqueue(sample_request)

        # Simulate worker setting result
        future.set_result("Response data")

        result = await future
        assert result == "Response data"
