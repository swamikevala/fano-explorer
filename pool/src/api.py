"""FastAPI HTTP API for the Browser Pool Service."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from shared.logging import get_logger

from .models import (
    SendRequest, SendResponse, Backend, Priority,
    BackendStatus, PoolStatus, HealthResponse, AuthResponse,
    JobSubmitRequest,
)
from .state import StateManager
from .queue import QueueManager, QueueFullError
from .workers import GeminiWorker, ChatGPTWorker, ClaudeWorker
from .jobs import JobStore, JobStatus

log = get_logger("pool", "api")


class BrowserPool:
    """
    The main Browser Pool service.

    Manages workers, queues, and state for all backends.
    """

    def __init__(self, config: dict):
        self.config = config
        self.start_time = time.time()

        # Initialize state manager
        state_file = Path(__file__).parent.parent / "pool_state.json"
        self.state = StateManager(state_file, config)

        # Initialize job store (async jobs)
        jobs_file = Path(__file__).parent.parent / "jobs_state.json"
        self.jobs = JobStore(persist_path=jobs_file)

        # Initialize queue manager (legacy sync mode)
        self.queues = QueueManager(config)

        # Initialize workers (but don't start yet)
        self.workers = {}
        backends_config = config.get("backends", {})

        if backends_config.get("gemini", {}).get("enabled", True):
            self.workers["gemini"] = GeminiWorker(
                config, self.state, self.queues.get_queue("gemini"), self.jobs
            )

        if backends_config.get("chatgpt", {}).get("enabled", True):
            self.workers["chatgpt"] = ChatGPTWorker(
                config, self.state, self.queues.get_queue("chatgpt"), self.jobs
            )

        if backends_config.get("claude", {}).get("enabled", True):
            self.workers["claude"] = ClaudeWorker(
                config, self.state, self.queues.get_queue("claude"), self.jobs
            )

        # Watchdog task for stuck detection
        self._watchdog_task: Optional[asyncio.Task] = None
        watchdog_config = config.get("watchdog", {})
        self._watchdog_enabled = watchdog_config.get("enabled", True)
        self._watchdog_interval = watchdog_config.get("check_interval_seconds", 60)
        self._backends_config = backends_config

    async def startup(self):
        """Start all workers and connect to backends."""
        log.info("pool.service.lifecycle", action="starting", backends=list(self.workers.keys()))

        # Restore any pending queue items from before restart
        restored = self.queues.restore_pending()
        if restored:
            log.info("pool.service.queue_restored", restored=restored)

        for name, worker in self.workers.items():
            try:
                await worker.connect()
            except Exception as e:
                log.warning("pool.backend.connect_failed", backend=name, error=str(e), will_retry=True)
            # Start worker loop regardless - it will wait for availability
            try:
                await worker.start()
            except Exception as e:
                log.error("pool.backend.worker_start_failed", backend=name, error=str(e))

        # Start watchdog for stuck detection
        if self._watchdog_enabled:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            log.info("pool.watchdog.started",
                     check_interval=self._watchdog_interval)

        log.info("pool.service.lifecycle", action="started", backends=list(self.workers.keys()))

    async def shutdown(self):
        """Stop all workers and disconnect."""
        log.info("pool.service.lifecycle", action="stopping")

        # Stop watchdog
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            log.info("pool.watchdog.stopped")

        for name, worker in self.workers.items():
            try:
                await worker.stop()
                await worker.disconnect()
            except Exception as e:
                log.error("pool.backend.stop_failed", backend=name, error=str(e))

        log.info("pool.service.lifecycle", action="stopped")

    async def _watchdog_loop(self):
        """
        Background task that monitors workers for stuck requests.

        Periodically checks each worker's elapsed time and auto-kicks
        workers that exceed their backend-specific timeout threshold.
        """
        while True:
            try:
                await asyncio.sleep(self._watchdog_interval)

                for name, worker in self.workers.items():
                    if worker._current_start_time is None:
                        continue  # Worker is idle

                    # Get per-backend timeout from config
                    backend_config = self._backends_config.get(name, {})
                    timeout = backend_config.get("response_timeout_seconds", 3600)

                    elapsed = time.time() - worker._current_start_time
                    if elapsed > timeout:
                        log.warning("pool.watchdog.stuck_detected",
                                    backend=name,
                                    elapsed_seconds=round(elapsed),
                                    timeout_threshold=timeout,
                                    request_id=worker._current_request_id)

                        # Auto-kick the stuck worker
                        result = await self._kick_worker(name)
                        log.info("pool.watchdog.auto_kicked",
                                 backend=name,
                                 result=result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(e, "pool.watchdog.error", {})
                await asyncio.sleep(10)  # Back off on error

    async def _kick_worker(self, backend: str) -> dict:
        """
        Internal method to kick a stuck worker.

        Attempts to recover the response first before failing the job.
        This preserves work if the LLM was genuinely still thinking.

        Args:
            backend: The backend name to kick

        Returns:
            Dict with kick result details
        """
        if backend not in self.workers:
            return {"error": f"Unknown backend: {backend}"}

        worker = self.workers[backend]
        result = {"backend": backend, "actions": []}

        try:
            # Save job info before clearing anything
            current_job = worker._current_job
            chat_url = None
            if current_job:
                chat_url = current_job.chat_url
            elif worker.browser and hasattr(worker.browser, 'page') and worker.browser.page:
                chat_url = worker.browser.page.url

            # Take screenshot for debugging
            if worker.browser and hasattr(worker.browser, 'page') and worker.browser.page:
                try:
                    screenshot_path = Path(__file__).parent.parent / "debug" / f"kick_{backend}_{int(time.time())}.png"
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    await worker.browser.page.screenshot(path=str(screenshot_path))
                    result["actions"].append(f"screenshot_saved:{screenshot_path}")
                    log.info("pool.kick.screenshot", backend=backend, path=str(screenshot_path))
                except Exception as e:
                    result["actions"].append(f"screenshot_failed:{e}")

            # Clear current work tracking (but don't fail job yet)
            worker._current_job = None
            worker._current_request_id = None
            worker._current_prompt = None
            worker._current_start_time = None
            result["actions"].append("work_tracking_cleared")

            # Clear state manager active work
            self.state.clear_active_work(backend)
            result["actions"].append("state_cleared")

            # Try to reconnect browser
            try:
                reconnected = await worker.try_reconnect()
                if reconnected:
                    result["actions"].append("browser_reconnected")
                    log.info("pool.kick.reconnected", backend=backend)
                else:
                    result["actions"].append("browser_reconnect_failed")
                    log.warning("pool.kick.reconnect_failed", backend=backend)
            except Exception as e:
                result["actions"].append(f"reconnect_error:{e}")
                reconnected = False

            # Attempt quick recovery - check if response exists (don't wait for generation)
            if current_job and chat_url and reconnected and worker.browser:
                try:
                    log.info("pool.kick.recovery_attempt", backend=backend,
                             job_id=current_job.job_id, chat_url=chat_url)
                    result["actions"].append(f"recovery_attempt:{current_job.job_id}")

                    # Navigate back to the chat
                    await worker.browser.page.goto(chat_url)
                    await worker.browser.page.wait_for_load_state("networkidle")
                    await asyncio.sleep(2)

                    # Quick check - try to extract existing response (no waiting)
                    response_text = None
                    if hasattr(worker.browser, 'try_get_response'):
                        response_text = await worker.browser.try_get_response()

                    if response_text:
                        # Found a response - complete the job
                        worker.jobs.complete(
                            job_id=current_job.job_id,
                            result=response_text,
                            deep_mode_used=current_job.deep_mode,
                        )
                        result["actions"].append(f"job_recovered:{current_job.job_id}")
                        result["recovered"] = True
                        log.info("pool.kick.recovery_success", backend=backend,
                                 job_id=current_job.job_id, response_length=len(response_text))
                    else:
                        # No response - timeout exceeded, fail the job
                        worker.jobs.fail(current_job.job_id, "Timeout exceeded - no response found")
                        result["actions"].append(f"job_failed:{current_job.job_id}")
                        result["recovered"] = False
                        log.warning("pool.kick.timeout_no_response", backend=backend,
                                    job_id=current_job.job_id)

                except Exception as e:
                    log.error("pool.kick.recovery_error", backend=backend, error=str(e))
                    result["actions"].append(f"recovery_error:{e}")
                    if current_job and worker.jobs:
                        worker.jobs.fail(current_job.job_id, f"Timeout exceeded - recovery failed: {e}")
                        result["actions"].append(f"job_failed:{current_job.job_id}")
                    result["recovered"] = False

            elif current_job and worker.jobs:
                # No chat URL or couldn't reconnect - just fail the job
                worker.jobs.fail(current_job.job_id, "Kicked by operator - browser stuck")
                result["actions"].append(f"job_failed:{current_job.job_id}")
                result["recovered"] = False

            result["success"] = True

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            log.exception(e, "pool.kick.error", {"backend": backend})

        return result

    async def send(self, request: SendRequest) -> SendResponse:
        """Send a prompt to a backend and wait for response."""
        backend = request.backend.value

        # Check if backend exists
        if backend not in self.workers:
            return SendResponse(
                success=False,
                error="unavailable",
                message=f"Backend '{backend}' not enabled",
            )

        # Check if backend is available
        if not self.state.is_available(backend):
            state = self.state.get_backend_state(backend)
            if state.get("rate_limited"):
                return SendResponse(
                    success=False,
                    error="rate_limited",
                    message=f"{backend} is rate limited",
                    retry_after_seconds=3600,
                )
            else:
                return SendResponse(
                    success=False,
                    error="auth_required",
                    message=f"{backend} requires authentication",
                )

        # Enqueue the request
        try:
            queue = self.queues.get_queue(backend)
            future = await queue.enqueue(request)
        except QueueFullError as e:
            return SendResponse(
                success=False,
                error="queue_full",
                message=str(e),
            )

        # Wait for response with timeout
        timeout = request.options.timeout_seconds
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return SendResponse(
                success=False,
                error="timeout",
                message=f"Request timed out after {timeout} seconds",
            )

    async def check_all_health(self, auto_recover: bool = True) -> dict[str, tuple[bool, str]]:
        """
        Run health checks on all browser workers.

        Args:
            auto_recover: If True, attempt to reconnect crashed browsers.

        Returns dict of backend -> (is_healthy, reason)
        """
        results = {}
        for name, worker in self.workers.items():
            try:
                is_healthy, reason = await worker.check_health()

                # If unhealthy and auto_recover enabled, try to reconnect
                if not is_healthy and auto_recover and name in ("gemini", "chatgpt"):
                    log.info("pool.health.auto_recover", backend=name, reason=reason)
                    if await worker.try_reconnect():
                        # Re-check health after reconnect
                        is_healthy, reason = await worker.check_health()

                results[name] = (is_healthy, reason)
            except Exception as e:
                results[name] = (False, f"health_check_error:{type(e).__name__}")
        return results

    async def get_status_async(self, run_health_check: bool = True) -> PoolStatus:
        """Get status of all backends with optional health check."""
        backends_config = self.config.get("backends", {})
        depths = self.queues.get_depths()

        # Run health checks if requested
        health_results = {}
        if run_health_check:
            health_results = await self.check_all_health()

        status = PoolStatus()

        if "gemini" in self.workers:
            state = self.state.get_backend_state("gemini")
            gemini_config = backends_config.get("gemini", {})

            # Use health check result if available, otherwise fall back to state
            if "gemini" in health_results:
                is_healthy, health_reason = health_results["gemini"]
                available = is_healthy and not state.get("rate_limited", False)
            else:
                available = self.state.is_available("gemini")

            status.gemini = BackendStatus(
                available=available,
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("gemini", 0),
                deep_mode_uses_today=state.get("deep_mode_uses_today", 0),
                deep_mode_limit=gemini_config.get("deep_mode", {}).get("daily_limit", 20),
            )

        if "chatgpt" in self.workers:
            state = self.state.get_backend_state("chatgpt")
            chatgpt_config = backends_config.get("chatgpt", {})

            # Use health check result if available
            if "chatgpt" in health_results:
                is_healthy, health_reason = health_results["chatgpt"]
                available = is_healthy and not state.get("rate_limited", False)
            else:
                available = self.state.is_available("chatgpt")

            status.chatgpt = BackendStatus(
                available=available,
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("chatgpt", 0),
                pro_mode_uses_today=state.get("pro_mode_uses_today", 0),
                pro_mode_limit=chatgpt_config.get("pro_mode", {}).get("daily_limit", 100),
            )

        if "claude" in self.workers:
            state = self.state.get_backend_state("claude")
            status.claude = BackendStatus(
                available=self.state.is_available("claude"),
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("claude", 0),
            )

        return status

    def get_status(self) -> PoolStatus:
        """Get status of all backends (sync version, no health check)."""
        backends_config = self.config.get("backends", {})
        depths = self.queues.get_depths()

        status = PoolStatus()

        if "gemini" in self.workers:
            state = self.state.get_backend_state("gemini")
            gemini_config = backends_config.get("gemini", {})
            status.gemini = BackendStatus(
                available=self.state.is_available("gemini"),
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("gemini", 0),
                deep_mode_uses_today=state.get("deep_mode_uses_today", 0),
                deep_mode_limit=gemini_config.get("deep_mode", {}).get("daily_limit", 20),
            )

        if "chatgpt" in self.workers:
            state = self.state.get_backend_state("chatgpt")
            chatgpt_config = backends_config.get("chatgpt", {})
            status.chatgpt = BackendStatus(
                available=self.state.is_available("chatgpt"),
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("chatgpt", 0),
                pro_mode_uses_today=state.get("pro_mode_uses_today", 0),
                pro_mode_limit=chatgpt_config.get("pro_mode", {}).get("daily_limit", 100),
            )

        if "claude" in self.workers:
            state = self.state.get_backend_state("claude")
            status.claude = BackendStatus(
                available=self.state.is_available("claude"),
                authenticated=state.get("authenticated", False),
                rate_limited=state.get("rate_limited", False),
                rate_limit_resets_at=state.get("rate_limit_resets_at"),
                queue_depth=depths.get("claude", 0),
            )

        return status

    async def authenticate(self, backend: str) -> bool:
        """Trigger interactive authentication for a backend."""
        if backend not in self.workers:
            return False

        return await self.workers[backend].authenticate()


def create_app(config: dict) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="Browser Pool Service",
        description="Shared LLM access layer for Fano platform",
        version="0.1.0",
    )

    pool = BrowserPool(config)

    @app.on_event("startup")
    async def startup():
        await pool.startup()

    @app.on_event("shutdown")
    async def shutdown():
        await pool.shutdown()

    @app.post("/send", response_model=SendResponse)
    async def send(request: SendRequest):
        """Send a prompt to an LLM and wait for response (legacy sync mode)."""
        return await pool.send(request)

    # ==================== ASYNC JOB ENDPOINTS ====================

    @app.post("/job/submit")
    async def job_submit(request: JobSubmitRequest):
        """
        Submit a job for async processing.

        Returns immediately with job status. Poll /job/{job_id}/status for completion.

        Returns:
            {status: "queued" | "exists" | "cached", job_id: str, cached_job_id?: str}
        """
        # Validate backend
        if request.backend not in pool.workers:
            raise HTTPException(
                status_code=400,
                detail=f"Backend '{request.backend}' not enabled"
            )

        # Check if backend is available
        if not pool.state.is_available(request.backend):
            state = pool.state.get_backend_state(request.backend)
            if state.get("rate_limited"):
                raise HTTPException(
                    status_code=503,
                    detail=f"{request.backend} is rate limited"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"{request.backend} requires authentication"
                )

        # Submit to job store
        result = pool.jobs.submit(
            job_id=request.job_id,
            backend=request.backend,
            prompt=request.prompt,
            thread_id=request.thread_id,
            deep_mode=request.deep_mode,
            new_chat=request.new_chat,
            priority=request.priority,
            images=[img.model_dump() for img in request.images] if request.images else [],
        )

        return result

    @app.get("/job/{job_id}/status")
    async def job_status(job_id: str):
        """
        Get the status of a job.

        Returns:
            {job_id, status, queue_position, backend, created_at, started_at, completed_at}
        """
        status = pool.jobs.get_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return status

    @app.get("/job/{job_id}/result")
    async def job_result(job_id: str):
        """
        Get the result of a completed job.

        Returns:
            {job_id, status, result?, error?, deep_mode_used, backend, thread_id}
        """
        result = pool.jobs.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return result

    @app.get("/jobs/queue")
    async def jobs_queue():
        """Get queue depths for all backends."""
        return {
            "queues": pool.jobs.get_all_queues(),
            "total": sum(pool.jobs.get_all_queues().values()),
        }

    # ==================== END ASYNC JOB ENDPOINTS ====================

    @app.get("/status", response_model=PoolStatus)
    async def status(health_check: bool = True):
        """
        Get status of all backends.

        Args:
            health_check: If True (default), runs actual browser health checks.
                         Set to False for faster response without health verification.
        """
        return await pool.get_status_async(run_health_check=health_check)

    @app.post("/auth/{backend}", response_model=AuthResponse)
    async def auth(backend: str):
        """Trigger interactive authentication for a backend."""
        if backend not in ["gemini", "chatgpt", "claude"]:
            raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")

        success = await pool.authenticate(backend)
        if success:
            return AuthResponse(
                success=True,
                message=f"Authentication window opened for {backend}. Please log in manually.",
            )
        else:
            return AuthResponse(
                success=False,
                message=f"Failed to start authentication for {backend}",
            )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            uptime_seconds=time.time() - pool.start_time,
            version="0.1.0",
        )

    @app.get("/activity")
    async def activity(include_history: bool = False, history_limit: int = 5):
        """
        Get current activity across all backends.

        Args:
            include_history: If True, include recent request history
            history_limit: Number of history entries per backend

        Returns what each LLM is currently working on.
        """
        result = {}
        for name, worker in pool.workers.items():
            work = worker.get_current_work()
            entry = {
                "status": "working" if work else "idle",
                "current_work": work,
            }
            if include_history:
                entry["history"] = worker.get_request_history(history_limit)
            result[name] = entry
        return result

    @app.get("/activity/detail")
    async def activity_detail():
        """
        Get detailed activity information for all backends.

        Includes full prompts, history, and queue depths.
        """
        depths = pool.queues.get_depths()
        result = {
            "backends": {},
            "queue_depths": depths,
            "uptime_seconds": time.time() - pool.start_time,
        }

        for name, worker in pool.workers.items():
            work = worker.get_current_work()
            history = worker.get_request_history(10)

            result["backends"][name] = {
                "status": "working" if work else "idle",
                "current_work": work,
                "history": history,
                "queue_depth": depths.get(name, 0),
            }

        return result

    @app.get("/recovery/status")
    async def recovery_status():
        """
        Get detailed recovery status for monitoring.

        Returns information about:
        - Active work per backend (with age and thread_id)
        - Pending queue items per backend
        - Pending async jobs per backend
        """
        from .state import MAX_ACTIVE_WORK_AGE_SECONDS

        result = {
            "backends": {},
            "pending_queue": pool.queues.get_depths(),
            "pending_jobs": pool.jobs.get_all_queues(),
            "max_active_work_age_seconds": MAX_ACTIVE_WORK_AGE_SECONDS,
        }

        for backend in ["gemini", "chatgpt", "claude"]:
            # Get active work without staleness check to see all work
            active = pool.state.get_active_work(backend, check_staleness=False)
            processing_job = pool.jobs.get_processing_job(backend)

            backend_info = {"has_active_work": False}

            if processing_job:
                backend_info = {
                    "has_active_work": True,
                    "job_id": processing_job.job_id,
                    "thread_id": processing_job.thread_id,
                    "chat_url": processing_job.chat_url,
                    "deep_mode": processing_job.deep_mode,
                    "is_async_job": True,
                }
            elif active:
                started_at = active.get("started_at", 0)
                age_seconds = time.time() - started_at
                backend_info = {
                    "has_active_work": True,
                    "request_id": active.get("request_id"),
                    "thread_id": active.get("thread_id"),
                    "chat_url": active.get("chat_url"),
                    "age_seconds": round(age_seconds),
                    "is_stale": age_seconds > MAX_ACTIVE_WORK_AGE_SECONDS,
                    "deep_mode": active.get("options", {}).get("deep_mode", False),
                    "is_async_job": False,
                }

            result["backends"][backend] = backend_info

        return result

    @app.post("/kick/{backend}")
    async def kick_worker(backend: str):
        """
        Force-reset a stuck worker.

        This will:
        1. Take a screenshot for debugging
        2. Fail any current job
        3. Clear active work state
        4. Attempt to reconnect the browser
        """
        result = await pool._kick_worker(backend)
        log.info("pool.kick.complete", backend=backend, actions=result.get("actions", []))
        return result

    @app.post("/shutdown")
    async def shutdown_endpoint():
        """
        Gracefully shutdown the pool service.

        This allows external controllers (like the control panel) to stop
        the pool even if they didn't start it.
        """
        import os
        import signal

        log.info("pool.service.lifecycle", action="shutdown_requested")

        # Schedule shutdown after response is sent
        async def delayed_shutdown():
            await asyncio.sleep(0.5)  # Give time for response to be sent
            await pool.shutdown()
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.create_task(delayed_shutdown())
        return {"status": "shutting_down", "message": "Pool will shutdown shortly"}

    # ==================== RECOVERY ENDPOINTS ====================

    @app.get("/recovered")
    async def get_recovered_responses():
        """
        Get all completed jobs awaiting pickup.

        Used by orchestrator on startup to recover responses from
        jobs that completed while it was disconnected.

        Returns:
            {
                "responses": [
                    {
                        "job_id": str,
                        "request_id": str,
                        "backend": str,
                        "thread_id": str,
                        "result": str,
                        "deep_mode_used": bool,
                        "completed_at": float
                    },
                    ...
                ]
            }
        """
        completed = pool.jobs.get_completed_jobs()
        log.info("pool.recovery.list_requested", count=len(completed))
        return {"responses": completed}

    @app.delete("/recovered/{job_id}")
    async def clear_recovered_response(job_id: str):
        """
        Remove a recovered response after it's been processed.

        Called by orchestrator after successfully applying a recovered response.

        Returns:
            {"success": bool, "message": str}
        """
        removed = pool.jobs.remove_job(job_id)
        if removed:
            log.info("pool.recovery.cleared", job_id=job_id)
            return {"success": True, "message": f"Removed job {job_id}"}
        else:
            log.warning("pool.recovery.not_found", job_id=job_id)
            return {"success": False, "message": f"Job {job_id} not found"}

    return app


def load_config() -> dict:
    """Load pool configuration."""
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {}


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    server_config = config.get("server", {})
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 9000)

    print(f"\n  Browser Pool Service")
    print(f"  =====================")
    print(f"  Running on http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")
