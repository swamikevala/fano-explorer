---

# Fano Orchestration Layer: Unified Design Specification (v3.0)

**Version:** 3.0 (Final)
**Date:** 2026-01-14
**Status:** Approved for Implementation
**Architecture:** Polling Orchestrator + JIT Pool Gateway + Delta WAL
**Constraint:** Single-Process Asyncio with Legacy Adapters

---

## 1. Executive Summary

### 1.1 The Problem

The current architecture suffers from "Split-Brain" scheduling. The **Pool Service** autonomously pulls work from legacy queues, while **Modules** (Explorer, Documenter) attempt to manage their own flows. This causes:

1. **Priority Inversion:** High-stakes tasks wait behind low-priority batch jobs in the Pool.
2. **Event Loop Blocking:** Legacy CPU-heavy operations starve network heartbeats, causing false failure detections.
3. **Double Billing:** Crashes leave orphaned tasks running in the Pool, which are re-submitted on restart.

### 1.2 The Solution

We introduce a unified **Orchestrator** that holds all tasks in a single priority queue and leases LLM backends on a **Just-In-Time (JIT)** basis.

* **JIT Submission:** The Orchestrator holds tasks in memory and submits to the Pool *only* when a backend is strictly free.
* **Priority Locking:** The Pool API is upgraded to ensure JIT requests preempt legacy background workers.
* **Crash Reconciliation:** On startup, the Orchestrator "adopts" active Pool requests, preventing zombie processes.

---

## 2. Architecture Overview

```mermaid
graph TD
    subgraph "Orchestrator Process (Asyncio)"
        SCH[Scheduler<br/>Priority Queue]
        STATE[State Manager<br/>WAL + Checkpoint]
        
        subgraph "Legacy Adapters"
            EXP[Explorer Adapter]
            DOC[Documenter Adapter]
        end
        
        WORK[JIT Worker Loops<br/>(One per Backend)]
    end
    
    subgraph "Pool Service (Legacy)"
        API[API Gateway]
        LOCK[Backend Locks]
        REG[Active Request Registry]
        BG[Legacy Workers<br/>(Low Priority)]
    end

    EXP -->|Poll Tasks<br/>(Thread Offload)| SCH
    DOC -->|Poll Tasks<br/>(Thread Offload)| SCH
    SCH -->|Next High Prio| WORK
    WORK -->|Submit Immediate<br/>+ Idempotency Token| API
    
    API -->|Acquire Lock| LOCK
    BG -.->|Try Acquire| LOCK
    
    LOCK -->|Execute| REG

```

### 2.1 Concurrency Model

* **Single Process Asyncio:** The Orchestrator runs on a single event loop.
* **Legacy Adapters:** Existing blocking code (in `explorer/src` and `documenter/main.py`) **must** be wrapped in `run_in_executor` to prevent the Orchestrator from freezing.
```python
# Example Adapter Pattern
async def get_pending_tasks(self):
    return await loop.run_in_executor(self.process_pool, legacy_module.scan)

```



---

## 3. Task Model & Stability

### 3.1 Task Definition

```python
@dataclass
class Task:
    id: str
    key: str                    # Deduplication key
    module: str                 # "explorer", "documenter"
    priority: int               # Computed score
    state: TaskState            # PENDING | RUNNING | FAILED | COMPLETED
    
    # Context
    payload: dict
    conversation: ConversationState
    
    # Recovery Handles
    pool_request_id: Optional[str] = None # Handle for active work in Pool
    attempts: int = 0

```

### 3.2 Submit-If-Absent (Duplicate Prevention)

Modules will repeatedly propose the same work every poll cycle.

* **Logic:** The Scheduler maintains `active_task_keys` for all non-terminal tasks. `submit(task)` is ignored if `task.key` is already active.

### 3.3 The Failure Cache (Infinite Loop Prevention)

If a task fails deterministically (e.g., "Context too long"), it falls out of `active_task_keys`. The module will re-propose it immediately, causing an infinite failure loop.

* **Fix:** Maintain a `RecentFailureCache` (LRU, 1-hour TTL).
* **Logic:** Reject submission if `key` is in `RecentFailureCache` unless `force_retry=True`.

---

## 4. Scheduling & Work Stealing (The "JIT" Model)

### 4.1 Orchestrator-Owned Priority

The Orchestrator holds the **only** valid priority queue. The Pool is treated as a dumb executor with Capacity=1 per backend.

### 4.2 JIT (Just-In-Time) Execution Loop

Each backend (ChatGPT, Gemini, Claude) has a dedicated worker loop in the Orchestrator:

```python
async def worker_loop(backend):
    while True:
        # 1. Check if Pool is effectively free
        if await pool.is_busy(backend):
            await asyncio.sleep(1)
            continue

        # 2. Get highest priority task for this backend
        task = scheduler.get_next_task(backend)
        if not task:
            await asyncio.sleep(5)
            continue

        # 3. Submit Immediate (Bypassing Pool Queue)
        try:
            # Generate Idempotency Token
            token = f"{task.id}:{task.attempts}"
            
            req_id = await pool.submit_immediate(backend, task.payload, token)
            
            task.pool_request_id = req_id
            await state.save_task(task) # Commit RUNNING state
            
            await wait_for_result(task)
            
        except PoolBusyError:
            # Race condition lost (Legacy worker snatched it); retry
            continue

```

### 4.3 Sticky Scheduling

Allocator must prefer backends that already hold the conversation context (`thread_id`) to avoid browser navigation overhead.

---

## 5. Pool Integration Requirements

The Pool needs minimal changes to support this "JIT" model, but requires **Priority Locking** to prevent legacy workers from starving the Orchestrator.

### 5.1 New APIs

* `is_backend_busy(backend) -> bool`: Returns true if a browser session is locked.
* `submit_immediate(backend, payload, token) -> request_id`:
* **Atomic Check:** If backend is busy, raise `PoolBusyError`.
* **Idempotency:** If `token` matches an active request, return existing `request_id`.


* `get_active_requests() -> list[request_info]`: Returns all currently running requests (for recovery).

### 5.2 Priority Locking (Critical)

**The Race Condition:** A legacy background worker (in `pool/src/workers.py`) might wake up and grab the lock *after* `is_backend_busy` returns False but *before* `submit_immediate` arrives.
**The Fix:**

* The `submit_immediate` API handler attempts to acquire the lock.
* Legacy workers in `workers.py` should check a `high_priority_pending` flag or rely on the API winning the mutex contention often enough (retry logic in Orchestrator handles the loss).

---

## 6. Persistence: Snapshot + Delta WAL

To prevent "Write Amplification" (logging 100MB of history for a 1KB change).

### 6.1 The Protocol

1. **WAL (Deltas):** Records only **Changes** since the last checkpoint.
* `Op: MSG_APPEND, Data: { role: "assistant", content: "..." }`
* `Op: STATE_CHANGE, Data: { task_id, new_state: RUNNING }`


2. **Checkpoint (Snapshot):** Every 60 seconds, serialize the **Full In-Memory State** (including full conversation history) to `checkpoint.json`.
* *Why?* You cannot replay a delta log without a valid base state.


3. **Atomic Writes:** Write `.tmp` -> `fsync` -> `os.replace`.

---

## 7. Recovery & Reconciliation

When the Orchestrator restarts, it must "adopt" any orphans running in the Pool to prevent Double Billing.

### 7.1 Reconciliation Protocol (On Startup)

1. **Query:** Call `pool.get_active_requests()`.
2. **Load:** Restore state from Checkpoint/WAL.
3. **Match:**
* If `Task A` is `RUNNING` in Orchestrator AND exists in Pool  **Re-attach** (Update `pool_request_id`, wait for result).
* If `Task A` is `RUNNING` in Orchestrator but NOT in Pool  **Mark FAILED** (Lost during crash).
* If Pool has a request unknown to Orchestrator  **Kill Request** (Zombie).



---

## 8. Migration Strategy

1. **Phase 1: Pool Gateway:** Implement `submit_immediate` with Idempotency and `get_active_requests`. Ensure API calls can preempt legacy workers.
2. **Phase 2: Orchestrator Skeleton:** Build the Scheduler, State Manager (Checkpoint+WAL), and JIT Worker Loop.
3. **Phase 3: Adapters:** Wrap `Explorer` and `Documenter` legacy logic in `ProcessPoolExecutor` wrappers.
4. **Phase 4: Switchover:** Disable the legacy internal orchestration loops. Start the Master Orchestrator.

---

## 9. Configuration Schema

```yaml
orchestrator:
  state:
    checkpoint_dir: "data/orchestrator"
    checkpoint_interval: 60
  scheduler:
    jit_poll_interval: 1.0
    failure_cache_ttl: 3600
  recovery:
    orphan_reconciliation: true

pool:
  url: "http://localhost:8000"
  timeout: 3600 # 1 hour max for massive reasoning tasks

```
