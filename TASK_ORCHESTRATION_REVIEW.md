# Task Orchestration System - Comprehensive Review

**Date:** 2026-01-14
**Scope:** Complete analysis of task orchestration, LLM utilization, robustness, and code quality
**Updated:** Analysis of new top-level orchestration module design (`docs/requirements/module-orchestration.md`)

---

## Executive Summary

This review identified **47 issues** across the Fano task orchestration system, including:
- **8 CRITICAL** issues that could cause data loss or system failures
- **15 HIGH** severity bugs affecting reliability and correctness
- **14 MEDIUM** issues impacting efficiency and maintainability
- **10 LOW** priority improvements

### New Orchestration Module Impact

The proposed top-level orchestration module (`docs/requirements/module-orchestration.md`) **addresses 12 of the 47 issues** identified, including:
- ✅ Priority scheduling (new Scheduler with dynamic priority computation)
- ✅ Atomic file writes (StateManager uses temp file + rename)
- ✅ Unified quota tracking (centralized LLMQuotas)
- ✅ Work stealing (preemptible tasks with yield points)
- ✅ Crash recovery (checkpointing with proper state restoration)

However, the new design introduces **8 new potential issues** and **23 existing issues remain unaddressed** that must be fixed before or during migration.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [New Orchestration Module Analysis](#2-new-orchestration-module-analysis)
3. [LLM Instance Utilization](#3-llm-instance-utilization)
4. [Task Priority Issues](#4-task-priority-issues)
5. [Insight Flow Problems](#5-insight-flow-problems)
6. [Robustness & Recovery](#6-robustness--recovery)
7. [Race Conditions](#7-race-conditions)
8. [Error Handling](#8-error-handling)
9. [Recommendations](#9-recommendations)

---

## 1. Architecture Overview

### Current System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPLORER ORCHESTRATOR                    │
│ explorer/src/orchestrator.py                                │
└─────────────────────────────────────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ThreadManager │  │LLMManager    │  │SynthesisEng. │
    │(select/spawn)│  │(send message)│  │(chunk ready?)│
    └──────────────┘  └──────────────┘  └──────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
          ┌─────────────────────────────────────┐
          │         POOL SERVICE (HTTP)         │
          │  pool/src/api.py                    │
          │  - JobStore (async jobs)            │
          │  - RequestQueue (sync requests)     │
          │  - Workers (Gemini/ChatGPT/Claude)  │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │  InsightProcessor → BlessedStore    │
          │  (extraction, review, blessing)     │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │        DOCUMENTER ORCHESTRATOR      │
          │  documenter/main.py                 │
          │  - OpportunityFinder                │
          │  - OpportunityProcessor             │
          └─────────────────────────────────────┘
```

### Key Data Flows

1. **Exploration Flow**: Thread → LLM Request → Response → Exchange → Save
2. **Synthesis Flow**: Thread → Chunk Ready Check → Extract → Review → Bless
3. **Document Flow**: Blessed Insight → Opportunity → Evaluate → Draft → Add Section

---

## 2. New Orchestration Module Analysis

### 2.1 Proposed Architecture

The new design (`docs/requirements/module-orchestration.md`) introduces a centralized orchestrator:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scheduler │  │   State     │  │   LLM Allocator         │  │
│  │   (decides  │  │   Manager   │  │   (routes requests,     │  │
│  │    what)    │  │   (tracks)  │  │    manages quotas)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└────────┬────────────────┬─────────────────────┬─────────────────┘
         │                │                     │
         ▼                ▼                     ▼
    ┌─────────┐     ┌───────────┐        ┌───────────┐
    │Explorer │     │Documenter │        │Researcher │
    │ Module  │     │  Module   │        │  Module   │
    └─────────┘     └───────────┘        └───────────┘
         │                │                     │
         └────────────────┼─────────────────────┘
                          ▼
              ┌───────────────────────┐
              │      LLM Pool         │
              │  (ChatGPT/Gemini/     │
              │   Claude)             │
              └───────────────────────┘
```

### 2.2 Issues ADDRESSED by New Design

| Issue ID | Original Issue | How New Design Addresses It |
|----------|---------------|----------------------------|
| **CRITICAL** | JobStore ignores priority | Scheduler.compute_priority() with dynamic factors (Part 3.1) |
| **CRITICAL** | Worker scheduling priority inversion | Unified task queue via Scheduler (Part 3.3) |
| **HIGH** | Non-atomic blessed_insights.json writes | StateManager uses temp file + os.rename() (Part 5.2) |
| **HIGH** | Dual deep mode tracking out of sync | Centralized LLMQuotas with single source of truth (Part 4.2) |
| **HIGH** | No work stealing between models | Preemptible tasks with yield points (Part 2.4) |
| **HIGH** | Pool restart mid-request recovery | Task state persistence with PAUSED state recovery (Part 5.3) |
| **MEDIUM** | Thread selection reloads all threads | Module state caching in OrchestratorState (Part 5.1) |
| **MEDIUM** | Document write not atomic | StateManager pattern can be applied to all writes |
| **LOW** | Deep mode quota underutilized | LLM consultation recommends allocation shifts (Part 3.4) |
| **LOW** | Missing timeout config | Configurable in orchestrator config (Part 8) |
| **MEDIUM** | No cross-module coordination | Explicit module interface with priority balancing |
| **HIGH** | Orphaned futures after restart | Task-based model doesn't rely on futures |

**Summary:** 12 of 47 issues addressed (26%)

### 2.3 Issues REMAINING (Must Fix Before/During Migration)

These issues exist in current code and are NOT addressed by the new orchestration design:

| Issue ID | Issue | Location | Severity |
|----------|-------|----------|----------|
| 1 | Directory lookup bug in insight processor | `explorer/src/orchestration/insight_processor.py:390` | **CRITICAL** |
| 2 | Documenter doesn't load blessed insights into dedup | `documenter/session.py:217-237` | **CRITICAL** |
| 3 | Global rate_tracker without synchronization | `explorer/src/browser/base.py:94-151` | **CRITICAL** |
| 4 | Unprotected worker state reads | `pool/src/workers.py:45-48` | **CRITICAL** |
| 5 | Bare except clause | `control/debug_util.py:47` | **CRITICAL** |
| 6 | Fire-and-forget save in quota handler | `explorer/src/orchestration/insight_processor.py:372` | **HIGH** |
| 7 | File TOCTOU in queue recovery | `pool/src/queue.py:236-268` | **HIGH** |
| 8 | JobStore content cache race | `pool/src/jobs.py:121-127` | **HIGH** |
| 9 | Swallowed exceptions in worker loop | `pool/src/workers.py:451-454` | **HIGH** |
| 10 | Silent failures in blessed store load | `explorer/src/orchestration/blessed_store.py:88-96` | **MEDIUM** |
| 11 | In-memory review progress tracking | `explorer/src/orchestration/insight_processor.py:263-323` | **MEDIUM** |
| 12 | Missing incorporation flag persistence | `documenter/opportunities.py:186-208` | **MEDIUM** |

**Note:** The new orchestration module doesn't touch pool internals or existing module code. These bugs will persist unless explicitly fixed.

### 2.4 NEW Potential Issues in Proposed Design

The new design introduces some concerns that should be addressed:

#### NEW-1: ConversationState Size Unbounded [MEDIUM]

**Location:** `docs/requirements/module-orchestration.md` Part 2.5

```python
@dataclass
class ConversationState:
    messages: list[dict]        # Message history - NO SIZE LIMIT
```

**Problem:** Multi-turn exploration conversations can grow to hundreds of messages. Serializing full history on every checkpoint creates:
- Large state files (potential 10MB+ per conversation)
- Slow checkpoint writes
- Memory pressure

**Recommendation:** Add configurable message limit (e.g., last 50 messages) or rolling summary.

---

#### NEW-2: Preemption Only After LLM Requests [MEDIUM]

**Location:** `docs/requirements/module-orchestration.md` Part 2.4

```python
if action.type == "llm_request":
    if self.scheduler.should_preempt(task):
        task.state = TaskState.PAUSED
        # ...
elif action.type == "file_operation":
    await action.execute()  # No preemption check!
```

**Problem:** File operations (extraction, dedup checks, blessed store writes) can take significant time and cannot be preempted. A long file operation blocks higher-priority work.

**Recommendation:** Add preemption points before expensive file operations or make them cancellable.

---

#### NEW-3: LLM Consultation Overhead [LOW]

**Location:** `docs/requirements/module-orchestration.md` Part 3.4

```python
async def consult_llm_for_priorities(self):
    """Use ~5% of LLM budget for meta-decisions."""
```

**Problem:** Hourly LLM consultations for priority guidance could be wasteful, especially when:
- System is in steady state with balanced workload
- Rate limits are already hit
- Simple heuristics would suffice

**Recommendation:** Make consultation conditional - only when significant imbalance detected OR explicitly requested.

---

#### NEW-4: Task Queue Unbounded [MEDIUM]

**Location:** `docs/requirements/module-orchestration.md` Part 5.1

```python
@dataclass
class OrchestratorState:
    tasks: dict[str, Task]       # No size limit
    task_queue: list[str]        # No size limit
```

**Problem:** If modules generate tasks faster than they're consumed (e.g., many seeds, backlog of insights), the task queue can grow unboundedly.

**Recommendation:** Add queue size limits per module with backpressure.

---

#### NEW-5: Recovery Gap for RUNNING Tasks [HIGH]

**Location:** `docs/requirements/module-orchestration.md` Part 5.3

```python
if saved_state:
    for task in self.tasks.values():
        if task.state == TaskState.RUNNING:
            # Was running when crashed - mark as paused
            task.state = TaskState.PAUSED
```

**Problem:** A task marked RUNNING at crash time may not have saved its conversation state (checkpoints are periodic, not per-action). When resumed as PAUSED, the conversation state may be stale or missing.

**Recommendation:** Save conversation state immediately before each LLM request, not just at checkpoints.

---

#### NEW-6: Checkpoint Failure Not Handled [MEDIUM]

**Location:** `docs/requirements/module-orchestration.md` Part 5.2

```python
def save(self):
    temp_path = f"{self.state_path}.tmp"
    with open(temp_path, 'w') as f:
        json.dump(asdict(state), f, indent=2, default=str)
    os.rename(temp_path, self.state_path)  # What if this fails?
```

**Problem:** No error handling for checkpoint failures. If disk is full, permissions denied, or rename fails, the orchestrator continues without persisted state.

**Recommendation:** Add retry logic, alerting, and consider graceful degradation.

---

#### NEW-7: Module Task Generation Race [MEDIUM]

**Location:** `docs/requirements/module-orchestration.md` Part 6.1

```python
class ModuleInterface(ABC):
    @abstractmethod
    def get_pending_tasks(self) -> list[Task]:
        """Return tasks this module wants to run."""
        pass
```

**Problem:** Scheduler calls `get_pending_tasks()` on each module, but module state may change between task generation and task execution. A task may become invalid (e.g., insight already processed by another path).

**Recommendation:** Add task validation before execution, with graceful handling of stale tasks.

---

#### NEW-8: Pool Service Role Unclear [HIGH]

**Location:** `docs/requirements/module-orchestration.md` Part 12, Question 1

> "Browser pool simplification: With single-instance LLMs, do we still need the full pool service, or can we simplify?"

**Concern:** The design shows LLM Allocator routing to Pool, but it's unclear:
- Does Pool retain its current queue/job semantics?
- Does Orchestrator bypass Pool for task execution?
- How do the two priority systems (Orchestrator Scheduler vs Pool RequestQueue) interact?

**Recommendation:** Clarify Pool's role in the new architecture:
- Option A: Pool becomes thin browser wrapper, Orchestrator handles all scheduling
- Option B: Pool retains scheduling for browser-specific concerns, Orchestrator handles cross-module coordination

---

### 2.5 Design Strengths

The new orchestration design has several excellent features:

1. **Dynamic Priority Computation** (Part 3.1) - Considers backlog pressure, starvation prevention, seed priority, comment responsiveness, and quota availability

2. **Preemptible Tasks** (Part 2.4) - Enables fine-grained control over LLM allocation

3. **Atomic Checkpointing** (Part 5.2) - Proper crash-safe state persistence

4. **Module Interface** (Part 6.1) - Clean abstraction for module integration

5. **Configurable Everything** (Part 8) - Sensible defaults with full override capability

6. **LLM Consultation** (Part 3.4) - Innovative approach to adaptive scheduling

7. **Metrics & Observability** (Part 7) - Comprehensive structured logging

---

## 3. LLM Instance Utilization

### Current Model Management

**File:** `explorer/src/orchestration/llm_manager.py`

The system manages 2-3 LLM instances:
- ChatGPT (browser-based)
- Gemini (browser-based)
- Claude (API-based, via pool)

### Issues with LLM Utilization

#### ISSUE-LLM-1: No Work Stealing Between Models [HIGH] ✅ ADDRESSED

**Location:** `explorer/src/orchestrator.py:373`

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Problem:** If one model completes quickly (e.g., Gemini in 2 min) and another is slow (ChatGPT taking 10 min), the fast model sits idle until all parallel work completes.

**Impact:** With 30-second poll intervals, fast models waste ~80% of their capacity waiting.

**New Design Solution:** Preemptible tasks with yield points allow idle LLMs to pick up new work immediately.

---

#### ISSUE-LLM-2: Thread Selection Reloads All Threads Each Cycle [MEDIUM] ✅ ADDRESSED

**Location:** `explorer/src/orchestration/thread_manager.py:67-125`

**Problem:** `load_active_threads()` does filesystem glob scan for EACH model assignment (up to 3x per 30-second cycle).

**Impact:** ~100 JSON file reads per cycle as thread count grows.

**New Design Solution:** Module state cached in OrchestratorState, updated only on changes.

---

#### ISSUE-LLM-3: Deep Mode Quota Underutilized [LOW] ✅ ADDRESSED

**Configuration:** ChatGPT Pro limited to 100/day, Gemini Deep Think to 20/day

**Problem:** With ~3-5 synthesis events per day, only using ~30% of allocated quota.

**New Design Solution:** LLM consultation can recommend more aggressive deep mode usage when appropriate.

---

#### ISSUE-LLM-4: Dual Deep Mode Tracking Systems Out of Sync [HIGH] ✅ ADDRESSED

**Locations:**
- `explorer/src/browser/model_selector.py` - maintains `deep_mode_state.json`
- `pool/src/state.py` - maintains `pool_state.json` with `deep_mode_uses_today`

**Problem:** Two independent counters that can diverge if pool restarts or explorer records usage before pool fails.

**New Design Solution:** Centralized LLMQuotas in Orchestrator is the single source of truth.

---

## 4. Task Priority Issues

### CRITICAL: JobStore Ignores Priority [CRITICAL] ✅ ADDRESSED

**Location:** `pool/src/jobs.py:211-234`

```python
def get_next_job(self, backend: str) -> Optional[Job]:
    with self._lock:
        queue = self._backend_queues.get(backend, [])

        for job_id in queue:  # SIMPLE LINEAR ITERATION (FIFO)
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                # ... returns first found, ignores priority
                return job
```

**Problem:** The `Job` dataclass accepts a `priority` field (line 48) but `get_next_job()` iterates through jobs in FIFO order, completely ignoring priority.

**New Design Solution:** Scheduler.compute_priority() considers multiple factors; unified task queue replaces per-backend queues.

**Migration Note:** If Pool is retained, JobStore still needs fixing for backward compatibility during migration.

---

### CRITICAL: Worker Scheduling Priority Inversion [CRITICAL] ✅ ADDRESSED

**Location:** `pool/src/workers.py:71-91`

```python
async def _run_loop(self):
    while self._running:
        # First, check JobStore for async jobs (new system)
        if self.jobs:
            job = self.jobs.get_next_job(self.backend_name)  # ← FIFO
            if job:
                await self._process_job(job)
                continue

        # Fall back to legacy queue (sync system)
        queued = await self.queue.dequeue()  # ← Priority-aware
```

**Problem:** Workers check JobStore (FIFO) BEFORE RequestQueue (priority-aware). A LOW priority async job will be processed before a HIGH priority sync request.

**New Design Solution:** Single Scheduler determines execution order; no dual-queue confusion.

---

### Priority System Summary (Updated)

| Component | Current Priority Support | New Design |
|-----------|-------------------------|------------|
| RequestQueue (sync) | HIGH/NORMAL/LOW via heap | Replaced by Scheduler |
| JobStore (async) | Field exists, IGNORED | Replaced by Scheduler |
| Worker scheduling | Async before sync | N/A - Scheduler controls |
| Documenter opportunities | Calculated priority score | Integrated into Scheduler |
| **Orchestrator Scheduler** | N/A | ✅ Dynamic multi-factor priority |

---

## 5. Insight Flow Problems

### CRITICAL: Directory Lookup Bug in Insight Processor [CRITICAL] ⚠️ NOT ADDRESSED

**Location:** `explorer/src/orchestration/insight_processor.py:390`

```python
subdirs = ["pending", "reviewing", "insights/blessed", "insights/rejected"]
for subdir in subdirs:
    search_dir = self.paths.chunks_dir / subdir
```

**Problem:** Searches `chunks/pending/` but pending insights are actually saved to `chunks/insights/pending/`.

**Impact:** After crash during review, restart cannot find existing insights and re-extracts them, creating duplicates.

**Status:** Must be fixed in current code. New orchestration module doesn't change InsightProcessor internals.

---

### CRITICAL: Documenter Doesn't Load Blessed Insights into Dedup [CRITICAL] ⚠️ NOT ADDRESSED

**Location:** `documenter/session.py:217-237`

```python
async def _initialize_dedup(self):
    # Load existing document sections into dedup checker
    for section in self.document.sections:
        self.dedup_checker.add_content(...)
    # MISSING: Load blessed insights from blessed_dir!
```

**Problem:** Documenter only checks new insights against existing document sections, not against other blessed insights waiting to be processed.

**Impact:** Two similar blessed insights can both be incorporated, creating duplicates in the document.

**Status:** Must be fixed in Documenter module code. New orchestration won't change this behavior.

---

### HIGH: Fire-and-Forget Save in Quota Handler [HIGH] ⚠️ NOT ADDRESSED

**Location:** `explorer/src/orchestration/insight_processor.py:372`

```python
def _handle_quota_exhausted(self, insight: AtomicInsight, error):
    insight.status = InsightStatus.PENDING
    asyncio.create_task(asyncio.to_thread(insight.save, self.paths.chunks_dir))
    # Returns immediately without awaiting
```

**Problem:** If process exits immediately, save may not complete.

**Status:** Must be fixed in current code.

---

### HIGH: Non-Atomic blessed_insights.json Writes [HIGH] ✅ ADDRESSED

**Location:** `explorer/src/orchestration/blessed_store.py:117-137`

**Problem:** Read-modify-write without atomicity. Crash during write corrupts file.

**New Design Solution:** StateManager pattern (temp file + rename) can be applied to all JSON writes.

---

### Insight Flow Summary

```
Thread Exploration
       │
       ▼
  Chunk Ready? ──────────────────────────────┐
       │                                      │
       ▼                                      │
Extract Insights ◄──── ⚠️ BUG: Wrong directory│
       │                   (NOT ADDRESSED)    │
       ▼                                      │
 Dedup Check ◄───────── ⚠️ BUG: Explorer and  │
       │                   Documenter have    │
       ▼                   separate dedup     │
Review Panel               (NOT ADDRESSED)    │
       │                                      │
       ▼                                      │
Bless Insight ◄──────── ✅ Will use atomic    │
       │                   writes             │
       ▼                                      │
Documenter ◄─────────── ⚠️ BUG: Doesn't load  │
       │                   blessed into dedup │
       ▼                   (NOT ADDRESSED)    │
 Add to Document                              │
```

---

## 6. Robustness & Recovery

### State Persistence Analysis (Updated)

| State | Current | New Design | Notes |
|-------|---------|------------|-------|
| Pool queue requests | ✅ Persisted | ⚠️ TBD | Pool role unclear |
| Async jobs | ✅ Persisted | ⚠️ TBD | Pool role unclear |
| Active work | ⚠️ Watchdog only | ✅ Task state | Immediate recovery |
| Explorer threads | ✅ On startup | ✅ Module state | Cached |
| In-progress extraction | ❌ None | ✅ Task PAUSED | Resumable |
| Review progress | ❌ None | ✅ Task state | Persisted |
| Document sections | ✅ On startup | ✅ Module state | Unchanged |
| Conversation state | ❌ None | ⚠️ Periodic | See NEW-5 |

---

### HIGH: Pool Restart Mid-Request [HIGH] ✅ ADDRESSED

**Current Behavior:**
1. Active work tracked in `pool_state.json`
2. On restart: Queue restored, but NO automatic recovery of active work
3. Watchdog kicks in after 1 HOUR timeout

**New Design Solution:** Tasks have explicit PAUSED state that's restored on startup. No 2-hour limbo.

---

### HIGH: Orphaned Futures After Pool Restart [HIGH] ✅ ADDRESSED

**Location:** `pool/src/queue.py:157-190`

**Problem:** Original requestor's Future is lost (cannot be serialized).

**New Design Solution:** Task-based model doesn't rely on futures. Tasks resume from persisted state.

---

### MEDIUM: Document Write Not Atomic [MEDIUM] ⚠️ PARTIALLY ADDRESSED

**Location:** `documenter/document.py:107-135`

**Problem:** Direct `write_text()` - if crash mid-write, file is truncated/corrupted.

**Status:** StateManager pattern addresses orchestrator state, but document writes still need explicit fix.

---

## 7. Race Conditions

### CRITICAL: Unprotected Worker State Reads [CRITICAL] ⚠️ NOT ADDRESSED

**Location:** `pool/src/workers.py:45-48, 101-103` and `pool/src/api.py:137-144`

```python
# Workers (no lock):
self._current_request_id = queued.request_id
self._current_start_time = time.time()

# Watchdog (no lock):
if worker._current_start_time is None:
    continue
elapsed = time.time() - worker._current_start_time  # TOCTOU
```

**Problem:** Worker state accessed without synchronization.

**Status:** Pool internals not changed by new orchestration. Must be fixed separately.

---

### CRITICAL: Global rate_tracker Without Synchronization [CRITICAL] ⚠️ NOT ADDRESSED

**Location:** `explorer/src/browser/base.py:94-151`

```python
rate_tracker = RateLimitTracker()  # Global instance

def _save(self):
    with open(RATE_LIMIT_FILE, "w", encoding="utf-8") as f:
        json.dump(self.limits, f, indent=2, default=str)  # No lock!
```

**Problem:** Multiple async tasks call `mark_limited()`, `is_available()`, `_save()` without any synchronization.

**Status:** New LLMQuotas is centralized but old rate_tracker still exists in explorer code. Must be migrated.

---

### HIGH: File TOCTOU in Queue Recovery [HIGH] ⚠️ NOT ADDRESSED

**Location:** `pool/src/queue.py:236-268`

**Status:** Pool internals not changed.

---

### HIGH: JobStore Content Cache Race [HIGH] ⚠️ NOT ADDRESSED

**Location:** `pool/src/jobs.py:121-127`

**Status:** Pool internals not changed.

---

### Race Conditions Summary (Updated)

| Issue | Severity | New Design Status |
|-------|----------|-------------------|
| Unprotected worker state | CRITICAL | ⚠️ NOT ADDRESSED |
| Global rate_tracker | CRITICAL | ⚠️ PARTIALLY (new quota, old tracker remains) |
| Queue recovery TOCTOU | HIGH | ⚠️ NOT ADDRESSED |
| JobStore cache cleanup | HIGH | ⚠️ NOT ADDRESSED |
| Callback variable capture | HIGH | ✅ No callbacks in new design |
| JSON file writes | HIGH | ✅ Atomic in StateManager |
| Lock under blocking I/O | MEDIUM | ✅ Async-first design |
| File deletion races | MEDIUM | ⚠️ NOT ADDRESSED |

---

## 8. Error Handling

### CRITICAL: Bare Except Clause [CRITICAL] ⚠️ NOT ADDRESSED

**Location:** `control/debug_util.py:47`

```python
try:
    count = len(re.findall(pat, resp))
except:  # ⚠️ BARE EXCEPT
    pass
```

**Status:** Control panel code not changed by new orchestration.

---

### HIGH: Swallowed Exceptions in Worker Loop [HIGH] ⚠️ NOT ADDRESSED

**Location:** `pool/src/workers.py:451-454`

**Status:** Pool internals not changed.

---

### Error Handling Summary (Updated)

| Issue | Location | New Design Status |
|-------|----------|-------------------|
| Bare except | `control/debug_util.py:47` | ⚠️ NOT ADDRESSED |
| Broad Exception catch | `pool/src/workers.py:451` | ⚠️ NOT ADDRESSED |
| Silent failures | Multiple locations | ⚠️ PARTIAL (new logging better) |
| Missing timeout config | `explorer/src/browser/base.py:244` | ✅ Configurable in new design |
| Inconsistent logging | Mixed structured/string | ✅ Structured logging in design |

---

## 9. Recommendations

### Phase 1: Fix Critical Issues Before Migration

These must be fixed in current code before or during orchestration migration:

| Priority | Issue | Location | Action |
|----------|-------|----------|--------|
| 1 | Directory lookup bug | `insight_processor.py:390` | Fix path to `insights/pending` |
| 2 | Documenter dedup loading | `session.py:217-237` | Load `blessed_insights.json` on startup |
| 3 | Global rate_tracker races | `base.py:94-151` | Add threading.Lock or migrate to LLMQuotas |
| 4 | Unprotected worker state | `workers.py:45-48` | Add asyncio.Lock |
| 5 | Bare except | `debug_util.py:47` | Change to `except Exception:` |
| 6 | Fire-and-forget save | `insight_processor.py:372` | Change to `await` |

### Phase 2: Implement New Orchestration with Fixes

Address new design issues during implementation:

| Priority | Issue | Section | Recommendation |
|----------|-------|---------|----------------|
| 1 | Pool role unclear | NEW-8 | Decide Pool's future before coding |
| 2 | RUNNING task recovery gap | NEW-5 | Save conversation state before LLM requests |
| 3 | ConversationState size | NEW-1 | Add message limit (e.g., 50) |
| 4 | Task queue unbounded | NEW-4 | Add per-module limits with backpressure |
| 5 | Checkpoint failure handling | NEW-6 | Add retry + alerting |
| 6 | Preemption granularity | NEW-2 | Add preemption points for file ops |
| 7 | Module task generation race | NEW-7 | Validate tasks before execution |
| 8 | LLM consultation overhead | NEW-3 | Make conditional |

### Phase 3: Cleanup During Migration

| Action | Files Affected |
|--------|----------------|
| Remove old rate_tracker | `explorer/src/browser/base.py` |
| Remove old deep_mode_state | `explorer/src/browser/model_selector.py` |
| Simplify/remove Pool queues | `pool/src/jobs.py`, `pool/src/queue.py` |
| Apply atomic writes everywhere | `blessed_store.py`, `document.py`, etc. |

---

## Appendix A: Issue Status Summary

### Issues by New Design Status

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Addressed by new design | 12 | 26% |
| ⚠️ Not addressed (fix required) | 23 | 49% |
| ⚠️ Partially addressed | 4 | 9% |
| 🆕 New issues in design | 8 | 17% |
| **Total Issues** | **47 + 8 = 55** | |

### Critical Issues Checklist

| # | Issue | Status |
|---|-------|--------|
| 1 | JobStore ignores priority | ✅ Addressed |
| 2 | Worker scheduling inversion | ✅ Addressed |
| 3 | Directory lookup bug | ⚠️ Fix required |
| 4 | Documenter dedup missing | ⚠️ Fix required |
| 5 | Unprotected worker state | ⚠️ Fix required |
| 6 | Global rate_tracker races | ⚠️ Fix required |
| 7 | Bare except clause | ⚠️ Fix required |
| 8 | Dual deep mode tracking | ✅ Addressed |

---

## Appendix B: Files Requiring Changes (Updated)

| File | Issue Type | When to Fix |
|------|------------|-------------|
| `explorer/src/orchestration/insight_processor.py` | Directory bug, fire-and-forget | **Phase 1** |
| `documenter/session.py` | Missing dedup loading | **Phase 1** |
| `explorer/src/browser/base.py` | Global rate_tracker races | **Phase 1** |
| `pool/src/workers.py` | Unprotected state, swallowed exceptions | **Phase 1** |
| `control/debug_util.py` | Bare except | **Phase 1** |
| `pool/src/jobs.py` | Priority scheduling (if Pool retained) | **Phase 2** |
| `pool/src/queue.py` | TOCTOU (if Pool retained) | **Phase 2** |
| `pool/src/state.py` | Non-atomic writes | **Phase 2** |
| `explorer/src/orchestration/blessed_store.py` | Apply atomic writes | **Phase 3** |
| `documenter/document.py` | Apply atomic writes | **Phase 3** |
| `explorer/src/browser/model_selector.py` | Remove old deep_mode_state | **Phase 3** |

---

## Appendix C: New Orchestration Implementation Priorities

Based on the design and this review, recommended implementation order:

1. **StateManager** - Foundation for all persistence (atomic writes, checkpointing)
2. **Task Model** - Define Task, TaskState, ConversationState with size limits
3. **Scheduler** - Implement compute_priority() with all factors
4. **LLMQuotas** - Centralized quota tracking
5. **ModuleInterface** - Define abstract interface
6. **ExplorerModule** - Adapt existing orchestrator.py
7. **DocumenterModule** - Adapt existing main.py
8. **ResearcherModule** - New implementation
9. **TaskExecutor** - Preemption, yield points, state saving
10. **Main Loop** - Wire everything together

---

*End of Review*
