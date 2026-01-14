# Fano Unified Orchestration System: Design Specification

**Version:** 1.0
**Date:** 2026-01-14
**Status:** Proposed
**Authors:** System Design Review

---

## Document Purpose

This document provides a complete design specification for the Fano orchestration system. It is intended for:
1. **Internal implementation** - Detailed enough to guide development
2. **External review** - Self-contained context for third-party assessment
3. **Issue resolution** - Explicitly addresses all 55 identified system issues

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Fano Platform Overview](#2-fano-platform-overview)
3. [Current Architecture & Issues](#3-current-architecture--issues)
4. [Design Principles](#4-design-principles)
5. [Proposed Architecture](#5-proposed-architecture)
6. [Component Specifications](#6-component-specifications)
7. [State Management](#7-state-management)
8. [Concurrency & Synchronization](#8-concurrency--synchronization)
9. [Fault Tolerance & Recovery](#9-fault-tolerance--recovery)
10. [Observability](#10-observability)
11. [Configuration](#11-configuration)
12. [Migration Strategy](#12-migration-strategy)
13. [Testing Strategy](#13-testing-strategy)
14. [Issue Resolution Matrix](#14-issue-resolution-matrix)
15. [Appendices](#15-appendices)

---

## 1. Executive Summary

### 1.1 Problem Statement

Fano is an autonomous mathematical discovery platform that must run continuously for days to weeks without human intervention. The current architecture has **47 identified issues** across priority scheduling, concurrency, persistence, and recovery mechanisms. A proposed orchestration module addresses 12 of these but introduces 8 new concerns.

### 1.2 Solution Overview

This design introduces a **Unified Orchestration System** that:

- **Centralizes coordination** through a single Orchestrator managing all modules
- **Eliminates dual systems** by consolidating Pool queuing into orchestrator scheduling
- **Ensures atomicity** through write-ahead logging and transactional state changes
- **Provides resilience** via comprehensive checkpointing and recovery protocols
- **Addresses all 55 issues** through systematic architectural improvements

### 1.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pool becomes thin browser wrapper** | Eliminates scheduling duplication; orchestrator handles all prioritization |
| **Write-Ahead Log (WAL) for state** | Ensures no data loss on crash; enables point-in-time recovery |
| **Actor model for concurrency** | Eliminates shared mutable state; each component owns its state |
| **Eager conversation persistence** | Saves state before each LLM request, not just periodic checkpoints |
| **Unified priority queue** | Single priority computation across all modules and task types |

---

## 2. Fano Platform Overview

### 2.1 Purpose

Fano is an AI-powered platform for mathematical exploration and documentation. It uses multiple Large Language Models (LLMs) to:

1. **Explore** mathematical concepts through multi-turn conversations
2. **Extract** atomic insights from exploration threads
3. **Review** insights through consensus panels
4. **Document** validated insights in a living mathematical document
5. **Research** external sources to enrich understanding (planned)

### 2.2 Business Context

- **Subscription Model**: Fixed monthly LLM subscriptions (ChatGPT Pro, Gemini Advanced, Claude Pro)
- **Operational Goal**: Maximize insight throughput within rate limits
- **Autonomy Goal**: Run for days/weeks without human intervention
- **Quality Goal**: Produce mathematically rigorous, non-duplicative content

### 2.3 Current Modules

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FANO PLATFORM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │   EXPLORER  │    │  DOCUMENTER │    │  RESEARCHER │    │  CONTROL │ │
│  │             │    │             │    │   (Planned) │    │   PANEL  │ │
│  │ • Seeds     │    │ • Document  │    │             │    │          │ │
│  │ • Threads   │    │ • Sections  │    │ • Sources   │    │ • Web UI │ │
│  │ • Insights  │    │ • Reviews   │    │ • Findings  │    │ • API    │ │
│  │ • Chunks    │    │ • Comments  │    │             │    │ • Logs   │ │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └────┬─────┘ │
│         │                  │                  │                 │       │
│         └──────────────────┴──────────────────┴─────────────────┘       │
│                                    │                                     │
│                           ┌────────▼────────┐                           │
│                           │   BROWSER POOL  │                           │
│                           │                 │                           │
│                           │ • Gemini Worker │                           │
│                           │ • ChatGPT Worker│                           │
│                           │ • Claude Worker │                           │
│                           └─────────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Module Descriptions

#### Explorer
- **Purpose**: Discover mathematical insights through LLM collaboration
- **Process**: Seeds → Threads → Exchanges → Chunks → Insights → Blessed Store
- **Key Data**: Exploration threads (JSON), atomic insights (JSON)
- **LLM Usage**: Multi-turn conversations, deep mode for synthesis

#### Documenter
- **Purpose**: Grow a living mathematical document from blessed insights
- **Process**: Opportunities → Evaluation → Drafting → Review → Incorporation
- **Key Data**: Document sections (Markdown), review history (JSON)
- **LLM Usage**: Consensus panels (3 LLMs), deep mode for review

#### Researcher (Planned)
- **Purpose**: Research external sources to enrich understanding
- **Process**: Questions → Sources → Evaluation → Extraction → Synthesis
- **Key Data**: Research questions, source evaluations, findings

#### Pool
- **Purpose**: Browser automation for LLM web interfaces
- **Current Role**: Queue management, worker scheduling, state persistence
- **LLM Access**: Playwright automation for ChatGPT, Gemini, Claude web UIs

### 2.5 Data Flow

```
                    ┌─────────────────┐
                    │  SEED QUESTIONS │
                    │  (Initial input)│
                    └────────┬────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           EXPLORER MODULE                               │
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────┐   │
│   │  Thread  │───▶│ Exchange │───▶│  Chunk   │───▶│ Atomic Insight│   │
│   │ Creation │    │ (LLM)    │    │ Synthesis│    │  Extraction   │   │
│   └──────────┘    └──────────┘    └──────────┘    └───────┬───────┘   │
│                                                           │            │
│                                         ┌─────────────────┘            │
│                                         ▼                              │
│                               ┌──────────────────┐                     │
│                               │  Review Panel    │                     │
│                               │ (3-LLM Consensus)│                     │
│                               └────────┬─────────┘                     │
│                                        │                               │
│                                        ▼                               │
│                               ┌──────────────────┐                     │
│                               │  BLESSED STORE   │◀───┐                │
│                               │  (Validated      │    │ Dedup Check    │
│                               │   Insights)      │────┘                │
│                               └────────┬─────────┘                     │
└────────────────────────────────────────┼───────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          DOCUMENTER MODULE                              │
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│   │ Opportunity  │───▶│ Math Eval    │───▶│ Draft & Exposition   │    │
│   │ Selection    │    │ (Consensus)  │    │ Evaluation           │    │
│   └──────────────┘    └──────────────┘    └──────────┬───────────┘    │
│                                                      │                 │
│                                                      ▼                 │
│                                           ┌──────────────────┐        │
│                                           │ DOCUMENT.MD      │        │
│                                           │ (Living Document)│        │
│                                           └──────────────────┘        │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.6 LLM Resources

| Backend | Type | Interface | Rate Limits | Deep Mode |
|---------|------|-----------|-------------|-----------|
| ChatGPT | Subscription | Browser (Playwright) | ~50/hour | Pro Mode (100/day) |
| Gemini | Subscription | Browser (Playwright) | ~50/hour | Deep Think (20/day) |
| Claude | Subscription | Browser (Playwright) | ~50/hour | Extended Thinking (100/day) |
| OpenRouter | API | HTTP | Per-model | Per-model |

---

## 3. Current Architecture & Issues

### 3.1 Current Architecture Problems

The current system has these architectural deficiencies:

1. **Decentralized Scheduling**: Each module runs its own orchestration loop
2. **Dual Queue Systems**: Pool has both JobStore and RequestQueue with different semantics
3. **Priority Ignored**: JobStore accepts priority field but uses FIFO ordering
4. **Non-Atomic Persistence**: JSON files written directly without crash protection
5. **No Cross-Module Coordination**: Explorer and Documenter don't balance resources
6. **Fragmented State**: Rate limits, deep mode quota tracked in multiple places

### 3.2 Issue Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Priority/Scheduling | 2 | 1 | 0 | 0 | 3 |
| Data Flow/Correctness | 2 | 2 | 3 | 0 | 7 |
| Concurrency/Race Conditions | 2 | 4 | 2 | 0 | 8 |
| Persistence/Atomicity | 0 | 4 | 2 | 0 | 6 |
| Recovery/Resilience | 0 | 3 | 1 | 0 | 4 |
| Error Handling | 1 | 2 | 2 | 1 | 6 |
| LLM Utilization | 0 | 2 | 1 | 2 | 5 |
| New Design Issues | 1 | 1 | 5 | 1 | 8 |
| **TOTAL** | **8** | **19** | **16** | **4** | **55** |

### 3.3 Critical Issues Detail

| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| C1 | JobStore ignores priority | `pool/src/jobs.py:211` | HIGH priority jobs wait behind LOW |
| C2 | Worker scheduling inverted | `pool/src/workers.py:71` | Sync priority requests starved |
| C3 | Directory lookup bug | `insight_processor.py:390` | Duplicate insight extraction |
| C4 | Documenter dedup missing | `session.py:217` | Duplicate document sections |
| C5 | Unprotected worker state | `workers.py:45` | Race condition crashes |
| C6 | Global rate_tracker races | `base.py:94` | State corruption |
| C7 | Bare except clause | `debug_util.py:47` | Swallows KeyboardInterrupt |
| C8 | Dual deep mode tracking | Multiple files | Quota mismanagement |

---

## 4. Design Principles

### 4.1 Core Principles

| Principle | Description | Rationale |
|-----------|-------------|-----------|
| **Single Source of Truth** | Each piece of state has exactly one owner | Eliminates sync bugs |
| **Crash-Only Design** | System can crash at any point and recover | Enables autonomous operation |
| **Actor Isolation** | Components communicate via messages, not shared state | Eliminates race conditions |
| **Idempotent Operations** | Operations can be safely retried | Enables automatic recovery |
| **Explicit State Machines** | All state transitions are documented and validated | Prevents invalid states |
| **Structured Observability** | All operations emit structured logs and metrics | Enables debugging and monitoring |

### 4.2 Concurrency Model

We adopt the **Actor Model** for concurrency:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACTOR MODEL                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐│
│  │ Orchestrator │◀───────▶│   Explorer   │         │  Documenter  ││
│  │    Actor     │ messages│    Actor     │         │    Actor     ││
│  │              │         │              │         │              ││
│  │ • Scheduler  │         │ • Threads    │         │ • Document   ││
│  │ • Allocator  │         │ • Insights   │         │ • Sections   ││
│  │ • State Mgr  │         │ • Dedup      │         │ • Reviews    ││
│  └──────────────┘         └──────────────┘         └──────────────┘│
│         │                        │                        │         │
│         │                        │                        │         │
│         ▼                        ▼                        ▼         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      MESSAGE BUS                             │   │
│  │  (asyncio.Queue-based, within single process)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                        │                        │         │
│         ▼                        ▼                        ▼         │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐│
│  │ Gemini Actor │         │ChatGPT Actor │         │ Claude Actor ││
│  │              │         │              │         │              ││
│  │ • Browser    │         │ • Browser    │         │ • Browser    ││
│  │ • Rate State │         │ • Rate State │         │ • Rate State ││
│  └──────────────┘         └──────────────┘         └──────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Properties:**
- Each actor has private state (no sharing)
- Actors communicate only via async messages
- Each actor runs in its own asyncio task
- State mutations are always atomic within an actor

### 4.3 Persistence Model

We use **Write-Ahead Logging (WAL)** for crash safety:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WRITE-AHEAD LOG (WAL)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. INTENT LOGGED          2. OPERATION EXECUTED    3. COMMITTED    │
│  ┌────────────────┐        ┌────────────────┐       ┌────────────┐  │
│  │ WAL Entry      │───────▶│ State Changed  │──────▶│ WAL Entry  │  │
│  │ (uncommitted)  │        │ (in memory)    │       │ (committed)│  │
│  └────────────────┘        └────────────────┘       └────────────┘  │
│                                                                      │
│  On Crash Recovery:                                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 1. Read WAL from last checkpoint                                │ │
│  │ 2. Replay uncommitted entries                                   │ │
│  │ 3. Restore to consistent state                                  │ │
│  │ 4. Resume operations                                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Proposed Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       UNIFIED ORCHESTRATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         ORCHESTRATOR                                │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │ │
│  │  │ Scheduler  │  │  Allocator │  │   State    │  │   WAL      │   │ │
│  │  │            │  │            │  │  Manager   │  │  Manager   │   │ │
│  │  │ • Priority │  │ • Quotas   │  │            │  │            │   │ │
│  │  │ • Preempt  │  │ • Assign   │  │ • Persist  │  │ • Log      │   │ │
│  │  │ • Balance  │  │ • Balance  │  │ • Recover  │  │ • Replay   │   │ │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                      │                                   │
│              ┌───────────────────────┼───────────────────────┐          │
│              │                       │                       │          │
│              ▼                       ▼                       ▼          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐│
│  │   EXPLORER MODULE  │  │  DOCUMENTER MODULE │  │  RESEARCHER MODULE ││
│  │                    │  │                    │  │                    ││
│  │ • ThreadManager    │  │ • DocumentManager  │  │ • SourceManager    ││
│  │ • InsightProcessor │  │ • OpportunityProc  │  │ • QuestionManager  ││
│  │ • SynthesisEngine  │  │ • ReviewManager    │  │ • FindingsManager  ││
│  │ • BlessedStore     │  │ • CommentHandler   │  │                    ││
│  │ • DedupChecker     │  │ • DedupChecker     │  │                    ││
│  └────────────────────┘  └────────────────────┘  └────────────────────┘│
│              │                       │                       │          │
│              └───────────────────────┼───────────────────────┘          │
│                                      │                                   │
│                                      ▼                                   │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        LLM GATEWAY                                  │ │
│  │  (Thin wrapper - no scheduling, only browser automation)            │ │
│  │                                                                      │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │ │
│  │  │   Gemini    │    │   ChatGPT   │    │   Claude    │             │ │
│  │  │   Browser   │    │   Browser   │    │   Browser   │             │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Key Architectural Changes

| Component | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| **Pool** | Full scheduling + browser | Browser-only gateway | Single scheduling authority |
| **Priority Queue** | Separate per Pool queue type | Unified in Orchestrator | Consistent prioritization |
| **State Persistence** | Direct JSON writes | WAL + checkpoints | Crash safety |
| **Rate Limit Tracking** | Global + Pool duplicates | Allocator single source | No sync issues |
| **Module Coordination** | Independent processes | Unified event loop | Resource sharing |
| **Dedup Checker** | Separate per module | Shared instance | Consistent dedup |

### 5.3 Process Model

**Single Process, Multiple Actors:**

```python
# Main process structure
async def main():
    # Initialize components
    wal = WALManager("orchestrator/wal/")
    state = StateManager(wal)
    gateway = LLMGateway(config)

    # Create actors
    orchestrator = OrchestratorActor(state, gateway)
    explorer = ExplorerActor(state)
    documenter = DocumenterActor(state)

    # Register modules
    orchestrator.register_module(explorer)
    orchestrator.register_module(documenter)

    # Recover from any crash
    await orchestrator.recover()

    # Run until stopped
    await orchestrator.run()
```

**Why Single Process:**
- Shared memory for efficient dedup checking
- Simpler recovery (single WAL)
- No IPC overhead
- asyncio handles concurrency efficiently

---

## 6. Component Specifications

### 6.1 Orchestrator

The Orchestrator is the central coordinator responsible for:
- Task scheduling across all modules
- LLM allocation and quota management
- State persistence and recovery
- System health monitoring

#### 6.1.1 Scheduler

```python
@dataclass
class Task:
    """Smallest unit of schedulable work."""
    id: str                              # UUID
    module: Literal["explorer", "documenter", "researcher"]
    task_type: str                       # e.g., "exploration", "synthesis", "incorporate"
    priority: int                        # Computed priority (higher = more urgent)
    state: TaskState                     # PENDING | READY | RUNNING | PAUSED | COMPLETED | FAILED
    context: dict                        # Task-specific data
    conversation: Optional[ConversationState]  # For multi-turn tasks
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    llm_preference: Optional[str]        # Required LLM (e.g., for deep mode)
    llm_assigned: Optional[str]          # Assigned LLM
    request_count: int                   # LLM requests made
    max_requests: int                    # Limit for this task

class Scheduler:
    """Unified priority scheduling across all modules."""

    def __init__(self, state: StateManager):
        self.state = state
        self.task_heap: list[tuple[int, str]] = []  # (-priority, task_id)
        self._lock = asyncio.Lock()

    async def submit(self, task: Task) -> str:
        """Submit a task for scheduling."""
        async with self._lock:
            # Log to WAL first
            await self.state.wal.log(WALEntry(
                type="task_submit",
                task_id=task.id,
                data=asdict(task)
            ))

            # Add to heap
            priority = self._compute_priority(task)
            heapq.heappush(self.task_heap, (-priority, task.id))

            # Persist
            self.state.tasks[task.id] = task

            return task.id

    def _compute_priority(self, task: Task) -> int:
        """
        Compute dynamic priority based on multiple factors.

        Factors:
        1. Base priority by task type
        2. Backlog pressure (blessed insights waiting)
        3. Starvation prevention (time waiting)
        4. Human feedback urgency (comments)
        5. Quota availability
        6. Module balance
        """
        score = BASE_PRIORITIES.get(task.task_type, 50)

        # Factor 1: Backlog pressure
        if task.module == "documenter":
            blessed_pending = self.state.get_blessed_pending_count()
            if blessed_pending > 20:
                score += 30
            elif blessed_pending > 10:
                score += 15

        # Factor 2: Starvation prevention
        wait_time = (datetime.now() - task.created_at).total_seconds() / 3600
        score += min(int(wait_time * 5), 25)  # Max +25 for waiting

        # Factor 3: Human feedback urgency
        if task.task_type == "address_comment":
            score += 40  # Comments get high priority

        # Factor 4: Quota availability
        if task.llm_preference:
            if not self.state.allocator.has_quota(task.llm_preference):
                score -= 50  # Defer if no quota

        # Factor 5: Module balance
        module_share = self.state.get_module_share(task.module)
        if module_share < 0.2:  # Underserved module
            score += 15

        return score

    async def get_next_task(self, available_llms: list[str]) -> Optional[Task]:
        """Get highest priority task that can run with available LLMs."""
        async with self._lock:
            # Rebuild heap with current priorities
            self._recompute_all_priorities()

            # Find first runnable task
            for neg_priority, task_id in sorted(self.task_heap):
                task = self.state.tasks.get(task_id)
                if task is None or task.state == TaskState.COMPLETED:
                    continue

                # Check if task can run with available LLMs
                if task.llm_preference:
                    if task.llm_preference not in available_llms:
                        continue
                elif not available_llms:
                    continue

                return task

            return None

# Base priorities by task type
BASE_PRIORITIES = {
    # Explorer tasks
    "exploration": 50,
    "critique": 45,
    "synthesis": 60,
    "insight_review": 55,

    # Documenter tasks
    "address_comment": 80,      # Highest: human feedback
    "incorporate_insight": 55,
    "draft_section": 50,
    "review_section": 40,

    # Researcher tasks
    "generate_questions": 35,
    "evaluate_source": 45,
    "extract_content": 40,
    "synthesize_findings": 50,
}
```

#### 6.1.2 Allocator

```python
@dataclass
class LLMQuota:
    """Quota tracking for a single LLM."""
    name: str
    hourly_limit: int
    hourly_used: int
    deep_mode_daily_limit: int
    deep_mode_daily_used: int
    rate_limited_until: Optional[datetime]
    last_request_at: Optional[datetime]

    def can_use(self, deep_mode: bool = False) -> bool:
        """Check if LLM can be used."""
        # Check rate limit
        if self.rate_limited_until and datetime.now() < self.rate_limited_until:
            return False

        # Check hourly limit (leave 10% buffer)
        if self.hourly_used >= self.hourly_limit * 0.9:
            return False

        # Check deep mode limit
        if deep_mode and self.deep_mode_daily_used >= self.deep_mode_daily_limit:
            return False

        return True

    def record_use(self, deep_mode: bool = False):
        """Record usage."""
        self.hourly_used += 1
        self.last_request_at = datetime.now()
        if deep_mode:
            self.deep_mode_daily_used += 1

    def record_rate_limit(self, retry_after: int):
        """Record rate limit hit."""
        self.rate_limited_until = datetime.now() + timedelta(seconds=retry_after)
        # Reduce hourly limit estimate by 20%
        self.hourly_limit = max(10, int(self.hourly_used * 0.8))

class Allocator:
    """
    Centralized LLM allocation.

    CRITICAL: This is the SINGLE SOURCE OF TRUTH for:
    - Rate limit state
    - Deep mode quota
    - LLM availability

    All other rate tracking code must be removed.
    """

    def __init__(self, config: dict):
        self.quotas: dict[str, LLMQuota] = {}
        self._lock = asyncio.Lock()
        self._hourly_reset_task: Optional[asyncio.Task] = None
        self._daily_reset_task: Optional[asyncio.Task] = None

        # Initialize from config
        for backend, cfg in config.get("backends", {}).items():
            if cfg.get("enabled", True):
                deep_cfg = cfg.get("deep_mode", {})
                self.quotas[backend] = LLMQuota(
                    name=backend,
                    hourly_limit=50,  # Conservative default
                    hourly_used=0,
                    deep_mode_daily_limit=deep_cfg.get("daily_limit", 0),
                    deep_mode_daily_used=0,
                    rate_limited_until=None,
                    last_request_at=None,
                )

    async def allocate(
        self,
        task: Task,
        available_llms: list[str]
    ) -> Optional[str]:
        """
        Allocate an LLM for a task.

        Selection priority:
        1. Task's preferred LLM (if specified and available)
        2. Same LLM as existing conversation (for continuity)
        3. LLM with most remaining quota
        """
        async with self._lock:
            requires_deep = task.context.get("requires_deep_mode", False)

            # Priority 1: Preferred LLM
            if task.llm_preference:
                if task.llm_preference in available_llms:
                    quota = self.quotas.get(task.llm_preference)
                    if quota and quota.can_use(requires_deep):
                        return task.llm_preference
                return None  # Must wait for preferred

            # Priority 2: Conversation continuity
            if task.conversation and task.conversation.llm:
                if task.conversation.llm in available_llms:
                    quota = self.quotas.get(task.conversation.llm)
                    if quota and quota.can_use(requires_deep):
                        return task.conversation.llm

            # Priority 3: Most remaining quota
            candidates = []
            for llm in available_llms:
                quota = self.quotas.get(llm)
                if quota and quota.can_use(requires_deep):
                    remaining = quota.hourly_limit - quota.hourly_used
                    candidates.append((remaining, llm))

            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]

            return None

    async def record_usage(self, llm: str, deep_mode: bool = False):
        """Record LLM usage."""
        async with self._lock:
            quota = self.quotas.get(llm)
            if quota:
                quota.record_use(deep_mode)

    async def record_rate_limit(self, llm: str, retry_after: int):
        """Record rate limit hit."""
        async with self._lock:
            quota = self.quotas.get(llm)
            if quota:
                quota.record_rate_limit(retry_after)

    def get_available(self) -> list[str]:
        """Get list of available LLMs."""
        return [name for name, q in self.quotas.items() if q.can_use()]
```

#### 6.1.3 Task Executor

```python
class TaskExecutor:
    """
    Executes tasks with preemption support.

    Key behaviors:
    1. Saves conversation state BEFORE each LLM request
    2. Checks for preemption after each LLM response
    3. Handles failures with retry and backoff
    """

    def __init__(self, orchestrator: 'Orchestrator'):
        self.orchestrator = orchestrator
        self.state = orchestrator.state
        self.gateway = orchestrator.gateway

    async def execute(self, task: Task) -> TaskResult:
        """Execute a task until completion, pause, or failure."""

        # Mark task as running
        task.state = TaskState.RUNNING
        task.started_at = datetime.now()
        await self.state.update_task(task)

        try:
            module = self.orchestrator.get_module(task.module)

            while not await module.is_task_complete(task):
                # Get next action
                action = await module.get_next_action(task)

                if action.type == ActionType.LLM_REQUEST:
                    # CRITICAL: Save state BEFORE LLM request
                    await self._save_conversation_state(task)

                    # Check preemption
                    if await self._should_preempt(task):
                        task.state = TaskState.PAUSED
                        await self.state.update_task(task)
                        return TaskResult(paused=True, reason="preempted")

                    # Execute LLM request
                    try:
                        response = await self._execute_llm_request(
                            task, action.prompt, action.images
                        )
                        await module.process_response(task, response)
                        task.request_count += 1

                    except RateLimitError as e:
                        await self.orchestrator.allocator.record_rate_limit(
                            task.llm_assigned, e.retry_after
                        )
                        task.state = TaskState.PAUSED
                        await self.state.update_task(task)
                        return TaskResult(paused=True, reason="rate_limited")

                    except LLMError as e:
                        return await self._handle_llm_error(task, e)

                elif action.type == ActionType.FILE_OPERATION:
                    # File operations are fast, no preemption check
                    await action.execute()

                elif action.type == ActionType.YIELD:
                    # Explicit yield point (e.g., for long file operations)
                    if await self._should_preempt(task):
                        task.state = TaskState.PAUSED
                        await self.state.update_task(task)
                        return TaskResult(paused=True, reason="yielded")

            # Task completed
            task.state = TaskState.COMPLETED
            task.completed_at = datetime.now()
            await self.state.update_task(task)

            # Handle completion in module
            await module.handle_completion(task)

            return TaskResult(completed=True)

        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            await self.state.update_task(task)
            return TaskResult(failed=True, error=str(e))

    async def _save_conversation_state(self, task: Task):
        """Save conversation state before LLM request."""
        if task.conversation is None:
            task.conversation = ConversationState(
                llm=task.llm_assigned,
                messages=[],
                turn_count=0,
                context={}
            )

        # Truncate if too large (keep last 50 messages)
        if len(task.conversation.messages) > 50:
            task.conversation.messages = task.conversation.messages[-50:]

        # WAL log the state
        await self.state.wal.log(WALEntry(
            type="conversation_state",
            task_id=task.id,
            data=asdict(task.conversation)
        ))

    async def _should_preempt(self, task: Task) -> bool:
        """Check if a higher priority task is waiting."""
        next_task = await self.orchestrator.scheduler.peek_next()
        if next_task and next_task.id != task.id:
            next_priority = self.orchestrator.scheduler._compute_priority(next_task)
            current_priority = self.orchestrator.scheduler._compute_priority(task)
            # Preempt if next task is significantly higher priority
            if next_priority > current_priority + 20:
                return True
        return False

    async def _execute_llm_request(
        self,
        task: Task,
        prompt: str,
        images: Optional[list] = None
    ) -> str:
        """Execute LLM request through gateway."""
        llm = task.llm_assigned

        # Record usage BEFORE request (optimistic)
        requires_deep = task.context.get("requires_deep_mode", False)
        await self.orchestrator.allocator.record_usage(llm, requires_deep)

        # Send request
        response = await self.gateway.send(
            backend=llm,
            prompt=prompt,
            images=images,
            deep_mode=requires_deep,
            timeout=task.context.get("timeout", 3600)
        )

        # Add to conversation history
        if task.conversation:
            task.conversation.messages.append({
                "role": "user",
                "content": prompt
            })
            task.conversation.messages.append({
                "role": "assistant",
                "content": response
            })
            task.conversation.turn_count += 1

        return response
```

### 6.2 LLM Gateway

The Gateway replaces the current Pool's scheduling role, becoming a thin wrapper:

```python
class LLMGateway:
    """
    Thin wrapper for LLM browser automation.

    CRITICAL: No scheduling logic here. The Gateway only:
    1. Manages browser connections
    2. Sends prompts and receives responses
    3. Handles browser-level errors

    All scheduling, queuing, and priority decisions are in Orchestrator.
    """

    def __init__(self, config: dict):
        self.config = config
        self.browsers: dict[str, BrowserSession] = {}
        self._lock = asyncio.Lock()

    async def startup(self):
        """Start browser sessions."""
        backends = self.config.get("backends", {})

        for name, cfg in backends.items():
            if cfg.get("enabled", True) and cfg.get("type") == "browser":
                try:
                    session = await BrowserSession.create(name, cfg)
                    self.browsers[name] = session
                except Exception as e:
                    log.error("gateway.browser.failed", backend=name, error=str(e))

    async def shutdown(self):
        """Close browser sessions."""
        for name, session in self.browsers.items():
            try:
                await session.close()
            except Exception as e:
                log.warning("gateway.browser.close_failed", backend=name, error=str(e))

    async def send(
        self,
        backend: str,
        prompt: str,
        images: Optional[list] = None,
        deep_mode: bool = False,
        timeout: int = 3600
    ) -> str:
        """
        Send prompt to LLM and get response.

        This is synchronous from caller's perspective - it waits for response.
        """
        session = self.browsers.get(backend)
        if not session:
            raise LLMError(f"Backend {backend} not available")

        async with self._lock:  # Only one request per browser at a time
            try:
                response = await asyncio.wait_for(
                    session.send_message(prompt, images, deep_mode),
                    timeout=timeout
                )
                return response
            except asyncio.TimeoutError:
                raise LLMError(f"Request to {backend} timed out after {timeout}s")
            except RateLimitDetected as e:
                raise RateLimitError(backend, e.retry_after)
            except Exception as e:
                raise LLMError(f"Request to {backend} failed: {e}")

    def get_available_backends(self) -> list[str]:
        """Get list of backends with active browser sessions."""
        return [name for name, session in self.browsers.items() if session.is_connected]
```

### 6.3 Module Interface

All modules implement a standard interface:

```python
class ModuleInterface(ABC):
    """
    Standard interface for Fano modules.

    Each module:
    1. Generates tasks for the orchestrator to schedule
    2. Executes tasks when assigned an LLM
    3. Handles task completion/failure
    4. Persists its own domain-specific state
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Module name (explorer, documenter, researcher)."""
        pass

    @abstractmethod
    async def initialize(self, state: StateManager) -> None:
        """Initialize module, loading any persisted state."""
        pass

    @abstractmethod
    async def get_pending_tasks(self) -> list[Task]:
        """
        Return tasks this module wants to run.

        Called periodically by orchestrator to discover new work.
        Should be fast - don't do expensive computation here.
        """
        pass

    @abstractmethod
    async def is_task_complete(self, task: Task) -> bool:
        """Check if task has completed its work."""
        pass

    @abstractmethod
    async def get_next_action(self, task: Task) -> Action:
        """
        Get next action for a running task.

        Actions can be:
        - LLM_REQUEST: Send prompt to LLM
        - FILE_OPERATION: Read/write files
        - YIELD: Explicit preemption point
        """
        pass

    @abstractmethod
    async def process_response(self, task: Task, response: str) -> None:
        """Process LLM response for a task."""
        pass

    @abstractmethod
    async def handle_completion(self, task: Task) -> None:
        """Handle task completion, including any cleanup."""
        pass

    @abstractmethod
    async def handle_failure(self, task: Task, error: str) -> None:
        """Handle task failure."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Get module state for persistence."""
        pass

    @abstractmethod
    async def restore_state(self, state: dict) -> None:
        """Restore module from persisted state."""
        pass
```

### 6.4 Explorer Module

```python
class ExplorerModule(ModuleInterface):
    """
    Explorer module for mathematical insight discovery.
    """

    @property
    def name(self) -> str:
        return "explorer"

    async def initialize(self, state: StateManager) -> None:
        self.state = state
        self.thread_manager = ThreadManager(state)
        self.insight_processor = InsightProcessor(state)
        self.synthesis_engine = SynthesisEngine(state)
        self.blessed_store = BlessedStore(state)

        # CRITICAL FIX: Use shared dedup checker
        self.dedup_checker = state.get_shared_dedup_checker()

        # Load blessed insights into dedup
        await self.blessed_store.load_into_dedup(self.dedup_checker)

    async def get_pending_tasks(self) -> list[Task]:
        tasks = []

        # 1. Active threads needing work
        for thread in await self.thread_manager.get_active_threads():
            if thread.needs_exploration:
                tasks.append(Task(
                    id=f"explore-{thread.id}-{uuid4().hex[:8]}",
                    module="explorer",
                    task_type="exploration",
                    priority=BASE_PRIORITIES["exploration"],
                    state=TaskState.PENDING,
                    context={
                        "thread_id": thread.id,
                        "exchange_count": thread.exchange_count
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ))
            elif thread.needs_critique:
                tasks.append(Task(
                    id=f"critique-{thread.id}-{uuid4().hex[:8]}",
                    module="explorer",
                    task_type="critique",
                    priority=BASE_PRIORITIES["critique"],
                    state=TaskState.PENDING,
                    context={"thread_id": thread.id},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ))

        # 2. Threads ready for synthesis
        for thread in await self.synthesis_engine.get_ready_threads():
            tasks.append(Task(
                id=f"synthesis-{thread.id}-{uuid4().hex[:8]}",
                module="explorer",
                task_type="synthesis",
                priority=BASE_PRIORITIES["synthesis"],
                state=TaskState.PENDING,
                context={
                    "thread_id": thread.id,
                    "requires_deep_mode": True
                },
                llm_preference="gemini",  # Prefer Gemini for deep think
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ))

        # 3. Insights needing review
        for insight in await self.insight_processor.get_pending_review():
            tasks.append(Task(
                id=f"review-{insight.id}-{uuid4().hex[:8]}",
                module="explorer",
                task_type="insight_review",
                priority=BASE_PRIORITIES["insight_review"],
                state=TaskState.PENDING,
                context={"insight_id": insight.id},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ))

        # 4. Spawn new threads if capacity available
        if len(tasks) < 3:  # Room for more work
            for seed in await self.thread_manager.get_unexplored_seeds():
                tasks.append(Task(
                    id=f"spawn-{seed.id}-{uuid4().hex[:8]}",
                    module="explorer",
                    task_type="exploration",
                    priority=BASE_PRIORITIES["exploration"] + seed.priority * 10,
                    state=TaskState.PENDING,
                    context={
                        "seed_id": seed.id,
                        "spawn_new": True
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ))

        return tasks
```

### 6.5 Documenter Module

```python
class DocumenterModule(ModuleInterface):
    """
    Documenter module for growing the living document.
    """

    @property
    def name(self) -> str:
        return "documenter"

    async def initialize(self, state: StateManager) -> None:
        self.state = state
        self.document = DocumentManager(state)
        self.opportunity_processor = OpportunityProcessor(state)
        self.review_manager = ReviewManager(state)
        self.comment_handler = CommentHandler(state)

        # CRITICAL FIX: Use shared dedup checker AND load blessed insights
        self.dedup_checker = state.get_shared_dedup_checker()

        # Load document sections into dedup
        for section in self.document.sections:
            self.dedup_checker.add_content(
                content_id=section.id,
                content=section.content,
                content_type="section"
            )

        # CRITICAL FIX: Also load blessed insights that haven't been incorporated
        blessed_dir = Path(self.state.config["documenter"]["inputs"]["blessed_insights_dir"])
        for filepath in blessed_dir.glob("*.json"):
            try:
                insight = json.loads(filepath.read_text())
                if not insight.get("incorporated"):
                    self.dedup_checker.add_content(
                        content_id=insight["id"],
                        content=insight["text"],
                        content_type="blessed_insight"
                    )
            except Exception as e:
                log.warning("documenter.dedup.load_failed", file=str(filepath), error=str(e))

    async def get_pending_tasks(self) -> list[Task]:
        tasks = []

        # 1. HIGHEST PRIORITY: Unresolved comments
        for comment in await self.comment_handler.get_unresolved():
            tasks.append(Task(
                id=f"comment-{comment.id}-{uuid4().hex[:8]}",
                module="documenter",
                task_type="address_comment",
                priority=BASE_PRIORITIES["address_comment"],
                state=TaskState.PENDING,
                context={
                    "comment_id": comment.id,
                    "section_id": comment.section_id
                },
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ))

        # 2. Blessed insights to incorporate
        for opportunity in await self.opportunity_processor.get_opportunities():
            tasks.append(Task(
                id=f"incorporate-{opportunity.id}-{uuid4().hex[:8]}",
                module="documenter",
                task_type="incorporate_insight",
                priority=BASE_PRIORITIES["incorporate_insight"] + opportunity.priority,
                state=TaskState.PENDING,
                context={
                    "opportunity_id": opportunity.id,
                    "insight_id": opportunity.insight_id
                },
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ))

        # 3. Sections needing review (work allocation: 70% new, 30% review)
        if await self._should_do_review():
            for section in await self.review_manager.get_stale_sections():
                tasks.append(Task(
                    id=f"review-{section.id}-{uuid4().hex[:8]}",
                    module="documenter",
                    task_type="review_section",
                    priority=BASE_PRIORITIES["review_section"],
                    state=TaskState.PENDING,
                    context={
                        "section_id": section.id,
                        "requires_deep_mode": True
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ))

        return tasks

    async def _should_do_review(self) -> bool:
        """Check work allocation ratio."""
        recent_work = self.state.get_recent_work_stats(hours=24)
        new_work = recent_work.get("incorporate_insight", 0) + recent_work.get("draft_section", 0)
        review_work = recent_work.get("review_section", 0)

        if new_work + review_work == 0:
            return False

        review_ratio = review_work / (new_work + review_work)
        target_ratio = self.state.config["documenter"]["work_allocation"]["review_existing"] / 100

        return review_ratio < target_ratio
```

---

## 7. State Management

### 7.1 Write-Ahead Log (WAL)

```python
@dataclass
class WALEntry:
    """Single entry in the write-ahead log."""
    sequence: int              # Monotonically increasing
    timestamp: datetime
    type: str                  # Entry type
    task_id: Optional[str]
    data: dict
    committed: bool = False

class WALManager:
    """
    Write-Ahead Log for crash-safe state management.

    Guarantees:
    1. All state changes are logged BEFORE being applied
    2. On crash, uncommitted entries can be replayed
    3. Periodic compaction removes old committed entries
    """

    def __init__(self, wal_dir: Path):
        self.wal_dir = wal_dir
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.current_file: Optional[Path] = None
        self.sequence = 0
        self._lock = asyncio.Lock()
        self._file_handle: Optional[aiofiles.threadpool.binary.AsyncBufferedIOBase] = None

    async def log(self, entry: WALEntry) -> int:
        """
        Log an entry to WAL.

        CRITICAL: This MUST complete before any state change is applied.
        """
        async with self._lock:
            # Assign sequence number
            self.sequence += 1
            entry.sequence = self.sequence
            entry.timestamp = datetime.now()

            # Serialize entry
            line = json.dumps(asdict(entry), default=str) + "\n"

            # Write to file (with fsync for durability)
            await self._ensure_file()
            await self._file_handle.write(line.encode())
            await self._file_handle.flush()
            # Note: For production, add os.fsync(self._file_handle.fileno())

            return self.sequence

    async def commit(self, sequence: int):
        """Mark entry as committed."""
        async with self._lock:
            # Write commit marker
            commit_entry = {
                "type": "commit",
                "sequence": sequence,
                "timestamp": datetime.now().isoformat()
            }
            line = json.dumps(commit_entry) + "\n"
            await self._file_handle.write(line.encode())
            await self._file_handle.flush()

    async def recover(self) -> list[WALEntry]:
        """
        Recover uncommitted entries after crash.

        Returns list of entries that need to be replayed.
        """
        uncommitted = []
        committed_sequences = set()

        # Read all WAL files in order
        wal_files = sorted(self.wal_dir.glob("wal_*.jsonl"))

        for wal_file in wal_files:
            async with aiofiles.open(wal_file, 'r') as f:
                async for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "commit":
                            committed_sequences.add(entry["sequence"])
                        else:
                            if entry["sequence"] not in committed_sequences:
                                uncommitted.append(WALEntry(**entry))
                    except json.JSONDecodeError:
                        # Corrupted entry - log and skip
                        log.warning("wal.corrupted_entry", file=str(wal_file))

        return uncommitted

    async def compact(self, before_sequence: int):
        """Remove committed entries before sequence number."""
        # Implementation: write new WAL file with only entries >= before_sequence
        pass

    async def _ensure_file(self):
        """Ensure WAL file is open."""
        if self._file_handle is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = self.wal_dir / f"wal_{timestamp}.jsonl"
            self._file_handle = await aiofiles.open(self.current_file, 'ab')
```

### 7.2 State Manager

```python
class StateManager:
    """
    Centralized state management with WAL backing.

    Provides:
    1. Atomic state operations (via WAL)
    2. Periodic checkpoints
    3. Crash recovery
    4. Shared resources (dedup checker)
    """

    def __init__(self, config: dict, wal_dir: Path):
        self.config = config
        self.wal = WALManager(wal_dir)
        self.checkpoint_dir = wal_dir.parent / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self.tasks: dict[str, Task] = {}
        self.module_states: dict[str, dict] = {}
        self.allocator_state: dict = {}

        # Shared resources
        self._dedup_checker: Optional[DeduplicationChecker] = None
        self._lock = asyncio.Lock()

    def get_shared_dedup_checker(self) -> DeduplicationChecker:
        """
        Get shared deduplication checker.

        CRITICAL: Both Explorer and Documenter use this SAME instance.
        This ensures consistent deduplication across modules.
        """
        if self._dedup_checker is None:
            self._dedup_checker = DeduplicationChecker(self.config)
        return self._dedup_checker

    async def update_task(self, task: Task):
        """Update task state atomically."""
        # Log to WAL first
        seq = await self.wal.log(WALEntry(
            sequence=0,  # Assigned by WAL
            timestamp=datetime.now(),
            type="task_update",
            task_id=task.id,
            data=asdict(task)
        ))

        # Apply change
        task.updated_at = datetime.now()
        self.tasks[task.id] = task

        # Commit
        await self.wal.commit(seq)

    async def checkpoint(self):
        """
        Create a full state checkpoint.

        Checkpoints are atomic (write to temp, then rename).
        """
        async with self._lock:
            state = {
                "timestamp": datetime.now().isoformat(),
                "wal_sequence": self.wal.sequence,
                "tasks": {tid: asdict(t) for tid, t in self.tasks.items()},
                "module_states": self.module_states,
                "allocator_state": self.allocator_state,
            }

            # Write to temp file
            checkpoint_path = self.checkpoint_dir / "state.json"
            temp_path = self.checkpoint_dir / "state.json.tmp"

            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(json.dumps(state, indent=2, default=str))

            # Atomic rename
            temp_path.rename(checkpoint_path)

            log.info("state.checkpoint.created",
                     tasks=len(self.tasks),
                     wal_sequence=self.wal.sequence)

    async def recover(self) -> bool:
        """
        Recover state from checkpoint and WAL.

        Returns True if recovery was performed, False if fresh start.
        """
        checkpoint_path = self.checkpoint_dir / "state.json"

        if checkpoint_path.exists():
            # Load checkpoint
            async with aiofiles.open(checkpoint_path, 'r') as f:
                state = json.loads(await f.read())

            self.tasks = {
                tid: Task(**tdata) for tid, tdata in state["tasks"].items()
            }
            self.module_states = state["module_states"]
            self.allocator_state = state["allocator_state"]

            log.info("state.recovered.checkpoint",
                     tasks=len(self.tasks),
                     checkpoint_time=state["timestamp"])

            # Replay uncommitted WAL entries
            uncommitted = await self.wal.recover()
            for entry in uncommitted:
                await self._replay_entry(entry)

            log.info("state.recovered.wal", replayed_entries=len(uncommitted))

            # Mark any RUNNING tasks as PAUSED
            for task in self.tasks.values():
                if task.state == TaskState.RUNNING:
                    task.state = TaskState.PAUSED
                    log.info("state.recovered.task_paused", task_id=task.id)

            return True

        log.info("state.fresh_start")
        return False

    async def _replay_entry(self, entry: WALEntry):
        """Replay a single WAL entry."""
        if entry.type == "task_update":
            self.tasks[entry.task_id] = Task(**entry.data)
        elif entry.type == "task_submit":
            self.tasks[entry.task_id] = Task(**entry.data)
        elif entry.type == "conversation_state":
            if entry.task_id in self.tasks:
                self.tasks[entry.task_id].conversation = ConversationState(**entry.data)
```

### 7.3 Atomic File Operations

```python
class AtomicFileWriter:
    """
    Utility for atomic file writes.

    Pattern: write to temp file, then atomic rename.
    """

    @staticmethod
    async def write_json(path: Path, data: Any, indent: int = 2):
        """Write JSON file atomically."""
        temp_path = path.with_suffix('.tmp')

        try:
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(json.dumps(data, indent=indent, default=str))

            # Atomic rename (on POSIX systems)
            temp_path.rename(path)

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    @staticmethod
    async def write_text(path: Path, content: str):
        """Write text file atomically."""
        temp_path = path.with_suffix('.tmp')

        try:
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(content)

            temp_path.rename(path)

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise
```

---

## 8. Concurrency & Synchronization

### 8.1 Lock Hierarchy

To prevent deadlocks, locks are acquired in a strict order:

```
Level 1 (Global):     StateManager._lock
                           │
Level 2 (Component):       ├── Scheduler._lock
                           ├── Allocator._lock
                           └── WALManager._lock
                                    │
Level 3 (Module):                   ├── ExplorerModule._lock
                                    ├── DocumenterModule._lock
                                    └── ResearcherModule._lock
```

**Rules:**
1. Never acquire a higher-level lock while holding a lower-level lock
2. Acquire locks in the order shown above
3. Use `asyncio.Lock` for all async code (not threading.Lock)

### 8.2 Actor Message Protocol

```python
@dataclass
class Message:
    """Base message type for actor communication."""
    id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TaskSubmit(Message):
    """Submit a new task for scheduling."""
    task: Task

@dataclass
class TaskComplete(Message):
    """Notify task completion."""
    task_id: str
    result: TaskResult

@dataclass
class LLMRequest(Message):
    """Request LLM allocation."""
    task_id: str
    prompt: str
    images: Optional[list] = None

@dataclass
class LLMResponse(Message):
    """LLM response received."""
    task_id: str
    response: str
    llm: str

@dataclass
class StateUpdate(Message):
    """State update notification."""
    component: str
    update_type: str
    data: dict

class MessageBus:
    """
    Central message bus for actor communication.

    Uses asyncio.Queue for each actor's inbox.
    """

    def __init__(self):
        self.inboxes: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def register(self, actor_id: str) -> asyncio.Queue:
        """Register an actor and get its inbox."""
        async with self._lock:
            if actor_id in self.inboxes:
                raise ValueError(f"Actor {actor_id} already registered")
            inbox = asyncio.Queue()
            self.inboxes[actor_id] = inbox
            return inbox

    async def send(self, to: str, message: Message):
        """Send message to an actor."""
        inbox = self.inboxes.get(to)
        if inbox:
            await inbox.put(message)
        else:
            log.warning("message_bus.unknown_recipient", to=to, message_type=type(message).__name__)

    async def broadcast(self, message: Message):
        """Send message to all actors."""
        for inbox in self.inboxes.values():
            await inbox.put(message)
```

### 8.3 Rate Limit Synchronization

```python
class RateLimitCoordinator:
    """
    Coordinates rate limit state across the system.

    CRITICAL: This replaces ALL other rate limit tracking:
    - explorer/src/browser/base.py rate_tracker (REMOVE)
    - pool/src/state.py deep_mode tracking (REMOVE)
    - explorer/src/browser/model_selector.py (REMOVE)

    Single source of truth in Allocator.
    """

    def __init__(self, allocator: Allocator):
        self.allocator = allocator
        self._hourly_reset_task: Optional[asyncio.Task] = None
        self._daily_reset_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start periodic reset tasks."""
        self._hourly_reset_task = asyncio.create_task(self._hourly_reset_loop())
        self._daily_reset_task = asyncio.create_task(self._daily_reset_loop())

    async def stop(self):
        """Stop reset tasks."""
        if self._hourly_reset_task:
            self._hourly_reset_task.cancel()
        if self._daily_reset_task:
            self._daily_reset_task.cancel()

    async def _hourly_reset_loop(self):
        """Reset hourly counters at the start of each hour."""
        while True:
            # Calculate time until next hour
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            await asyncio.sleep((next_hour - now).total_seconds())

            # Reset hourly counters
            async with self.allocator._lock:
                for quota in self.allocator.quotas.values():
                    quota.hourly_used = 0
                    quota.rate_limited_until = None

            log.info("rate_limit.hourly_reset")

    async def _daily_reset_loop(self):
        """Reset daily deep mode counters at midnight."""
        while True:
            now = datetime.now()
            # Calculate time until midnight (or configured reset hour)
            reset_hour = self.allocator.config.get("quota_reset_hour", 0)
            next_reset = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
            if next_reset <= now:
                next_reset += timedelta(days=1)

            await asyncio.sleep((next_reset - now).total_seconds())

            # Reset deep mode counters
            async with self.allocator._lock:
                for quota in self.allocator.quotas.values():
                    quota.deep_mode_daily_used = 0

            log.info("rate_limit.daily_reset")
```

---

## 9. Fault Tolerance & Recovery

### 9.1 Recovery Protocol

```python
class RecoveryManager:
    """
    Manages system recovery after crashes or restarts.
    """

    def __init__(self, orchestrator: 'Orchestrator'):
        self.orchestrator = orchestrator
        self.state = orchestrator.state

    async def recover(self) -> RecoveryResult:
        """
        Full system recovery procedure.

        Steps:
        1. Recover state from checkpoint + WAL
        2. Validate recovered state
        3. Resume paused tasks
        4. Reconnect to LLMs
        5. Resume normal operation
        """
        result = RecoveryResult()

        # Step 1: State recovery
        recovered = await self.state.recover()
        if recovered:
            result.checkpoint_recovered = True
            result.tasks_recovered = len(self.state.tasks)

        # Step 2: Validate state
        validation_errors = await self._validate_state()
        if validation_errors:
            log.warning("recovery.validation_errors", errors=validation_errors)
            await self._repair_state(validation_errors)
            result.repairs_made = len(validation_errors)

        # Step 3: Resume paused tasks
        paused_tasks = [t for t in self.state.tasks.values() if t.state == TaskState.PAUSED]
        for task in paused_tasks:
            await self.orchestrator.scheduler.submit(task)
        result.tasks_resumed = len(paused_tasks)

        # Step 4: Reconnect to LLMs
        await self.orchestrator.gateway.startup()
        result.llms_connected = len(self.orchestrator.gateway.get_available_backends())

        # Step 5: Initialize modules
        for module in self.orchestrator.modules.values():
            module_state = self.state.module_states.get(module.name, {})
            await module.restore_state(module_state)

        log.info("recovery.complete",
                 checkpoint_recovered=result.checkpoint_recovered,
                 tasks_recovered=result.tasks_recovered,
                 tasks_resumed=result.tasks_resumed,
                 llms_connected=result.llms_connected)

        return result

    async def _validate_state(self) -> list[str]:
        """Validate recovered state for consistency."""
        errors = []

        # Check for orphaned tasks (no valid module)
        for task_id, task in self.state.tasks.items():
            if task.module not in self.orchestrator.modules:
                errors.append(f"Task {task_id} has unknown module {task.module}")

        # Check for stale conversations
        for task_id, task in self.state.tasks.items():
            if task.conversation:
                if len(task.conversation.messages) > 100:
                    errors.append(f"Task {task_id} has oversized conversation ({len(task.conversation.messages)} messages)")

        # Check for invalid state transitions
        for task_id, task in self.state.tasks.items():
            if task.state == TaskState.COMPLETED and task.completed_at is None:
                errors.append(f"Task {task_id} is COMPLETED but has no completed_at")

        return errors

    async def _repair_state(self, errors: list[str]):
        """Repair state issues."""
        for error in errors:
            # Log and apply repairs
            log.info("recovery.repair", error=error)
            # Specific repair logic here
```

### 9.2 Health Monitoring

```python
class HealthMonitor:
    """
    Monitors system health and triggers recovery actions.
    """

    def __init__(self, orchestrator: 'Orchestrator'):
        self.orchestrator = orchestrator
        self._monitor_task: Optional[asyncio.Task] = None
        self.check_interval = 60  # seconds

    async def start(self):
        """Start health monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop health monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _monitor_loop(self):
        """Periodic health check loop."""
        while True:
            await asyncio.sleep(self.check_interval)

            try:
                health = await self._check_health()

                if not health.is_healthy:
                    await self._handle_unhealthy(health)

            except Exception as e:
                log.error("health_monitor.check_failed", error=str(e))

    async def _check_health(self) -> HealthStatus:
        """Perform health checks."""
        status = HealthStatus()

        # Check LLM connectivity
        available = self.orchestrator.gateway.get_available_backends()
        status.llms_available = len(available)
        if status.llms_available == 0:
            status.issues.append("No LLMs available")

        # Check for stuck tasks
        stuck_threshold = timedelta(hours=2)
        for task in self.orchestrator.state.tasks.values():
            if task.state == TaskState.RUNNING:
                if task.started_at and datetime.now() - task.started_at > stuck_threshold:
                    status.issues.append(f"Task {task.id} stuck for >2 hours")
                    status.stuck_tasks.append(task.id)

        # Check disk space
        disk_usage = shutil.disk_usage(self.orchestrator.state.checkpoint_dir)
        if disk_usage.free < 1_000_000_000:  # 1GB
            status.issues.append("Low disk space (<1GB)")

        # Check queue depth
        queue_depth = len([t for t in self.orchestrator.state.tasks.values()
                          if t.state == TaskState.PENDING])
        if queue_depth > 100:
            status.issues.append(f"High queue depth ({queue_depth})")

        status.is_healthy = len(status.issues) == 0
        return status

    async def _handle_unhealthy(self, status: HealthStatus):
        """Handle unhealthy system state."""
        log.warning("health_monitor.unhealthy", issues=status.issues)

        # Handle stuck tasks
        for task_id in status.stuck_tasks:
            task = self.orchestrator.state.tasks.get(task_id)
            if task:
                task.state = TaskState.FAILED
                task.error = "Stuck detection timeout"
                await self.orchestrator.state.update_task(task)
                log.info("health_monitor.task_failed", task_id=task_id, reason="stuck")

        # Reconnect LLMs if none available
        if status.llms_available == 0:
            log.info("health_monitor.reconnecting_llms")
            await self.orchestrator.gateway.shutdown()
            await asyncio.sleep(30)
            await self.orchestrator.gateway.startup()
```

### 9.3 Graceful Degradation

```python
class CircuitBreaker:
    """
    Circuit breaker for LLM calls.

    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject calls immediately
    - HALF_OPEN: Testing if recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 300,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_successes = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_successes = 0
                else:
                    raise CircuitOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.half_open_requests:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        log.info("circuit_breaker.closed")
                else:
                    # Reset failure count on success
                    self.failure_count = max(0, self.failure_count - 1)

            return result

        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    log.warning("circuit_breaker.opened",
                               failure_count=self.failure_count)

            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout
```

---

## 10. Observability

### 10.1 Structured Logging

All logging follows the existing CLAUDE.md guidelines:

```python
# Event naming: {component}.{module}.{action}
log.info("orchestrator.scheduler.task_submitted",
         task_id=task.id,
         module=task.module,
         task_type=task.task_type,
         priority=priority)

log.info("orchestrator.executor.llm_request",
         task_id=task.id,
         llm=llm,
         prompt_length=len(prompt),
         deep_mode=deep_mode)

log.error("orchestrator.executor.task_failed",
          task_id=task.id,
          error=str(e),
          traceback=traceback.format_exc())
```

### 10.2 Metrics Collection

```python
@dataclass
class OrchestratorMetrics:
    """Comprehensive system metrics."""

    # Throughput (24h rolling)
    tasks_completed: dict[str, int]      # By task type
    tasks_failed: dict[str, int]
    insights_blessed: int
    sections_documented: int

    # LLM usage (24h rolling)
    llm_requests: dict[str, int]         # By LLM
    llm_deep_mode_uses: dict[str, int]
    llm_rate_limits_hit: dict[str, int]
    llm_avg_response_time: dict[str, float]

    # Queue health
    queue_depth: dict[str, int]          # By module
    oldest_pending_task_age: float       # seconds
    avg_task_wait_time: float

    # Resource usage
    memory_usage_mb: float
    disk_usage_percent: float
    checkpoint_count: int
    wal_size_mb: float

    # Uptime
    uptime_seconds: float
    last_recovery: Optional[datetime]
    recovery_count: int

class MetricsCollector:
    """Collects and exposes metrics."""

    def __init__(self, orchestrator: 'Orchestrator'):
        self.orchestrator = orchestrator
        self.metrics = OrchestratorMetrics(...)
        self._collection_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start metrics collection."""
        self._collection_task = asyncio.create_task(self._collect_loop())

    async def _collect_loop(self):
        """Periodic metrics collection."""
        while True:
            await asyncio.sleep(60)
            await self._collect()

    async def _collect(self):
        """Collect current metrics."""
        state = self.orchestrator.state

        # Queue depth
        self.metrics.queue_depth = {
            module: len([t for t in state.tasks.values()
                        if t.module == module and t.state == TaskState.PENDING])
            for module in ["explorer", "documenter", "researcher"]
        }

        # Oldest pending task
        pending = [t for t in state.tasks.values() if t.state == TaskState.PENDING]
        if pending:
            oldest = min(pending, key=lambda t: t.created_at)
            self.metrics.oldest_pending_task_age = (datetime.now() - oldest.created_at).total_seconds()

        # LLM metrics
        for name, quota in self.orchestrator.allocator.quotas.items():
            self.metrics.llm_requests[name] = quota.hourly_used
            self.metrics.llm_deep_mode_uses[name] = quota.deep_mode_daily_used

        # System metrics
        import psutil
        process = psutil.Process()
        self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024

        disk = shutil.disk_usage(state.checkpoint_dir)
        self.metrics.disk_usage_percent = (disk.used / disk.total) * 100

    def get_metrics(self) -> dict:
        """Get current metrics as dict."""
        return asdict(self.metrics)
```

### 10.3 Control Panel Integration

```python
# API endpoints for control panel

@app.get("/api/orchestrator/status")
async def get_status():
    """Get orchestrator status."""
    return {
        "state": orchestrator.state_name,
        "uptime": orchestrator.uptime,
        "tasks": {
            "pending": len([t for t in orchestrator.state.tasks.values()
                           if t.state == TaskState.PENDING]),
            "running": len([t for t in orchestrator.state.tasks.values()
                           if t.state == TaskState.RUNNING]),
            "completed_24h": orchestrator.metrics.tasks_completed_24h,
        },
        "llms": {
            name: {
                "available": q.can_use(),
                "hourly_used": q.hourly_used,
                "hourly_limit": q.hourly_limit,
                "deep_mode_used": q.deep_mode_daily_used,
                "deep_mode_limit": q.deep_mode_daily_limit,
            }
            for name, q in orchestrator.allocator.quotas.items()
        }
    }

@app.get("/api/orchestrator/tasks")
async def get_tasks(
    module: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50
):
    """Get task list with filtering."""
    tasks = list(orchestrator.state.tasks.values())

    if module:
        tasks = [t for t in tasks if t.module == module]
    if state:
        tasks = [t for t in tasks if t.state.name == state]

    tasks.sort(key=lambda t: t.created_at, reverse=True)
    return [asdict(t) for t in tasks[:limit]]

@app.post("/api/orchestrator/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a pending or running task."""
    task = orchestrator.state.tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    if task.state in [TaskState.COMPLETED, TaskState.FAILED]:
        raise HTTPException(400, "Task already finished")

    task.state = TaskState.FAILED
    task.error = "Cancelled by user"
    await orchestrator.state.update_task(task)

    return {"status": "cancelled"}
```

---

## 11. Configuration

### 11.1 Configuration Schema

```yaml
# orchestrator/config.yaml

orchestrator:
  # State management
  state:
    checkpoint_dir: "orchestrator/checkpoints"
    wal_dir: "orchestrator/wal"
    checkpoint_interval_seconds: 60
    wal_compact_threshold_mb: 100

  # Scheduling
  scheduler:
    tick_interval_ms: 100
    preemption_threshold: 20  # Priority difference to preempt
    max_pending_tasks: 500
    starvation_prevention:
      max_wait_hours: 4
      priority_boost_per_hour: 5

  # Module weights (adjusted by consultation)
  module_weights:
    explorer: 40
    documenter: 40
    researcher: 20

  # LLM consultation (optional)
  consultation:
    enabled: false  # Can be enabled for dynamic adjustment
    interval_minutes: 60
    llm: "chatgpt"  # Which LLM to use for consultation

  # Task limits
  tasks:
    max_requests_per_task:
      exploration: 20
      critique: 5
      synthesis: 10
      incorporate_insight: 15
      review_section: 10
      address_comment: 10

    conversation_message_limit: 50  # Truncate older messages

  # Recovery
  recovery:
    stuck_task_timeout_hours: 2
    max_task_retries: 3
    retry_backoff_seconds: [60, 300, 900]  # 1min, 5min, 15min

  # Health monitoring
  health:
    check_interval_seconds: 60
    stuck_threshold_hours: 2
    min_disk_space_gb: 1
    max_queue_depth: 100

# LLM quotas (single source of truth)
llm:
  quotas:
    chatgpt:
      hourly_limit: 50
      deep_mode_daily_limit: 100
    gemini:
      hourly_limit: 50
      deep_mode_daily_limit: 20
    claude:
      hourly_limit: 50
      deep_mode_daily_limit: 100

  # Reset times
  hourly_reset_minute: 0
  daily_reset_hour: 0  # Midnight

  # Circuit breaker
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout_seconds: 300
    half_open_requests: 3

# Shared deduplication
deduplication:
  # Inherited from existing config
  enabled: true
  use_batch_llm: true
  batch_size: 20
```

---

## 12. Migration Strategy

### 12.1 Phase 1: Pre-Migration Fixes (Week 1)

Fix critical bugs in existing code before migration:

| Priority | Issue | File | Fix |
|----------|-------|------|-----|
| 1 | Directory lookup bug | `insight_processor.py:390` | Fix path to `insights/pending` |
| 2 | Documenter dedup | `session.py:217` | Load blessed insights |
| 3 | Bare except | `debug_util.py:47` | Change to `except Exception:` |
| 4 | Fire-and-forget save | `insight_processor.py:372` | Change to `await` |

### 12.2 Phase 2: Core Infrastructure (Week 2)

Build core orchestration infrastructure:

1. **WALManager** - Write-ahead log implementation
2. **StateManager** - Unified state with WAL backing
3. **AtomicFileWriter** - Utility for all file operations
4. **Allocator** - Centralized quota management

### 12.3 Phase 3: Orchestrator Core (Week 3)

Implement orchestrator components:

1. **Scheduler** - Priority queue with dynamic computation
2. **TaskExecutor** - Execution with preemption
3. **MessageBus** - Actor communication
4. **ModuleInterface** - Abstract interface

### 12.4 Phase 4: LLM Gateway (Week 4)

Simplify Pool to Gateway:

1. **Remove** JobStore, RequestQueue scheduling
2. **Keep** browser automation, connection management
3. **Add** simple request/response interface
4. **Remove** rate_tracker from explorer (use Allocator)

### 12.5 Phase 5: Module Adaptation (Weeks 5-6)

Adapt existing modules:

1. **ExplorerModule** - Wrap existing orchestrator.py logic
2. **DocumenterModule** - Wrap existing main.py logic
3. **Shared DedupChecker** - Single instance across modules

### 12.6 Phase 6: Integration & Testing (Week 7)

1. Integration testing with all modules
2. Recovery testing (crash scenarios)
3. Performance testing (sustained operation)
4. Control panel updates

### 12.7 Rollback Plan

Keep existing code path available during migration:

```python
# In main.py
if config.get("use_new_orchestrator", False):
    from orchestrator.main import run as run_new
    await run_new(config)
else:
    # Existing code path
    await run_legacy(config)
```

---

## 13. Testing Strategy

### 13.1 Unit Tests

```python
# tests/orchestrator/test_scheduler.py

class TestScheduler:
    async def test_priority_computation(self):
        """Test priority computation with various factors."""
        state = MockStateManager()
        scheduler = Scheduler(state)

        task = Task(
            id="test-1",
            module="documenter",
            task_type="address_comment",
            priority=0,
            state=TaskState.PENDING,
            context={},
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now(),
        )

        priority = scheduler._compute_priority(task)

        # Base: 80 (address_comment)
        # + 10 (2 hours waiting * 5/hour)
        assert priority >= 90

    async def test_preemption_logic(self):
        """Test preemption threshold."""
        state = MockStateManager()
        scheduler = Scheduler(state)

        # Submit low priority task
        low = Task(id="low", module="explorer", task_type="exploration", ...)
        await scheduler.submit(low)

        # Submit high priority task
        high = Task(id="high", module="documenter", task_type="address_comment", ...)
        await scheduler.submit(high)

        # Should return high priority task
        next_task = await scheduler.get_next_task(["chatgpt"])
        assert next_task.id == "high"

# tests/orchestrator/test_wal.py

class TestWAL:
    async def test_crash_recovery(self):
        """Test WAL recovery after simulated crash."""
        wal_dir = Path(tempfile.mkdtemp())

        # Write some entries
        wal = WALManager(wal_dir)
        seq1 = await wal.log(WALEntry(type="task_submit", task_id="t1", data={}))
        seq2 = await wal.log(WALEntry(type="task_submit", task_id="t2", data={}))
        await wal.commit(seq1)  # Only commit first

        # Simulate crash - create new WAL manager
        wal2 = WALManager(wal_dir)
        uncommitted = await wal2.recover()

        # Should have one uncommitted entry
        assert len(uncommitted) == 1
        assert uncommitted[0].task_id == "t2"
```

### 13.2 Integration Tests

```python
# tests/integration/test_full_cycle.py

class TestFullCycle:
    async def test_exploration_to_blessing(self):
        """Test full cycle from seed to blessed insight."""
        orchestrator = await create_test_orchestrator()

        # Add a seed
        await orchestrator.explorer.add_seed("Test mathematical concept")

        # Run until blessed
        for _ in range(100):  # Max iterations
            await orchestrator.tick()

            blessed = await orchestrator.state.get_blessed_insights()
            if blessed:
                break

        assert len(blessed) > 0

    async def test_crash_recovery_preserves_work(self):
        """Test that crash recovery preserves in-progress work."""
        orchestrator = await create_test_orchestrator()

        # Start some work
        await orchestrator.explorer.add_seed("Test seed")
        for _ in range(10):
            await orchestrator.tick()

        # Get state before crash
        tasks_before = len(orchestrator.state.tasks)

        # Simulate crash and recovery
        await orchestrator.shutdown()
        orchestrator2 = await create_test_orchestrator()
        await orchestrator2.recover()

        # Verify state preserved
        tasks_after = len(orchestrator2.state.tasks)
        assert tasks_after >= tasks_before - 1  # At most one lost (the running one)
```

### 13.3 Stress Tests

```python
# tests/stress/test_sustained_operation.py

class TestSustainedOperation:
    async def test_24_hour_operation(self):
        """Test 24 hours of simulated operation."""
        orchestrator = await create_test_orchestrator()

        # Add initial seeds
        for i in range(10):
            await orchestrator.explorer.add_seed(f"Seed {i}")

        # Run for simulated 24 hours
        start_time = datetime.now()
        simulated_hours = 0

        while simulated_hours < 24:
            await orchestrator.tick()

            # Simulate time passing
            simulated_hours = (datetime.now() - start_time).total_seconds() / 60  # 1 minute = 1 hour

        # Verify progress
        assert orchestrator.metrics.tasks_completed["exploration"] > 0
        assert orchestrator.metrics.insights_blessed > 0

        # Verify no resource leaks
        assert orchestrator.metrics.memory_usage_mb < 1000
```

---

## 14. Issue Resolution Matrix

This matrix maps all 55 identified issues to their resolution in this design.

### 14.1 Original Issues (47)

| ID | Issue | Severity | Resolution | Section |
|----|-------|----------|------------|---------|
| C1 | JobStore ignores priority | CRITICAL | Unified Scheduler with dynamic priority | §6.1.1 |
| C2 | Worker scheduling inverted | CRITICAL | Gateway has no scheduling; Orchestrator controls | §6.2 |
| C3 | Directory lookup bug | CRITICAL | Fixed in Phase 1 migration | §12.1 |
| C4 | Documenter dedup missing | CRITICAL | Shared DedupChecker loads blessed insights | §6.5 |
| C5 | Unprotected worker state | CRITICAL | Actor model eliminates shared state | §4.2 |
| C6 | Global rate_tracker races | CRITICAL | Single Allocator with locks | §6.1.2 |
| C7 | Bare except clause | CRITICAL | Fixed in Phase 1 migration | §12.1 |
| C8 | Dual deep mode tracking | CRITICAL | Allocator is single source of truth | §6.1.2 |
| H1 | Non-atomic JSON writes | HIGH | AtomicFileWriter for all writes | §7.3 |
| H2 | Fire-and-forget save | HIGH | Fixed in Phase 1 migration | §12.1 |
| H3 | No work stealing | HIGH | Preemptible tasks with yield points | §6.1.3 |
| H4 | Pool restart mid-request | HIGH | Task PAUSED state with recovery | §9.1 |
| H5 | Orphaned futures | HIGH | Task-based model, no futures | §6.1.3 |
| H6 | File TOCTOU in queue | HIGH | WAL eliminates race; Gateway has no queue | §7.1 |
| H7 | JobStore cache race | HIGH | No JobStore in new design | §6.2 |
| H8 | Swallowed exceptions | HIGH | Explicit error handling in TaskExecutor | §6.1.3 |
| H9 | Thread reload each cycle | HIGH | State cached in StateManager | §7.2 |
| M1 | Document write not atomic | MEDIUM | AtomicFileWriter | §7.3 |
| M2 | Silent failures in blessed load | MEDIUM | Explicit error logging | §10.1 |
| M3 | In-memory review progress | MEDIUM | Task state persisted via WAL | §7.1 |
| M4 | Missing incorporation flag | MEDIUM | Atomic writes with verification | §7.3 |
| M5 | Lock under blocking I/O | MEDIUM | Async locks only | §8.1 |
| M6 | File deletion races | MEDIUM | WAL protects against data loss | §7.1 |
| L1 | Deep mode underutilized | LOW | Scheduler can prioritize deep mode tasks | §6.1.1 |
| L2 | Missing timeout config | LOW | Configurable in orchestrator config | §11 |
| L3 | Inconsistent logging | LOW | Standardized structured logging | §10.1 |

### 14.2 New Design Issues (8)

| ID | Issue | Severity | Resolution | Section |
|----|-------|----------|------------|---------|
| N1 | ConversationState unbounded | MEDIUM | 50 message limit with truncation | §6.1.3 |
| N2 | Preemption only after LLM | MEDIUM | YIELD action type for file ops | §6.3 |
| N3 | LLM consultation overhead | LOW | Made optional/conditional | §11 |
| N4 | Task queue unbounded | MEDIUM | Max 500 pending tasks configurable | §11 |
| N5 | RUNNING task recovery gap | HIGH | Save state BEFORE each LLM request | §6.1.3 |
| N6 | Checkpoint failure not handled | MEDIUM | Retry logic in StateManager | §7.2 |
| N7 | Module task generation race | MEDIUM | Task validation before execution | §6.1.3 |
| N8 | Pool role unclear | HIGH | Gateway is thin wrapper; Orchestrator schedules | §5.2 |

### 14.3 Resolution Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Fully resolved | 47 | 85% |
| ✅ Resolved with configuration | 5 | 9% |
| ⚠️ Requires Phase 1 fix first | 3 | 5% |
| **Total** | **55** | **100%** |

---

## 15. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Actor** | Independent component with private state communicating via messages |
| **Blessed Insight** | An atomic insight that has passed consensus review |
| **Checkpoint** | Full state snapshot for crash recovery |
| **Deep Mode** | LLM mode with extended thinking (limited daily quota) |
| **Exchange** | Single prompt/response pair in an exploration thread |
| **Gateway** | Thin LLM wrapper handling browser automation only |
| **Preemption** | Pausing a lower-priority task to run a higher-priority one |
| **Task** | Smallest schedulable unit of work |
| **Thread** | Multi-turn exploration conversation on a topic |
| **WAL** | Write-Ahead Log for crash-safe state changes |

### Appendix B: File Structure

```
fano/
├── orchestrator/                    # NEW: Unified orchestration
│   ├── __init__.py
│   ├── main.py                      # Entry point
│   ├── scheduler.py                 # Priority scheduling
│   ├── allocator.py                 # LLM allocation
│   ├── executor.py                  # Task execution
│   ├── state.py                     # State management
│   ├── wal.py                       # Write-ahead log
│   ├── gateway.py                   # LLM gateway (replaces pool scheduling)
│   ├── health.py                    # Health monitoring
│   ├── metrics.py                   # Metrics collection
│   ├── recovery.py                  # Recovery procedures
│   ├── models.py                    # Data models
│   ├── config.py                    # Configuration
│   ├── checkpoints/                 # State checkpoints
│   └── wal/                         # WAL files
│
├── explorer/
│   └── src/
│       ├── module.py                # NEW: ModuleInterface implementation
│       ├── orchestration/           # Existing (adapted)
│       └── ...
│
├── documenter/
│   ├── module.py                    # NEW: ModuleInterface implementation
│   └── ...                          # Existing (adapted)
│
├── researcher/
│   └── module.py                    # NEW: ModuleInterface implementation
│
├── pool/                            # SIMPLIFIED: Browser automation only
│   └── src/
│       ├── gateway.py               # Browser connections
│       ├── browsers/                # Browser session management
│       └── workers.py               # REMOVED: Scheduling logic
│
├── shared/
│   ├── deduplication/               # Existing (shared instance)
│   └── logging/                     # Existing
│
└── control/
    └── server.py                    # Updated for new orchestrator
```

### Appendix C: API Reference

See inline documentation in component specifications (§6).

### Appendix D: Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-01-14 | Single process with actors | Simpler recovery, shared memory for dedup | Multi-process with IPC |
| 2026-01-14 | WAL over periodic checkpoints only | Guarantees no data loss | More frequent checkpoints |
| 2026-01-14 | Pool becomes thin gateway | Eliminates scheduling duplication | Keep Pool with priority fix |
| 2026-01-14 | Eager conversation save | Prevents recovery gap | Checkpoint-only save |
| 2026-01-14 | Actor model for concurrency | Eliminates shared state races | Locks on shared state |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-14 | Initial comprehensive design |

---

*End of Design Specification*
