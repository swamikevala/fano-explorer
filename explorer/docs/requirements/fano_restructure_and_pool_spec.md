# Fano Platform — Restructuring & Browser Pool Service

## Overview

We're evolving from a single app (fano-explorer) to a platform of autonomous agents that share infrastructure. This document covers:

1. **Repository restructuring** — rename and reorganize
2. **Browser Pool Service** — shared LLM access layer
3. **Consensus Library** — multi-LLM validation (next phase, not this doc)

---

## Part 1: Repository Restructuring

### Current State

```
fano-explorer/
├── fano_explorer.py
├── config.yaml
├── src/
│   ├── orchestrator.py
│   ├── browser/
│   ├── models/
│   ├── storage/
│   ├── chunking/
│   ├── review_panel/
│   ├── augmentation/
│   └── ui/
├── data/
├── templates/
└── ...
```

### Target State

```
fano/
├── README.md                    # Platform overview
├── explorer/                    # The existing fano-explorer app
│   ├── fano_explorer.py         # Entry point (renamed from root)
│   ├── config.yaml
│   ├── src/
│   ├── data/
│   ├── templates/
│   └── ...
├── pool/                        # NEW: Browser pool service
│   ├── browser_pool.py          # Entry point
│   ├── config.yaml
│   ├── src/
│   └── ...
├── consensus/                   # NEW: Consensus library (Phase 2)
│   └── (placeholder for now)
├── improver/                    # NEW: Meta-improvement agent (Phase 3)
│   └── (placeholder for now)
└── shared/                      # NEW: Common utilities
    ├── __init__.py
    └── (will contain shared types, utils)
```

### Migration Steps

1. **Rename repo** from `fano-explorer` to `fano` (do this on GitHub)

2. **Create new directory structure** — add `explorer/`, `pool/`, `consensus/`, `improver/`, `shared/`

3. **Move existing code** into `explorer/` subdirectory
   - All existing files move under `explorer/`
   - Preserve git history (use `git mv`)

4. **Update imports** — anything that was `from src.X` may need adjustment based on new paths

5. **Update entry point** — `explorer/fano_explorer.py` should work from within the `explorer/` directory

6. **Test** — verify explorer still runs correctly after restructuring

### Important Notes

- The `data/` directory should stay within `explorer/` for now (it's explorer's data)
- Later, some data may become shared (e.g., blessed insights used by documenter)
- Browser session data (`browser_data/`) will eventually move to `pool/` but keep it in `explorer/` for now until pool is working

---

## Part 2: Browser Pool Service

### Purpose

A long-running service that owns and manages browser instances for LLM access. Apps connect to it via HTTP API instead of managing browsers themselves.

### Why

- **Single point of auth** — log in once, all apps benefit
- **Unified rate limiting** — one tracker, no risk of exceeding limits
- **Request queuing** — serialize access, prevent conflicts
- **Warm browsers** — no startup cost per request

### Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Communication | HTTP API (localhost) | Standard, debuggable, language-agnostic |
| Lifecycle | Long-running service | Browsers are expensive to start |
| Queueing | Priority queue per backend | Different apps have different urgency |
| Auth | Interactive on-demand | User triggers via CLI when needed |

### Backends to Support

1. **Gemini** (browser automation via Playwright)
   - Deep Think mode support
   - Rate limit detection
   
2. **ChatGPT** (browser automation via Playwright)
   - Pro mode support
   - Thinking mode support
   - Rate limit detection

3. **Claude** (API)
   - Direct API calls, no browser needed
   - Simpler, but still goes through pool for unified interface

4. **OpenRouter** (API) — for DeepSeek, other models
   - Future expansion, placeholder for now

### HTTP API Specification

#### POST /send

Send a prompt to an LLM and wait for response.

**Request:**
```json
{
    "backend": "gemini | chatgpt | claude",
    "prompt": "Your prompt text here",
    "options": {
        "deep_mode": false,
        "timeout_seconds": 300,
        "priority": "low | normal | high",
        "new_chat": true
    }
}
```

**Response (success):**
```json
{
    "success": true,
    "response": "The LLM's response text",
    "metadata": {
        "backend": "gemini",
        "deep_mode_used": false,
        "response_time_seconds": 23.5,
        "session_id": "abc123"
    }
}
```

**Response (error):**
```json
{
    "success": false,
    "error": "rate_limited | timeout | auth_required | unavailable",
    "message": "Human readable error message",
    "retry_after_seconds": 300
}
```

#### GET /status

Get status of all backends.

**Response:**
```json
{
    "gemini": {
        "available": true,
        "authenticated": true,
        "rate_limited": false,
        "rate_limit_resets_at": null,
        "queue_depth": 0,
        "deep_mode_uses_today": 5,
        "deep_mode_limit": 20
    },
    "chatgpt": {
        "available": true,
        "authenticated": true,
        "rate_limited": false,
        "rate_limit_resets_at": null,
        "queue_depth": 1,
        "pro_mode_uses_today": 12,
        "pro_mode_limit": 100
    },
    "claude": {
        "available": true,
        "authenticated": true,
        "rate_limited": false,
        "queue_depth": 0
    }
}
```

#### POST /auth/{backend}

Trigger interactive authentication for a backend.

**Response:**
```json
{
    "success": true,
    "message": "Authentication window opened. Please log in manually."
}
```

This opens a visible browser window for the user to log in. Once they close it or the pool detects successful login, auth is complete.

#### GET /health

Simple health check.

**Response:**
```json
{
    "status": "ok",
    "uptime_seconds": 3600,
    "version": "0.1.0"
}
```

### Internal Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BROWSER POOL                             │
│                                                              │
│  ┌─────────────┐                                            │
│  │  HTTP API   │◄─── incoming requests                      │
│  │  (FastAPI)  │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │  DISPATCHER │                                            │
│  │             │                                            │
│  │  • Routes to correct backend                             │
│  │  • Manages priority queue per backend                    │
│  │  • Enforces rate limits                                  │
│  └──────┬──────┘                                            │
│         │                                                    │
│    ┌────┴────┬──────────┐                                   │
│    ▼         ▼          ▼                                   │
│ ┌──────┐ ┌──────┐ ┌──────┐                                  │
│ │Gemini│ │ChatGP│ │Claude│                                  │
│ │Worker│ │Worker│ │Worker│                                  │
│ └──┬───┘ └──┬───┘ └──┬───┘                                  │
│    │        │        │                                       │
│    ▼        ▼        ▼                                       │
│ [Browser] [Browser] [API]                                    │
│                                                              │
│  ┌─────────────────────────────────────────┐                │
│  │            STATE MANAGER                 │                │
│  │                                          │                │
│  │  • Rate limit tracking                   │                │
│  │  • Deep/Pro mode daily counters          │                │
│  │  • Session persistence                   │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

1. Request arrives at HTTP API
2. Dispatcher checks if backend is available (not rate limited, authenticated)
3. If not available, return error immediately
4. If available, add to backend's priority queue
5. Worker pulls from queue (highest priority first)
6. Worker sends prompt to LLM (browser or API)
7. Worker waits for response (with timeout)
8. Worker checks for rate limit signals in response
9. Response returned to caller
10. State manager updated (rate limits, counters)

### Priority Levels

- **high**: User-initiated actions (researcher, documenter)
- **normal**: Active research (explorer during exploration)
- **low**: Background tasks (improver analysis)

Within same priority, FIFO ordering.

### Rate Limit Handling

**Detection:**
- Gemini: Look for "try again tomorrow", "quota", overload messages
- ChatGPT: Look for "You've reached the limit", "Usage cap"
- Claude: HTTP 429 response

**Response:**
- Mark backend as rate_limited
- Set retry_after timestamp
- Return error to caller with retry_after
- Auto-clear rate limit when time passes

### Deep Mode Tracking

Track daily usage of premium features:
- Gemini Deep Think: limit per day (configurable, default 20)
- ChatGPT Pro: limit per day (configurable, default 100)

When limit reached:
- Still allow requests, but `deep_mode` option is ignored
- Return `deep_mode_used: false` in metadata
- Caller can decide if that's acceptable

### Browser Session Management

**Storage:**
- Session data stored in `pool/browser_data/{backend}/`
- Separate directories for each backend
- Playwright's persistent context handles cookies/auth

**Health Checks:**
- Periodic check that browsers are still responsive
- Detect logged-out state (look for login buttons, etc.)
- If unhealthy, mark backend as `auth_required`

**Recovery:**
- If browser crashes, attempt restart
- If auth lost, mark `authenticated: false`, require `/auth/{backend}` call

### Configuration

```yaml
# pool/config.yaml

server:
  host: "127.0.0.1"
  port: 9000

backends:
  gemini:
    enabled: true
    browser_data_dir: "./browser_data/gemini"
    url: "https://gemini.google.com/app"
    response_timeout_seconds: 3600  # Deep Think can be slow
    deep_mode:
      daily_limit: 20
      reset_hour: 0  # Midnight local time

  chatgpt:
    enabled: true
    browser_data_dir: "./browser_data/chatgpt"
    url: "https://chatgpt.com"
    response_timeout_seconds: 3600
    pro_mode:
      daily_limit: 100
      reset_hour: 0

  claude:
    enabled: true
    api_key_env: "ANTHROPIC_API_KEY"
    model: "claude-sonnet-4-20250514"
    
  openrouter:
    enabled: false  # Future
    api_key_env: "OPENROUTER_API_KEY"

queue:
  max_depth_per_backend: 10  # Reject if queue too deep

browser:
  headless: false  # Set true for production
  viewport_width: 1280
  viewport_height: 720
  slow_mo: 100

logging:
  level: "INFO"
  file: "./logs/pool.log"
```

### CLI

```bash
# Start the pool service (foreground)
python browser_pool.py start

# Start in background (daemon mode) — optional, nice to have
python browser_pool.py start --daemon

# Authenticate a backend
python browser_pool.py auth gemini
python browser_pool.py auth chatgpt

# Check status
python browser_pool.py status

# Stop (if running as daemon)
python browser_pool.py stop
```

### What to Reuse from Explorer

The existing `fano-explorer/src/browser/` has working code for:
- Gemini browser automation (`gemini.py`)
- ChatGPT browser automation (`chatgpt.py`)
- Base browser class (`base.py`)
- Rate tracking (`base.py`)
- Deep mode tracking (`model_selector.py`)
- Chat logging (`base.py`)

**Approach:**
1. Copy/extract the browser automation logic
2. Adapt to work as a service (remove orchestrator coupling)
3. Add HTTP API layer on top
4. Add queue management

Don't rewrite browser automation from scratch — it's battle-tested and handles many edge cases.

### Error Handling Principles

- **Never crash the service** — catch and log errors, return error responses
- **Never block forever** — all operations have timeouts
- **Graceful degradation** — if one backend fails, others keep working
- **Clear error messages** — callers need to know what went wrong and what to do

### Testing

At minimum:
- API endpoints respond correctly
- Queue prioritization works
- Rate limit detection works
- Can send/receive from each backend (manual test with real browsers)

Nice to have:
- Mock browser backends for unit tests
- Load testing with concurrent requests

---

## Part 3: Migration Path for Explorer

Once the pool is working:

1. **Create pool client** in explorer — simple HTTP client wrapper

2. **Add pool dependency** to explorer — check pool is running at startup

3. **Replace direct browser usage** — instead of `self.gemini.send_message()`, call pool

4. **Remove browser management** from explorer — no more browser startup/auth in explorer

5. **Test thoroughly** — explorer should work exactly as before, just using pool

This is Phase 2 work. First get the pool working standalone.

---

## Questions for User

If anything is unclear during implementation, ask Swami about:

1. **Port number** — is 9000 okay, or preference for another?

2. **Auth flow** — should auth open browser in foreground always, or try headless first?

3. **Queue limits** — reject requests if queue too deep, or always accept?

4. **Startup behavior** — should pool auto-start browsers, or wait for first request?

5. **Shutdown behavior** — graceful drain of queue, or immediate stop?

---

## Success Criteria

Phase 1 (Pool) is complete when:

- [ ] Pool service starts and runs
- [ ] Can authenticate with Gemini via CLI
- [ ] Can authenticate with ChatGPT via CLI  
- [ ] Can send prompt to Gemini and get response via HTTP API
- [ ] Can send prompt to ChatGPT and get response via HTTP API
- [ ] Can send prompt to Claude and get response via HTTP API
- [ ] Rate limiting is detected and tracked
- [ ] Deep/Pro mode usage is tracked
- [ ] Priority queue works (high priority jumps queue)
- [ ] Status endpoint shows accurate state
- [ ] Service recovers gracefully from browser issues

---

## What NOT to Build Yet

- Consensus library (Phase 2)
- Explorer refactoring to use pool (Phase 2)  
- Improver (Phase 3)
- OpenRouter backend (future)
- WebSocket streaming (future, if needed)
- Multi-machine support (future, if needed)
