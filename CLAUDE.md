# CLAUDE.md - Project Guidelines for Fano

## Project Overview

Fano is a multi-component platform for mathematical exploration and documentation:
- **Explorer**: Discovers and reviews mathematical insights using LLM collaboration
- **Documenter**: Generates and maintains a living mathematical document
- **Control**: Web UI for monitoring and managing all components
- **Pool**: Browser pool for web-based LLM interactions

## Logging Standards

**All logging must be in structured, machine-readable format (JSON) to enable automated analysis.**

### Why
- Automated agents will analyze logs to diagnose issues and suggest fixes
- Structured logs enable programmatic querying and alerting
- Performance metrics can be extracted and tracked over time

### How

Use the shared logging module which outputs structured JSON:

```python
from shared.logging import get_logger

log = get_logger("component_name", "module_name")

# Good - structured with context
log.info("document.section.generated",
    section_id="prereq-123",
    word_count=450,
    duration_ms=1234
)

# Good - errors with context
log.error("llm.request.failed",
    backend="claude",
    error_type="timeout",
    retry_count=3
)

# Bad - unstructured string
log.info("Generated section prereq-123 with 450 words in 1.2s")
```

### Log Event Naming Convention

Use dot-separated hierarchical names: `{component}.{module}.{action}`

Examples:
- `explorer.review.started`
- `documenter.section.saved`
- `control.api.request`
- `pool.browser.crashed`

### Required Fields

Every log event should include relevant context:
- Identifiers (section_id, insight_id, etc.)
- Counts and metrics (word_count, duration_ms, retry_count)
- Status information (success, error_type)

## Code Style

- Python: Follow existing patterns in the codebase
- JavaScript: Inline in HTML templates, keep functions focused
- Use explicit variable names over clever abbreviations
- Prefer clarity over brevity

## Common Commands

```bash
# Start control server
python control/server.py

# Run explorer
python explorer/src/orchestrator.py

# Run documenter
python documenter/main.py
```

## Project Structure

```
fano/
├── control/          # Web UI and API server
│   ├── server.py
│   └── templates/
├── documenter/       # Document generation
│   ├── main.py
│   ├── document/     # Generated content
│   └── formatting.py # Math/markdown fixing
├── explorer/         # Insight discovery
│   └── src/
├── shared/           # Common utilities
│   ├── logging.py
│   └── llm.py
└── pool/             # Browser automation
```
