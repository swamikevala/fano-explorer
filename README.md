# Fano Explorer

An autonomous multi-agent research system for exploring deep mathematical connections between Fano-plane geometry, Yogic/Tantric systems, Sanskrit grammar, and Indian music theory—guided by Sadhguru's teachings.

## Philosophy

The system operates on the principle that mathematical truth should feel **discovered, not invented**. Good findings are:
- **Natural** — they arise without forcing
- **Elegant** — minimal, symmetric, beautiful
- **Inevitable** — once seen, they couldn't be otherwise
- **Decodifying** — they explain the specific numbers in the source teachings

## Key Features

- **Panel-based extraction**: All 3 LLMs (Gemini, ChatGPT, Claude) propose insights independently, then consolidate
- **Automated review panel**: 3-round LLM consensus process with deep analysis modes
- **Atomic insights**: Extracts multiple 1-3 sentence aphorisms per exploration thread
- **4-dimension scoring**: Impact, Intuition, Surprise, Sadhguru Alignment (1-10 each)
- **Mathematical verification**: DeepSeek Prover V2 validates formal claims
- **Augmentations**: Auto-generates diagrams, tables, proofs, and verification code
- **Seed aphorisms**: User-provided starting points for guided exploration
- **Deduplication**: Prevents near-duplicate insights with semantic matching

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AXIOM STORE                              │
│  Seed aphorisms, Sadhguru excerpts, target numbers,         │
│  blessed insights                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│  Exploration → Panel Extraction → Review → Augmentation     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ ChatGPT  │    │ Gemini   │    │ Claude   │
       │ Pro/5.2  │◄──►│ Deep     │◄──►│ Opus     │
       │ Thinking │    │ Think    │    │ API      │
       └──────────┘    └──────────┘    └──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 AUTOMATED REVIEW PANEL                       │
│  Round 1: Independent (standard modes)                       │
│  Round 2: Deep analysis (deep think/pro/extended thinking)  │
│  Round 3: Structured deliberation (if split)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AUGMENTATION                              │
│  Diagrams (matplotlib) • Tables • Proofs • Code             │
└─────────────────────────────────────────────────────────────┘
```

## Review Panel Process

| Round | Mode | Purpose |
|-------|------|---------|
| **Round 1** | Standard | Independent review by all 3 LLMs |
| **Round 2** | Deep Think/Pro/Extended | Deep analysis with Round 1 visible |
| **Round 3** | Deliberation | Minority argues, majority responds, final votes |

### Outcomes

| Result | Action |
|--------|--------|
| Unanimous ⚡ | Auto-bless, generate augmentations |
| Unanimous ? | Needs development |
| Unanimous ✗ | Auto-reject |
| 2-1 Split | Majority wins, flagged as disputed |

## Scoring System

Each insight is scored on 4 dimensions (1-10 scale):

| Dimension | Question |
|-----------|----------|
| **IMPACT** | If true, how much would it change our understanding? |
| **INTUITION** | How much does it "feel" true in your bones? |
| **SURPRISE** | How unexpected is this cross-domain correspondence? |
| **SADHGURU_ALIGNMENT** | Does this align with yogic/mystical teachings? |

**Total Score** = sum of all 4 (max 40). Prioritize insights with TOTAL ≥ 25.

## Setup

### Prerequisites
- Python 3.10+
- Chrome browser
- Google account (for ChatGPT/Gemini SSO)

### Environment Variables

```bash
# Required for Claude API (review panel, extraction, augmentation)
export ANTHROPIC_API_KEY="sk-ant-..."

# Required for DeepSeek math verification
export OPENROUTER_API_KEY="sk-or-..."
```

### Installation

```bash
cd fano-explorer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### First Run (Authentication)

```bash
python fano_explorer.py auth
```

This opens Chrome and lets you log into ChatGPT and Gemini manually. Sessions are saved for future runs.

### Configuration

Edit `config.yaml` to customize:

```yaml
# Exploration seed prompt
exploration:
  intro: |
    You are exploring deep mathematical connections between:
    - Fano plane incidence geometry...
  key_numbers: "3, 5, 7, 14, 16, 21, 22, 24, 36, 72, 84, 108, 112, 114"
  goals: |
    Your goal is to find mathematical structures that...

# Panel extraction (recommended)
chunking:
  use_panel_extraction: true
  max_insights_per_thread: 15
  max_final_insights: 10

# Review panel
review_panel:
  enabled: true
  claude_model: "claude-opus-4-20250514"

# Math verification
math_verification:
  enabled: true
  model: "deepseek/deepseek-prover-v2"

# Augmentation
augmentation:
  enabled: true
  auto_generate: true
```

## Axioms

### Seed Aphorisms

Add your conjectured connections to `data/axioms/seeds.yaml`:

```yaml
seeds:
  - text: "The Fano plane's 7 points correspond to the 7 chakra dimensions..."
    tags: [fano, chakras]
    confidence: high
    notes: "Starting point for exploration"
```

Seeds are treated as pre-blessed axioms and picked up automatically (no restart needed).

### Sadhguru Excerpts

Place source texts in `data/axioms/sadhguru_excerpts/*.md`:

```markdown
---
title: On the Five Elements
source: Inner Engineering, Ch. 4
tags: [pancha_bhuta, elements]
numbers: [72, 12, 5]
---

The actual excerpt content here...
```

### Target Numbers

Edit `data/axioms/target_numbers.yaml` with specific numbers to decode.

## Usage

### Start Exploration

```bash
python fano_explorer.py start
```

The system runs continuously, exploring and generating insights. Stop with `Ctrl+C`.

### Review Interface

```bash
python fano_explorer.py review
```

Opens a local web interface at `http://localhost:8765`:
- `/` — Pending insights
- `/blessed` — Blessed insights with augmentations
- `/interesting` — Needs development
- `/rejected` — Discarded insights

### Check Status

```bash
python fano_explorer.py status
```

Shows current exploration threads, pending insights, and rate limit status.

## Project Structure

```
fano-explorer/
├── fano_explorer.py          # Main entry point
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── src/
│   ├── orchestrator.py       # Main coordinator
│   ├── browser/              # Browser automation
│   │   ├── chatgpt.py        # ChatGPT interface
│   │   └── gemini.py         # Gemini interface
│   ├── models/
│   │   ├── thread.py         # Exploration threads
│   │   └── axiom.py          # Axioms, seeds, insights
│   ├── chunking/             # Atomic insight extraction
│   │   ├── panel_extractor.py  # Panel-based extraction
│   │   ├── prompts.py        # Extraction prompts
│   │   ├── deduplication.py  # Duplicate detection
│   │   └── dependencies.py   # Dependency resolution
│   ├── review_panel/         # Automated review
│   │   ├── reviewer.py       # Main coordinator
│   │   ├── round1.py         # Independent review
│   │   ├── round2.py         # Deep analysis
│   │   ├── round3.py         # Deliberation
│   │   └── claude_api.py     # Claude API client
│   ├── augmentation/         # Insight augmentation
│   │   ├── augmenter.py      # Main coordinator
│   │   └── models.py         # Augmentation types
│   └── ui/
│       └── review_server.py  # Web review interface
├── data/
│   └── axioms/
│       ├── seeds.yaml          # Your seed aphorisms
│       ├── target_numbers.yaml # Numbers to decode
│       ├── sadhguru_excerpts/  # Source texts
│       └── blessed_insights/   # Blessed findings
├── templates/
│   └── review.html             # Review interface
└── logs/
    └── exploration.log
```

## Rate Limit Handling

The system automatically handles rate limits:

| Condition | Response |
|-----------|----------|
| ChatGPT "Usage cap reached" | Backs off, switches to standard mode |
| Gemini "Deep Think overload" | Waits 10 min, retries up to 3 times |
| "Try again tomorrow" | Pauses that model until reset |

State is persisted, so you can stop and restart without losing progress.

## License

Private research tool.
