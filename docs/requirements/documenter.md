# Fano Documenter: Requirements

## Overview

The Documenter creates and continuously updates a living mathematical document. It orchestrates LLM collaboration through the existing `llm` library to decide what to document, how to document it, and whether the documentation meets quality standards.

The document grows organically from a seed, with each addition validated through multi-LLM consensus. The goal is a document that is **natural, elegant, beautiful, interesting, and inevitable** — capturing mathematical structure that must exist, not structure we impose.

---

## Part 1: Extending the LLM Consensus Library

The existing `llm/src/consensus.py` provides `ConsensusReviewer` for validating insights. We extend it with a general-purpose `run()` method that the Documenter (and other components) can use for any consensus task.

### 1.1 New General-Purpose API

```python
# In llm/src/consensus.py

async def run(
    self,
    context: str,
    task: str,
    *,
    response_format: Optional[str] = None,  # e.g., "DECISION: [yes/no]\nREASONING: ..."
    max_rounds: int = 3,
    use_deep_mode: bool = False,
) -> ConsensusRunResult:
    """
    Run a general-purpose consensus task.

    Args:
        context: Background information for the task
        task: What the LLMs should do (in natural language)
        response_format: Optional format hint for structured responses
        max_rounds: Maximum deliberation rounds (default 3)
        use_deep_mode: Whether to use deep/pro modes

    Returns:
        ConsensusRunResult with outcome and transcript
    """
```

### 1.2 Result Structure

```python
@dataclass
class ConsensusRunResult:
    success: bool
    outcome: str              # The synthesized answer/decision/content
    converged: bool           # Did all LLMs agree?
    confidence: float         # 0.0-1.0 based on agreement strength
    rounds: list[dict]        # Full transcript of all rounds
    dissent: Optional[str]    # Minority view if not converged
    duration_seconds: float
```

### 1.3 Convergence Detection

Convergence is determined by response similarity:

1. **Explicit voting**: If `response_format` includes a decision field (e.g., "DECISION: yes/no"), count matching decisions
2. **Semantic agreement**: If responses are open-ended, use the final round to ask: "Do you all agree? Summarize the consensus."
3. **Confidence calculation**: `confidence = matching_votes / total_votes` for explicit voting, or derived from the synthesis round

### 1.4 Deliberation Process

1. **Round 1**: Send `context + task` to all available LLMs in parallel
2. **Round 2+**: Share all responses: "Here's what others said: [responses]. Continue the discussion or confirm your position."
3. **Synthesis**: After max rounds or early convergence, one LLM synthesizes: "Given this discussion, what is the final outcome?"
4. **Early exit**: If all responses in a round express the same decision, skip remaining rounds

### 1.5 Backward Compatibility

The existing `review()` method remains unchanged for Explorer compatibility. It internally uses the same deliberation logic but with domain-specific prompts and the bless/uncertain/reject rating system.

---

## Part 2: Documenter Component

### 2.1 Purpose

Creates and maintains a living mathematical document that:
- Starts from a seed (a quote about "three as first manifestation")
- Grows by incorporating blessed insights from the Explorer
- Validates all additions through LLM consensus
- Meets strict criteria for mathematical content
- Flows logically with proper dependencies
- Periodically reviews and improves existing content

### 2.2 The Seed Document

```markdown
# [Title TBD]

> "So, when it's un-manifest, it's nothing. But the first number
> is three not one, it never happened as one, always it was three,
> whichever way you look at it. From whichever perspective you
> look at it, the first manifestation is always in the form of three."
> — Sadhguru

This document explores the mathematical structure that emerges from
this principle, and its appearance across traditions.

We seek what is natural, elegant, beautiful, interesting, and
inevitable — structure that must exist, not structure we impose.
```

### 2.3 Two Separate Evaluations

The documenter distinguishes between evaluating the MATHEMATICS and evaluating the EXPOSITION:

**For mathematics (should this be included?):**
- Natural — does it arise on its own, not artificially constructed?
- Elegant — is this the simplest form?
- Beautiful — does it have aesthetic quality, rightness?
- Interesting — does it connect, illuminate, open doors?
- Inevitable — given the axioms, does this HAVE to exist?

**For exposition (is this explanation acceptable?):**
- Clear — can it be understood?
- Correct — is it accurate?
- Flows — does it follow from what precedes?
- No dangling dependencies — only references established concepts

### 2.4 Concept Tracking with Explicit Markers

The document forms a directed acyclic graph of concepts. To reliably track dependencies, we use explicit markers (not LLM extraction):

```markdown
<!-- ESTABLISHES: triangular_numbers -->
<!-- ESTABLISHES: tetrahedral_numbers -->
<!-- REQUIRES: natural_numbers -->
```

When adding new content:
1. The documenter checks that all `REQUIRES` concepts are already `ESTABLISHES`ed earlier in the document
2. New content must declare what concepts it establishes
3. The consensus task includes: "Does this content properly introduce [concept] before using it?"

### 2.5 Proof Status

The document distinguishes between:
- **Theorem**: Proven in the document
- **Observation**: Noted, awaiting proof
- **Conjecture**: Hypothesis, may or may not be provable

### 2.6 Cross-Domain Bridges

The document may include "bridges" to other domains (music, Sanskrit, Hindu cosmology) that validate mathematical findings. A bridge is appropriate when:

- A significant number has emerged from the math
- That number appears independently in another tradition
- The correspondence is precise, not vague

**Bridge timing heuristics** (at least one must apply):
- After completing a proof (natural pause)
- After establishing 3+ new concepts without a bridge
- When transitioning between major mathematical themes
- When the number is particularly significant (e.g., 108, 84)

### 2.7 Visual Elements

When consensus determines a diagram would aid understanding:
- Generate programmatically if possible (SVG, Mermaid)
- Otherwise insert placeholder: `<!-- DIAGRAM: [description] -->`

### 2.8 Context Management for Cost Control

To avoid sending the full document on every consensus call:

1. **Document summary**: Maintain a running summary (updated after each addition)
2. **Relevant sections**: Include only sections that the current task depends on
3. **Progressive expansion**: Start with summary; if LLMs need more context, expand to relevant sections
4. **Token budget**: Maximum 8000 tokens of context per consensus call

```python
def build_context(self, task_type: str, relevant_concepts: list[str]) -> str:
    """Build context appropriate for the task, respecting token budget."""
    context_parts = [self.document_summary]  # ~500 tokens

    for concept in relevant_concepts:
        section = self.get_section_establishing(concept)
        if section and self.token_count(context_parts) + len(section) < 8000:
            context_parts.append(section)

    return "\n\n---\n\n".join(context_parts)
```

### 2.9 Work Allocation

The documenter splits time between:
- **New material**: Incorporating unaddressed blessed items, extending the frontier
- **Review**: Re-reading existing sections, critiquing, improving

Configurable ratio (default 70% new, 30% review):

```yaml
documenter:
  work_allocation:
    new_material: 70
    review_existing: 30
```

### 2.10 Review Mechanism

When reviewing existing content:

1. **Select section** using priority: never_reviewed > stale (>7 days) > recently_reviewed
2. **Within each tier**: Select randomly to avoid bias
3. **Provide context**: Section text, when written, what's been learned since
4. **Consensus task**: Is this still accurate? Clear? Optimally placed? Missing connections?
5. **If improvements suggested**: Revision cycle
6. **If confirmed**: Mark as reviewed with timestamp

### 2.11 Section Metadata

Each section tracks metadata via HTML comments:

```markdown
<!-- SECTION
id: section_003
created: 2026-01-10
last_reviewed: 2026-01-15
review_count: 2
status: stable | needs_work | provisional
establishes: [triangular_numbers, tetrahedral_numbers]
requires: [natural_numbers]
-->
```

### 2.12 Author Comments

The author can add inline comments to request changes:

```markdown
<!-- COMMENT: This doesn't flow well -->
<!-- COMMENT: Should be 168 not 186 -->
```

On the next cycle:
1. Documenter detects the comment
2. Consensus task: "The author says [comment]. How should we address this?"
3. If resolved → remove comment, apply change
4. If unable to resolve → keep comment, add `attempted: true`

### 2.13 Handling Disputed Consensus

When LLMs cannot reach consensus after max rounds:

1. **For new additions**: Skip the item, log as "disputed", try again next session with different framing
2. **For reviews**: Keep existing content unchanged, log the disagreement
3. **Consecutive disputes**: After 3 consecutive disputes on the same item, flag for human review

```markdown
<!-- NEEDS_HUMAN_REVIEW: [item description] - disputed 3 times -->
```

### 2.14 Daily Snapshots

At configured time (default midnight):
1. Copy current document to `archive/YYYY-MM-DD.md`
2. Archive is append-only, never modified

### 2.15 Main Loop

```python
async def run(self):
    while not self.exhausted:

        # Check for author comments first (highest priority)
        comment = self.find_unresolved_comment()
        if comment:
            await self.address_comment(comment)
            continue

        # Decide: new work or review?
        if self.should_do_review():  # Based on configured ratio
            section = self.select_section_for_review()
            if section:
                await self.review_section(section)
                continue

        # Work on new material
        opportunity = self.find_next_opportunity()  # From blessed items
        if opportunity:
            await self.work_on_opportunity(opportunity)
            continue

        # Nothing new to do — if nothing to document, don't document
        if self.all_sections_recently_reviewed():
            log.info("documenter.exhausted", reason="no_new_material")
            break
        else:
            # Force review mode
            section = self.select_section_for_review()
            if section:
                await self.review_section(section)
```

### 2.16 Termination Conditions

The documenter exits when:
1. **Exhausted**: No blessed items to incorporate, no comments pending, all sections recently reviewed
2. **Stuck**: Same item disputed N times consecutively (configurable, default 3)
3. **Budget**: Token/cost limit reached for session
4. **Manual stop**: User terminates

On exit, log summary:
- Sections added/modified
- Items remaining
- Why it stopped

### 2.17 Integration Points

**Input — Blessed Insights:**
- Reads from `blessed_insights/` directory
- Each insight is a JSON file from Explorer
- Documenter tracks which are incorporated via `incorporated: true` field

**Output — The Document:**
- Lives at configured path (e.g., `document/main.md`)
- Updated in place
- Daily snapshots in `document/archive/`

**Logging:**
- Uses `shared/logging` module (already implemented)
- All consensus calls logged with full transcripts
- Documenter actions logged as structured events

---

## Part 3: Configuration

```yaml
# Documenter Config
documenter:
  document:
    path: document/main.md
    archive_dir: document/archive
    snapshot_time: "00:00"  # Daily at midnight

  inputs:
    blessed_insights_dir: blessed_insights/

  work_allocation:
    new_material: 70
    review_existing: 30

  review:
    max_age_days: 7  # Review sections older than this

  context:
    max_tokens: 8000  # Token budget per consensus call
    summary_update_frequency: 5  # Update summary every N additions

  termination:
    max_consecutive_disputes: 3
    max_consensus_calls_per_session: 100

  logging:
    component: documenter

# LLM Consensus Config (in existing llm config)
llm:
  consensus:
    max_rounds: 3
    use_deep_mode: false  # Default; can override per-call
```

---

## Part 4: Example Consensus Tasks

### 4.1 Deciding What to Work On

```
CONTEXT:
Document summary: [500-word summary]
Established concepts: [list]
Unincorporated blessed items:
  1. [item with its concepts]
  2. [item with its concepts]

TASK:
What should we add next? Consider:
- Which items have all their required concepts already established?
- What would flow most naturally from the current document state?
- What would be most valuable to add?

Respond with:
RECOMMENDATION: [item number or "none ready"]
REASONING: [why this is the best next step]
```

### 4.2 Evaluating Mathematics for Inclusion

```
CONTEXT:
Document summary: [summary]
Proposed mathematical content:
[the proposed math]

TASK:
Should this mathematics be included? Evaluate against these criteria:
- Natural: Does it arise on its own, not artificially constructed?
- Elegant: Is this the simplest form?
- Beautiful: Does it have aesthetic quality, rightness?
- Interesting: Does it connect, illuminate, open doors?
- Inevitable: Given the axioms, does this HAVE to exist?

Respond with:
DECISION: [INCLUDE or EXCLUDE]
NATURAL: [assessment]
ELEGANT: [assessment]
BEAUTIFUL: [assessment]
INTERESTING: [assessment]
INEVITABLE: [assessment]
REASONING: [overall justification]
```

### 4.3 Evaluating Exposition Quality

```
CONTEXT:
Document summary: [summary]
Preceding section: [the section this follows]
Established concepts: [list]

Proposed addition:
[the draft text]

TASK:
Evaluate this exposition:
- Is it clear and understandable?
- Is it correct?
- Does it flow from what precedes?
- Does it only reference established concepts (or properly introduce new ones)?

Respond with:
DECISION: [APPROVE, REVISE, or REJECT]
CLARITY: [assessment]
CORRECTNESS: [assessment]
FLOW: [assessment]
DEPENDENCIES: [any missing concepts?]
REVISION_NOTES: [if REVISE, what to fix]
```

### 4.4 Drafting Content

```
CONTEXT:
Document summary: [summary]
Preceding section: [what comes before]
Established concepts: [list]

We need to add content about: [topic/blessed item]

TASK:
Draft a section that:
- Introduces [topic] clearly
- Connects to established concepts
- Maintains neutral expository voice
- Includes proof if this is a theorem
- Specifies what concepts it ESTABLISHES and REQUIRES

Provide the draft including section metadata markers.
```

### 4.5 Deciding on a Bridge

```
CONTEXT:
Document summary: [summary]
Numbers established so far: [3, 7, 21, 108, ...]
Recent additions: [last 3 sections added]

Available bridge candidates:
- 108: Appears in [yoga tradition details]
- 84: Appears in [asana count details]

TASK:
Should we include a cross-domain bridge now?
- Is there a number we've established that a tradition validates?
- Would it reinforce the mathematical development or distract?
- Have we reached a natural pause point?

Respond with:
DECISION: [INCLUDE bridge_name, WAIT, or NO_BRIDGE]
REASONING: [justification]
PLACEMENT: [if INCLUDE, suggested placement relative to current content]
```

### 4.6 Reviewing Existing Content

```
CONTEXT:
Section to review:
[section text with metadata]

Written: [date]
Concepts known at writing: [list]
Concepts established since: [list]

TASK:
Re-evaluate this section:
- Is it still accurate given current knowledge?
- Is it still clear in current context?
- Are there connections to later material we should add?

Respond with:
DECISION: [CONFIRMED or REVISE]
ACCURACY: [still accurate?]
CLARITY: [still clear?]
CONNECTIONS: [any new connections to add?]
REVISION_NOTES: [if REVISE, specific changes]
```

### 4.7 Addressing Author Comment

```
CONTEXT:
Section with comment:
[section text]

Author's comment: "[the comment text]"

TASK:
Address the author's concern:
- If correction: verify and fix
- If question: elaborate or clarify
- If style concern: revise for clarity

Respond with:
UNDERSTOOD: [restate what the author wants]
RESOLUTION: [how to address it]
REVISED_SECTION: [the fixed section, or UNCHANGED if no fix needed]
```

---

## Part 5: File Structure

```
fano/
├── llm/
│   └── src/
│       ├── consensus.py      # Extended with run() method
│       └── models.py         # Add ConsensusRunResult
│
├── documenter/
│   ├── __init__.py
│   ├── main.py               # Entry point, main loop
│   ├── document.py           # Document loading, saving, parsing, metadata
│   ├── concepts.py           # Concept tracking (ESTABLISHES/REQUIRES)
│   ├── opportunities.py      # Finding what to work on next
│   ├── tasks.py              # Building consensus task prompts
│   ├── review.py             # Review mode logic
│   └── snapshots.py          # Daily archiving
│
├── document/
│   ├── main.md               # The living document
│   └── archive/              # Daily snapshots
│       └── YYYY-MM-DD.md
│
├── blessed_insights/         # Input from Explorer
│   └── insight_*.json
│
├── logs/                     # Uses shared/logging
│   └── documenter.jsonl
│
└── config.yaml
```

---

## Part 6: Success Criteria

The documenter is working correctly when:

1. It starts from the seed and grows the document when blessed insights are available
2. It does nothing when there's nothing to document (no blessed items, no comments)
3. All additions pass through LLM consensus with logged transcripts
4. Mathematical content meets the five criteria (natural, elegant, beautiful, interesting, inevitable)
5. The document flows logically with explicit concept dependencies (ESTABLISHES/REQUIRES)
6. Author comments are detected and addressed
7. Daily snapshots are created reliably
8. Disputed items are skipped and flagged, not forced through
9. Review mode catches and improves stale content
10. Context is managed to stay within token budgets
11. It terminates gracefully when exhausted or budget exceeded
