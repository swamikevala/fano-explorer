# Chunk Authorship and Refinement Process

## Overview

This document specifies who writes chunks at each stage and how chunks get refined during the review process. The goal is to capture improved articulations that emerge during review rather than discarding them.

---

## Authorship Roles

| Stage | Author | Rationale |
|-------|--------|-----------|
| **Exploration** | Gemini + ChatGPT | Divergent thinking, raw idea generation |
| **Initial Extraction** | Claude Opus 4.5 | Best at precise, structured articulation |
| **Refinement** | Claude Opus 4.5 | Synthesizes critiques coherently |
| **Review/Rating** | All 3 LLMs | Independent perspectives |

Claude Opus writes; all three judge. This separation prevents the author from reviewing their own work.

---

## Updated Flow

```
Exploration (Gemini ↔ ChatGPT)
    │
    ▼
Extraction (Claude Opus writes atomic chunks)
    │
    ▼
Review Round 1 (all 3 rate + provide detailed critique)
    │
    ├── All ⚡ → Bless as-is
    │
    ├── All ✗ → Reject
    │
    ├── All ? → Park as "needs development"
    │
    ├── Mixed BUT fixable → Refinement Round
    │   │
    │   ▼
    │   Claude Opus rewrites incorporating critiques
    │   │
    │   ▼
    │   Review Round 2 (all 3 rate refined version)
    │   │
    │   ├── All ⚡ → Bless refined version
    │   ├── All ✗ → Reject
    │   └── Still mixed → Deliberation (Round 3)
    │
    └── Mixed with fundamental disagreement → Deliberation (existing Round 2-3)
```

---

## Decision Criteria: Refine vs Deliberate vs Reject

After Round 1, categorize the outcome:

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| All ⚡ | Consensus approval | Bless |
| All ✗ | Consensus rejection | Reject |
| All ? | Promising but underdeveloped | Park for later |
| 2×⚡ + 1×? | Strong with minor hesitation | Bless |
| 2×✗ + 1×? | Weak with slight hope | Reject |
| ⚡ + ? + ✗ (full spread) | Needs diagnosis | Analyze critiques → Refine or Deliberate |
| 2×⚡ + 1×✗ | Fundamental disagreement | Deliberate |
| 2×✗ + 1×⚡ | Likely flawed, one holdout | Deliberate (give dissenter a chance) |

**To determine Refine vs Deliberate, analyze the critiques:**

| Critique Type | Action |
|---------------|--------|
| "Vague wording but idea is solid" | Refine |
| "Math is correct but framing is unclear" | Refine |
| "Missing precision that could be added" | Refine |
| "Hedging language weakens a solid claim" | Refine |
| "The core claim is wrong" | Deliberate or Reject |
| "This is numerology, not structure" | Deliberate |
| "I disagree with the other reviewers' reasoning" | Deliberate |
| "Unfalsifiable / too vague to evaluate" | Reject |
| "Restates input without insight" | Reject |

---

## Refinement Round

**Trigger:** Mixed reviews where critiques indicate fixable issues (articulation, precision, framing) rather than fundamental flaws.

**Author:** Claude Opus 4.5

**Input:** Original chunk + all three Round 1 reviews

**Refinement Prompt:**

```
An insight was extracted but received mixed reviews. Your task is to 
REFINE the articulation based on the critiques, not to change the 
underlying claim.

ORIGINAL CHUNK:
{original_insight}

EXTRACTION CONFIDENCE: {confidence}
TAGS: {tags}
DEPENDS ON: {dependencies}

REVIEWER CRITIQUES:

GEMINI ({gemini_rating}):
- Mathematical verification: {gemini_math}
- Structural analysis: {gemini_structure}
- Naturalness assessment: {gemini_natural}
- Reasoning: {gemini_reasoning}

CHATGPT ({chatgpt_rating}):
- Mathematical verification: {chatgpt_math}
- Structural analysis: {chatgpt_structure}
- Naturalness assessment: {chatgpt_natural}
- Reasoning: {chatgpt_reasoning}

CLAUDE ({claude_rating}):
- Mathematical verification: {claude_math}
- Structural analysis: {claude_structure}
- Naturalness assessment: {claude_natural}
- Reasoning: {claude_reasoning}

TASK:
Rewrite the insight to address valid critiques while preserving 
what is genuinely valuable.

YOU MAY:
- Sharpen vague language
- Correct minor errors noted by reviewers
- Reframe to make the structure clearer
- Add precision that reviewers noted was missing
- Remove hedging if the claim is actually solid
- Strengthen the mathematical grounding

YOU MAY NOT:
- Change the fundamental claim
- Add new claims not present in the original
- Weaken the insight just to avoid criticism
- Ignore valid critiques

RESPOND:

REFINED_INSIGHT: [1-3 sentences, precise, standalone]

CHANGES_MADE: [List each change and why]

ADDRESSED_CRITIQUES: [Which specific reviewer concerns this resolves]

UNRESOLVED_ISSUES: [Any critiques that couldn't be addressed 
                    without changing the core claim — these may 
                    need deliberation]

REFINEMENT_CONFIDENCE: [high/medium/low — how much better is this?]
```

---

## Review Round 2 (Post-Refinement)

**Mode:** Standard mode (Gemini 3 Pro, ChatGPT 5.2, Claude Opus 4.5)

**Process:** All 3 review the refined version. They see both original and refined.

**Prompt:**

```
A chunk has been refined based on initial review feedback. 
Evaluate the refined version.

ORIGINAL CHUNK:
{original_insight}

REFINED CHUNK:
{refined_insight}

CHANGES MADE:
{changes_made}

CRITIQUES THAT PROMPTED REFINEMENT:
{summary_of_critiques}

YOUR ORIGINAL RATING: {this_llm_round1_rating}
YOUR ORIGINAL CRITIQUE: {this_llm_round1_critique}

TASK:
Evaluate the refined version. Has the refinement addressed the issues?

RESPOND:

ISSUES_ADDRESSED: [yes/partially/no — which specific concerns were fixed?]

NEW_ISSUES: [Did refinement introduce any problems?]

RATING: ⚡ / ? / ✗

REASONING: [Justify rating for the refined version]

PREFER_ORIGINAL: [yes/no — in rare cases the original might be better]
```

**Outcome routing:**
- All ⚡ → Bless refined version
- All ✗ → Reject
- Mixed → Proceed to Deliberation (Round 3)

---

## Version Tracking

Chunks that go through refinement should track their history:

```python
@dataclass
class ChunkVersion:
    version: int  # 1 = original, 2 = refined, etc.
    insight: str
    author: str  # "extraction" or "refinement"
    created_at: datetime
    review_round: int
    ratings: Dict[str, str]  # {llm: rating}

@dataclass
class Chunk:
    id: str
    versions: List[ChunkVersion]  # Full history
    current_version: int  # Which version is active
    insight: str  # Current/best version text
    # ... rest of existing fields
    
    @property
    def was_refined(self) -> bool:
        return len(self.versions) > 1
    
    @property
    def original_insight(self) -> str:
        return self.versions[0].insight
```

---

## Refinement Limits

To prevent infinite loops:

```yaml
refinement:
  max_refinement_rounds: 2  # Original + up to 2 refinements
  escalate_after_failed_refinement: true  # Go to deliberation if still mixed
```

**Flow with limits:**

```
Round 1 (original) → Mixed → Refine
Round 2 (v1 refined) → Still mixed → Refine again OR Deliberate
Round 3 (v2 refined) → Still mixed → Deliberate (forced)
```

---

## When Refinement Fails

If a refined chunk still gets mixed reviews:

1. **Analyze why:** Are the same critiques recurring, or new ones?
2. **If same critiques:** The issue may be fundamental → Deliberate
3. **If new critiques:** Refinement may have introduced problems → Consider reverting
4. **If spread narrows** (e.g., ⚡+?+✗ → ⚡+⚡+?): Refinement helped → One more round
5. **If spread widens or stays same:** Deliberate or Reject

---

## Data Model Updates

**ChunkReview (updated):**

```python
@dataclass
class ChunkReview:
    chunk_id: str
    rounds: List[ReviewRound]
    refinements: List[Refinement]  # NEW
    final_rating: str
    final_version: int  # NEW — which version was blessed/rejected
    is_unanimous: bool
    is_disputed: bool
    was_refined: bool  # NEW
    mind_changes: List[MindChange]
    total_tokens_used: int
    review_duration_seconds: float
    reviewed_at: datetime

@dataclass
class Refinement:
    from_version: int
    to_version: int
    original_insight: str
    refined_insight: str
    changes_made: List[str]
    addressed_critiques: List[str]
    unresolved_issues: List[str]
    refinement_confidence: str
    triggered_by_ratings: Dict[str, str]  # What ratings prompted this
    timestamp: datetime
```

---

## Logging

Log refinement activity:

```json
{
  "event": "refinement",
  "chunk_id": "chunk_042",
  "from_version": 1,
  "to_version": 2,
  "original": "The 84 appears in both body composition and cosmic cycles",
  "refined": "The sum 72% (water) + 12% (earth) = 84% solid matter precisely equals the 84 cosmic cycles, suggesting body composition encodes cosmological structure",
  "trigger_ratings": {"gemini": "?", "chatgpt": "⚡", "claude": "?"},
  "changes": ["Added specific percentages", "Made structural claim explicit"],
  "timestamp": "2024-01-15T14:30:00Z"
}
```

---

## Success Metrics

Track refinement effectiveness:

- % of mixed-review chunks that get blessed after refinement
- Average version number of blessed chunks (1.0 = no refinement needed)
- % of refinements that address all critiques
- % of refinements that introduce new issues
- Correlation between refinement rounds and final quality

---

## Summary

| Role | Actor |
|------|-------|
| Exploration | Gemini + ChatGPT |
| Extraction (initial write) | Claude Opus |
| Review/Rating | All 3 LLMs |
| Refinement (rewrite) | Claude Opus |
| Deliberation | All 3 LLMs (deep mode) |

This ensures insights improve through the review process rather than being lost to imprecise initial articulation.
