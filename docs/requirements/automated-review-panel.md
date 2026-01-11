# Automated Review Panel: LLM Consensus with Deliberation

## Overview

Replace human chunk review with a panel of three LLMs (Gemini, ChatGPT, Claude) that evaluate chunks for mathematical rigor and alignment with Sadhguru's axioms. Disagreements trigger escalation to deep reasoning modes and structured deliberation until consensus is reached.

---

## Core Principles

**The panel is:**
- Open-minded about conclusions — willing to change ratings when presented with compelling arguments
- Uncompromising about method — criteria never soften to reach agreement

**The goal is truth, not consensus.** Agreement must emerge from genuine persuasion, not compromise.

---

## Review Criteria (Fixed, Non-Negotiable)

**For blessing (⚡):**
1. **Mathematically rigorous** — verifiable, precise, no hand-waving
2. **Structurally connected** — genuine relationship to Sadhguru axioms, not surface numerology
3. **Natural and inevitable** — feels discovered, not constructed; "of course it had to be this way"
4. **Precisely stated** — could be formalized as theorem or well-defined conjecture

**For rejection (✗):**
1. Mathematically incorrect or unverifiable
2. Vague, hedging, or unfalsifiable
3. Surface pattern-matching without structural explanation
4. Merely restates inputs in different words

**For uncertain (?):**
1. Mathematically sound but connection to axioms unclear
2. Interesting but needs further development
3. Promising direction but not yet precise enough

---

## Models

**Round 1 (Standard — already top-tier):**
- Gemini 3 Pro
- ChatGPT 5.2
- Claude Opus 4.5

**Round 2 & 3 (Deep Reasoning activated):**
- Gemini 3 Pro Deep Think
- ChatGPT 5.2 Pro mode
- Claude Opus 4.5 with extended thinking

---

## Three-Round Process

```
ROUND 1: Independent Review (Standard Mode)
    │
    ├── Unanimous → Done (auto-bless / auto-reject / needs-work)
    │
    └── Disagreement → ROUND 2

ROUND 2: Deep Analysis (Deep Think / Pro Mode / Extended Thinking)
    │
    ├── Unanimous → Done
    │
    └── Still Split → ROUND 3

ROUND 3: Structured Deliberation
    │
    ├── Consensus Reached → Done
    │
    └── Persistent Split → Majority wins, flag as "disputed"
```

---

## Round 1: Independent Review

**Mode:** Standard (Gemini 3 Pro, ChatGPT 5.2, Claude Opus 4.5)

**Process:** Each LLM reviews independently, without seeing others' responses.

**Prompt:**
```
You are a rigorous mathematician reviewing a proposed insight for inclusion 
in an axiomatic knowledge system exploring connections between pure mathematics 
and Sadhguru's teachings on yogic science.

SADHGURU'S CORE AXIOMS (ground truth):
{blessed_axioms_summary}

CHUNK TO REVIEW:
{chunk_insight}

CONFIDENCE LEVEL FROM EXTRACTION: {confidence}
TAGS: {tags}
DEPENDS ON: {dependencies}

REVIEW CRITERIA:

1. MATHEMATICAL RIGOR
   - Is every mathematical claim verifiable?
   - Are the numbers, properties, theorems cited accurate?
   - Is the logic valid?

2. STRUCTURAL DEPTH
   - Is this a genuine structural connection or surface numerology?
   - Does it reveal WHY these things connect, not just THAT they share a number?
   - Could this be formalized as a theorem or precise conjecture?

3. NATURALNESS
   - Does this feel discovered or constructed?
   - Is there an "of course!" quality — inevitability?
   - Would a skeptical mathematician find this interesting or dismiss it?

RESPOND:
RATING: ⚡ (bless) / ? (uncertain) / ✗ (reject)

MATHEMATICAL_VERIFICATION: [Verify or refute specific claims. Be precise.]

STRUCTURAL_ANALYSIS: [Is the connection deep or superficial? Why?]

NATURALNESS_ASSESSMENT: [Does it feel inevitable? Explain.]

REASONING: [Overall justification for your rating, 2-4 sentences]

CONFIDENCE: [high / medium / low in your rating]
```

**Outcome routing:**
- 3× ⚡ → Auto-bless
- 3× ✗ → Auto-reject
- 3× ? → Auto-mark as "needs development"
- Any disagreement → Proceed to Round 2

---

## Round 2: Deep Analysis

**Mode:** Deep reasoning (Gemini 3 Pro Deep Think, ChatGPT 5.2 Pro, Claude Opus 4.5 extended thinking)

**Process:** Each LLM sees all Round 1 responses and re-evaluates with extended reasoning.

**Prompt:**
```
You are a rigorous mathematician engaged in collaborative truth-seeking.
A proposed insight has received conflicting reviews. Your task is to 
deeply analyze and determine the correct assessment.

DELIBERATION PRINCIPLES:
- OPEN-MINDED about conclusions — you may have missed something
- UNCOMPROMISING about method — criteria never soften
- The goal is TRUTH, not consensus
- Changing your mind when presented with good arguments is intellectual honesty
- Refusing to change despite good arguments is stubbornness

SADHGURU'S CORE AXIOMS:
{blessed_axioms_summary}

CHUNK UNDER REVIEW:
{chunk_insight}

ROUND 1 ASSESSMENTS:

GEMINI ({gemini_rating}):
- Mathematical verification: {gemini_math}
- Structural analysis: {gemini_structure}
- Naturalness: {gemini_natural}
- Reasoning: {gemini_reasoning}

CHATGPT ({chatgpt_rating}):
- Mathematical verification: {chatgpt_math}
- Structural analysis: {chatgpt_structure}
- Naturalness: {chatgpt_natural}
- Reasoning: {chatgpt_reasoning}

CLAUDE ({claude_rating}):
- Mathematical verification: {claude_math}
- Structural analysis: {claude_structure}
- Naturalness: {claude_natural}
- Reasoning: {claude_reasoning}

YOUR ROUND 1 ASSESSMENT WAS: {this_llm_round1_rating}

TASK:
Consider the other perspectives seriously. They may have seen something 
you missed. But do not lower your standards — only change if genuinely persuaded.

RESPOND:

NEW_INFORMATION: [What, if anything, did the other reviewers point out 
                  that you hadn't fully considered?]

REASSESSMENT:
- Mathematical claims: [Any errors found by others? Any corrections?]
- Structural depth: [Did others reveal deeper/shallower connections?]
- Naturalness: [Did others' framing change how inevitable this feels?]

DOES_THIS_CHANGE_THINGS: [yes/no — and why]

UPDATED_RATING: ⚡ / ? / ✗

UPDATED_REASONING: [If changed: what convinced you. 
                    If unchanged: why the arguments don't meet the bar 
                    despite serious consideration.]

CONFIDENCE: [high / medium / low]
```

**Outcome routing:**
- Unanimous → Done
- Still split → Proceed to Round 3

---

## Round 3: Structured Deliberation

**Mode:** Deep reasoning continues

**Process:** Direct exchange between positions until resolution.

**Step 3A — Minority States Case:**
```
There remains disagreement after deep analysis.

THE CONTESTED INSIGHT:
{chunk_insight}

CURRENT POSITIONS:
- MAJORITY ({majority_rating}): {majority_count} reviewers
  {majority_reasoning_summary}

- MINORITY ({minority_rating}): {minority_count} reviewer(s)
  {minority_reasoning_summary}

YOU HOLD THE MINORITY POSITION.

TASK:
State your single strongest argument. Be maximally specific.
What exact criterion does this chunk meet or fail that the majority is 
mis-evaluating? Point to specific mathematical facts, structural features, 
or axiom connections.

Do not restate your general position. Make your best case in one focused argument.

STRONGEST_ARGUMENT: [Your single most compelling point]
```

**Step 3B — Majority Responds:**
```
THE CONTESTED INSIGHT:
{chunk_insight}

YOU HOLD THE MAJORITY POSITION ({majority_rating}).

THE MINORITY'S STRONGEST ARGUMENT:
{minority_strongest_argument}

TASK:
Respond directly to this specific argument. Do not restate your general case.
Either:
- Refute it with specific counter-evidence
- Acknowledge it changes your assessment

RESPONSE: [Direct engagement with their specific point]

DOES_THIS_CHANGE_YOUR_RATING: [yes/no]
UPDATED_RATING: [if changed]
```

**Step 3C — Final Resolution:**
```
FINAL RESOLUTION ROUND

THE CONTESTED INSIGHT:
{chunk_insight}

MINORITY'S ARGUMENT:
{minority_strongest_argument}

MAJORITY'S RESPONSE:
{majority_response}

{if_minority}
You made the minority argument. The majority has responded.

TASK: Either concede or maintain.
- "CONCEDE: [what convinced you]"
- "MAINTAIN: [why their response fails to address your point]"

FINAL_RATING: ⚡ / ? / ✗
{/if_minority}

{if_majority}
You held the majority position. You have heard the full exchange.

FINAL_RATING: ⚡ / ? / ✗
ONE_SENTENCE_JUSTIFICATION: [final reasoning]
{/if_majority}
```

**Outcome routing:**
- All three converge → Done, use consensus rating
- 2-1 split persists → Majority wins, flag chunk as "disputed"

---

## Data Models

**ReviewRound:**
```python
@dataclass
class ReviewRound:
    round_number: int  # 1, 2, or 3
    mode: str  # "standard" or "deep"
    responses: Dict[str, ReviewResponse]  # keyed by llm name
    outcome: str  # "unanimous" / "split" / "resolved"
    timestamp: datetime
```

**ReviewResponse:**
```python
@dataclass
class ReviewResponse:
    llm: str  # "gemini" / "chatgpt" / "claude"
    mode: str  # "standard" / "deep_think" / "pro" / "extended_thinking"
    rating: str  # "⚡" / "?" / "✗"
    mathematical_verification: str
    structural_analysis: str
    naturalness_assessment: str
    reasoning: str
    confidence: str
    # Round 2+ fields
    new_information: Optional[str]
    changed_mind: Optional[bool]
    previous_rating: Optional[str]
    # Round 3 fields
    strongest_argument: Optional[str]
    response_to_argument: Optional[str]
    final_stance: Optional[str]  # "concede" / "maintain"
```

**ChunkReview:**
```python
@dataclass
class ChunkReview:
    chunk_id: str
    rounds: List[ReviewRound]
    final_rating: str
    is_unanimous: bool
    is_disputed: bool
    mind_changes: List[MindChange]
    total_tokens_used: int
    review_duration_seconds: float
    reviewed_at: datetime
```

**MindChange:**
```python
@dataclass
class MindChange:
    llm: str
    round_number: int
    from_rating: str
    to_rating: str
    reason: str  # what convinced them
```

---

## Integration with Chunk System

**After chunk extraction:**
```python
def process_new_chunks(chunks: List[Chunk], blessed_axioms: List[Chunk]):
    for chunk in chunks:
        # Run review panel
        review = run_review_panel(chunk, blessed_axioms)
        
        # Apply outcome
        chunk.rating = review.final_rating
        chunk.review = review
        chunk.is_disputed = review.is_disputed
        
        # If blessed, add to blessed collection
        if chunk.rating == "⚡":
            add_to_blessed(chunk)
        
        # Save
        save_chunk(chunk)
        save_review_log(review)
```

**Blessed axioms context:**

Include both:
1. Sadhguru's source axioms (from the excerpts files)
2. Previously blessed chunks (machine-discovered insights now treated as axioms)

This allows the knowledge to compound — new insights can build on blessed discoveries.

---

## Dispute Handling

Disputed chunks (persistent 2-1 splits) are:
- Blessed/rejected per majority vote
- Flagged with `is_disputed: true`
- Full deliberation transcript preserved
- Periodically analyzed for patterns:
  - Which LLM is frequently the dissenter?
  - What types of insights cause splits?
  - Do disputed chunks that get blessed hold up over time?

---

## Configuration

```yaml
review_panel:
  enabled: true
  
  round1:
    gemini_model: "gemini-3-pro"
    chatgpt_model: "chatgpt-5.2"
    claude_model: "claude-opus-4.5"
  
  round2:
    gemini_model: "gemini-3-pro-deep-think"
    chatgpt_model: "chatgpt-5.2-pro"
    claude_model: "claude-opus-4.5"  # extended thinking enabled
  
  round3:
    # Same as round 2
    max_exchanges: 3  # Minority argument, majority response, final
  
  outcomes:
    unanimous_bless: "auto_bless"
    unanimous_reject: "auto_reject"
    unanimous_uncertain: "needs_development"
    disputed_majority_bless: "bless_with_flag"
    disputed_majority_reject: "reject_with_flag"
  
  analysis:
    flag_disputed: true
    log_mind_changes: true
    track_llm_agreement_rates: true
```

---

## Logging Requirements

**Log file:** `logs/review_panel.log`

Log every review with:
- Chunk ID and insight text
- Each round's responses (all fields)
- Mind changes with full reasoning
- Final outcome
- Token usage
- Duration

Format: JSON lines for parseability

---

## File Structure

```
fano-explorer/
├── review_panel/
│   ├── __init__.py
│   ├── reviewer.py       # Main orchestration
│   ├── prompts.py        # All prompt templates
│   ├── round1.py         # Independent review logic
│   ├── round2.py         # Deep analysis logic
│   ├── round3.py         # Deliberation logic
│   ├── models.py         # Data classes
│   └── analysis.py       # Dispute pattern analysis
├── data/
│   └── reviews/
│       ├── completed/    # Full review records
│       └── disputed/     # Disputed chunks for analysis
└── logs/
    └── review_panel.log
```

---

## Success Metrics

Track over time:
- % resolved in Round 1 (efficiency)
- % resolved in Round 2 (deep mode effectiveness)
- % requiring Round 3 (deliberation rate)
- % disputed after Round 3 (persistent disagreement rate)
- Mind change rate per LLM (flexibility)
- Correlation between disputed chunks and later invalidation (dispute predictiveness)

---

## Remove Human Review Queue

The existing human review queue (`display_chunk_for_review`, rating input, etc.) should be removed or converted to an optional "audit mode" for periodic spot-checking. The review panel is now the primary rating mechanism.
