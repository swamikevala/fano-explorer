# Refactor Chunking System: Atomic Aphorisms with Knowledge Graph

## Problem

1. Currently one monolithic chunk per exploration thread, mixing profound insights with vague speculation — impossible to grade fairly
2. On restart, entire explorations pile into context creating bloated chunks
3. No way to build on previously validated insights

## Requirements

### Atomic Extraction

- Each chunk must be a single, atomic insight (1-3 sentences max)
- One exploration thread should produce multiple chunks
- Skip anything vague, hedging, or speculative
- Each chunk gets a confidence level (high/medium/low) and tags

### Dependency Graph

- A chunk can reference blessed (⚡-rated) chunks as its foundation
- Dependencies are resolved by semantic matching to existing blessed chunks
- Unresolved dependencies are flagged as "pending" for review awareness
- When a blessed chunk is demoted, flag all chunks that depend on it

### Extraction Prompt

- Provide list of existing blessed axioms as available foundations
- Instruct LLM to extract individual insights, not summaries
- Require structured output: insight, confidence, tags, dependencies
- Allow "depends on: none" for standalone insights

### Extraction Prompt Template

```
Extract INDIVIDUAL insights from this exploration as separate aphorisms.

RULES:
- Each insight must be ONE standalone claim or connection
- Maximum 1-3 sentences per insight
- Must be understandable on its own OR explicitly state what it depends on
- Only extract claims that are SPECIFIC and TESTABLE or PROFOUND and PRECISE
- Skip anything vague, hedging, or speculative ("might be", "could possibly", "perhaps")
- Do NOT bundle multiple ideas together
- Do NOT repeat the same insight with different wording

DEPENDENCIES:
- If an insight REQUIRES a prior concept to make sense, note it in DEPENDS_ON
- Only list dependencies that are themselves worthy of being standalone insights
- Reference existing blessed axioms when possible (provided below)

BLESSED AXIOMS (available as foundations):
{blessed_chunks_summary}

FORMAT YOUR RESPONSE EXACTLY AS:
===
INSIGHT: [single atomic aphorism here]
CONFIDENCE: [high/medium/low]
TAGS: [comma-separated concepts]
DEPENDS_ON: [comma-separated descriptions of required prior insights, or "none"]
===
INSIGHT: [next aphorism]
CONFIDENCE: [high/medium/low]
TAGS: [concepts]
DEPENDS_ON: [dependencies or "none"]
===
(continue for each distinct insight worth extracting)

If no insights meet the quality bar, respond with:
===
NO_INSIGHTS: [brief explanation why]
===
```

### Restart Handling

- Mark threads as `chunks_extracted: true` after processing
- Skip re-extraction on restart
- No duplicate chunks from reprocessing old threads

### Config Options

```yaml
chunking:
  mode: "atomic"  # vs "monolithic" (old behavior)
  min_confidence_to_keep: "low"  # filter during extraction: "high", "medium", or "low"
  max_insights_per_thread: 10  # prevent over-extraction

dependencies:
  semantic_match_threshold: 0.7  # for matching deps to blessed chunks
  max_blessed_in_prompt: 20  # limit context size
  warn_on_unresolved: true  # flag during review for pending deps
```

### Data Models

**Chunk:**
```python
@dataclass
class Chunk:
    id: str
    insight: str  # The atomic aphorism (1-3 sentences max)
    confidence: str  # high/medium/low from extraction
    tags: List[str]  # Relevant concepts for searchability
    source_thread_id: str  # Which exploration it came from
    source_exchange_indices: List[int]  # Which exchanges contributed
    depends_on: List[str]  # IDs of blessed chunks this builds upon
    pending_dependencies: List[str]  # Described deps not yet matched to blessed chunks
    rating: Optional[str] = None  # ⚡ (blessed) / ? (interesting) / ✗ (discard)
    is_disputed: bool = False  # Flag for chunks that had split reviews
    created_at: datetime
    
    def is_foundation_solid(self, blessed_chunks: dict) -> bool:
        """Check if all dependencies are blessed."""
        return all(
            dep_id in blessed_chunks and blessed_chunks[dep_id].rating == "⚡"
            for dep_id in self.depends_on
        )
```

**ExplorationThread (updated):**
```python
@dataclass
class ExplorationThread:
    id: str
    topic: str
    exchanges: List[Exchange]
    created_at: datetime
    updated_at: datetime
    chunks_extracted: bool = False  # NEW: prevent re-extraction
    extraction_note: Optional[str] = None  # NEW: note if no insights found
```

### Dependency Resolution

```python
def resolve_dependencies(dep_descriptions: List[str], blessed_chunks: List[Chunk]) -> Tuple[List[str], List[str]]:
    """Match dependency descriptions to blessed chunk IDs."""
    resolved = []
    pending = []
    
    for desc in dep_descriptions:
        # Semantic similarity search against blessed chunks
        match = find_best_semantic_match(desc, blessed_chunks, threshold=0.7)
        
        if match and match.rating == "⚡":
            resolved.append(match.id)
        else:
            # Not yet blessed or no match - keep as pending
            pending.append(desc)
    
    return resolved, pending
```

### Validation Cascade

When a blessed chunk is demoted:
```python
def demote_chunk(chunk_id: str, chunks: List[Chunk], blessed_chunks: dict):
    """Handle demotion of a blessed chunk - flag dependents."""
    
    demoted = blessed_chunks.pop(chunk_id, None)
    if not demoted:
        return
    
    # Find all chunks that depend on this one
    affected = []
    for chunk in chunks:
        if chunk_id in chunk.depends_on:
            affected.append(chunk)
            # Move from resolved to pending
            chunk.depends_on.remove(chunk_id)
            chunk.pending_dependencies.append(demoted.insight)
    
    if affected:
        log(f"⚠️ {len(affected)} chunks had their foundation weakened")
```

### File Structure

```
fano-explorer/
├── chunking/
│   ├── __init__.py
│   ├── extractor.py      # Atomic extraction logic
│   ├── prompts.py        # Extraction prompt templates
│   ├── dependencies.py   # Dependency resolution
│   └── models.py         # Chunk data classes
├── data/
│   ├── chunks/           # All extracted chunks
│   ├── blessed/          # Blessed chunks (⚡ rated)
│   └── explorations/     # Thread data with chunks_extracted flag
```

### Summary of Changes

| Component | Old Behavior | New Behavior |
|-----------|-------------|--------------|
| Chunks per thread | 1 monolithic | Multiple atomic |
| Chunk content | Summary of everything | Single aphorism (1-3 sentences) |
| Dependencies | None | Can reference blessed chunks |
| Quality filtering | None | Confidence levels, skip vague |
| Re-extraction | Every restart | Skip if `chunks_extracted=True` |
| Knowledge structure | Flat list | Graph with blessed foundations |
