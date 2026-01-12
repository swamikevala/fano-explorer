"""
Main DeduplicationChecker class.

LLM-first deduplication checker that uses a cheap/fast LLM for semantic
duplicate detection. This approach prioritizes accuracy over speed,
recognizing that mathematical duplicates require understanding.
"""

import time
from typing import Callable, Optional, Awaitable

from shared.logging import get_logger

from .models import ContentItem, ContentType, SimilarityScore, DuplicateResult
from .text_processing import calculate_similarity
from .llm_prompts import (
    build_pairwise_llm_prompt,
    build_batch_llm_prompt,
    parse_llm_duplicate_response,
)

log = get_logger("shared", "deduplication.checker")

# Type alias for LLM callback
LLMCallback = Callable[[str], Awaitable[str]]


class DeduplicationChecker:
    """
    LLM-first deduplication checker.

    Uses a cheap/fast LLM (like Haiku or Sonnet) for semantic duplicate detection.
    This approach prioritizes accuracy over speed, recognizing that mathematical
    duplicates require understanding, not just keyword matching.

    Detection layers:
    1. Signature matching (instant) - catches exact duplicates
    2. LLM semantic check (primary) - catches semantic duplicates

    Heuristics are available but disabled by default since they often miss
    semantic duplicates in mathematical content.

    Design principles:
    - Use cheap LLM for dedup, save expensive quota for exploration
    - LLM-agnostic: Works with any LLM via simple callback
    - Shared: Used by both explorer and documenter modules
    """

    def __init__(
        self,
        llm_callback: Optional[LLMCallback] = None,
        *,
        # Features (LLM-first by default)
        use_signature_check: bool = True,
        use_heuristic_check: bool = False,  # Disabled by default - LLM is better for math
        use_llm_check: bool = True,
        use_batch_llm: bool = True,
        batch_size: int = 20,
        # Heuristic thresholds (only used if use_heuristic_check=True)
        keyword_threshold: float = 0.40,
        concept_threshold: float = 0.45,
        combined_threshold: float = 0.50,
        # LLM confidence requirements
        require_high_confidence: bool = False,
        # Stats logging
        stats_log_interval: int = 50,  # Log stats every N checks (0 to disable)
    ):
        """
        Initialize the deduplication checker.

        Args:
            llm_callback: Async function that takes a prompt and returns LLM response text.
                         Use a cheap/fast model (Haiku, Sonnet) to preserve expensive quota.
                         If None, falls back to heuristics only.
            use_signature_check: Enable exact signature matching (recommended)
            use_heuristic_check: Enable keyword/concept heuristics (disabled by default)
            use_llm_check: Enable LLM semantic check (primary detection method)
            use_batch_llm: Check multiple items in one LLM call (more efficient)
            batch_size: Max items per batch LLM call
            keyword_threshold: Heuristic threshold (if enabled)
            concept_threshold: Heuristic threshold (if enabled)
            combined_threshold: Heuristic threshold (if enabled)
            require_high_confidence: Only accept LLM duplicates with high confidence
            stats_log_interval: Log stats every N checks (0 to disable)
        """
        self.llm_callback = llm_callback
        self.keyword_threshold = keyword_threshold
        self.concept_threshold = concept_threshold
        self.combined_threshold = combined_threshold
        self.use_signature_check = use_signature_check
        self.use_heuristic_check = use_heuristic_check
        self.use_llm_check = use_llm_check and llm_callback is not None
        self.use_batch_llm = use_batch_llm
        self.batch_size = batch_size
        self.require_high_confidence = require_high_confidence
        self.stats_log_interval = stats_log_interval

        # Known content registry
        self._known_items: dict[str, ContentItem] = {}
        self._signatures: dict[str, str] = {}  # signature -> item_id

        # Statistics
        self._stats = {
            "checks": 0,
            "duplicates_found": 0,
            "by_signature": 0,
            "by_heuristic": 0,
            "by_llm": 0,
        }

        log.info(
            "deduplication.checker.initialized",
            use_signature=self.use_signature_check,
            use_heuristics=self.use_heuristic_check,
            use_llm=self.use_llm_check,
            use_batch=self.use_batch_llm,
            batch_size=self.batch_size,
            stats_log_interval=self.stats_log_interval,
        )

    def add_content(self, item: ContentItem) -> None:
        """
        Add content to the known registry.

        Args:
            item: ContentItem to add
        """
        if item.id in self._known_items:
            log.debug("deduplication.content.already_known", item_id=item.id)
            return

        self._known_items[item.id] = item
        self._signatures[item.signature] = item.id

        log.debug(
            "deduplication.content.added",
            item_id=item.id,
            content_type=item.content_type.value,
            keyword_count=len(item.keywords),
            concept_count=len(item.concepts),
        )

    def add_contents(self, items: list[ContentItem]) -> None:
        """Add multiple content items."""
        for item in items:
            self.add_content(item)

    def load_from_dicts(
        self,
        items: list[dict],
        content_type: ContentType = ContentType.UNKNOWN,
    ) -> None:
        """
        Load content from list of dicts.

        Expected dict format: {"id": str, "text": str, ...}
        """
        for item_dict in items:
            item = ContentItem(
                id=item_dict.get("id", str(len(self._known_items))),
                text=item_dict.get("text", item_dict.get("content", "")),
                content_type=content_type,
                metadata=item_dict,
            )
            self.add_content(item)

        log.info(
            "deduplication.content.loaded",
            count=len(items),
            content_type=content_type.value,
            total_known=len(self._known_items),
        )

    async def check_duplicate(
        self,
        text: str,
        item_id: str = "new",
        content_type: ContentType = ContentType.UNKNOWN,
        skip_llm: bool = False,
    ) -> DuplicateResult:
        """
        Check if text is a duplicate of any known content.

        Args:
            text: The text to check
            item_id: ID for logging/tracking
            content_type: Type of content being checked
            skip_llm: If True, skip LLM check even if enabled

        Returns:
            DuplicateResult with details of the check
        """
        start_time = time.time()

        self._stats["checks"] += 1

        # Create content item for the new text
        new_item = ContentItem(
            id=item_id,
            text=text,
            content_type=content_type,
        )

        # Layer 1: Signature check (instant)
        if self.use_signature_check:
            result = self._check_signature(new_item)
            if result.is_duplicate:
                result.check_time_ms = (time.time() - start_time) * 1000
                self._stats["duplicates_found"] += 1
                self._stats["by_signature"] += 1
                self._maybe_log_stats()
                return result

        # Layer 2: Heuristic check (fast)
        if self.use_heuristic_check:
            result = self._check_heuristics(new_item)
            if result.is_duplicate:
                result.check_time_ms = (time.time() - start_time) * 1000
                self._stats["duplicates_found"] += 1
                self._stats["by_heuristic"] += 1
                self._maybe_log_stats()
                return result

        # Layer 3: LLM semantic check (thorough)
        if self.use_llm_check and not skip_llm and self._known_items:
            result = await self._check_with_llm(new_item)
            if result.is_duplicate:
                result.check_time_ms = (time.time() - start_time) * 1000
                self._stats["duplicates_found"] += 1
                self._stats["by_llm"] += 1
                self._maybe_log_stats()
                return result

        # No duplicate found
        elapsed = (time.time() - start_time) * 1000
        log.debug(
            "deduplication.check.no_duplicate",
            item_id=item_id,
            known_count=len(self._known_items),
            time_ms=round(elapsed, 2),
        )

        self._maybe_log_stats()
        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item_id,
            check_method="all",
            reason="No duplicate found",
            check_time_ms=elapsed,
        )

    def _check_signature(self, item: ContentItem) -> DuplicateResult:
        """Check for exact signature match."""
        if item.signature in self._signatures:
            dup_id = self._signatures[item.signature]
            log.info(
                "deduplication.signature.match",
                new_id=item.id,
                duplicate_of=dup_id,
            )
            return DuplicateResult(
                is_duplicate=True,
                checked_item_id=item.id,
                duplicate_of=dup_id,
                check_method="signature",
                reason="Exact content match (identical normalized text)",
            )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            check_method="signature",
        )

    def _check_heuristics(self, item: ContentItem) -> DuplicateResult:
        """Check using keyword/concept/ngram heuristics."""
        best_match: Optional[tuple[str, SimilarityScore]] = None
        best_score = 0.0

        for known_id, known_item in self._known_items.items():
            score = calculate_similarity(item, known_item)

            # Check if this is a potential duplicate
            is_potential = (
                score.keyword_similarity >= self.keyword_threshold or
                score.concept_similarity >= self.concept_threshold or
                score.combined_score >= self.combined_threshold
            )

            if is_potential and score.combined_score > best_score:
                best_score = score.combined_score
                best_match = (known_id, score)

        if best_match and best_score >= self.combined_threshold:
            dup_id, similarity = best_match
            log.info(
                "deduplication.heuristic.match",
                new_id=item.id,
                duplicate_of=dup_id,
                keyword_sim=round(similarity.keyword_similarity, 3),
                concept_sim=round(similarity.concept_similarity, 3),
                combined=round(similarity.combined_score, 3),
            )
            return DuplicateResult(
                is_duplicate=True,
                checked_item_id=item.id,
                duplicate_of=dup_id,
                similarity=similarity,
                check_method="heuristic",
                reason=f"High similarity (keyword={similarity.keyword_similarity:.2f}, "
                       f"concept={similarity.concept_similarity:.2f}, "
                       f"combined={similarity.combined_score:.2f})",
            )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            similarity=best_match[1] if best_match else None,
            check_method="heuristic",
        )

    async def _check_with_llm(self, item: ContentItem) -> DuplicateResult:
        """Check using LLM semantic comparison."""
        if not self.llm_callback:
            return DuplicateResult(
                is_duplicate=False,
                checked_item_id=item.id,
                check_method="llm_skipped",
                reason="No LLM callback configured",
            )

        if self.use_batch_llm:
            return await self._check_with_batch_llm(item)
        else:
            return await self._check_with_pairwise_llm(item)

    async def _check_with_batch_llm(self, item: ContentItem) -> DuplicateResult:
        """Efficient batch LLM check against all known content."""
        # Prepare items for batch check
        items_for_check = [
            {"id": k, "text": v.text}
            for k, v in list(self._known_items.items())[:self.batch_size]
        ]

        if not items_for_check:
            return DuplicateResult(
                is_duplicate=False,
                checked_item_id=item.id,
                check_method="batch_llm",
                reason="No known items to check against",
            )

        prompt = build_batch_llm_prompt(item.text, items_for_check)

        try:
            response = await self.llm_callback(prompt)
            is_dup, dup_idx, confidence, reason = parse_llm_duplicate_response(response)

            # Check confidence requirement
            if is_dup and self.require_high_confidence and confidence != "high":
                log.info(
                    "deduplication.llm.low_confidence",
                    new_id=item.id,
                    confidence=confidence,
                )
                is_dup = False
                reason = f"LLM match with {confidence} confidence (requires high)"

            if is_dup and dup_idx is not None and 0 <= dup_idx < len(items_for_check):
                dup_id = items_for_check[dup_idx]["id"]
                log.info(
                    "deduplication.llm.batch_match",
                    new_id=item.id,
                    duplicate_of=dup_id,
                    confidence=confidence,
                    reason=reason,
                )

                # Build similarity score with LLM confirmation
                known_item = self._known_items[dup_id]
                similarity = calculate_similarity(item, known_item)
                similarity.llm_confirmed = True
                similarity.llm_explanation = reason

                return DuplicateResult(
                    is_duplicate=True,
                    checked_item_id=item.id,
                    duplicate_of=dup_id,
                    similarity=similarity,
                    check_method="batch_llm",
                    reason=reason,
                )

            log.debug(
                "deduplication.llm.no_match",
                new_id=item.id,
                checked_count=len(items_for_check),
            )

        except Exception as e:
            log.warning(
                "deduplication.llm.error",
                new_id=item.id,
                error=str(e),
            )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            check_method="batch_llm",
            reason="No LLM-confirmed duplicate",
        )

    async def _check_with_pairwise_llm(self, item: ContentItem) -> DuplicateResult:
        """Pairwise LLM check (slower but more thorough)."""
        # First, find candidates with some heuristic similarity
        candidates = []
        for known_id, known_item in self._known_items.items():
            score = calculate_similarity(item, known_item)
            if score.combined_score >= 0.30:  # Low threshold for candidates
                candidates.append((known_id, known_item, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[2].combined_score, reverse=True)

        # Check top candidates with LLM
        for known_id, known_item, heuristic_score in candidates[:5]:
            prompt = build_pairwise_llm_prompt(item.text, known_item.text)

            try:
                response = await self.llm_callback(prompt)
                is_dup, _, confidence, reason = parse_llm_duplicate_response(response)

                if is_dup:
                    if self.require_high_confidence and confidence != "high":
                        continue

                    heuristic_score.llm_confirmed = True
                    heuristic_score.llm_explanation = reason

                    log.info(
                        "deduplication.llm.pairwise_match",
                        new_id=item.id,
                        duplicate_of=known_id,
                        confidence=confidence,
                    )

                    return DuplicateResult(
                        is_duplicate=True,
                        checked_item_id=item.id,
                        duplicate_of=known_id,
                        similarity=heuristic_score,
                        check_method="pairwise_llm",
                        reason=reason,
                    )

            except Exception as e:
                log.warning(
                    "deduplication.llm.pairwise_error",
                    new_id=item.id,
                    known_id=known_id,
                    error=str(e),
                )

        return DuplicateResult(
            is_duplicate=False,
            checked_item_id=item.id,
            check_method="pairwise_llm",
            reason="No LLM-confirmed duplicate",
        )

    def clear(self) -> None:
        """Clear all known content."""
        self._known_items.clear()
        self._signatures.clear()
        log.info("deduplication.checker.cleared")

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            **self._stats,
            "known_items": len(self._known_items),
        }

    def log_stats(self) -> None:
        """Log current deduplication statistics."""
        stats = self.get_stats()
        dup_rate = (
            stats["duplicates_found"] / stats["checks"] * 100
            if stats["checks"] > 0 else 0.0
        )
        log.info(
            "deduplication.stats",
            total_checks=stats["checks"],
            duplicates_found=stats["duplicates_found"],
            duplicate_rate_pct=round(dup_rate, 1),
            by_signature=stats["by_signature"],
            by_heuristic=stats["by_heuristic"],
            by_llm=stats["by_llm"],
            known_items=stats["known_items"],
        )

    def _maybe_log_stats(self) -> None:
        """Log stats if interval reached."""
        if (
            self.stats_log_interval > 0 and
            self._stats["checks"] > 0 and
            self._stats["checks"] % self.stats_log_interval == 0
        ):
            self.log_stats()

    @property
    def known_count(self) -> int:
        """Number of known content items."""
        return len(self._known_items)

    # =========================================================================
    # Backward-Compatible Methods (for explorer module migration)
    # =========================================================================

    def add_known_insight(self, insight_id: str, text: str) -> None:
        """
        Add an insight to the known set (backward-compatible method).

        This method provides compatibility with the explorer's existing interface.

        Args:
            insight_id: The insight's unique identifier
            text: The insight text
        """
        self.add_content(ContentItem(
            id=insight_id,
            text=text,
            content_type=ContentType.INSIGHT,
        ))

    def load_known_insights(self, insights: list[dict]) -> None:
        """
        Load multiple known insights (backward-compatible method).

        This method provides compatibility with the explorer's existing interface.

        Args:
            insights: List of dicts with 'id' and 'text' (or 'insight') keys
        """
        for insight in insights:
            insight_id = insight.get("id", str(len(self._known_items)))
            text = insight.get("text") or insight.get("insight", "")
            self.add_known_insight(insight_id, text)

        log.info(
            "deduplication.insights.loaded",
            count=len(insights),
            total_known=len(self._known_items),
        )

    async def is_duplicate(
        self,
        new_text: str,
        new_id: str = "new",
    ) -> tuple[bool, Optional[str]]:
        """
        Check if text is a duplicate (backward-compatible method).

        This method provides compatibility with the explorer's existing interface.

        Args:
            new_text: The new insight text
            new_id: ID for the new insight

        Returns:
            Tuple of (is_duplicate, duplicate_of_id or None)
        """
        result = await self.check_duplicate(
            new_text,
            item_id=new_id,
            content_type=ContentType.INSIGHT,
        )
        return result.is_duplicate, result.duplicate_of
