"""
Automated Review Panel Coordinator

Orchestrates the review process with refinement support:
1. Round 1: Independent parallel review (standard modes)
2. If mixed with fixable issues → Refinement Round (Claude Opus rewrites)
3. Post-Refinement Review (all 3 review refined version)
4. If still mixed → Round 3: Structured deliberation
5. Final outcome applied with appropriate flags

Claude Opus writes extraction and refinement; all three judge.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    ChunkReview,
    ReviewDecision,
    should_refine_vs_deliberate,
    get_rating_pattern,
)
from .round1 import run_round1
from .round2 import run_round2
from .round3 import run_round3
from .refinement import run_refinement_round, run_post_refinement_review
from .claude_api import get_claude_reviewer, ClaudeReviewer

logger = logging.getLogger(__name__)


class AutomatedReviewer:
    """
    Coordinates the three-round automated review process.

    Uses browser automation for Gemini/ChatGPT and API for Claude.
    """

    def __init__(
        self,
        gemini_browser,
        chatgpt_browser,
        config: dict,
        data_dir: Path,
    ):
        """
        Initialize the automated reviewer.

        Args:
            gemini_browser: GeminiBrowser instance (or None)
            chatgpt_browser: ChatGPTBrowser instance (or None)
            config: Full configuration dict
            data_dir: Base data directory for saving reviews
        """
        self.gemini_browser = gemini_browser
        self.chatgpt_browser = chatgpt_browser
        self.config = config
        self.data_dir = data_dir

        # Get review panel config
        self.panel_config = config.get("review_panel", {})

        # Initialize Claude reviewer
        self.claude_reviewer = get_claude_reviewer(self.panel_config)
        if self.claude_reviewer:
            logger.info("[reviewer] Claude API available")
        else:
            logger.warning("[reviewer] Claude API not available")

        # Check we have at least 2 reviewers
        available = sum([
            1 if gemini_browser else 0,
            1 if chatgpt_browser else 0,
            1 if self.claude_reviewer else 0,
        ])
        if available < 2:
            logger.warning(f"[reviewer] Only {available} reviewers available, need at least 2")

    async def review_insight(
        self,
        chunk_id: str,
        insight_text: str,
        confidence: str,
        tags: list[str],
        dependencies: list[str],
        blessed_axioms_summary: str,
    ) -> ChunkReview:
        """
        Run the full review process on an insight.

        Flow:
        1. Round 1: Independent review
        2. If unanimous → Done
        3. If mixed, check if refinement or deliberation is needed
        4. If refinement → Claude Opus rewrites, then post-refinement review
        5. If still mixed → Round 3 deliberation
        6. Final rating applied

        Args:
            chunk_id: Unique ID for this insight
            insight_text: The insight text to review
            confidence: Confidence level from extraction
            tags: Tags assigned to the insight
            dependencies: Dependencies listed for the insight
            blessed_axioms_summary: Summary of blessed axioms for context

        Returns:
            ChunkReview with complete review history and final rating
        """
        start_time = time.time()
        logger.info(f"[reviewer] Starting review for {chunk_id}")

        review = ChunkReview(chunk_id=chunk_id)
        current_insight = insight_text
        current_version = 1
        max_refinement_rounds = self.panel_config.get("refinement", {}).get("max_refinement_rounds", 2)

        try:
            # Round 1: Independent review
            round1 = await run_round1(
                chunk_insight=current_insight,
                confidence=confidence,
                tags=tags,
                dependencies=dependencies,
                blessed_axioms_summary=blessed_axioms_summary,
                gemini_browser=self.gemini_browser,
                chatgpt_browser=self.chatgpt_browser,
                claude_reviewer=self.claude_reviewer,
                config=self.panel_config,
            )
            review.add_round(round1)

            pattern = get_rating_pattern(round1)
            logger.info(f"[reviewer] Round 1 complete: {pattern}")

            # Check for early exit (unanimous)
            if round1.outcome == "unanimous":
                final_rating = list(round1.get_ratings().values())[0]
                review.finalize(
                    rating=final_rating,
                    unanimous=True,
                    disputed=False,
                )
                logger.info(f"[reviewer] Unanimous after Round 1: {final_rating}")
                self._save_review(review, start_time)
                return review

            # Determine: Refine or Deliberate?
            should_refine, refine_reason = should_refine_vs_deliberate(round1)
            logger.info(f"[reviewer] Decision: {'Refine' if should_refine else 'Deliberate'} ({refine_reason})")

            last_round = round1
            refinement_attempts = 0

            # Refinement loop (up to max_refinement_rounds)
            while should_refine and refinement_attempts < max_refinement_rounds:
                refinement_attempts += 1
                logger.info(f"[reviewer] Refinement attempt {refinement_attempts}/{max_refinement_rounds}")

                # Run refinement round (Claude Opus rewrites)
                refinement, refined_insight = await run_refinement_round(
                    original_insight=current_insight,
                    confidence=confidence,
                    tags=tags,
                    dependencies=dependencies,
                    round1=last_round,
                    claude_reviewer=self.claude_reviewer,
                    current_version=current_version,
                )

                if not refinement:
                    logger.warning("[reviewer] Refinement failed, proceeding to deliberation")
                    break

                # Record refinement
                review.add_refinement(refinement)
                current_insight = refined_insight
                current_version = refinement.to_version

                # Post-refinement review
                post_refine_round, mind_changes = await run_post_refinement_review(
                    original_insight=insight_text,
                    refined_insight=refined_insight,
                    refinement=refinement,
                    round1=last_round,
                    gemini_browser=self.gemini_browser,
                    chatgpt_browser=self.chatgpt_browser,
                    claude_reviewer=self.claude_reviewer,
                    config=self.panel_config,
                )
                review.add_round(post_refine_round)
                review.mind_changes.extend(mind_changes)

                pattern = get_rating_pattern(post_refine_round)
                logger.info(f"[reviewer] Post-refinement review: {pattern}")

                # Check if refinement resolved the split
                if post_refine_round.outcome == "unanimous":
                    final_rating = list(post_refine_round.get_ratings().values())[0]
                    review.finalize(
                        rating=final_rating,
                        unanimous=True,
                        disputed=False,
                    )
                    logger.info(f"[reviewer] Unanimous after refinement: {final_rating}")
                    self._save_review(review, start_time)
                    return review

                # Check if we should try another refinement or deliberate
                last_round = post_refine_round
                should_refine, refine_reason = should_refine_vs_deliberate(post_refine_round)

                if not should_refine:
                    logger.info(f"[reviewer] Moving to deliberation: {refine_reason}")

            # Round 2/3: Deep analysis or deliberation
            # At this point we've either skipped refinement or exhausted refinement attempts
            round2, mind_changes_r2 = await run_round2(
                chunk_insight=current_insight,
                blessed_axioms_summary=blessed_axioms_summary,
                round1=last_round,
                gemini_browser=self.gemini_browser,
                chatgpt_browser=self.chatgpt_browser,
                claude_reviewer=self.claude_reviewer,
                config=self.panel_config,
            )
            review.add_round(round2)
            review.mind_changes.extend(mind_changes_r2)

            pattern = get_rating_pattern(round2)
            logger.info(f"[reviewer] Round 2 (deep analysis): {pattern}")

            # Check for resolution
            if round2.outcome == "unanimous":
                final_rating = list(round2.get_ratings().values())[0]
                review.finalize(
                    rating=final_rating,
                    unanimous=True,
                    disputed=False,
                )
                logger.info(f"[reviewer] Unanimous after Round 2: {final_rating}")
                self._save_review(review, start_time)
                return review

            # Round 3: Structured deliberation
            if self.panel_config.get("round3", {}).get("enabled", True):
                round3, mind_changes_r3, is_disputed = await run_round3(
                    chunk_insight=current_insight,
                    round2=round2,
                    gemini_browser=self.gemini_browser,
                    chatgpt_browser=self.chatgpt_browser,
                    claude_reviewer=self.claude_reviewer,
                    config=self.panel_config,
                )
                review.add_round(round3)
                review.mind_changes.extend(mind_changes_r3)

                # Determine final rating (majority wins)
                final_rating = round3.get_majority_rating() or "?"
                review.finalize(
                    rating=final_rating,
                    unanimous=(round3.outcome == "resolved"),
                    disputed=is_disputed,
                )
            else:
                # Skip Round 3, use Round 2 majority
                final_rating = round2.get_majority_rating() or "?"
                review.finalize(
                    rating=final_rating,
                    unanimous=False,
                    disputed=True,
                )

            logger.info(f"[reviewer] Final rating: {final_rating} "
                       f"(disputed={review.is_disputed}, refined={review.was_refined})")
            self._save_review(review, start_time)
            return review

        except Exception as e:
            logger.error(f"[reviewer] Review failed: {e}")
            review.finalize(rating="?", unanimous=False, disputed=True)
            self._save_review(review, start_time)
            raise

    def _save_review(self, review: ChunkReview, start_time: float):
        """Save review to disk and update duration."""
        review.review_duration_seconds = time.time() - start_time
        review.save(self.data_dir)
        logger.info(f"[reviewer] Saved review for {review.chunk_id} ({review.review_duration_seconds:.1f}s)")

    def get_outcome_action(self, review: ChunkReview) -> str:
        """
        Determine what action to take based on review outcome.

        Uses the outcomes config to map results to actions.

        Args:
            review: Completed review

        Returns:
            Action string: "auto_bless", "auto_reject", "needs_development",
                          "bless_with_flag", "reject_with_flag"
        """
        outcomes_config = self.panel_config.get("outcomes", {})

        if review.is_unanimous:
            if review.final_rating == "⚡":
                return outcomes_config.get("unanimous_bless", "auto_bless")
            elif review.final_rating == "✗":
                return outcomes_config.get("unanimous_reject", "auto_reject")
            else:
                return outcomes_config.get("unanimous_uncertain", "needs_development")
        else:
            # Disputed
            if review.final_rating == "⚡":
                return outcomes_config.get("disputed_majority_bless", "bless_with_flag")
            elif review.final_rating == "✗":
                return outcomes_config.get("disputed_majority_reject", "reject_with_flag")
            else:
                return "needs_development"

    async def review_batch(
        self,
        insights: list[dict],
        blessed_axioms_summary: str,
    ) -> list[ChunkReview]:
        """
        Review a batch of insights sequentially.

        Args:
            insights: List of insight dicts with id, text, confidence, tags, dependencies
            blessed_axioms_summary: Summary of blessed axioms

        Returns:
            List of ChunkReview results
        """
        results = []

        for insight in insights:
            try:
                review = await self.review_insight(
                    chunk_id=insight["id"],
                    insight_text=insight["text"],
                    confidence=insight.get("confidence", "medium"),
                    tags=insight.get("tags", []),
                    dependencies=insight.get("dependencies", []),
                    blessed_axioms_summary=blessed_axioms_summary,
                )
                results.append(review)
            except Exception as e:
                logger.error(f"[reviewer] Failed to review {insight['id']}: {e}")
                # Create a failed review
                review = ChunkReview(chunk_id=insight["id"])
                review.finalize(rating="?", unanimous=False, disputed=True)
                results.append(review)

        return results
