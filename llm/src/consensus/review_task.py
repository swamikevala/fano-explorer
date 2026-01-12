"""
Review Task - Multi-round insight validation.

Implements the 3-round review process:
1. Round 1: Independent parallel review
2. Round 2: Deep analysis with Round 1 context
3. Round 3: Deliberation if still split
"""

import time
from typing import Optional, TYPE_CHECKING

from shared.logging import get_logger

from ..models import ConsensusResult, ReviewResponse
from .prompts import build_review_round1_prompt, build_review_round2_prompt
from .response_parser import parse_review_response, create_error_response
from .convergence import check_rating_convergence

if TYPE_CHECKING:
    from ..client import LLMClient

log = get_logger("llm", "consensus.review")


class ReviewTask:
    """
    Multi-round consensus review for insight validation.

    Orchestrates 3 rounds of review to reach consensus on whether
    an insight should be blessed, rejected, or needs more work.
    """

    def __init__(self, client: "LLMClient"):
        """
        Initialize review task.

        Args:
            client: LLMClient for sending requests
        """
        self.client = client

    async def run(
        self,
        text: str,
        *,
        tags: Optional[list[str]] = None,
        context: str = "",
        confidence: str = "medium",
        dependencies: Optional[list[str]] = None,
        use_deep_mode: bool = True,
    ) -> ConsensusResult:
        """
        Run multi-LLM consensus review on a piece of text.

        Process:
        1. Round 1: Independent parallel review (standard modes)
        2. Round 2: Deep analysis with Round 1 responses visible
        3. Round 3: Structured deliberation if still split

        Args:
            text: The insight/claim to review
            tags: Tags for context
            context: Additional context (e.g., blessed axioms)
            confidence: Confidence level from extraction
            dependencies: Dependencies on other insights
            use_deep_mode: Whether to use deep/pro modes in Round 2

        Returns:
            ConsensusResult with final rating and review history
        """
        start_time = time.time()
        tags = tags or []
        dependencies = dependencies or []

        # Get available backends
        available = await self.client.get_available_backends()
        if len(available) < 2:
            return ConsensusResult(
                success=False,
                final_rating="uncertain",
                is_unanimous=False,
                is_disputed=True,
                rounds=[{"error": f"Need at least 2 backends, only {len(available)} available"}],
            )

        log.info(
            "llm.consensus.review_start",
            backends=available,
            text_length=len(text),
            tags=tags,
            use_deep_mode=use_deep_mode,
        )

        # Round 1: Independent parallel review
        round1_responses = await self._run_round1(
            text, tags, context, confidence, dependencies, available
        )

        # Check for early exit (unanimous)
        ratings = [r.rating for r in round1_responses.values()]
        is_unanimous, majority_rating = check_rating_convergence(ratings)

        log.info(
            "llm.consensus.round_complete",
            round=1,
            ratings={k: v.rating for k, v in round1_responses.items()},
            unanimous=is_unanimous,
        )

        if is_unanimous:
            elapsed = time.time() - start_time
            log.info(
                "llm.consensus.review_complete",
                final_rating=majority_rating,
                rounds_needed=1,
                is_unanimous=True,
                duration_ms=round(elapsed * 1000, 2),
            )
            return ConsensusResult(
                success=True,
                final_rating=majority_rating,
                is_unanimous=True,
                is_disputed=False,
                rounds=[{"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}}],
                review_duration_seconds=elapsed,
            )

        # Round 2: Deep analysis with Round 1 context
        round2_responses = await self._run_round2(
            text, round1_responses, context, available, use_deep_mode
        )

        # Check for resolution and track mind changes
        ratings = [r.rating for r in round2_responses.values()]
        self._log_mind_changes(round1_responses, round2_responses)

        is_unanimous, majority_rating = check_rating_convergence(ratings)

        log.info(
            "llm.consensus.round_complete",
            round=2,
            ratings={k: v.rating for k, v in round2_responses.items()},
            unanimous=is_unanimous,
        )

        if is_unanimous:
            elapsed = time.time() - start_time
            log.info(
                "llm.consensus.review_complete",
                final_rating=majority_rating,
                rounds_needed=2,
                is_unanimous=True,
                duration_ms=round(elapsed * 1000, 2),
            )
            return ConsensusResult(
                success=True,
                final_rating=majority_rating,
                is_unanimous=True,
                is_disputed=False,
                rounds=[
                    {"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}},
                    {"round": 2, "responses": {k: v.to_dict() for k, v in round2_responses.items()}},
                ],
                review_duration_seconds=elapsed,
            )

        # Round 3: Deliberation if still split
        final_rating, is_disputed = self._run_round3(round2_responses)

        elapsed = time.time() - start_time
        log.info(
            "llm.consensus.review_complete",
            final_rating=final_rating,
            rounds_needed=3,
            is_unanimous=not is_disputed,
            is_disputed=is_disputed,
            duration_ms=round(elapsed * 1000, 2),
        )

        return ConsensusResult(
            success=True,
            final_rating=final_rating,
            is_unanimous=not is_disputed,
            is_disputed=is_disputed,
            rounds=[
                {"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}},
                {"round": 2, "responses": {k: v.to_dict() for k, v in round2_responses.items()}},
                {"round": 3, "final_rating": final_rating, "disputed": is_disputed},
            ],
            review_duration_seconds=elapsed,
        )

    async def _run_round1(
        self,
        text: str,
        tags: list[str],
        context: str,
        confidence: str,
        dependencies: list[str],
        backends: list[str],
    ) -> dict[str, ReviewResponse]:
        """Run Round 1: Independent parallel review."""
        prompt = build_review_round1_prompt(text, tags, context, confidence, dependencies)

        # Send to all backends in parallel
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=False)

        # Parse responses
        parsed = {}
        for backend, response in responses.items():
            if response.success:
                parsed[backend] = parse_review_response(backend, response.text, "standard")
            else:
                parsed[backend] = create_error_response(
                    backend, "standard", response.error, response.message
                )

        return parsed

    async def _run_round2(
        self,
        text: str,
        round1: dict[str, ReviewResponse],
        context: str,
        backends: list[str],
        use_deep_mode: bool,
    ) -> dict[str, ReviewResponse]:
        """Run Round 2: Deep analysis with Round 1 context."""
        prompt = build_review_round2_prompt(text, round1, context)

        # Send to all backends in parallel with deep mode
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=use_deep_mode)

        # Parse responses
        parsed = {}
        for backend, response in responses.items():
            mode = "deep" if use_deep_mode and response.deep_mode_used else "standard"
            if response.success:
                parsed[backend] = parse_review_response(backend, response.text, mode)
            else:
                parsed[backend] = create_error_response(
                    backend, mode, response.error, response.message
                )

        return parsed

    def _run_round3(
        self,
        round2: dict[str, ReviewResponse],
    ) -> tuple[str, bool]:
        """Run Round 3: Deliberation to reach final decision."""
        # Get majority rating from Round 2
        ratings = [r.rating for r in round2.values()]
        rating_counts = {}
        for r in ratings:
            rating_counts[r] = rating_counts.get(r, 0) + 1

        # Find majority (2 out of 3)
        for rating, count in rating_counts.items():
            if count >= 2:
                return rating, False

        # No majority - use "uncertain" as default
        return "uncertain", True

    def _log_mind_changes(
        self,
        round1: dict[str, ReviewResponse],
        round2: dict[str, ReviewResponse],
    ) -> None:
        """Log any mind changes between rounds."""
        for backend in round1:
            if backend in round2:
                r1_rating = round1[backend].rating
                r2_rating = round2[backend].rating
                if r1_rating != r2_rating:
                    log.info(
                        "llm.consensus.mind_change",
                        llm=backend,
                        from_rating=r1_rating,
                        to_rating=r2_rating,
                    )
