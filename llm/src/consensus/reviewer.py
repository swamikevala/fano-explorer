"""
Consensus Reviewer - Main orchestrator for multi-LLM validation.

This is a lightweight coordinator that delegates to specialized task modules.
"""

from typing import Optional

from ..client import LLMClient
from ..models import ConsensusResult, ConsensusRunResult
from .review_task import ReviewTask
from .consensus_task import ConsensusTask
from .quick_check import QuickChecker


class ConsensusReviewer:
    """
    Multi-LLM consensus reviewer for validating insights.

    Orchestrates multiple rounds of review across available LLMs
    to reach reliable consensus on whether an insight should be
    blessed (accepted), rejected, or needs more development.

    This class provides a unified API while delegating to specialized
    task modules for different consensus modes:
    - review(): Multi-round insight validation (ReviewTask)
    - run(): General-purpose consensus (ConsensusTask)
    - quick_check(): Single-LLM fast validation (QuickChecker)

    Usage:
        client = LLMClient()
        reviewer = ConsensusReviewer(client)

        result = await reviewer.review(
            text="The number 84 appears in yoga as 84 asanas...",
            tags=["yoga", "numbers"],
        )

        if result.final_rating == "bless":
            print("Insight validated!")
    """

    def __init__(
        self,
        client: LLMClient,
        config: Optional[dict] = None,
    ):
        """
        Initialize the reviewer.

        Args:
            client: LLMClient for sending requests
            config: Optional configuration dict (reserved for future use)
        """
        self.client = client
        self.config = config or {}

        # Initialize task modules
        self._review_task = ReviewTask(client)
        self._consensus_task = ConsensusTask(client)
        self._quick_checker = QuickChecker(client)

    async def review(
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
        return await self._review_task.run(
            text,
            tags=tags,
            context=context,
            confidence=confidence,
            dependencies=dependencies,
            use_deep_mode=use_deep_mode,
        )

    async def quick_check(
        self,
        text: str,
        *,
        context: str = "",
    ) -> tuple[str, str]:
        """
        Quick single-LLM check (no consensus, just one opinion).

        Useful for fast validation when consensus isn't needed.

        Args:
            text: The text to check
            context: Optional context

        Returns:
            Tuple of (rating, reasoning)
        """
        return await self._quick_checker.check(text, context=context)

    async def run(
        self,
        context: str,
        task: str,
        *,
        response_format: Optional[str] = None,
        max_rounds: int = 3,
        use_deep_mode: bool = False,
        select_best: bool = False,
    ) -> ConsensusRunResult:
        """
        Run a general-purpose consensus task.

        This is the generic API for any consensus task - the caller shapes
        everything through context and task description.

        Args:
            context: Background information for the task
            task: What the LLMs should do (in natural language)
            response_format: Optional format hint (e.g., "DECISION: [yes/no]\\nREASONING: ...")
            max_rounds: Maximum deliberation rounds (default 3)
            use_deep_mode: Whether to use deep/pro modes
            select_best: If True, run a final selection round where LLMs vote on
                        the best response from all candidates (useful for generative tasks)

        Returns:
            ConsensusRunResult with outcome and transcript
        """
        return await self._consensus_task.run(
            context,
            task,
            response_format=response_format,
            max_rounds=max_rounds,
            use_deep_mode=use_deep_mode,
            select_best=select_best,
        )
