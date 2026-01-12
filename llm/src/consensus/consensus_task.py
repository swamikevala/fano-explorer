"""
Consensus Task - General-purpose multi-LLM consensus.

Provides generic consensus orchestration for arbitrary tasks,
with configurable rounds and convergence detection.
"""

import time
from typing import Optional, TYPE_CHECKING

from shared.logging import get_logger

from ..models import ConsensusRunResult
from .prompts import (
    build_consensus_initial_prompt,
    build_consensus_deliberation_prompt,
    build_selection_prompt,
)
from .convergence import (
    check_convergence,
    find_dissent,
    synthesize_outcome,
    CONFIDENCE_THRESHOLD,
)
from .response_parser import parse_vote

if TYPE_CHECKING:
    from ..client import LLMClient

log = get_logger("llm", "consensus.task")

# Configuration constants
MIN_BACKENDS = 2
MAX_CANDIDATES = 6


class ConsensusTask:
    """
    General-purpose consensus task runner.

    Orchestrates multi-round deliberation for arbitrary tasks,
    with convergence detection and optional best-response selection.
    """

    def __init__(self, client: "LLMClient"):
        """
        Initialize consensus task.

        Args:
            client: LLMClient for sending requests
        """
        self.client = client

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

        Args:
            context: Background information for the task
            task: What the LLMs should do (in natural language)
            response_format: Optional format hint
            max_rounds: Maximum deliberation rounds (default 3)
            use_deep_mode: Whether to use deep/pro modes
            select_best: If True, run a final selection round

        Returns:
            ConsensusRunResult with outcome and transcript
        """
        start_time = time.time()

        # Get available backends
        available = await self.client.get_available_backends()
        if len(available) < MIN_BACKENDS:
            error_msg = f"Need at least {MIN_BACKENDS} backends, only {len(available)} available: {available}"
            log.error(
                "llm.consensus.insufficient_backends",
                available_backends=available,
                required=MIN_BACKENDS,
                hint="Check that pool is running (for gemini/chatgpt) or API keys are set",
            )
            return ConsensusRunResult(
                success=False,
                outcome=f"[ERROR: {error_msg}]",
                converged=False,
                confidence=0.0,
                rounds=[{"error": error_msg}],
            )

        log.info(
            "llm.consensus.run_start",
            backends=available,
            task_length=len(task),
            context_length=len(context),
            max_rounds=max_rounds,
            use_deep_mode=use_deep_mode,
        )

        rounds = []
        all_responses: dict[int, dict[str, str]] = {}

        # Round 1: Independent parallel responses
        round1_responses = await self._run_initial_round(
            context, task, response_format, available, use_deep_mode
        )
        all_responses[1] = round1_responses
        rounds.append({"round": 1, "responses": round1_responses})

        # Check for early convergence
        converged, confidence, decision = check_convergence(round1_responses, response_format)
        log.info("llm.consensus.round_complete", round=1, converged=converged, confidence=confidence)

        if converged and confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(
                success=True,
                outcome=decision or synthesize_outcome(round1_responses),
                converged=True,
                confidence=confidence,
                rounds=rounds,
                start_time=start_time,
                rounds_needed=1,
            )

        # Subsequent rounds: Share responses and continue discussion
        prev_responses = round1_responses
        for round_num in range(2, max_rounds + 1):
            round_responses = await self._run_deliberation_round(
                context, task, response_format, prev_responses, available, use_deep_mode
            )
            all_responses[round_num] = round_responses
            rounds.append({"round": round_num, "responses": round_responses})

            # Check for convergence
            converged, confidence, decision = check_convergence(round_responses, response_format)
            log.info(
                "llm.consensus.round_complete",
                round=round_num,
                converged=converged,
                confidence=confidence,
            )

            if converged and confidence >= CONFIDENCE_THRESHOLD:
                return self._build_result(
                    success=True,
                    outcome=decision or synthesize_outcome(round_responses),
                    converged=True,
                    confidence=confidence,
                    rounds=rounds,
                    start_time=start_time,
                    rounds_needed=round_num,
                )

            prev_responses = round_responses

        # Max rounds reached - synthesize final outcome
        final_responses = all_responses[max_rounds]
        converged, confidence, decision = check_convergence(final_responses, response_format)

        # Find dissent if not converged
        dissent = None
        if not converged:
            dissent = find_dissent(final_responses, response_format)

        # Select best outcome if requested
        selection_stats = None
        if select_best and not decision:
            outcome, selection_stats = await self._select_best_outcome(
                all_responses, context, task
            )
            log.info("llm.consensus.selection_complete", selection_stats=selection_stats)
        else:
            outcome = decision or synthesize_outcome(final_responses)

        return self._build_result(
            success=True,
            outcome=outcome,
            converged=converged,
            confidence=confidence,
            rounds=rounds,
            start_time=start_time,
            rounds_needed=max_rounds,
            dissent=dissent,
            selection_stats=selection_stats,
        )

    async def _run_initial_round(
        self,
        context: str,
        task: str,
        response_format: Optional[str],
        backends: list[str],
        use_deep_mode: bool,
    ) -> dict[str, str]:
        """Run initial round with independent parallel responses."""
        prompt = build_consensus_initial_prompt(context, task, response_format)
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=use_deep_mode)

        result = {}
        for backend, response in responses.items():
            if response.success:
                result[backend] = response.text
            else:
                result[backend] = f"[Error: {response.error} - {response.message}]"

        return result

    async def _run_deliberation_round(
        self,
        context: str,
        task: str,
        response_format: Optional[str],
        prev_responses: dict[str, str],
        backends: list[str],
        use_deep_mode: bool,
    ) -> dict[str, str]:
        """Run deliberation round with previous responses visible."""
        prompt = build_consensus_deliberation_prompt(
            context, task, prev_responses, response_format
        )
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=use_deep_mode)

        result = {}
        for backend, response in responses.items():
            if response.success:
                result[backend] = response.text
            else:
                result[backend] = f"[Error: {response.error} - {response.message}]"

        return result

    async def _select_best_outcome(
        self,
        all_responses: dict[int, dict[str, str]],
        context: str,
        task: str,
    ) -> tuple[str, dict]:
        """
        Have LLMs vote on which response is best.

        Returns: (winning_text, selection_stats)
        """
        # Collect all unique valid responses
        candidates = []
        seen_texts = set()
        for round_num, responses in all_responses.items():
            for backend, text in responses.items():
                if text.startswith("[Error"):
                    continue
                # Deduplicate identical responses
                text_hash = hash(text[:500])
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    candidates.append({
                        "round": round_num,
                        "backend": backend,
                        "text": text,
                    })

        if not candidates:
            return "No valid responses received.", {"error": "no_candidates"}

        if len(candidates) == 1:
            return candidates[0]["text"], {"single_candidate": True}

        # Limit to best candidates if too many
        if len(candidates) > MAX_CANDIDATES:
            candidates = sorted(candidates, key=lambda c: -c["round"])[:MAX_CANDIDATES]

        # Build selection prompt and get votes
        prompt = build_selection_prompt(context, task, candidates)
        available = await self.client.get_available_backends()
        prompts = {backend: prompt for backend in available}
        responses = await self.client.send_parallel(prompts, deep_mode=False)

        # Count votes
        votes: dict[str, list[str]] = {}
        for backend, response in responses.items():
            if not response.success:
                continue
            vote = parse_vote(response.text)
            if vote:
                if vote not in votes:
                    votes[vote] = []
                votes[vote].append(backend)

        log.info(
            "llm.consensus.selection_votes",
            votes={k: len(v) for k, v in votes.items()},
            total_candidates=len(candidates),
        )

        # Find winner
        if votes:
            winner_letter = max(votes, key=lambda k: len(votes[k]))
            winner_index = ord(winner_letter) - 65
            if 0 <= winner_index < len(candidates):
                winner = candidates[winner_index]
                return winner["text"], {
                    "winner": winner_letter,
                    "votes": {k: len(v) for k, v in votes.items()},
                    "winner_backend": winner["backend"],
                    "winner_round": winner["round"],
                }

        # Fallback to longest if voting failed
        longest = max(candidates, key=lambda c: len(c["text"]))
        return longest["text"], {"fallback": "longest", "reason": "voting_failed"}

    def _build_result(
        self,
        success: bool,
        outcome: str,
        converged: bool,
        confidence: float,
        rounds: list,
        start_time: float,
        rounds_needed: int,
        dissent: Optional[str] = None,
        selection_stats: Optional[dict] = None,
    ) -> ConsensusRunResult:
        """Build ConsensusRunResult with timing and logging."""
        elapsed = time.time() - start_time
        log.info(
            "llm.consensus.run_complete",
            rounds_needed=rounds_needed,
            converged=converged,
            confidence=confidence,
            has_dissent=dissent is not None,
            used_selection=selection_stats is not None,
            duration_ms=round(elapsed * 1000, 2),
        )

        return ConsensusRunResult(
            success=success,
            outcome=outcome,
            converged=converged,
            confidence=confidence,
            rounds=rounds,
            dissent=dissent,
            duration_seconds=elapsed,
            selection_stats=selection_stats,
        )
