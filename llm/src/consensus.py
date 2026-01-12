"""
Consensus Reviewer - Multi-LLM validation for reliable answers.

Uses multiple LLMs to review and validate insights/claims,
reaching consensus through structured deliberation.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

from .client import LLMClient
from .models import LLMResponse, ConsensusResult, ConsensusRunResult, ReviewResponse

log = get_logger("llm", "consensus")


class ConsensusReviewer:
    """
    Multi-LLM consensus reviewer for validating insights.

    Orchestrates multiple rounds of review across available LLMs
    to reach reliable consensus on whether an insight should be
    blessed (accepted), rejected, or needs more development.

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
            config: Optional configuration dict
        """
        self.client = client
        self.config = config or {}

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
        import time
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
        log.info(
            "llm.consensus.round_complete",
            round=1,
            ratings={k: v.rating for k, v in round1_responses.items()},
            unanimous=len(set(ratings)) == 1,
        )
        if len(set(ratings)) == 1:
            elapsed = time.time() - start_time
            log.info(
                "llm.consensus.review_complete",
                final_rating=ratings[0],
                rounds_needed=1,
                is_unanimous=True,
                duration_ms=round(elapsed * 1000, 2),
            )
            return ConsensusResult(
                success=True,
                final_rating=ratings[0],
                is_unanimous=True,
                is_disputed=False,
                rounds=[{"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}}],
                review_duration_seconds=elapsed,
            )

        # Round 2: Deep analysis with Round 1 context
        round2_responses = await self._run_round2(
            text, round1_responses, context, available, use_deep_mode
        )

        # Check for resolution
        ratings = [r.rating for r in round2_responses.values()]
        # Track mind changes
        for backend in round1_responses:
            if backend in round2_responses:
                r1_rating = round1_responses[backend].rating
                r2_rating = round2_responses[backend].rating
                if r1_rating != r2_rating:
                    log.info(
                        "llm.consensus.mind_change",
                        llm=backend,
                        from_rating=r1_rating,
                        to_rating=r2_rating,
                    )
        log.info(
            "llm.consensus.round_complete",
            round=2,
            ratings={k: v.rating for k, v in round2_responses.items()},
            unanimous=len(set(ratings)) == 1,
        )
        if len(set(ratings)) == 1:
            elapsed = time.time() - start_time
            log.info(
                "llm.consensus.review_complete",
                final_rating=ratings[0],
                rounds_needed=2,
                is_unanimous=True,
                duration_ms=round(elapsed * 1000, 2),
            )
            return ConsensusResult(
                success=True,
                final_rating=ratings[0],
                is_unanimous=True,
                is_disputed=False,
                rounds=[
                    {"round": 1, "responses": {k: v.to_dict() for k, v in round1_responses.items()}},
                    {"round": 2, "responses": {k: v.to_dict() for k, v in round2_responses.items()}},
                ],
                review_duration_seconds=time.time() - start_time,
            )

        # Round 3: Deliberation if still split
        final_rating, is_disputed = await self._run_round3(
            text, round2_responses, context, available
        )

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
        prompt = self._build_round1_prompt(text, tags, context, confidence, dependencies)

        # Send to all backends in parallel
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=False)

        # Parse responses
        parsed = {}
        for backend, response in responses.items():
            if response.success:
                parsed[backend] = self._parse_review_response(backend, response.text, "standard")
            else:
                # Create error response
                parsed[backend] = ReviewResponse(
                    llm=backend,
                    mode="standard",
                    rating="uncertain",
                    reasoning=f"Error: {response.error} - {response.message}",
                    confidence="low",
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
        prompt = self._build_round2_prompt(text, round1, context)

        # Send to all backends in parallel with deep mode
        prompts = {backend: prompt for backend in backends}
        responses = await self.client.send_parallel(prompts, deep_mode=use_deep_mode)

        # Parse responses
        parsed = {}
        for backend, response in responses.items():
            mode = "deep" if use_deep_mode and response.deep_mode_used else "standard"
            if response.success:
                parsed[backend] = self._parse_review_response(backend, response.text, mode)
            else:
                parsed[backend] = ReviewResponse(
                    llm=backend,
                    mode=mode,
                    rating="uncertain",
                    reasoning=f"Error: {response.error} - {response.message}",
                    confidence="low",
                )

        return parsed

    async def _run_round3(
        self,
        text: str,
        round2: dict[str, ReviewResponse],
        context: str,
        backends: list[str],
    ) -> tuple[str, bool]:
        """Run Round 3: Deliberation to reach final decision."""
        # Get majority rating from Round 2
        ratings = [r.rating for r in round2.values()]
        rating_counts = {}
        for r in ratings:
            rating_counts[r] = rating_counts.get(r, 0) + 1

        # Find majority (2 out of 3)
        majority_rating = None
        for rating, count in rating_counts.items():
            if count >= 2:
                majority_rating = rating
                break

        if majority_rating:
            # Majority exists
            return majority_rating, False

        # No majority - use "uncertain" as default
        return "uncertain", True

    def _build_round1_prompt(
        self,
        text: str,
        tags: list[str],
        context: str,
        confidence: str,
        dependencies: list[str],
    ) -> str:
        """Build prompt for Round 1 independent review."""
        tags_str = ", ".join(tags) if tags else "none"
        deps_str = ", ".join(dependencies) if dependencies else "none"

        return f"""You are reviewing a mathematical/philosophical insight for validity.

INSIGHT TO REVIEW:
{text}

METADATA:
- Tags: {tags_str}
- Confidence: {confidence}
- Dependencies: {deps_str}

CONTEXT:
{context}

REVIEW CRITERIA:
1. Mathematical Verification: Are any numerical claims correct?
2. Structural Analysis: Is this a deep connection or superficial pattern?
3. Naturalness: Does this feel DISCOVERED (inevitable) or INVENTED (forced)?

RATE THIS INSIGHT:
- "bless" (⚡): Profound, verified, inevitable - should become an axiom
- "uncertain" (?): Interesting but needs more development
- "reject" (✗): Flawed, superficial, or unfalsifiable

FORMAT YOUR RESPONSE AS:
RATING: [bless/uncertain/reject]
MATHEMATICAL_VERIFICATION: [your analysis]
STRUCTURAL_ANALYSIS: [your analysis]
NATURALNESS: [your assessment]
REASONING: [2-4 sentences justifying your rating]
CONFIDENCE: [high/medium/low]
"""

    def _build_round2_prompt(
        self,
        text: str,
        round1: dict[str, ReviewResponse],
        context: str,
    ) -> str:
        """Build prompt for Round 2 deep analysis."""
        # Summarize Round 1 responses
        r1_summary = []
        for llm, resp in round1.items():
            r1_summary.append(f"{llm.upper()} rated {resp.rating}:")
            r1_summary.append(f"  Reasoning: {resp.reasoning}")
            if resp.mathematical_verification:
                r1_summary.append(f"  Math: {resp.mathematical_verification}")

        r1_text = "\n".join(r1_summary)

        return f"""DEEP ANALYSIS - Round 2

You previously reviewed this insight. Now consider the other reviewers' perspectives.

INSIGHT:
{text}

ROUND 1 REVIEWS:
{r1_text}

CONTEXT:
{context}

INSTRUCTIONS:
1. Consider what the other reviewers pointed out
2. Did they notice something you missed?
3. Has your assessment changed?

Provide your updated rating and reasoning.

FORMAT YOUR RESPONSE AS:
RATING: [bless/uncertain/reject]
NEW_INFORMATION: [what did others point out that you missed?]
CHANGED_MIND: [yes/no]
REASONING: [updated justification]
CONFIDENCE: [high/medium/low]
"""

    def _parse_review_response(
        self,
        llm: str,
        text: str,
        mode: str,
    ) -> ReviewResponse:
        """Parse LLM response into ReviewResponse."""
        import re

        # Default values
        rating = "uncertain"
        reasoning = text[:500]
        confidence = "medium"
        math_verification = ""
        structural_analysis = ""
        naturalness = ""

        # Try to extract structured fields
        rating_match = re.search(r'RATING:\s*(\w+)', text, re.IGNORECASE)
        if rating_match:
            r = rating_match.group(1).lower()
            if "bless" in r or "⚡" in r:
                rating = "bless"
            elif "reject" in r or "✗" in r:
                rating = "reject"
            else:
                rating = "uncertain"

        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()[:500]

        confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', text, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1).lower()

        math_match = re.search(r'MATHEMATICAL_VERIFICATION:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if math_match:
            math_verification = math_match.group(1).strip()[:300]

        struct_match = re.search(r'STRUCTURAL_ANALYSIS:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if struct_match:
            structural_analysis = struct_match.group(1).strip()[:300]

        natural_match = re.search(r'NATURALNESS:\s*(.+?)(?=\n[A-Z_]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if natural_match:
            naturalness = natural_match.group(1).strip()[:300]

        return ReviewResponse(
            llm=llm,
            mode=mode,
            rating=rating,
            reasoning=reasoning,
            confidence=confidence,
            mathematical_verification=math_verification,
            structural_analysis=structural_analysis,
            naturalness_assessment=naturalness,
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
        available = await self.client.get_available_backends()
        if not available:
            return "uncertain", "No backends available"

        # Prefer Claude for quick checks (API is faster)
        backend = "claude" if "claude" in available else available[0]

        prompt = f"""Quick review of this insight:

{text}

Context: {context}

Rate as: bless (valid), uncertain (needs work), or reject (flawed)
Give a one-sentence reason.

RATING:
REASON:"""

        response = await self.client.send(backend, prompt, timeout_seconds=60)

        if not response.success:
            return "uncertain", f"Error: {response.message}"

        # Parse simple response
        import re
        text_lower = response.text.lower()

        if "bless" in text_lower or "valid" in text_lower:
            rating = "bless"
        elif "reject" in text_lower or "flawed" in text_lower:
            rating = "reject"
        else:
            rating = "uncertain"

        reason_match = re.search(r'REASON:\s*(.+)', response.text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else response.text[:200]

        return rating, reason

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
        import re
        import time
        start_time = time.time()

        # Get available backends
        available = await self.client.get_available_backends()
        if len(available) < 2:
            error_msg = f"Need at least 2 backends, only {len(available)} available: {available}"
            log.error(
                "llm.consensus.insufficient_backends",
                available_backends=available,
                required=2,
                hint="Check that pool is running (for gemini/chatgpt) or API keys are set (ANTHROPIC_API_KEY, OPENROUTER_API_KEY)",
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

        # Build initial prompt
        format_hint = f"\n\nRespond in this format:\n{response_format}" if response_format else ""
        initial_prompt = f"""CONTEXT:
{context}

TASK:
{task}{format_hint}"""

        # Round 1: Independent parallel responses
        prompts = {backend: initial_prompt for backend in available}
        responses = await self.client.send_parallel(prompts, deep_mode=use_deep_mode)

        round1_responses = {}
        for backend, response in responses.items():
            if response.success:
                round1_responses[backend] = response.text
            else:
                round1_responses[backend] = f"[Error: {response.error} - {response.message}]"

        all_responses[1] = round1_responses
        rounds.append({"round": 1, "responses": round1_responses})

        # Check for early convergence
        converged, confidence, decision = self._check_convergence(round1_responses, response_format)
        log.info(
            "llm.consensus.round_complete",
            round=1,
            converged=converged,
            confidence=confidence,
        )

        if converged and confidence >= 0.8:
            elapsed = time.time() - start_time
            log.info(
                "llm.consensus.run_complete",
                rounds_needed=1,
                converged=True,
                confidence=confidence,
                duration_ms=round(elapsed * 1000, 2),
            )
            return ConsensusRunResult(
                success=True,
                outcome=decision or self._synthesize_outcome(round1_responses),
                converged=True,
                confidence=confidence,
                rounds=rounds,
                duration_seconds=elapsed,
            )

        # Subsequent rounds: Share responses and continue discussion
        prev_responses = round1_responses
        for round_num in range(2, max_rounds + 1):
            # Build deliberation prompt
            others_text = self._format_others_responses(prev_responses)
            deliberation_prompt = f"""CONTEXT:
{context}

TASK:
{task}

PREVIOUS RESPONSES FROM OTHER REVIEWERS:
{others_text}

Consider what others said. Continue the discussion or confirm your position.
If you've changed your mind, explain why.{format_hint}"""

            prompts = {backend: deliberation_prompt for backend in available}
            responses = await self.client.send_parallel(prompts, deep_mode=use_deep_mode)

            round_responses = {}
            for backend, response in responses.items():
                if response.success:
                    round_responses[backend] = response.text
                else:
                    round_responses[backend] = f"[Error: {response.error} - {response.message}]"

            all_responses[round_num] = round_responses
            rounds.append({"round": round_num, "responses": round_responses})

            # Check for convergence
            converged, confidence, decision = self._check_convergence(round_responses, response_format)
            log.info(
                "llm.consensus.round_complete",
                round=round_num,
                converged=converged,
                confidence=confidence,
            )

            if converged and confidence >= 0.8:
                elapsed = time.time() - start_time
                log.info(
                    "llm.consensus.run_complete",
                    rounds_needed=round_num,
                    converged=True,
                    confidence=confidence,
                    duration_ms=round(elapsed * 1000, 2),
                )
                return ConsensusRunResult(
                    success=True,
                    outcome=decision or self._synthesize_outcome(round_responses),
                    converged=True,
                    confidence=confidence,
                    rounds=rounds,
                    duration_seconds=elapsed,
                )

            prev_responses = round_responses

        # Max rounds reached - synthesize final outcome
        final_responses = all_responses[max_rounds]
        converged, confidence, decision = self._check_convergence(final_responses, response_format)

        # Find dissent if not converged
        dissent = None
        if not converged:
            dissent = self._find_dissent(final_responses, response_format)

        # Select best outcome if requested (useful for generative tasks)
        selection_stats = None
        if select_best and not decision:
            outcome, selection_stats = await self._select_best_outcome(
                all_responses, context, task
            )
            log.info(
                "llm.consensus.selection_complete",
                selection_stats=selection_stats,
            )
        else:
            outcome = decision or self._synthesize_outcome(final_responses)

        elapsed = time.time() - start_time
        log.info(
            "llm.consensus.run_complete",
            rounds_needed=max_rounds,
            converged=converged,
            confidence=confidence,
            has_dissent=dissent is not None,
            used_selection=select_best,
            duration_ms=round(elapsed * 1000, 2),
        )

        return ConsensusRunResult(
            success=True,
            outcome=outcome,
            converged=converged,
            confidence=confidence,
            rounds=rounds,
            dissent=dissent,
            duration_seconds=elapsed,
            selection_stats=selection_stats,
        )

    def _check_convergence(
        self,
        responses: dict[str, str],
        response_format: Optional[str],
    ) -> tuple[bool, float, Optional[str]]:
        """
        Check if responses have converged.

        Returns: (converged, confidence, extracted_decision)
        """
        import re

        # Try to extract structured decisions if format specified
        if response_format and "DECISION:" in response_format.upper():
            decisions = {}
            for backend, text in responses.items():
                match = re.search(r'DECISION:\s*(\w+)', text, re.IGNORECASE)
                if match:
                    decisions[backend] = match.group(1).upper()

            if decisions:
                # Count votes
                vote_counts: dict[str, int] = {}
                for decision in decisions.values():
                    vote_counts[decision] = vote_counts.get(decision, 0) + 1

                total = len(decisions)
                if total > 0:
                    max_votes = max(vote_counts.values())
                    confidence = max_votes / total
                    majority_decision = max(vote_counts, key=vote_counts.get)

                    # Converged if all agree
                    converged = max_votes == total
                    return converged, confidence, majority_decision

        # Fallback: check if responses seem similar (simple heuristic)
        # This is a basic check - just see if they start with similar text
        if len(responses) >= 2:
            texts = list(responses.values())
            # Check first 100 chars for similarity (rough heuristic)
            first_chunks = [t[:100].lower() for t in texts if not t.startswith("[Error")]
            if len(first_chunks) >= 2:
                # Very basic similarity check
                all_similar = all(
                    self._text_similarity(first_chunks[0], chunk) > 0.5
                    for chunk in first_chunks[1:]
                )
                if all_similar:
                    return True, 0.7, None

        return False, 0.5, None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0

    def _format_others_responses(self, responses: dict[str, str]) -> str:
        """Format other responses for deliberation prompt."""
        parts = []
        for backend, text in responses.items():
            parts.append(f"[{backend.upper()}]:\n{text}\n")
        return "\n".join(parts)

    def _synthesize_outcome(self, responses: dict[str, str]) -> str:
        """Synthesize outcome from multiple responses (fallback - picks longest)."""
        valid_responses = [
            (backend, text) for backend, text in responses.items()
            if not text.startswith("[Error")
        ]
        if not valid_responses:
            return "No valid responses received."

        # Return the longest response (often most complete)
        return max(valid_responses, key=lambda x: len(x[1]))[1]

    async def _select_best_outcome(
        self,
        all_responses: dict[int, dict[str, str]],
        context: str,
        task: str,
    ) -> tuple[str, dict]:
        """
        Have LLMs vote on which response is best.

        Collects all valid responses across rounds, presents them as candidates,
        and asks LLMs to vote on the best one.

        Returns: (winning_text, selection_stats)
        """
        import re

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

        # Limit to best candidates if too many (take from later rounds preferentially)
        if len(candidates) > 6:
            # Sort by round (descending) and take top 6
            candidates = sorted(candidates, key=lambda c: -c["round"])[:6]

        # Build selection prompt
        candidate_texts = []
        for i, c in enumerate(candidates):
            letter = chr(65 + i)  # A, B, C, ...
            candidate_texts.append(f"[{letter}] (Round {c['round']}, {c['backend']}):\n{c['text']}\n")

        selection_prompt = f"""CONTEXT:
{context}

ORIGINAL TASK:
{task}

CANDIDATE RESPONSES:
{"".join(candidate_texts)}

SELECTION TASK:
Review all candidate responses above. Vote for the BEST one based on:
1. Accuracy and correctness
2. Clarity and completeness
3. How well it fulfills the original task

VOTE: [A/B/C/etc - single letter only]
REASONING: [Brief explanation of why this is best]
"""

        # Get votes from all available backends
        available = await self.client.get_available_backends()
        prompts = {backend: selection_prompt for backend in available}
        responses = await self.client.send_parallel(prompts, deep_mode=False)

        # Count votes
        votes: dict[str, list[str]] = {}
        for backend, response in responses.items():
            if not response.success:
                continue
            # Extract vote (look for single letter)
            match = re.search(r'VOTE:\s*\[?([A-Z])\]?', response.text, re.IGNORECASE)
            if match:
                vote = match.group(1).upper()
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

    def _find_dissent(self, responses: dict[str, str], response_format: Optional[str]) -> Optional[str]:
        """Find dissenting view if not converged."""
        import re

        if not response_format or "DECISION:" not in response_format.upper():
            return None

        # Extract decisions
        decisions = {}
        for backend, text in responses.items():
            match = re.search(r'DECISION:\s*(\w+)', text, re.IGNORECASE)
            if match:
                decisions[backend] = match.group(1).upper()

        if len(set(decisions.values())) <= 1:
            return None

        # Find the minority decision
        vote_counts: dict[str, list[str]] = {}
        for backend, decision in decisions.items():
            if decision not in vote_counts:
                vote_counts[decision] = []
            vote_counts[decision].append(backend)

        # Get the minority
        minority_decision = min(vote_counts, key=lambda d: len(vote_counts[d]))
        minority_backends = vote_counts[minority_decision]

        # Return the minority view
        if minority_backends:
            backend = minority_backends[0]
            return f"[{backend}] voted {minority_decision}: {responses[backend][:500]}"

        return None
