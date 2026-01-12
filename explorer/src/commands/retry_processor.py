"""
Retry processor for disputed and interesting insights.

Handles Round 4 (Modification Focus) execution for insights that
need another review attempt.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from shared.logging import get_logger

from .llm_connections import LLMConnections

log = get_logger("explorer", "retry_processor")


class RetryMode(Enum):
    """Mode for retry processing."""
    DISPUTED = "disputed"
    INTERESTING = "interesting"


@dataclass
class RetryResult:
    """Result of a single retry attempt."""
    chunk_id: str
    insight_text: str
    success: bool
    outcome: str  # "resolved", "majority", "disputed", "intractable", "error"
    final_rating: Optional[str] = None
    was_modified: bool = False
    modified_text: Optional[str] = None
    error: Optional[str] = None


class RetryProcessor:
    """
    Processes retry attempts for insights needing another review round.

    Supports two modes:
    - DISPUTED: Re-runs Round 4 for insights in disputed/ directory
    - INTERESTING: Re-runs Round 4 for insights stuck with '?' rating
    """

    def __init__(self, data_dir: Path, connections: LLMConnections):
        """
        Initialize retry processor.

        Args:
            data_dir: Path to the explorer data directory
            connections: Connected LLM interfaces
        """
        self.data_dir = data_dir
        self.connections = connections
        self.chunks_dir = data_dir / "chunks" / "insights"

    def find_disputed_reviews(self) -> list[Path]:
        """Find all disputed reviews that can be retried."""
        disputed_dir = self.data_dir / "reviews" / "disputed"
        if not disputed_dir.exists():
            return []
        return list(disputed_dir.glob("*.json"))

    def find_interesting_insights(self) -> list[tuple[Path, Path]]:
        """
        Find interesting insights with completed reviews.

        Returns:
            List of (chunk_path, review_path) tuples
        """
        interesting_dir = self.chunks_dir / "interesting"
        completed_dir = self.data_dir / "reviews" / "completed"

        if not interesting_dir.exists():
            return []

        results = []
        for chunk_file in interesting_dir.glob("*.json"):
            chunk_id = chunk_file.stem
            review_path = completed_dir / f"{chunk_id}.json"
            if review_path.exists():
                results.append((chunk_file, review_path))

        return results

    async def process_disputed(
        self,
        on_progress: Optional[Callable[[int, int, RetryResult], None]] = None
    ) -> list[RetryResult]:
        """
        Process all disputed reviews.

        Args:
            on_progress: Optional callback(current, total, result) for progress updates

        Returns:
            List of RetryResult for each processed review
        """
        reviews = self.find_disputed_reviews()
        results = []

        for i, review_path in enumerate(reviews, 1):
            result = await self._process_single_disputed(review_path)
            results.append(result)
            if on_progress:
                on_progress(i, len(reviews), result)

        return results

    async def process_interesting(
        self,
        on_progress: Optional[Callable[[int, int, RetryResult], None]] = None
    ) -> list[RetryResult]:
        """
        Process all interesting insights.

        Args:
            on_progress: Optional callback(current, total, result) for progress updates

        Returns:
            List of RetryResult for each processed insight
        """
        insights = self.find_interesting_insights()
        results = []

        for i, (chunk_path, review_path) in enumerate(insights, 1):
            result = await self._process_single_interesting(chunk_path, review_path)
            results.append(result)
            if on_progress:
                on_progress(i, len(insights), result)

        return results

    async def _process_single_disputed(self, review_path: Path) -> RetryResult:
        """Process a single disputed review."""
        from explorer.src.review_panel.models import ChunkReview
        from explorer.src.review_panel.round4 import run_round4

        try:
            review = ChunkReview.load(review_path)
            chunk_id = review.chunk_id

            # Find the chunk file
            insight_text = self._find_insight_text(chunk_id)
            if not insight_text:
                return RetryResult(
                    chunk_id=chunk_id,
                    insight_text="",
                    success=False,
                    outcome="error",
                    error="Could not find chunk file"
                )

            log.info("retry.processing", chunk_id=chunk_id, mode="disputed")

            # Run Round 4
            round4, modified_insight, refinement_record, is_intractable = await run_round4(
                chunk_insight=insight_text,
                review_rounds=review.rounds,
                gemini_browser=self.connections.gemini,
                chatgpt_browser=self.connections.chatgpt,
                claude_reviewer=self.connections.claude,
                config={},
            )

            # Update the review
            review.rounds = [r for r in review.rounds if r.round_number != 4]
            review.rounds.append(round4)

            # Handle intractable case
            if is_intractable:
                review.is_disputed = True
                review.final_rating = "?"
                review.save(self.data_dir)
                log.info("retry.intractable", chunk_id=chunk_id)
                return RetryResult(
                    chunk_id=chunk_id,
                    insight_text=insight_text,
                    success=False,
                    outcome="intractable",
                    final_rating="?"
                )

            # Handle modification
            was_modified = False
            if modified_insight and refinement_record:
                current_version = review.final_version or 1
                refinement_record.from_version = current_version
                refinement_record.to_version = current_version + 1
                review.refinements.append(refinement_record)
                review.was_refined = True
                review.final_version = refinement_record.to_version
                review.final_insight_text = modified_insight
                was_modified = True

            # Determine final outcome
            final_ratings = list(round4.get_ratings().values())
            outcome = round4.outcome

            if outcome == "resolved":
                review.final_rating = final_ratings[0]
                review.is_unanimous = True
                review.is_disputed = False
            elif outcome == "majority":
                review.final_rating = round4.get_majority_rating()
                review.is_unanimous = False
                review.is_disputed = True  # Majority = still disputed
            else:
                review.final_rating = round4.get_majority_rating() or "?"
                review.is_unanimous = False
                review.is_disputed = True

            review.save(self.data_dir)

            # Move if resolved unanimously
            if not review.is_disputed and review.is_unanimous:
                self._apply_final_rating(
                    chunk_id, review.final_rating, review.final_insight_text,
                    "Updated via retry-disputed Round 4 (unanimous)"
                )
                if review_path.exists():
                    review_path.unlink()
            elif review.final_insight_text:
                # Apply modified text but keep in disputed
                self._update_insight_text(chunk_id, review.final_insight_text)

            log.info(
                "retry.completed",
                chunk_id=chunk_id,
                outcome=outcome,
                rating=review.final_rating,
                modified=was_modified,
            )

            return RetryResult(
                chunk_id=chunk_id,
                insight_text=insight_text,
                success=True,
                outcome=outcome,
                final_rating=review.final_rating,
                was_modified=was_modified,
                modified_text=modified_insight if was_modified else None,
            )

        except Exception as e:
            log.error("retry.error", chunk_id=review_path.stem, error=str(e))
            return RetryResult(
                chunk_id=review_path.stem,
                insight_text="",
                success=False,
                outcome="error",
                error=str(e)
            )

    async def _process_single_interesting(
        self, chunk_path: Path, review_path: Path
    ) -> RetryResult:
        """Process a single interesting insight."""
        from explorer.src.review_panel.models import ChunkReview
        from explorer.src.review_panel.round4 import run_round4
        from explorer.src.chunking import AtomicInsight

        try:
            review = ChunkReview.load(review_path)
            chunk_id = review.chunk_id

            # Read insight text
            with open(chunk_path, encoding="utf-8") as f:
                chunk_data = json.load(f)
            insight_text = chunk_data.get("insight", "")

            if not insight_text:
                # Try markdown file
                md_path = chunk_path.with_suffix(".md")
                if md_path.exists():
                    content = md_path.read_text(encoding="utf-8")
                    for line in content.split("\n"):
                        if line.startswith("> "):
                            insight_text = line[2:].strip()
                            break

            if not insight_text:
                return RetryResult(
                    chunk_id=chunk_id,
                    insight_text="",
                    success=False,
                    outcome="error",
                    error="Could not extract insight text"
                )

            log.info("retry.processing", chunk_id=chunk_id, mode="interesting")

            # Run Round 4
            claude = self.connections.claude
            if claude and hasattr(claude, 'is_available') and not claude.is_available():
                claude = None

            round4, modified_insight, refinement_record, is_intractable = await run_round4(
                chunk_insight=insight_text,
                review_rounds=review.rounds,
                gemini_browser=self.connections.gemini,
                chatgpt_browser=self.connections.chatgpt,
                claude_reviewer=claude,
                config={},
            )

            review.add_round(round4)

            if is_intractable:
                review.is_disputed = True
                review.save(self.data_dir)
                # Remove from completed
                completed_path = self.data_dir / "reviews" / "completed" / f"{chunk_id}.json"
                if completed_path.exists():
                    completed_path.unlink()
                log.info("retry.intractable", chunk_id=chunk_id)
                return RetryResult(
                    chunk_id=chunk_id,
                    insight_text=insight_text,
                    success=False,
                    outcome="intractable",
                )

            # Handle modification
            was_modified = False
            if modified_insight and refinement_record:
                current_version = review.final_version or 1
                refinement_record.from_version = current_version
                refinement_record.to_version = current_version + 1
                review.refinements.append(refinement_record)
                review.final_version = current_version + 1
                review.final_insight_text = modified_insight
                was_modified = True

            # Determine outcome
            final_ratings = [r.rating for r in round4.responses.values()]
            outcome = round4.outcome

            if outcome == "resolved":
                review.final_rating = final_ratings[0]
                review.is_unanimous = True
                review.is_disputed = False
            elif outcome == "majority":
                review.final_rating = round4.get_majority_rating()
                review.is_unanimous = False
                review.is_disputed = True
            else:
                review.final_rating = round4.get_majority_rating() or "?"
                review.is_unanimous = False
                review.is_disputed = True

            review.save(self.data_dir)

            # Handle final placement
            if review.final_rating == "⚡" and review.is_unanimous:
                insight = AtomicInsight.load(chunk_path)
                if review.final_insight_text:
                    insight.insight = review.final_insight_text
                insight.apply_rating("⚡", notes="Upgraded via retry-interesting Round 4 (unanimous)")
                insight.save(self.data_dir / "chunks")
                chunk_path.unlink()
                md_path = chunk_path.with_suffix(".md")
                if md_path.exists():
                    md_path.unlink()
            elif review.final_rating == "✗" and review.is_unanimous:
                insight = AtomicInsight.load(chunk_path)
                if review.final_insight_text:
                    insight.insight = review.final_insight_text
                insight.apply_rating("✗", notes="Rejected via retry-interesting Round 4 (unanimous)")
                insight.save(self.data_dir / "chunks")
                chunk_path.unlink()
                md_path = chunk_path.with_suffix(".md")
                if md_path.exists():
                    md_path.unlink()
            elif review.final_insight_text:
                # Apply modified text but stay in interesting
                insight = AtomicInsight.load(chunk_path)
                insight.insight = review.final_insight_text
                insight.save(self.data_dir / "chunks")

            log.info(
                "retry.completed",
                chunk_id=chunk_id,
                outcome=outcome,
                rating=review.final_rating,
                modified=was_modified,
            )

            return RetryResult(
                chunk_id=chunk_id,
                insight_text=insight_text,
                success=True,
                outcome=outcome,
                final_rating=review.final_rating,
                was_modified=was_modified,
                modified_text=modified_insight if was_modified else None,
            )

        except Exception as e:
            log.error("retry.error", chunk_id=chunk_path.stem, error=str(e))
            return RetryResult(
                chunk_id=chunk_path.stem,
                insight_text="",
                success=False,
                outcome="error",
                error=str(e)
            )

    def _find_insight_text(self, chunk_id: str) -> Optional[str]:
        """Find insight text for a chunk ID."""
        for subdir in ["interesting", "blessed", "rejected", "pending"]:
            chunk_path = self.chunks_dir / subdir / f"{chunk_id}.md"
            if chunk_path.exists():
                content = chunk_path.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    if line.startswith("> "):
                        return line[2:].strip()
        return None

    def _apply_final_rating(
        self, chunk_id: str, rating: str, modified_text: Optional[str], notes: str
    ):
        """Apply final rating and move chunk to appropriate directory."""
        from explorer.src.chunking import AtomicInsight

        for subdir in ["interesting", "blessed", "rejected", "pending"]:
            chunk_json = self.chunks_dir / subdir / f"{chunk_id}.json"
            chunk_md = self.chunks_dir / subdir / f"{chunk_id}.md"
            if chunk_json.exists():
                insight = AtomicInsight.load(chunk_json)
                if modified_text:
                    insight.insight = modified_text
                insight.apply_rating(rating, notes=notes)
                insight.save(self.data_dir / "chunks")
                chunk_json.unlink()
                if chunk_md.exists():
                    chunk_md.unlink()
                return

    def _update_insight_text(self, chunk_id: str, new_text: str):
        """Update insight text without changing its location."""
        for subdir in ["interesting", "blessed", "rejected", "pending"]:
            chunk_json = self.chunks_dir / subdir / f"{chunk_id}.json"
            if chunk_json.exists():
                with open(chunk_json, encoding="utf-8") as f:
                    data = json.load(f)
                data["insight"] = new_text
                with open(chunk_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return
