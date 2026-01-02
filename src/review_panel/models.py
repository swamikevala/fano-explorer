"""
Data models for the automated review panel.

Models for tracking review rounds, responses, mind changes,
and final review outcomes.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class ReviewDecision(Enum):
    """Possible review decisions."""
    BLESS = "⚡"          # Approve as axiom
    UNCERTAIN = "?"       # Interesting but needs work
    REJECT = "✗"          # Discard


@dataclass
class ReviewResponse:
    """
    A single reviewer's response in a review round.
    """
    llm: str                              # "gemini" | "chatgpt" | "claude"
    mode: str                             # "standard" | "deep_think" | "pro" | "extended_thinking"
    rating: str                           # "⚡" | "?" | "✗"
    mathematical_verification: str         # Verify or refute specific claims
    structural_analysis: str               # Is connection deep or superficial?
    naturalness_assessment: str            # Does it feel inevitable?
    reasoning: str                         # Overall justification (2-4 sentences)
    confidence: str                        # "high" | "medium" | "low"

    # Round 2+ fields (when reviewing others' responses)
    new_information: Optional[str] = None  # What did others point out?
    changed_mind: Optional[bool] = None    # Did this change things?
    previous_rating: Optional[str] = None  # Rating before reconsideration

    # Round 3 fields (deliberation)
    strongest_argument: Optional[str] = None    # Minority's best case
    response_to_argument: Optional[str] = None  # Majority's response
    final_stance: Optional[str] = None          # "concede" | "maintain"

    def to_dict(self) -> dict:
        return {
            "llm": self.llm,
            "mode": self.mode,
            "rating": self.rating,
            "mathematical_verification": self.mathematical_verification,
            "structural_analysis": self.structural_analysis,
            "naturalness_assessment": self.naturalness_assessment,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "new_information": self.new_information,
            "changed_mind": self.changed_mind,
            "previous_rating": self.previous_rating,
            "strongest_argument": self.strongest_argument,
            "response_to_argument": self.response_to_argument,
            "final_stance": self.final_stance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewResponse":
        return cls(
            llm=data["llm"],
            mode=data["mode"],
            rating=data["rating"],
            mathematical_verification=data["mathematical_verification"],
            structural_analysis=data["structural_analysis"],
            naturalness_assessment=data["naturalness_assessment"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            new_information=data.get("new_information"),
            changed_mind=data.get("changed_mind"),
            previous_rating=data.get("previous_rating"),
            strongest_argument=data.get("strongest_argument"),
            response_to_argument=data.get("response_to_argument"),
            final_stance=data.get("final_stance"),
        )


@dataclass
class MindChange:
    """
    Record of a reviewer changing their position.
    """
    llm: str                # Which model changed
    round_number: int       # When they changed (2 or 3)
    from_rating: str        # Original rating
    to_rating: str          # New rating
    reason: str             # What convinced them

    def to_dict(self) -> dict:
        return {
            "llm": self.llm,
            "round_number": self.round_number,
            "from_rating": self.from_rating,
            "to_rating": self.to_rating,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MindChange":
        return cls(
            llm=data["llm"],
            round_number=data["round_number"],
            from_rating=data["from_rating"],
            to_rating=data["to_rating"],
            reason=data["reason"],
        )


@dataclass
class ReviewRound:
    """
    A single round of the review process.
    """
    round_number: int                      # 1, 2, or 3
    mode: str                              # "standard" or "deep"
    responses: dict[str, ReviewResponse]   # Keyed by LLM name
    outcome: str                           # "unanimous" | "split" | "resolved"
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "mode": self.mode,
            "responses": {k: v.to_dict() for k, v in self.responses.items()},
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewRound":
        return cls(
            round_number=data["round_number"],
            mode=data["mode"],
            responses={k: ReviewResponse.from_dict(v) for k, v in data["responses"].items()},
            outcome=data["outcome"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def get_ratings(self) -> dict[str, str]:
        """Get rating from each reviewer."""
        return {llm: resp.rating for llm, resp in self.responses.items()}

    def is_unanimous(self) -> bool:
        """Check if all reviewers agree."""
        ratings = list(self.get_ratings().values())
        return len(set(ratings)) == 1

    def get_majority_rating(self) -> Optional[str]:
        """Get the majority rating (2 out of 3)."""
        ratings = list(self.get_ratings().values())
        for rating in ["⚡", "?", "✗"]:
            if ratings.count(rating) >= 2:
                return rating
        return None

    def get_minority_llms(self) -> list[str]:
        """Get LLMs that hold minority position."""
        majority = self.get_majority_rating()
        if not majority:
            return []
        return [llm for llm, resp in self.responses.items() if resp.rating != majority]

    def get_majority_llms(self) -> list[str]:
        """Get LLMs that hold majority position."""
        majority = self.get_majority_rating()
        if not majority:
            return list(self.responses.keys())
        return [llm for llm, resp in self.responses.items() if resp.rating == majority]


@dataclass
class ChunkReview:
    """
    Complete review record for an insight.
    """
    chunk_id: str                          # ID of insight being reviewed
    rounds: list[ReviewRound] = field(default_factory=list)
    final_rating: Optional[str] = None     # "⚡" | "?" | "✗"
    is_unanimous: bool = False             # Did all reviewers agree?
    is_disputed: bool = False              # Persistent 2-1 split?
    mind_changes: list[MindChange] = field(default_factory=list)
    total_tokens_used: int = 0             # For cost tracking
    review_duration_seconds: float = 0.0   # How long the review took
    reviewed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_rating": self.final_rating,
            "is_unanimous": self.is_unanimous,
            "is_disputed": self.is_disputed,
            "mind_changes": [m.to_dict() for m in self.mind_changes],
            "total_tokens_used": self.total_tokens_used,
            "review_duration_seconds": self.review_duration_seconds,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkReview":
        return cls(
            chunk_id=data["chunk_id"],
            rounds=[ReviewRound.from_dict(r) for r in data.get("rounds", [])],
            final_rating=data.get("final_rating"),
            is_unanimous=data.get("is_unanimous", False),
            is_disputed=data.get("is_disputed", False),
            mind_changes=[MindChange.from_dict(m) for m in data.get("mind_changes", [])],
            total_tokens_used=data.get("total_tokens_used", 0),
            review_duration_seconds=data.get("review_duration_seconds", 0.0),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data.get("reviewed_at") else None,
        )

    def save(self, base_dir: Path):
        """Save review to disk."""
        # Determine subdirectory based on disputed status
        if self.is_disputed:
            review_dir = base_dir / "reviews" / "disputed"
        else:
            review_dir = base_dir / "reviews" / "completed"

        review_dir.mkdir(parents=True, exist_ok=True)
        filepath = review_dir / f"{self.chunk_id}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Path) -> "ChunkReview":
        """Load review from disk."""
        with open(filepath, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def add_round(self, round: ReviewRound):
        """Add a review round."""
        self.rounds.append(round)

    def finalize(self, rating: str, unanimous: bool, disputed: bool):
        """Finalize the review with outcome."""
        self.final_rating = rating
        self.is_unanimous = unanimous
        self.is_disputed = disputed
        self.reviewed_at = datetime.now()


def detect_mind_changes(
    round1: ReviewRound,
    round2: ReviewRound,
) -> list[MindChange]:
    """
    Detect which reviewers changed their minds between rounds.

    Args:
        round1: Earlier round
        round2: Later round

    Returns:
        List of MindChange records
    """
    changes = []

    for llm in round1.responses:
        if llm not in round2.responses:
            continue

        r1_rating = round1.responses[llm].rating
        r2_rating = round2.responses[llm].rating

        if r1_rating != r2_rating:
            # Get the reason from Round 2 response
            r2_resp = round2.responses[llm]
            reason = r2_resp.new_information or r2_resp.reasoning or "No reason given"

            changes.append(MindChange(
                llm=llm,
                round_number=round2.round_number,
                from_rating=r1_rating,
                to_rating=r2_rating,
                reason=reason[:500],  # Truncate if too long
            ))

    return changes
