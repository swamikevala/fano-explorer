"""
Automated review panel module.

Three-round LLM review process (Gemini, ChatGPT, Claude) with
consensus-based rating for atomic insights.
"""

from .models import (
    ReviewResponse,
    ReviewRound,
    ChunkReview,
    MindChange,
    ReviewDecision,
    detect_mind_changes,
)
from .reviewer import AutomatedReviewer
from .claude_api import ClaudeReviewer, get_claude_reviewer
from .round1 import run_round1
from .round2 import run_round2
from .round3 import run_round3

__all__ = [
    # Models
    "ReviewResponse",
    "ReviewRound",
    "ChunkReview",
    "MindChange",
    "ReviewDecision",
    "detect_mind_changes",
    # Main coordinator
    "AutomatedReviewer",
    # Claude API
    "ClaudeReviewer",
    "get_claude_reviewer",
    # Rounds (for advanced usage)
    "run_round1",
    "run_round2",
    "run_round3",
]
