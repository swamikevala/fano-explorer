"""
Consensus package - Multi-LLM validation and consensus.

This package provides tools for reaching consensus across multiple LLMs:

Main Classes:
- ConsensusReviewer: Main orchestrator for multi-LLM validation

Task Modules:
- ReviewTask: 3-round insight validation
- ConsensusTask: General-purpose consensus
- QuickChecker: Single-LLM fast validation

Utilities:
- response_parser: Extract structured data from responses
- prompts: Prompt building functions
- convergence: Convergence detection
"""

from .reviewer import ConsensusReviewer
from .review_task import ReviewTask
from .consensus_task import ConsensusTask
from .quick_check import QuickChecker

__all__ = [
    "ConsensusReviewer",
    "ReviewTask",
    "ConsensusTask",
    "QuickChecker",
]
