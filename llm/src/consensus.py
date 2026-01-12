"""
Consensus Reviewer - Multi-LLM validation for reliable answers.

Uses multiple LLMs to review and validate insights/claims,
reaching consensus through structured deliberation.

This module provides backward-compatible imports from the refactored
consensus package. The implementation has been split into focused modules:

- consensus/reviewer.py: Main ConsensusReviewer class
- consensus/review_task.py: Multi-round insight validation
- consensus/consensus_task.py: General-purpose consensus
- consensus/quick_check.py: Single-LLM fast validation
- consensus/response_parser.py: Response parsing utilities
- consensus/prompts.py: Prompt building functions
- consensus/convergence.py: Convergence detection
"""

# Re-export for backward compatibility
from .consensus import ConsensusReviewer

__all__ = ["ConsensusReviewer"]
