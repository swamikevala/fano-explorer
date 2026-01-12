"""
Researcher module - discovers and analyzes external sources for the Fano platform.

This module observes explorer and documenter activity, searches for relevant
sources, extracts findings, and makes them available for validation and
exploration direction.
"""

from .orchestrator import Orchestrator
from .api import ResearcherAPI

__all__ = [
    "Orchestrator",
    "ResearcherAPI",
]
