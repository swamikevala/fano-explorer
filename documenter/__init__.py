"""
Documenter - Living mathematical document generator.

Creates and maintains a mathematical document that grows organically
from a seed, validated through multi-LLM consensus.
"""

from .main import Documenter
from .document import Document, Section

__all__ = ["Documenter", "Document", "Section"]
