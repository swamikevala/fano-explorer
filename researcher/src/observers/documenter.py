"""
Observer for documenter module activity.

Watches documenter's document sections to understand what's being written.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import BaseObserver


class DocumenterObserver(BaseObserver):
    """Observer for documenter module activity."""

    def __init__(self, documenter_path: Path):
        """
        Initialize documenter observer.

        Args:
            documenter_path: Path to documenter directory
        """
        super().__init__(documenter_path)
        self.document_path = documenter_path / "document"
        self.main_doc_path = self.document_path / "main.md"

    def get_recent_activity(self) -> list[dict]:
        """
        Get recent documenter activity (document changes).

        Returns:
            List of activity records with document sections
        """
        activities = []

        # Check main document
        if self.main_doc_path.exists():
            mtime = self.main_doc_path.stat().st_mtime
            key = str(self.main_doc_path)

            if key not in self._file_mtimes or self._file_mtimes[key] < mtime:
                content = self._read_document()
                if content:
                    activities.append({
                        "type": "document_update",
                        "content": content,
                        "timestamp": datetime.now(),
                        "filepath": str(self.main_doc_path),
                    })
                self._file_mtimes[key] = mtime

        # Check history directory for recent changes
        history_dir = self.document_path / "history"
        if history_dir.exists():
            changed_files = self._get_changed_files(history_dir, "*.md")
            for filepath in changed_files[:5]:  # Limit to recent
                content = self._read_markdown_file(filepath)
                if content:
                    activities.append({
                        "type": "history_entry",
                        "content": {"text": content, "filename": filepath.name},
                        "timestamp": datetime.now(),
                        "filepath": str(filepath),
                    })

        self.last_check = datetime.now()
        return activities

    def extract_concepts(self, activity: dict) -> list[str]:
        """
        Extract concept names from documenter activity.

        Looks at section headers, defined terms, and key phrases.
        """
        concepts = []
        content = activity.get("content", {})

        if activity["type"] == "document_update":
            # From section headers
            if "sections" in content:
                for section in content["sections"]:
                    concepts.append(section["title"].lower())

            # From full text
            if "full_text" in content:
                concepts.extend(self._extract_concepts_from_text(content["full_text"]))

        elif activity["type"] == "history_entry":
            text = content.get("text", "")
            concepts.extend(self._extract_concepts_from_text(text))

        return list(set(concepts))

    def extract_numbers(self, activity: dict) -> list[int]:
        """
        Extract significant numbers from documenter activity.
        """
        content = activity.get("content", {})
        numbers = []

        if activity["type"] == "document_update":
            if "full_text" in content:
                numbers = self._extract_numbers_from_text(content["full_text"])
        elif activity["type"] == "history_entry":
            text = content.get("text", "")
            numbers = self._extract_numbers_from_text(text)

        return list(set(numbers))

    def _read_document(self) -> Optional[dict]:
        """
        Read and parse the main document.

        Returns:
            Dict with sections and full text
        """
        try:
            with open(self.main_doc_path, encoding="utf-8") as f:
                full_text = f.read()

            sections = self._parse_sections(full_text)

            return {
                "full_text": full_text,
                "sections": sections,
            }
        except (FileNotFoundError, PermissionError):
            return None

    def _read_markdown_file(self, filepath: Path) -> Optional[str]:
        """Read a markdown file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                return f.read()
        except (FileNotFoundError, PermissionError):
            return None

    def _parse_sections(self, text: str) -> list[dict]:
        """
        Parse markdown into sections.

        Returns:
            List of dicts with title, level, and content
        """
        sections = []
        current_section = None

        for line in text.split("\n"):
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                if current_section:
                    sections.append(current_section)

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    "level": level,
                    "title": title,
                    "content": "",
                }
            elif current_section:
                current_section["content"] += line + "\n"

        if current_section:
            sections.append(current_section)

        return sections

    def _extract_concepts_from_text(self, text: str) -> list[str]:
        """Extract potential concept names from text."""
        concepts = []

        # Look for emphasized terms (bold or italic in markdown)
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', text)
        italic_terms = re.findall(r'\*([^*]+)\*', text)
        concepts.extend(bold_terms)
        concepts.extend(italic_terms)

        # Look for terms in definitions (e.g., "X is defined as")
        definitions = re.findall(r'(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:defined|called|known)', text)
        concepts.extend(definitions)

        # Look for capitalized proper nouns
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        concepts.extend(proper_nouns)

        return [c.lower().strip() for c in concepts if len(c) > 2]

    def get_current_topics(self) -> list[str]:
        """Get list of current document topics/sections."""
        content = self._read_document()
        if not content:
            return []

        topics = []
        for section in content.get("sections", []):
            topics.append(section["title"])

        return topics

    def get_section_content(self, section_title: str) -> Optional[str]:
        """Get content of a specific section."""
        content = self._read_document()
        if not content:
            return None

        for section in content.get("sections", []):
            if section["title"].lower() == section_title.lower():
                return section["content"]

        return None
