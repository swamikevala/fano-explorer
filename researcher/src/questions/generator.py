"""
Research question generation from context observations.
"""

import random
from pathlib import Path
from typing import Optional
import yaml

from shared.logging import get_logger
from ..store.models import ResearchContext

log = get_logger("researcher", "questions")


class QuestionGenerator:
    """
    Generates research questions based on observations.

    Uses templates and context to formulate relevant search queries.
    """

    def __init__(self, templates_path: Path, domains_path: Path):
        """
        Initialize generator.

        Args:
            templates_path: Path to question_templates.yaml
            domains_path: Path to domains.yaml
        """
        self.templates = self._load_templates(templates_path)
        self.domains = self._load_domains(domains_path)

        # Extract domain modifiers and search terms
        self.domain_modifiers = self.templates.get("domain_modifiers", {})
        self.priorities = self.templates.get("priorities", {
            "supporting": 1.0,
            "contextual": 0.8,
            "challenging": 0.6,
            "alternative": 0.5
        })

    def _load_templates(self, path: Path) -> dict:
        """Load question templates."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {"templates": {}}

    def _load_domains(self, path: Path) -> dict:
        """Load domains config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    def generate(
        self,
        context: ResearchContext,
        max_questions: int = 20
    ) -> list[dict]:
        """
        Generate research questions from context.

        Args:
            context: Current research context
            max_questions: Maximum questions to generate

        Returns:
            List of question dicts with query, type, priority
        """
        questions = []

        # Generate from numbers
        for number in context.active_numbers[:5]:
            qs = self._generate_number_questions(number, context)
            questions.extend(qs)

        # Generate from concepts
        for concept in context.active_concepts[:5]:
            qs = self._generate_concept_questions(concept, context)
            questions.extend(qs)

        # Generate from documenter topics
        for topic in context.documenter_topics[:3]:
            qs = self._generate_topic_questions(topic, context)
            questions.extend(qs)

        # Generate cross-domain questions
        if context.active_numbers and context.active_domains:
            qs = self._generate_cross_domain_questions(context)
            questions.extend(qs)

        # Sort by priority and limit
        questions.sort(key=lambda q: q.get("priority", 0), reverse=True)

        log.info("questions.generated", count=len(questions[:max_questions]))
        return questions[:max_questions]

    def _generate_number_questions(
        self,
        number: int,
        context: ResearchContext
    ) -> list[dict]:
        """Generate questions about a number."""
        questions = []
        templates = self.templates.get("templates", {}).get("number_observed", {})

        for qtype, template_list in templates.items():
            priority = self.priorities.get(qtype, 0.5)

            for template in template_list[:2]:  # Limit templates per type
                # Substitute variables
                query = template.format(
                    number=number,
                    domain=context.active_domains[0] if context.active_domains else "Hindu tradition",
                    related_domain="Indian philosophy",
                    context="sacred texts"
                )

                questions.append({
                    "query": query,
                    "type": qtype,
                    "priority": priority,
                    "source": "number",
                    "source_value": number,
                })

        return questions

    def _generate_concept_questions(
        self,
        concept: str,
        context: ResearchContext
    ) -> list[dict]:
        """Generate questions about a concept."""
        questions = []
        templates = self.templates.get("templates", {}).get("concept_observed", {})

        for qtype, template_list in templates.items():
            priority = self.priorities.get(qtype, 0.5)

            for template in template_list[:2]:
                # Find related concept if available
                related = context.active_concepts[0] if context.active_concepts else "philosophy"
                if related == concept and len(context.active_concepts) > 1:
                    related = context.active_concepts[1]

                query = template.format(
                    concept=concept,
                    domain=context.active_domains[0] if context.active_domains else "Hindu",
                    tradition="Indian",
                    related_text="scripture",
                    related_concept=related
                )

                questions.append({
                    "query": query,
                    "type": qtype,
                    "priority": priority,
                    "source": "concept",
                    "source_value": concept,
                })

        return questions

    def _generate_topic_questions(
        self,
        topic: str,
        context: ResearchContext
    ) -> list[dict]:
        """Generate questions for documenter topics."""
        questions = []
        templates = self.templates.get("templates", {}).get("documenter_section", {})

        for qtype, template_list in templates.items():
            priority = self.priorities.get(qtype, 0.5)

            for template in template_list[:1]:  # Fewer for topics
                query = template.format(
                    topic=topic,
                    claim=topic
                )

                questions.append({
                    "query": query,
                    "type": qtype,
                    "priority": priority,
                    "source": "documenter",
                    "source_value": topic,
                })

        return questions

    def _generate_cross_domain_questions(
        self,
        context: ResearchContext
    ) -> list[dict]:
        """Generate cross-domain pattern questions."""
        questions = []
        templates = self.templates.get("templates", {}).get("cross_domain", [])

        if len(context.active_domains) >= 2:
            domain1, domain2 = context.active_domains[:2]

            for template in templates[:2]:
                if context.active_numbers:
                    number = context.active_numbers[0]
                    query = template.format(
                        number=number,
                        domain1=domain1.replace("_", " "),
                        domain2=domain2.replace("_", " "),
                        concept1=context.active_concepts[0] if context.active_concepts else "",
                        concept2=context.active_concepts[1] if len(context.active_concepts) > 1 else "",
                        pattern1="structure",
                        pattern2="pattern"
                    )

                    questions.append({
                        "query": query,
                        "type": "cross_domain",
                        "priority": 0.9,  # High priority for cross-domain
                        "source": "cross_domain",
                        "source_value": f"{domain1}+{domain2}",
                    })

        return questions

    def prioritize_by_context(
        self,
        questions: list[dict],
        context: ResearchContext
    ) -> list[dict]:
        """
        Adjust question priorities based on recent insights.

        Boosts questions related to recently blessed insights.
        """
        # Extract concepts from recent insights
        recent_concepts = set()
        for insight in context.recent_insights:
            if isinstance(insight, dict):
                recent_concepts.update(insight.get("tags", []))

        # Boost questions matching recent concepts
        for q in questions:
            query_lower = q["query"].lower()
            for concept in recent_concepts:
                if concept.lower() in query_lower:
                    q["priority"] = min(1.0, q["priority"] + 0.2)
                    break

        questions.sort(key=lambda q: q["priority"], reverse=True)
        return questions
