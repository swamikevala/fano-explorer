"""
Content extraction using LLM.

Extracts concepts, numbers, relationships, and quotes from fetched content.
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

from shared.logging import get_logger
from ..store.models import Finding, FindingType, NumberMention, MentionType

log = get_logger("researcher", "extractor")


class ContentExtractor:
    """
    Extracts structured information from source content.

    Uses LLM for intelligent extraction and pattern recognition.
    """

    def __init__(self, config_path: Path, domains_config_path: Path, llm_client=None):
        """
        Initialize extractor.

        Args:
            config_path: Path to settings.yaml
            domains_config_path: Path to domains.yaml
            llm_client: LLM client for extraction
        """
        self.config = self._load_config(config_path)
        self.domains_config = self._load_domains(domains_config_path)
        self.llm_client = llm_client

        # Key numbers to look for
        self.key_numbers = set(self.domains_config.get("global_key_numbers", []))
        for domain in self.domains_config.get("domains", []):
            self.key_numbers.update(domain.get("key_numbers", []))

        # Extraction settings
        extraction_config = self.config.get("extraction", {})
        self.min_confidence = extraction_config.get("min_confidence", 0.5)
        self.extract_types = extraction_config.get(
            "extract_types",
            ["concepts", "numbers", "relationships", "quotes"]
        )

    def _load_config(self, path: Path) -> dict:
        """Load settings config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    def _load_domains(self, path: Path) -> dict:
        """Load domains config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    async def extract(
        self,
        source_id: str,
        content: dict,
        research_context: dict,
        use_llm: bool = True
    ) -> dict:
        """
        Extract findings from content.

        Args:
            source_id: ID of the source
            content: Fetched content dict
            research_context: Current research context
            use_llm: Whether to use LLM for extraction

        Returns:
            Dict with findings, concepts, numbers
        """
        text = content.get("text", "")
        domain = self._detect_domain(text)

        # Rule-based extraction
        basic_findings = self._extract_basic(text, source_id, domain)
        basic_numbers = self._extract_numbers(text, source_id, domain)

        # LLM-based extraction
        llm_findings = []
        llm_numbers = []

        if use_llm and self.llm_client:
            llm_result = await self._extract_with_llm(
                text, source_id, domain, research_context
            )
            if llm_result:
                llm_findings = llm_result.get("findings", [])
                llm_numbers = llm_result.get("numbers", [])

        # Combine results
        all_findings = basic_findings + llm_findings
        all_numbers = basic_numbers + llm_numbers

        # Deduplicate
        all_findings = self._dedupe_findings(all_findings)
        all_numbers = self._dedupe_numbers(all_numbers)

        log.info(
            "extractor.complete",
            source_id=source_id,
            findings_count=len(all_findings),
            numbers_count=len(all_numbers)
        )

        return {
            "findings": all_findings,
            "numbers": all_numbers,
            "domain": domain,
        }

    def _detect_domain(self, text: str) -> str:
        """Detect research domain from text content."""
        # Simple keyword matching
        domain_keywords = {}
        for domain in self.domains_config.get("domains", []):
            name = domain["name"]
            keywords = domain.get("search_terms", []) + domain.get("related_concepts", [])
            domain_keywords[name] = keywords

        # Count keyword matches
        text_lower = text.lower()
        scores = {}
        for domain_name, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[domain_name] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _extract_basic(
        self,
        text: str,
        source_id: str,
        domain: str
    ) -> list[Finding]:
        """
        Rule-based extraction of findings.

        Looks for patterns like definitions, assertions, etc.
        """
        findings = []

        # Extract definitions (X is defined as Y, X means Y)
        definition_patterns = [
            r'(?:The\s+)?(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:defined as|called|known as|means?)\s+([^.]+)\.',
            r'(\w+(?:\s+\w+)?)\s*[-:]\s*([^.]+(?:means?|refers to)[^.]+)\.',
        ]

        for pattern in definition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group(1).strip()
                definition = match.group(2).strip()

                # Get surrounding context for the quote
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                finding = Finding.create(
                    source_id=source_id,
                    finding_type=FindingType.CONTEXTUAL,
                    summary=f"Definition of {term}",
                    original_quote=match.group(0),
                    domain=domain,
                    extraction_model="rule-based",
                    confidence=0.7
                )
                findings.append(finding)

        # Extract enumerated lists (The N Xs are: ...)
        enum_pattern = r'[Tt]he\s+(\d+)\s+(\w+(?:\s+\w+)?)\s+(?:are|include)[:\s]+([^.]+)\.'
        for match in re.finditer(enum_pattern, text):
            number = int(match.group(1))
            subject = match.group(2)
            items = match.group(3)

            if number in self.key_numbers:
                finding = Finding.create(
                    source_id=source_id,
                    finding_type=FindingType.SUPPORTING,
                    summary=f"Enumeration: {number} {subject}",
                    original_quote=match.group(0),
                    domain=domain,
                    extraction_model="rule-based",
                    confidence=0.8
                )
                findings.append(finding)

        return findings

    def _extract_numbers(
        self,
        text: str,
        source_id: str,
        domain: str
    ) -> list[NumberMention]:
        """
        Extract significant number mentions.

        Looks for key numbers and their contexts.
        """
        mentions = []

        # Pattern for numbers with context
        for number in self.key_numbers:
            # Look for the number in various contexts
            patterns = [
                rf'\b{number}\s+(\w+(?:\s+\w+)?)',  # 7 chakras
                rf'(\w+(?:\s+\w+)?)\s+{number}\b',  # chakras 7
                rf'{number}\s+(?:is|are|represents?)',  # 7 represents
            ]

            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Get broader context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()

                    # Determine mention type
                    mention_type = self._classify_number_mention(match.group(0), context)

                    mention = NumberMention.create(
                        value=number,
                        context=context[:200],  # Limit context length
                        mention_type=mention_type,
                        finding_id="",  # Will be linked later
                        source_id=source_id,
                        domain=domain,
                        significance=self._get_number_significance(number, context)
                    )
                    mentions.append(mention)

        return mentions

    def _classify_number_mention(self, match_text: str, context: str) -> MentionType:
        """Classify how a number is mentioned."""
        match_lower = match_text.lower()
        context_lower = context.lower()

        # Structural indicators
        if any(w in context_lower for w in ["structure", "organize", "level", "layer", "stage"]):
            return MentionType.STRUCTURAL

        # Symbolic indicators
        if any(w in context_lower for w in ["symbol", "represent", "sacred", "signif"]):
            return MentionType.SYMBOLIC

        # Enumerative indicators
        if any(w in context_lower for w in ["are:", "include", "namely", "consist"]):
            return MentionType.ENUMERATIVE

        return MentionType.INCIDENTAL

    def _get_number_significance(self, number: int, context: str) -> str:
        """Try to determine what a number signifies."""
        context_lower = context.lower()

        # Common significances for key numbers
        common_meanings = {
            3: ["gunas", "worlds", "aspects", "bodies"],
            5: ["elements", "pranas", "sheaths", "senses"],
            7: ["chakras", "svaras", "rishis", "planes", "colors"],
            9: ["planets", "forms", "nights", "durgas"],
            12: ["zodiac", "jyotirlingas", "months", "adityas"],
            27: ["nakshatras", "stars", "lunar mansions"],
            28: ["nakshatras", "lunar days"],
            36: ["tattvas", "principles"],
            64: ["arts", "tantras", "squares"],
            72: ["melakartas", "nadis"],
            84: ["asanas", "siddhas", "postures"],
            108: ["beads", "names", "upanishads"],
            112: ["chakras", "ways", "methods"],
        }

        # Check if any known meaning appears in context
        if number in common_meanings:
            for meaning in common_meanings[number]:
                if meaning in context_lower:
                    return meaning

        # Extract noun following the number
        pattern = rf'{number}\s+(\w+)'
        match = re.search(pattern, context)
        if match:
            return match.group(1)

        return ""

    async def _extract_with_llm(
        self,
        text: str,
        source_id: str,
        domain: str,
        context: dict
    ) -> Optional[dict]:
        """
        Use LLM for intelligent extraction.

        Args:
            text: Source text
            source_id: Source ID
            domain: Detected domain
            context: Research context with active concepts/numbers

        Returns:
            Dict with findings and numbers
        """
        if not self.llm_client:
            return None

        # Prepare text sample
        sample = text[:4000] if len(text) > 4000 else text

        # Build context string
        active_concepts = context.get("active_concepts", [])[:10]
        active_numbers = context.get("active_numbers", [])[:10]

        prompt = f"""Extract key information from this text about {domain}.

Current research interests:
- Concepts: {', '.join(active_concepts) if active_concepts else 'general'}
- Numbers of interest: {', '.join(str(n) for n in active_numbers) if active_numbers else 'any significant numbers'}

Text:
{sample}

Extract and return as JSON:
{{
    "findings": [
        {{
            "summary": "brief summary of finding",
            "quote": "exact quote from text",
            "type": "supporting|challenging|contextual|alternative",
            "confidence": 0.0-1.0,
            "concepts": ["concept1", "concept2"]
        }}
    ],
    "numbers": [
        {{
            "value": 7,
            "context": "context of the number",
            "significance": "what it represents",
            "mention_type": "structural|symbolic|enumerative|incidental"
        }}
    ]
}}

Focus on:
1. Definitions and explanations of key terms
2. Structural patterns and number symbolism
3. Connections to the concepts of interest
4. Any claims that support or challenge understanding
"""

        try:
            response = await self.llm_client.query(prompt)
            if not response:
                return None

            # Parse response
            parsed = self._parse_llm_extraction(response, source_id, domain)
            return parsed

        except Exception as e:
            log.error("extractor.llm_failed", error=str(e))
            return None

    def _parse_llm_extraction(
        self,
        response: str,
        source_id: str,
        domain: str
    ) -> Optional[dict]:
        """Parse LLM extraction response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            findings = []
            for f in data.get("findings", []):
                finding = Finding.create(
                    source_id=source_id,
                    finding_type=FindingType(f.get("type", "contextual")),
                    summary=f.get("summary", ""),
                    original_quote=f.get("quote", ""),
                    domain=domain,
                    extraction_model="llm",
                    confidence=f.get("confidence", 0.7)
                )
                findings.append(finding)

            numbers = []
            for n in data.get("numbers", []):
                mention_type_str = n.get("mention_type", "incidental")
                try:
                    mention_type = MentionType(mention_type_str)
                except ValueError:
                    mention_type = MentionType.INCIDENTAL

                mention = NumberMention.create(
                    value=n.get("value", 0),
                    context=n.get("context", ""),
                    mention_type=mention_type,
                    finding_id="",
                    source_id=source_id,
                    domain=domain,
                    significance=n.get("significance", "")
                )
                numbers.append(mention)

            return {
                "findings": findings,
                "numbers": numbers
            }

        except json.JSONDecodeError:
            log.warning("extractor.json_parse_failed")
            return None

    def _dedupe_findings(self, findings: list[Finding]) -> list[Finding]:
        """Remove duplicate findings based on summary."""
        seen = set()
        unique = []
        for f in findings:
            key = f.summary.lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def _dedupe_numbers(self, numbers: list[NumberMention]) -> list[NumberMention]:
        """Remove duplicate number mentions."""
        seen = set()
        unique = []
        for n in numbers:
            key = (n.value, n.context[:50])
            if key not in seen:
                seen.add(key)
                unique.append(n)
        return unique
