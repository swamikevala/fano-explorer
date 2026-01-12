"""
Trust evaluation for web sources using LLM consensus.

Evaluates sources based on observable features and LLM judgment.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import yaml

from shared.logging import get_logger
from ...store.models import Source, SourceTier

log = get_logger("researcher", "trust")


class TrustEvaluator:
    """
    Evaluates trustworthiness of web sources.

    Uses a combination of:
    - Observable features (citations, verse references, etc.)
    - Domain reputation (from seed list)
    - LLM consensus (for new sources)
    """

    def __init__(
        self,
        trusted_sources_path: Path,
        config_path: Path,
        llm_client=None
    ):
        """
        Initialize trust evaluator.

        Args:
            trusted_sources_path: Path to trusted_sources.yaml
            config_path: Path to settings.yaml
            llm_client: Optional LLM client for consensus evaluation
        """
        self.trusted_sources_path = trusted_sources_path
        self.trusted_sources = self._load_trusted_sources()
        self.config = self._load_config(config_path)
        self.llm_client = llm_client

        # Settings
        trust_config = self.config.get("trust", {})
        self.min_trust_score = trust_config.get("min_trust_score", 50)
        self.consensus_count = trust_config.get("consensus_evaluations", 3)
        self.primary_citation_boost = trust_config.get("primary_citation_boost", 15)
        self.academic_domain_boost = trust_config.get("academic_domain_boost", 10)

        # Academic domain patterns
        self.academic_patterns = [
            r'\.edu$', r'\.ac\.[a-z]{2}$', r'\.edu\.[a-z]{2}$',
            r'uni-.*\.de$', r'university', r'\.gov$'
        ]

    def _load_trusted_sources(self) -> dict:
        """Load trusted sources configuration."""
        try:
            with open(self.trusted_sources_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {"seed_sources": {"tier_1": [], "tier_2": []}}

    def _load_config(self, path: Path) -> dict:
        """Load settings config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    async def evaluate(
        self,
        url: str,
        content: dict,
        use_llm: bool = True
    ) -> dict:
        """
        Evaluate trustworthiness of a source.

        Args:
            url: Source URL
            content: Fetched content dict with text, title, etc.
            use_llm: Whether to use LLM for evaluation

        Returns:
            Dict with trust_score, tier, reasoning, observable features
        """
        domain = urlparse(url).netloc.lower()
        text = content.get("text", "")

        # Check seed sources first
        seed_result = self._check_seed_sources(domain)
        if seed_result:
            log.info(
                "trust.seed_source_match",
                domain=domain,
                score=seed_result["trust_score"]
            )
            return seed_result

        # Check if domain is excluded
        if self._is_excluded(domain):
            return {
                "trust_score": 0,
                "trust_tier": SourceTier.UNKNOWN,
                "reasoning": "Domain is in exclusion list",
                "observable_features": {},
                "evaluated_at": datetime.now().isoformat(),
            }

        # Evaluate observable features
        observable = self._evaluate_observable_features(domain, text)

        # Calculate base score from observables
        base_score = self._calculate_base_score(observable)

        # Optionally use LLM for additional evaluation
        llm_evaluation = None
        if use_llm and self.llm_client:
            llm_evaluation = await self._llm_evaluate(url, content, observable)
            if llm_evaluation:
                # Blend scores
                base_score = (base_score + llm_evaluation.get("score", base_score)) / 2

        # Determine tier
        tier = self._score_to_tier(base_score)

        result = {
            "trust_score": int(base_score),
            "trust_tier": tier,
            "reasoning": self._generate_reasoning(observable, llm_evaluation),
            "observable_features": observable,
            "llm_evaluation": llm_evaluation,
            "evaluated_at": datetime.now().isoformat(),
        }

        log.info(
            "trust.evaluation_complete",
            domain=domain,
            score=result["trust_score"],
            tier=tier.value
        )

        return result

    def _check_seed_sources(self, domain: str) -> Optional[dict]:
        """Check if domain is in seed sources."""
        seed_sources = self.trusted_sources.get("seed_sources", {})

        # Check tier 1
        for source in seed_sources.get("tier_1", []):
            if self._domain_matches(domain, source.get("domain", "")):
                return {
                    "trust_score": source.get("trust_score", 95),
                    "trust_tier": SourceTier.TIER_1,
                    "reasoning": f"Tier 1 seed source: {source.get('notes', '')}",
                    "observable_features": {"seed_source": True},
                    "evaluated_at": datetime.now().isoformat(),
                }

        # Check tier 2
        for source in seed_sources.get("tier_2", []):
            if self._domain_matches(domain, source.get("domain", "")):
                return {
                    "trust_score": source.get("trust_score", 75),
                    "trust_tier": SourceTier.TIER_2,
                    "reasoning": f"Tier 2 seed source: {source.get('notes', '')}",
                    "observable_features": {"seed_source": True},
                    "evaluated_at": datetime.now().isoformat(),
                }

        return None

    def _domain_matches(self, actual: str, pattern: str) -> bool:
        """Check if domain matches pattern (supports wildcards)."""
        if pattern.startswith("*."):
            return actual.endswith(pattern[1:])
        return actual == pattern or actual.endswith("." + pattern)

    def _is_excluded(self, domain: str) -> bool:
        """Check if domain is in exclusion list."""
        excluded = self.trusted_sources.get("excluded", [])
        for item in excluded:
            pattern = item.get("domain", "") if isinstance(item, dict) else item
            if self._domain_matches(domain, pattern):
                return True
        return False

    def _evaluate_observable_features(self, domain: str, text: str) -> dict:
        """
        Evaluate observable features of the source.

        These are objective, verifiable characteristics.
        """
        features = {}

        # Check for Sanskrit citations (Devanagari script)
        features["has_sanskrit"] = bool(re.search(r'[\u0900-\u097F]', text))

        # Check for verse references (e.g., "1.23", "verse 5", "śloka 12")
        verse_patterns = [
            r'\b\d+\.\d+\b',  # 1.23 format
            r'\bverse\s+\d+\b',
            r'\bśloka\s+\d+\b',
            r'\bsūtra\s+\d+\b',
            r'\bchapter\s+\d+\b',
            r'\badhyāya\s+\d+\b',
        ]
        features["has_verse_references"] = any(
            re.search(p, text, re.IGNORECASE) for p in verse_patterns
        )

        # Check for bibliography/references section
        bib_patterns = [
            r'\breferences\b',
            r'\bbibliography\b',
            r'\bworks cited\b',
            r'\bsources\b',
        ]
        features["has_bibliography"] = any(
            re.search(p, text, re.IGNORECASE) for p in bib_patterns
        )

        # Check for academic domain
        features["is_academic"] = any(
            re.search(p, domain, re.IGNORECASE) for p in self.academic_patterns
        )

        # Check for IAST transliteration (common diacritics)
        features["has_iast"] = bool(re.search(r'[āīūṛṝḷḹṃḥśṣṇṭḍ]', text))

        # Check for scholarly language patterns
        scholarly_patterns = [
            r'\baccording to\b',
            r'\bcommentary\b',
            r'\btranslation\b',
            r'\bsanskrit\b',
            r'\bvedic\b',
        ]
        features["scholarly_indicators"] = sum(
            1 for p in scholarly_patterns if re.search(p, text, re.IGNORECASE)
        )

        # Check for sensationalist language (negative indicator)
        sensationalist_patterns = [
            r'\bsecret\b',
            r'\bshocking\b',
            r'\bmiracle\b',
            r'\bunbelievable\b',
            r'\byou won\'t believe\b',
        ]
        features["sensationalist_indicators"] = sum(
            1 for p in sensationalist_patterns if re.search(p, text, re.IGNORECASE)
        )

        return features

    def _calculate_base_score(self, features: dict) -> float:
        """Calculate trust score from observable features."""
        score = 50  # Start at neutral

        # Positive indicators
        if features.get("has_sanskrit"):
            score += 10
        if features.get("has_verse_references"):
            score += self.primary_citation_boost
        if features.get("has_bibliography"):
            score += 10
        if features.get("is_academic"):
            score += self.academic_domain_boost
        if features.get("has_iast"):
            score += 5

        # Scholarly indicators (up to +15)
        scholarly = features.get("scholarly_indicators", 0)
        score += min(scholarly * 3, 15)

        # Negative indicators
        sensational = features.get("sensationalist_indicators", 0)
        score -= sensational * 10

        # Clamp to 0-100
        return max(0, min(100, score))

    async def _llm_evaluate(
        self,
        url: str,
        content: dict,
        observable: dict
    ) -> Optional[dict]:
        """
        Use LLM to evaluate source trustworthiness.

        Runs multiple evaluations for consensus.
        """
        if not self.llm_client:
            return None

        # Prepare content sample
        text = content.get("text", "")
        sample = text[:2000] if len(text) > 2000 else text

        prompt = f"""Evaluate this source for scholarly trustworthiness regarding Hindu scriptures and Sanskrit texts.

URL: {url}
Title: {content.get('title', 'Unknown')}

Content sample:
{sample}

Observable features detected:
- Has Sanskrit/Devanagari: {observable.get('has_sanskrit', False)}
- Has verse references: {observable.get('has_verse_references', False)}
- Has bibliography: {observable.get('has_bibliography', False)}
- Academic domain: {observable.get('is_academic', False)}

Evaluate on these criteria:
1. Does it cite primary texts with specific references?
2. Is the tone scholarly or sensationalist?
3. Does it show understanding of the tradition?
4. Are claims appropriately hedged?

Return a JSON object with:
- score: 0-100 trust score
- scholarly_tone: true/false
- cites_primary: true/false
- reasoning: brief explanation (1-2 sentences)
"""

        try:
            # Run multiple evaluations
            evaluations = []
            for _ in range(self.consensus_count):
                response = await self.llm_client.query(prompt)
                if response:
                    eval_data = self._parse_llm_response(response)
                    if eval_data:
                        evaluations.append(eval_data)

            if not evaluations:
                return None

            # Aggregate consensus
            avg_score = sum(e.get("score", 50) for e in evaluations) / len(evaluations)
            consensus_scholarly = sum(1 for e in evaluations if e.get("scholarly_tone")) > len(evaluations) / 2
            consensus_cites = sum(1 for e in evaluations if e.get("cites_primary")) > len(evaluations) / 2

            return {
                "score": avg_score,
                "scholarly_tone": consensus_scholarly,
                "cites_primary": consensus_cites,
                "evaluation_count": len(evaluations),
                "individual_scores": [e.get("score", 50) for e in evaluations],
            }

        except Exception as e:
            log.error("trust.llm_evaluation_failed", error=str(e))
            return None

    def _parse_llm_response(self, response: str) -> Optional[dict]:
        """Parse LLM evaluation response."""
        import json

        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Try to extract score from text
        score_match = re.search(r'score[:\s]+(\d+)', response, re.IGNORECASE)
        if score_match:
            return {"score": int(score_match.group(1))}

        return None

    def _score_to_tier(self, score: float) -> SourceTier:
        """Convert score to trust tier."""
        if score >= 80:
            return SourceTier.TIER_1
        elif score >= 60:
            return SourceTier.TIER_2
        elif score >= 40:
            return SourceTier.TIER_3
        else:
            return SourceTier.UNKNOWN

    def _generate_reasoning(
        self,
        observable: dict,
        llm_eval: Optional[dict]
    ) -> str:
        """Generate human-readable reasoning for trust evaluation."""
        reasons = []

        if observable.get("has_sanskrit"):
            reasons.append("Contains Sanskrit text")
        if observable.get("has_verse_references"):
            reasons.append("Cites specific verses")
        if observable.get("has_bibliography"):
            reasons.append("Has bibliography/references")
        if observable.get("is_academic"):
            reasons.append("Academic domain")
        if observable.get("sensationalist_indicators", 0) > 0:
            reasons.append("Contains sensationalist language (negative)")

        if llm_eval:
            if llm_eval.get("scholarly_tone"):
                reasons.append("Scholarly tone (LLM)")
            if llm_eval.get("cites_primary"):
                reasons.append("Cites primary sources (LLM)")

        return "; ".join(reasons) if reasons else "No specific indicators"

    def update_trusted_sources(self, domain: str, evaluation: dict) -> None:
        """
        Add a newly evaluated source to the auto-evaluated list.

        This updates the trusted_sources.yaml file.
        """
        try:
            # Load current file
            with open(self.trusted_sources_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Add to auto_evaluated section
            if "auto_evaluated" not in data:
                data["auto_evaluated"] = {}

            data["auto_evaluated"][domain] = {
                "trust_score": evaluation["trust_score"],
                "tier": evaluation["trust_tier"].value if hasattr(evaluation["trust_tier"], "value") else evaluation["trust_tier"],
                "reasoning": evaluation["reasoning"],
                "evaluated_at": evaluation["evaluated_at"],
            }

            # Write back
            with open(self.trusted_sources_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            log.info("trust.sources_updated", domain=domain)

        except Exception as e:
            log.error("trust.sources_update_failed", domain=domain, error=str(e))
