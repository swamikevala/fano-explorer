"""
Web search functionality for discovering sources.

Provides abstraction over search APIs and direct site queries.
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
import yaml

from shared.logging import get_logger

log = get_logger("researcher", "searcher")


class WebSearcher:
    """
    Web search for discovering relevant sources.

    Supports:
    - Direct queries to known authoritative sites
    - Web search API (when configured)
    - Rate limiting and caching
    """

    def __init__(self, config_path: Path, trusted_sources_path: Path):
        """
        Initialize searcher.

        Args:
            config_path: Path to settings.yaml
            trusted_sources_path: Path to trusted_sources.yaml
        """
        self.config = self._load_config(config_path)
        self.trusted_sources = self._load_trusted_sources(trusted_sources_path)

        # Rate limiting
        self._search_count = 0
        self._search_reset_time = datetime.now()
        self._max_searches_per_hour = self.config.get("search", {}).get(
            "max_searches_per_hour", 30
        )

    def _load_config(self, path: Path) -> dict:
        """Load settings config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    def _load_trusted_sources(self, path: Path) -> dict:
        """Load trusted sources config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {"seed_sources": {"tier_1": [], "tier_2": []}}

    async def search(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> list[dict]:
        """
        Search for relevant sources.

        Args:
            query: Search query
            domain: Optional domain to filter by (from domains.yaml)
            limit: Maximum results to return

        Returns:
            List of search results with url, title, snippet
        """
        # Check rate limit
        if not self._check_rate_limit():
            log.warning("searcher.rate_limited", query=query)
            return []

        results = []

        # First, check for known authoritative URLs
        direct_urls = self._get_direct_urls(query, domain)
        for url in direct_urls[:limit // 2]:
            results.append({
                "url": url,
                "title": f"Direct: {urlparse(url).netloc}",
                "snippet": "",
                "source_type": "direct",
            })

        # Then try web search if configured and needed
        if len(results) < limit:
            web_results = await self._web_search(query, limit - len(results))
            results.extend(web_results)

        self._search_count += 1
        log.info(
            "searcher.search.complete",
            query=query,
            domain=domain,
            result_count=len(results)
        )

        return results[:limit]

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()

        # Reset counter every hour
        if (now - self._search_reset_time).total_seconds() > 3600:
            self._search_count = 0
            self._search_reset_time = now

        return self._search_count < self._max_searches_per_hour

    def _get_direct_urls(self, query: str, domain: Optional[str]) -> list[str]:
        """
        Get direct URLs from known authoritative sites.

        Constructs URLs for sites we know have relevant content.
        """
        urls = []

        # Known site patterns for Hindu scriptures
        site_patterns = {
            "wisdomlib.org": [
                "https://www.wisdomlib.org/search?q={query}",
            ],
            "sacred-texts.com": [
                "https://www.sacred-texts.com/search.htm?q={query}",
            ],
        }

        # Construct search URLs for tier 1 sources
        tier1 = self.trusted_sources.get("seed_sources", {}).get("tier_1", [])
        for source in tier1:
            source_domain = source.get("domain", "")
            if source_domain in site_patterns:
                for pattern in site_patterns[source_domain]:
                    urls.append(pattern.format(query=query.replace(" ", "+")))

        # Add domain-specific known pages
        domain_specific_urls = self._get_domain_specific_urls(query, domain)
        urls.extend(domain_specific_urls)

        return urls

    def _get_domain_specific_urls(
        self,
        query: str,
        domain: Optional[str]
    ) -> list[str]:
        """Get URLs specific to a research domain."""
        urls = []

        # Map domains to known resource URLs
        domain_urls = {
            "kashmiri_shaivism": [
                "https://www.wisdomlib.org/hinduism/essay/kashmir-shaivism",
                "https://www.wisdomlib.org/hinduism/book/shiva-sutras",
            ],
            "tantraloka": [
                "https://www.wisdomlib.org/hinduism/book/tantraloka",
            ],
            "soundarya_lahari": [
                "https://www.wisdomlib.org/hinduism/book/soundarya-lahari",
            ],
            "hatha_yoga": [
                "https://www.wisdomlib.org/hinduism/book/hatha-yoga-pradipika",
            ],
            "vedas": [
                "https://www.sacred-texts.com/hin/rigveda/index.htm",
                "https://www.wisdomlib.org/hinduism/book/rig-veda",
            ],
            "indian_music": [
                "https://www.wisdomlib.org/hinduism/book/sangita-ratnakara",
                "https://www.wisdomlib.org/hinduism/book/natyashastra",
            ],
            "yantras": [
                "https://www.wisdomlib.org/definition/yantra",
            ],
            "sanskrit_grammar": [
                "https://www.wisdomlib.org/hinduism/book/ashtadhyayi",
            ],
        }

        if domain and domain in domain_urls:
            urls.extend(domain_urls[domain])

        return urls

    async def _web_search(self, query: str, limit: int) -> list[dict]:
        """
        Perform web search using available API.

        Currently returns empty - to be implemented with search API.
        """
        # TODO: Implement actual web search API
        # Options:
        # - Google Custom Search API
        # - Bing Search API
        # - SerpAPI
        # - Brave Search API

        log.debug("searcher.web_search.not_implemented", query=query)
        return []

    def get_search_suggestions(self, context: dict) -> list[str]:
        """
        Generate search suggestions based on research context.

        Args:
            context: Research context with concepts, numbers, domains

        Returns:
            List of suggested search queries
        """
        suggestions = []

        # Concept-based suggestions
        for concept in context.get("active_concepts", [])[:5]:
            suggestions.append(f"{concept} Hindu scripture")
            suggestions.append(f"{concept} Sanskrit meaning")

        # Number-based suggestions
        for number in context.get("active_numbers", [])[:5]:
            suggestions.append(f"{number} significance Hindu")
            suggestions.append(f"{number} sacred number Indian tradition")

        # Domain-based suggestions
        for domain in context.get("active_domains", [])[:3]:
            domain_clean = domain.replace("_", " ")
            suggestions.append(f"{domain_clean} texts")

        return suggestions[:20]
