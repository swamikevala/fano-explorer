"""
Content fetcher for retrieving web pages.

Handles HTML fetching, basic parsing, and content extraction.
"""

import asyncio
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import yaml

from shared.logging import get_logger

log = get_logger("researcher", "fetcher")

# Try to import optional dependencies
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class ContentFetcher:
    """
    Fetches and processes web content.

    Supports:
    - HTML page fetching
    - Basic content extraction
    - Rate limiting per domain
    """

    def __init__(self, config_path: Path, cache_dir: Path):
        """
        Initialize fetcher.

        Args:
            config_path: Path to settings.yaml
            cache_dir: Path to cache directory
        """
        self.config = self._load_config(config_path)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting per domain
        self._domain_last_fetch: dict[str, datetime] = {}
        self._min_delay_seconds = 2  # Minimum delay between requests to same domain

        # Settings
        search_config = self.config.get("search", {})
        self._timeout = search_config.get("request_timeout_seconds", 30)
        self._user_agent = search_config.get(
            "user_agent",
            "FanoResearcher/1.0 (Academic Research Tool)"
        )

    def _load_config(self, path: Path) -> dict:
        """Load settings config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    async def fetch(self, url: str, use_cache: bool = True) -> Optional[dict]:
        """
        Fetch content from URL.

        Args:
            url: URL to fetch
            use_cache: Whether to use cached content if available

        Returns:
            Dict with content, title, hash, or None if failed
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                log.debug("fetcher.cache_hit", url=url)
                return cached

        # Check rate limit for domain
        domain = urlparse(url).netloc
        if not self._check_domain_rate_limit(domain):
            log.warning("fetcher.rate_limited", domain=domain)
            await asyncio.sleep(self._min_delay_seconds)

        # Fetch content
        try:
            content = await self._fetch_url(url)
            if content:
                # Parse and extract
                result = self._process_content(url, content)
                if result:
                    # Cache the result
                    self._cache_content(url, result)
                    return result

        except Exception as e:
            log.error("fetcher.fetch_failed", url=url, error=str(e))

        return None

    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch raw content from URL."""
        if not AIOHTTP_AVAILABLE:
            log.warning("fetcher.aiohttp_not_available")
            return await self._fetch_url_fallback(url)

        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                    ssl=False  # Some older sites have certificate issues
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        log.warning(
                            "fetcher.http_error",
                            url=url,
                            status=response.status
                        )
                        return None

        except asyncio.TimeoutError:
            log.warning("fetcher.timeout", url=url)
            return None
        except Exception as e:
            log.error("fetcher.request_failed", url=url, error=str(e))
            return None

    async def _fetch_url_fallback(self, url: str) -> Optional[str]:
        """Fallback fetcher using urllib (synchronous)."""
        import urllib.request
        import urllib.error

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self._user_agent}
            )
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            log.error("fetcher.fallback_failed", url=url, error=str(e))
            return None

    def _process_content(self, url: str, html: str) -> Optional[dict]:
        """
        Process HTML content and extract useful information.

        Args:
            url: Source URL
            html: Raw HTML content

        Returns:
            Dict with extracted content, title, text, etc.
        """
        if not BS4_AVAILABLE:
            # Basic extraction without BeautifulSoup
            return self._basic_extract(url, html)

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string or ""

            # Extract main content
            main_content = soup.find("main") or soup.find("article") or soup.find("body")
            text = main_content.get_text(separator="\n", strip=True) if main_content else ""

            # Clean up text
            text = self._clean_text(text)

            # Compute hash
            content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

            # Check for Sanskrit/verse references
            has_sanskrit = bool(re.search(r'[\u0900-\u097F]', text))  # Devanagari
            has_verse_refs = bool(re.search(r'\b\d+\.\d+|\bverse\s+\d+|\bśloka\s+\d+', text, re.I))

            return {
                "url": url,
                "domain": urlparse(url).netloc,
                "title": title.strip(),
                "text": text,
                "content_hash": content_hash,
                "content_type": "html",
                "has_sanskrit": has_sanskrit,
                "has_verse_references": has_verse_refs,
                "fetched_at": datetime.now().isoformat(),
                "word_count": len(text.split()),
            }

        except Exception as e:
            log.error("fetcher.process_failed", url=url, error=str(e))
            return self._basic_extract(url, html)

    def _basic_extract(self, url: str, html: str) -> dict:
        """Basic content extraction without BeautifulSoup."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        text = self._clean_text(text)

        # Extract title from <title> tag
        title_match = re.search(r'<title>([^<]+)</title>', html, re.I)
        title = title_match.group(1) if title_match else ""

        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        return {
            "url": url,
            "domain": urlparse(url).netloc,
            "title": title.strip(),
            "text": text,
            "content_hash": content_hash,
            "content_type": "html",
            "has_sanskrit": bool(re.search(r'[\u0900-\u097F]', text)),
            "has_verse_references": bool(re.search(r'\b\d+\.\d+', text)),
            "fetched_at": datetime.now().isoformat(),
            "word_count": len(text.split()),
        }

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _check_domain_rate_limit(self, domain: str) -> bool:
        """Check if we can fetch from this domain."""
        if domain not in self._domain_last_fetch:
            self._domain_last_fetch[domain] = datetime.now()
            return True

        elapsed = (datetime.now() - self._domain_last_fetch[domain]).total_seconds()
        if elapsed >= self._min_delay_seconds:
            self._domain_last_fetch[domain] = datetime.now()
            return True

        return False

    def _get_cached(self, url: str) -> Optional[dict]:
        """Get cached content for URL."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                import json
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        return None

    def _cache_content(self, url: str, content: dict) -> None:
        """Cache content for URL."""
        import json

        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("fetcher.cache_write_failed", url=url, error=str(e))
