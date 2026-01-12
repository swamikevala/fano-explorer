"""
Content cache management.

Handles caching of fetched content with TTL and size limits.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml

from shared.logging import get_logger

log = get_logger("researcher", "cache")


class ContentCache:
    """
    Manages cached web content.

    Features:
    - TTL-based expiration
    - Size-based pruning
    - Metadata tracking
    """

    def __init__(self, cache_dir: Path, config_path: Path):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cached content
            config_path: Path to settings.yaml
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = cache_dir / "_metadata.json"

        # Load config
        self.config = self._load_config(config_path)
        cache_config = self.config.get("cache", {})
        self.ttl_days = cache_config.get("content_ttl_days", 365)
        self.max_size_mb = cache_config.get("max_size_mb", 500)

        # Load or initialize metadata
        self._metadata = self._load_metadata()

    def _load_config(self, path: Path) -> dict:
        """Load settings config."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"entries": {}, "total_size_bytes": 0}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            log.warning("cache.metadata_save_failed", error=str(e))

    def get(self, url: str) -> Optional[dict]:
        """
        Get cached content for URL.

        Args:
            url: URL to look up

        Returns:
            Cached content dict or None
        """
        cache_key = self._url_to_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check metadata for expiration
        entry = self._metadata.get("entries", {}).get(cache_key)
        if entry:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            if datetime.now() - cached_at > timedelta(days=self.ttl_days):
                # Expired
                self._remove_entry(cache_key)
                return None

        # Read cached file
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    content = json.load(f)

                # Update access time in metadata
                if cache_key in self._metadata.get("entries", {}):
                    self._metadata["entries"][cache_key]["last_accessed"] = (
                        datetime.now().isoformat()
                    )
                    self._metadata["entries"][cache_key]["access_count"] = (
                        self._metadata["entries"][cache_key].get("access_count", 0) + 1
                    )

                return content

            except (json.JSONDecodeError, FileNotFoundError):
                return None

        return None

    def put(self, url: str, content: dict) -> bool:
        """
        Cache content for URL.

        Args:
            url: Source URL
            content: Content dict to cache

        Returns:
            True if cached successfully
        """
        # Check size limit
        self._ensure_space()

        cache_key = self._url_to_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            content_json = json.dumps(content, ensure_ascii=False, indent=2)
            content_bytes = content_json.encode("utf-8")
            content_size = len(content_bytes)

            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content_json)

            # Update metadata
            if "entries" not in self._metadata:
                self._metadata["entries"] = {}

            self._metadata["entries"][cache_key] = {
                "url": url,
                "cached_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "size_bytes": content_size,
                "access_count": 0,
            }
            self._metadata["total_size_bytes"] = (
                self._metadata.get("total_size_bytes", 0) + content_size
            )
            self._save_metadata()

            log.debug("cache.put", url=url, size_bytes=content_size)
            return True

        except Exception as e:
            log.error("cache.put_failed", url=url, error=str(e))
            return False

    def has(self, url: str) -> bool:
        """Check if URL is in cache."""
        cache_key = self._url_to_key(url)
        return cache_key in self._metadata.get("entries", {})

    def remove(self, url: str) -> bool:
        """Remove URL from cache."""
        cache_key = self._url_to_key(url)
        return self._remove_entry(cache_key)

    def _remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry by key."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            if cache_file.exists():
                entry = self._metadata.get("entries", {}).get(cache_key, {})
                size = entry.get("size_bytes", 0)

                cache_file.unlink()

                if cache_key in self._metadata.get("entries", {}):
                    del self._metadata["entries"][cache_key]
                    self._metadata["total_size_bytes"] = max(
                        0, self._metadata.get("total_size_bytes", 0) - size
                    )
                    self._save_metadata()

                return True

        except Exception as e:
            log.warning("cache.remove_failed", cache_key=cache_key, error=str(e))

        return False

    def _ensure_space(self) -> None:
        """Ensure cache is within size limits."""
        max_bytes = self.max_size_mb * 1024 * 1024
        current_size = self._metadata.get("total_size_bytes", 0)

        if current_size < max_bytes:
            return

        # Need to prune - remove least recently accessed entries
        entries = self._metadata.get("entries", {})
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].get("last_accessed", ""),
        )

        bytes_to_free = current_size - (max_bytes * 0.8)  # Free 20% extra
        bytes_freed = 0

        for cache_key, entry in sorted_entries:
            if bytes_freed >= bytes_to_free:
                break

            size = entry.get("size_bytes", 0)
            if self._remove_entry(cache_key):
                bytes_freed += size
                log.debug("cache.pruned", cache_key=cache_key, size_bytes=size)

        log.info("cache.pruned_complete", bytes_freed=bytes_freed)

    def _url_to_key(self, url: str) -> str:
        """Convert URL to cache key."""
        return hashlib.md5(url.encode()).hexdigest()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        entries = self._metadata.get("entries", {})

        return {
            "entry_count": len(entries),
            "total_size_bytes": self._metadata.get("total_size_bytes", 0),
            "total_size_mb": round(
                self._metadata.get("total_size_bytes", 0) / (1024 * 1024), 2
            ),
            "max_size_mb": self.max_size_mb,
            "ttl_days": self.ttl_days,
        }

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        entries = list(self._metadata.get("entries", {}).items())
        removed = 0
        cutoff = datetime.now() - timedelta(days=self.ttl_days)

        for cache_key, entry in entries:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            if cached_at < cutoff:
                if self._remove_entry(cache_key):
                    removed += 1

        if removed > 0:
            log.info("cache.cleanup_complete", entries_removed=removed)

        return removed
