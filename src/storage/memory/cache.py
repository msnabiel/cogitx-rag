"""Simple file-based document cache"""

import os
import json
import hashlib
from typing import Optional, Dict, Any, Set


class DocumentCache:
    """File-based cache for document processing results"""

    def __init__(self, cache_dir: str, logger):
        self.cache_dir = cache_dir
        self.logger = logger
        os.makedirs(cache_dir, exist_ok=True)
        self.index_registry_file = os.path.join(cache_dir, "ingested_hashes.json")

    def _get_cache_key(self, source: str) -> str:
        """Generate cache key from source"""
        return hashlib.md5(source.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{key}.json")

    def _load_index_registry(self) -> Set[str]:
        if not os.path.exists(self.index_registry_file):
            return set()
        try:
            with open(self.index_registry_file, "r") as f:
                data = json.load(f)
            return set(data.get("ingested_hashes", []))
        except Exception as e:
            self.logger.warning(f"Failed to load index registry: {e}")
            return set()

    def _save_index_registry(self, hashes: Set[str]) -> None:
        try:
            with open(self.index_registry_file, "w") as f:
                json.dump({"ingested_hashes": sorted(hashes)}, f)
        except Exception as e:
            self.logger.warning(f"Failed to save index registry: {e}")

    def get(self, source: str) -> Optional[Dict[str, Any]]:
        """Get cached document data"""
        key = self._get_cache_key(source)
        cache_path = self._get_cache_path(key)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {source}: {e}")
        return None

    def set(self, source: str, data: Dict[str, Any]):
        """Cache document data"""
        key = self._get_cache_key(source)
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'w') as f:
                json.dump({"data": data}, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache {source}: {e}")

    def is_ingested(self, content_hash: str) -> bool:
        """Check whether a content hash has already been indexed."""
        return content_hash in self._load_index_registry()

    def mark_ingested(self, content_hash: str) -> None:
        """Record a content hash as indexed."""
        hashes = self._load_index_registry()
        if content_hash not in hashes:
            hashes.add(content_hash)
            self._save_index_registry(hashes)

    def clear_registry(self) -> None:
        """Clear the ingestion registry."""
        if os.path.exists(self.index_registry_file):
            try:
                os.remove(self.index_registry_file)
                self.logger.info("Ingestion registry cleared.")
            except Exception as e:
                self.logger.error(f"Failed to clear ingestion registry: {e}")

    def clear_all(self) -> bool:
        """Clear all cache entries"""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in files
            )
            return {
                "total_entries": len(files),
                "total_size_bytes": total_size,
                "cache_dir": self.cache_dir
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
