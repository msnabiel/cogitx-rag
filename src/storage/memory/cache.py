"""Simple file-based document cache"""

import os
import json
import hashlib
from typing import Optional, Dict, Any


class DocumentCache:
    """File-based cache for document processing results"""

    def __init__(self, cache_dir: str, logger):
        self.cache_dir = cache_dir
        self.logger = logger
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, source: str) -> str:
        """Generate cache key from source"""
        return hashlib.md5(source.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{key}.json")

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
