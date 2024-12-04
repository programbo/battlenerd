from typing import Dict, Optional, Any, List
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
from collections import defaultdict

class QueryCache:
    def __init__(self, cache_dir: str = "cache", ttl_days: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(days=ttl_days)

        # In-memory cache for faster access
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        # Index to track which documents are used in which cache entries
        self.document_cache_index: Dict[str, set] = defaultdict(set)

        # Load existing cache from disk
        self._load_cache()

    def _load_cache(self):
        """Load all cache files into memory"""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Check if cache is still valid
                    if datetime.fromisoformat(cache_data['timestamp']) + self.ttl > datetime.now():
                        self.memory_cache[cache_file.stem] = cache_data
                        # Rebuild document index
                        for doc_id in cache_data.get('document_ids', []):
                            self.document_cache_index[doc_id].add(cache_file.stem)
                    else:
                        # Remove expired cache file
                        cache_file.unlink()
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}")

    def _create_cache_key(self, query: str, context: str, offline: bool) -> str:
        """Create a unique cache key based on query parameters"""
        # Simple cache key for demonstration
        # Could be enhanced with more sophisticated hashing
        return f"{hash(query)}_{hash(context)}_{offline}"

    def _extract_document_ids(self, metadata: List[Dict[str, Any]]) -> List[str]:
        """Extract unique document IDs from response metadata"""
        return list(set(
            f"{meta.get('filename')}_{meta.get('section_title')}"
            for meta in metadata
            if meta.get('filename') and meta.get('section_title')
        ))

    def get(self, query: str, context: str, offline: bool) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        cache_key = self._create_cache_key(query, context, offline)
        cache_data = self.memory_cache.get(cache_key)

        if cache_data:
            # Check if cache is still valid
            if datetime.fromisoformat(cache_data['timestamp']) + self.ttl > datetime.now():
                return cache_data['response']
            else:
                # Remove expired cache
                self._remove(cache_key)
        return None

    def set(self, query: str, context: str, offline: bool, response: Dict[str, Any]):
        """Cache a new response"""
        cache_key = self._create_cache_key(query, context, offline)

        # Extract document IDs from response metadata
        document_ids = self._extract_document_ids(response.get('metadata', []))

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'context': context,
            'offline': offline,
            'response': response,
            'document_ids': document_ids
        }

        # Save to memory cache
        self.memory_cache[cache_key] = cache_data

        # Update document index
        for doc_id in document_ids:
            self.document_cache_index[doc_id].add(cache_key)

        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    def _remove(self, cache_key: str):
        """Remove a cache entry"""
        # Remove from document index
        if cache_key in self.memory_cache:
            for doc_id in self.memory_cache[cache_key].get('document_ids', []):
                self.document_cache_index[doc_id].discard(cache_key)

        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            cache_file.unlink()

    def invalidate_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Invalidate cache entries affected by new or updated documents.
        Returns number of cache entries invalidated.
        """
        invalidated_count = 0
        for doc in documents:
            filename = doc['filename']
            # Find all cache entries that use this document
            affected_keys = set()
            for doc_id in self.document_cache_index:
                if doc_id.startswith(filename):
                    affected_keys.update(self.document_cache_index[doc_id])

            # Remove affected cache entries
            for cache_key in affected_keys:
                self._remove(cache_key)
                invalidated_count += 1

        return invalidated_count

    def clear(self):
        """Clear all cache entries"""
        self.memory_cache.clear()
        self.document_cache_index.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
