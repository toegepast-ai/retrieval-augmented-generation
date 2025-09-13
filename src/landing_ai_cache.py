#!/usr/bin/env python3
"""
Landing AI Results Manager - Save and load costly Landing AI results
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LandingAICache:
    """Cache Landing AI results to avoid costly re-processing"""
    
    def __init__(self, cache_dir: str = "./data/landing_ai_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file to use as cache key"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def _get_cache_path(self, file_path: str) -> Path:
        """Get the cache file path for a document"""
        file_hash = self._get_file_hash(file_path)
        filename = Path(file_path).stem
        cache_file = f"{filename}_{file_hash}.json"
        return self.cache_dir / cache_file
    
    def save_result(self, file_path: str, result: Dict[str, Any]) -> bool:
        """Save Landing AI result to cache"""
        try:
            cache_path = self._get_cache_path(file_path)
            
            cache_data = {
                "file_path": file_path,
                "file_hash": self._get_file_hash(file_path),
                "result": result,
                "timestamp": str(Path(file_path).stat().st_mtime)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved Landing AI result to cache: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def load_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load Landing AI result from cache if available and valid"""
        try:
            cache_path = self._get_cache_path(file_path)
            
            if not cache_path.exists():
                logger.info(f"No cache found for {Path(file_path).name}")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Verify file hasn't changed
            current_hash = self._get_file_hash(file_path)
            if cache_data.get("file_hash") != current_hash:
                logger.warning(f"File changed since cache - cache invalid for {Path(file_path).name}")
                return None
            
            logger.info(f"Loaded Landing AI result from cache: {cache_path}")
            return cache_data.get("result")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def list_cached_files(self) -> list:
        """List all cached files"""
        cache_files = list(self.cache_dir.glob("*.json"))
        return [f.name for f in cache_files]
    
    def clear_cache(self) -> bool:
        """Clear all cached results"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared Landing AI cache")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

def check_existing_results():
    """Check if we have any cached Landing AI results"""
    print("🔍 Checking for existing Landing AI results...")
    
    cache = LandingAICache()
    cached_files = cache.list_cached_files()
    
    if cached_files:
        print(f"✅ Found {len(cached_files)} cached results:")
        for filename in cached_files:
            print(f"  📄 {filename}")
    else:
        print("❌ No cached Landing AI results found")
        print("💡 Future runs will be cached to avoid re-processing costs")
    
    # Check if the test file exists and what would be the cost
    test_files = list(Path("data/uploads").glob("*.pdf"))
    if test_files:
        test_file = test_files[0]
        cached_result = cache.load_result(str(test_file))
        if cached_result:
            print(f"✅ {test_file.name} has cached Landing AI results!")
            print(f"📊 Cached chunks: {len(cached_result.get('chunks', []))}")
        else:
            print(f"⚠️  {test_file.name} would require Landing AI processing (costs credits)")

if __name__ == "__main__":
    check_existing_results()
