"""
Caching utilities for NeuronScope

This module provides caching functionality for expensive operations like
weight analysis and pruning impact calculations.
"""

import json
import hashlib
import time
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
import pickle
import os

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for expensive operations."""
    
    def __init__(self, cache_dir: str = "data/cache", max_age_hours: int = 24):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        
    def _get_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for an operation with parameters."""
        # Create a deterministic string representation of parameters
        param_str = json.dumps(params, sort_keys=True)
        # Create hash of operation + parameters
        key_data = f"{operation}:{param_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached data if it exists and is not expired.
        
        Args:
            operation: Name of the operation (e.g., 'weight_analysis', 'pruning_candidates')
            params: Parameters used for the operation
            
        Returns:
            Cached data if valid, None otherwise
        """
        cache_key = self._get_cache_key(operation, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check if cache is expired
            if time.time() - cache_path.stat().st_mtime > self.max_age_seconds:
                logger.info(f"Cache expired for {operation}, removing: {cache_path}")
                cache_path.unlink()
                return None
            
            # Load cached data
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            logger.info(f"Cache hit for {operation}: {cache_path}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Error loading cache for {operation}: {e}")
            # Remove corrupted cache file
            if cache_path.exists():
                cache_path.unlink()
            return None
    
    def set(self, operation: str, params: Dict[str, Any], data: Any) -> None:
        """Store data in cache.
        
        Args:
            operation: Name of the operation
            params: Parameters used for the operation
            data: Data to cache
        """
        cache_key = self._get_cache_key(operation, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Cached data for {operation}: {cache_path}")
            
        except Exception as e:
            logger.error(f"Error caching data for {operation}: {e}")
    
    def invalidate(self, operation: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Invalidate cache entries.
        
        Args:
            operation: Operation name to invalidate
            params: Specific parameters to invalidate (if None, invalidates all for operation)
        """
        if params is None:
            # Invalidate all cache entries for this operation
            pattern = f"*_{operation}_*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"Invalidated cache: {cache_file}")
        else:
            # Invalidate specific cache entry
            cache_key = self._get_cache_key(operation, params)
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Invalidated cache: {cache_path}")
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cleared all cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "total_entries": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "max_age_hours": self.max_age_seconds / 3600
        }

# Global cache instance
cache_manager = CacheManager()

def cached(operation: str, max_age_hours: Optional[int] = None):
    """Decorator to cache function results.
    
    Args:
        operation: Name of the operation for caching
        max_age_hours: Override default max age for this operation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create parameters dict from function arguments
            params = {
                "args": args,
                "kwargs": kwargs
            }
            
            # Check cache first
            cached_result = cache_manager.get(operation, params)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(operation, params, result)
            
            return result
        return wrapper
    return decorator

# Cache operation names
CACHE_OPERATIONS = {
    "weight_analysis": "weight_analysis",
    "pruning_candidates": "pruning_candidates", 
    "pruning_impact": "pruning_impact",
    "neuron_importance": "neuron_importance",
    "batch_analysis": "batch_analysis"
} 