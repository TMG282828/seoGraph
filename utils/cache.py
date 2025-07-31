"""
Cache utilities for the SEO Content Knowledge Graph System.

This module provides caching decorators and utilities for FastAPI endpoints.
"""

import asyncio
import json
import hashlib
from functools import wraps
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone, timedelta
import structlog

from cachetools import TTLCache

logger = structlog.get_logger(__name__)

# Global cache instance
_cache = TTLCache(maxsize=1000, ttl=300)  # Default 5 minutes


def cache_response(ttl: int = 300, key_prefix: str = ""):
    """
    Decorator to cache FastAPI endpoint responses.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Optional prefix for cache keys
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs, key_prefix)
            
            # Try to get from cache
            if cache_key in _cache:
                logger.debug(f"Cache hit for {cache_key}")
                return _cache[cache_key]
            
            # Execute function and cache result
            try:
                result = await func(*args, **kwargs)
                _cache[cache_key] = result
                logger.debug(f"Cached result for {cache_key}")
                return result
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


def cache_key(*args, **kwargs):
    """
    Generate cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        str: Cache key
    """
    key_data = {
        'args': args,
        'kwargs': kwargs
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict, prefix: str = "") -> str:
    """Generate cache key for function call."""
    # Filter out non-serializable arguments
    clean_args = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool, type(None))):
            clean_args.append(arg)
        else:
            clean_args.append(str(type(arg).__name__))
    
    clean_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            clean_kwargs[key] = value
        else:
            clean_kwargs[key] = str(type(value).__name__)
    
    key_data = {
        'function': func_name,
        'args': clean_args,
        'kwargs': clean_kwargs
    }
    
    key_str = json.dumps(key_data, sort_keys=True)
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    
    if prefix:
        cache_key = f"{prefix}:{cache_key}"
    
    return cache_key


def invalidate_cache(pattern: str = "") -> int:
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Pattern to match cache keys
    
    Returns:
        int: Number of invalidated entries
    """
    if not pattern:
        # Clear all cache
        count = len(_cache)
        _cache.clear()
        return count
    
    # Clear matching entries
    keys_to_remove = [key for key in _cache.keys() if pattern in key]
    for key in keys_to_remove:
        del _cache[key]
    
    return len(keys_to_remove)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "size": len(_cache),
        "maxsize": _cache.maxsize,
        "ttl": _cache.ttl,
        "hits": getattr(_cache, 'hits', 0),
        "misses": getattr(_cache, 'misses', 0),
        "current_time": datetime.now(timezone.utc).isoformat()
    }


class CacheManager:
    """Cache manager for advanced caching operations."""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 300):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.cache[key]
            self.stats['hits'] += 1
            return value
        except KeyError:
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if ttl:
            # Create new cache with custom TTL
            custom_cache = TTLCache(maxsize=self.cache.maxsize, ttl=ttl)
            custom_cache[key] = value
        else:
            self.cache[key] = value
        
        self.stats['sets'] += 1
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            del self.cache[key]
            self.stats['deletes'] += 1
            return True
        except KeyError:
            return False
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.stats['deletes'] += len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            'size': len(self.cache),
            'maxsize': self.cache.maxsize,
            'ttl': self.cache.ttl
        }


# Global cache manager instance
cache_manager = CacheManager()


def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """
    Advanced caching decorator with custom key function.
    
    Args:
        ttl: Time to live in seconds
        key_func: Custom function to generate cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


# Distributed cache support (placeholder for Redis integration)
class DistributedCache:
    """Distributed cache using Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._client = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis
            self._client = redis.from_url(self.redis_url)
            await self._client.ping()
            logger.info("Connected to Redis cache")
        except ImportError:
            logger.warning("Redis not available, using in-memory cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        if not self._client:
            return None
        
        try:
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from distributed cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value in distributed cache."""
        if not self._client:
            return
        
        try:
            serialized_value = json.dumps(value, default=str)
            await self._client.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Error setting in distributed cache: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from distributed cache."""
        if not self._client:
            return False
        
        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting from distributed cache: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        if not self._client:
            return 0
        
        try:
            keys = await self._client.keys(pattern)
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern: {e}")
            return 0


# Global distributed cache instance
distributed_cache = DistributedCache()


async def init_distributed_cache(redis_url: str = "redis://localhost:6379"):
    """Initialize distributed cache."""
    global distributed_cache
    distributed_cache = DistributedCache(redis_url)
    await distributed_cache.initialize()


def distributed_cached(ttl: int = 300, key_prefix: str = ""):
    """
    Distributed caching decorator using Redis.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs, key_prefix)
            
            # Try to get from distributed cache
            cached_result = await distributed_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await distributed_cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator