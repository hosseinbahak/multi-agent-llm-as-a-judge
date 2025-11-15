# core/cache_manager.py
"""Caching system for LLM responses."""

import hashlib
import json
import pickle
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple, Callable
from datetime import datetime, timedelta
from functools import wraps
import aiofiles
import redis.asyncio as aioredis
from pathlib import Path
from loguru import logger

from .exceptions import CacheError


class CacheBackend(ABC):
    """Abstract cache backend."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str):
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self):
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Tuple[Any, Optional[datetime]]] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self._lock:
            if key not in self.cache:
                return None
            
            value, expiry = self.cache[key]
            
            # Check expiry
            if expiry and datetime.now() > expiry:
                del self.cache[key]
                return None
            
            # Update access count for LRU
            self.access_count[key] = self.access_count.get(key, 0) + 1
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in memory cache."""
        async with self._lock:
            # Implement LRU eviction if needed
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove least recently used
                lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
                del self.cache[lru_key]
                del self.access_count[lru_key]
            
            expiry = datetime.now() + timedelta(seconds=ttl) if ttl else None
            self.cache[key] = (value, expiry)
            self.access_count[key] = 1
    
    async def delete(self, key: str):
        """Delete from memory cache."""
        async with self._lock:
            self.cache.pop(key, None)
            self.access_count.pop(key, None)
    
    async def clear(self):
        """Clear memory cache."""
        async with self._lock:
            self.cache.clear()
            self.access_count.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        return key in self.cache


class DiskCache(CacheBackend):
    """Disk-based cache backend."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    def _get_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars for directory sharding
        shard = key[:2]
        return self.cache_dir / shard / f"{key}.pkl"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        path = self._get_path(key)
        
        if not path.exists():
            return None
        
        try:
            async with aiofiles.open(path, 'rb') as f:
                data = await f.read()
                cached = pickle.loads(data)
                
                # Check expiry
                if cached['expiry'] and datetime.now() > cached['expiry']:
                    await self.delete(key)
                    return None
                
                return cached['value']
                
        except Exception as e:
            logger.error(f"Failed to read cache {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in disk cache."""
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cached = {
                'value': value,
                'expiry': datetime.now() + timedelta(seconds=ttl) if ttl else None,
                'created': datetime.now()
            }
            
            async with aiofiles.open(path, 'wb') as f:
                await f.write(pickle.dumps(cached))
                
        except Exception as e:
            logger.error(f"Failed to write cache {key}: {e}")
            raise CacheError(f"Failed to write cache: {e}")
    
    async def delete(self, key: str):
        """Delete from disk cache."""
        path = self._get_path(key)
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete cache {key}: {e}")
    
    async def clear(self):
        """Clear disk cache."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists on disk."""
        return self._get_path(key).exists()


class RedisCache(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self.redis:
            raise CacheError("Redis not initialized")
        
        try:
            data = await self.redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis."""
        if not self.redis:
            raise CacheError("Redis not initialized")
        
        try:
            data = pickle.dumps(value)
            await self.redis.set(key, data, ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            raise CacheError(f"Redis set error: {e}")
    
    async def delete(self, key: str):
        """Delete from Redis."""
        if not self.redis:
            raise CacheError("Redis not initialized")
        
        await self.redis.delete(key)
    
    async def clear(self):
        """Clear Redis cache (use with caution)."""
        if not self.redis:
            raise CacheError("Redis not initialized")
        
        await self.redis.flushdb()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.redis:
            raise CacheError("Redis not initialized")
        
        return await self.redis.exists(key) > 0


class CacheManager:
    """Main cache manager supporting multiple backends."""
    
    def __init__(
        self,
        primary_backend: CacheBackend,
        secondary_backend: Optional[CacheBackend] = None,
        namespace: str = "judge",
        stats_enabled: bool = True
    ):
        self.primary = primary_backend
        self.secondary = secondary_backend
        self.namespace = namespace
        self.stats_enabled = stats_enabled
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0
        }
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        full_key = self._make_key(key)
        
        # Try primary cache
        try:
            value = await self.primary.get(full_key)
            if value is not None:
                if self.stats_enabled:
                    self.stats["hits"] += 1
                return value
        except Exception as e:
            logger.error(f"Primary cache error: {e}")
            self.stats["errors"] += 1
        
        # Try secondary cache if available
        if self.secondary:
            try:
                value = await self.secondary.get(full_key)
                if value is not None:
                    # Promote to primary cache
                    await self.primary.set(full_key, value)
                    if self.stats_enabled:
                        self.stats["hits"] += 1
                    return value
            except Exception as e:
                logger.error(f"Secondary cache error: {e}")
                self.stats["errors"] += 1
        
        if self.stats_enabled:
            self.stats["misses"] += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache."""
        full_key = self._make_key(key)
        
        # Set in primary cache
        try:
            await self.primary.set(full_key, value, ttl)
            if self.stats_enabled:
                self.stats["sets"] += 1
        except Exception as e:
            logger.error(f"Failed to set primary cache: {e}")
            self.stats["errors"] += 1
        
        # Set in secondary cache if available
        if self.secondary:
            try:
                await self.secondary.set(full_key, value, ttl)
            except Exception as e:
                logger.error(f"Failed to set secondary cache: {e}")
    
    async def delete(self, key: str):
        """Delete from cache."""
        full_key = self._make_key(key)
        
        await self.primary.delete(full_key)
        if self.secondary:
            await self.secondary.delete(full_key)
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def cached(
        self,
        ttl: Optional[int] = 3600,
        key_prefix: Optional[str] = None
    ) -> Callable:
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.generate_key(*args, **kwargs)
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # Try to get from cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # Call function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total
        }
