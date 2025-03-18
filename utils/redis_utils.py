import asyncio
import json
import logging
import time
from typing import Any, Optional
import redis.asyncio as redis
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis_cache")

# Import from your config file
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB_EM, REDIS_URL

# Global cache flags
# _global_cache_initialized = False
# _embedding_caches = {}

class RedisClient:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT):
        self.redis = redis.Redis(host=host, port=port, db=2, decode_responses=True)
        self.lock_redis = redis.Redis(host=host, port=port, db=3, decode_responses=True)
        # self._cache_initialized = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            await self.redis.ping()
            await self.lock_redis.ping()
            logger.info("Redis connections initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            return False
    
    async def set_with_ttl(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set a key with TTL."""
        try:
            await self.redis.set(key, json.dumps(value), ex=ttl_seconds)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key."""
        try:
            result = await self.redis.get(key)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None
    
    async def increment_counter(self, key: str, ttl_seconds: int = 60) -> int:
        """Increment a counter with TTL and return its new value."""
        pipe = self.redis.pipeline()
        try:
            await pipe.incr(key)
            await pipe.expire(key, ttl_seconds)
            results = await pipe.execute()
            return results[0]  # Return the incremented value
        except Exception as e:
            logger.error(f"Redis increment error: {str(e)}")
            return 0
    
    async def acquire_lock(self, lock_name: str, owner: str, ttl_seconds: int = 30) -> bool:
        """
        Acquire a distributed lock with a given TTL.
        Returns True if the lock was acquired, False otherwise.
        """
        try:
            result = await self.lock_redis.set(f"lock:{lock_name}", owner, nx=True, ex=ttl_seconds)
            return result is not None
        except Exception as e:
            logger.error(f"Redis lock acquisition error: {str(e)}")
            return False
    
    async def release_lock(self, lock_name: str, owner: str) -> bool:
        """
        Release a distributed lock if owned by the specified owner.
        Returns True if the lock was released, False otherwise.
        """
        try:
            pipe = self.lock_redis.pipeline()
            await pipe.get(f"lock:{lock_name}")
            await pipe.delete(f"lock:{lock_name}")
            results = await pipe.execute()
            return results[0] == owner
        except Exception as e:
            logger.error(f"Redis lock release error: {str(e)}")
            return False
    
    async def close(self):
        """Close the Redis connections."""
        try:
            await self.redis.close()
            await self.lock_redis.close()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Redis close error: {str(e)}")

# Rate limiter decorator with exponential backoff
def rate_limit(limit: int, period: int = 60, max_delay: int = 30):
    """
    Rate limiting decorator using Redis with exponential backoff.
    
    Args:
        limit: Maximum number of calls per period
        period: Time period in seconds
        max_delay: Maximum delay for retries in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a key based on function name
            rate_key = f"ratelimit:{func.__name__}"
            
            # Get Redis client instance
            redis_client = redis_client_instance
            
            # Try with exponential backoff if needed
            retry_count = 0
            max_retries = 3
            base_delay = 1
            
            while retry_count <= max_retries:
                # Check rate limit
                current = await redis_client.increment_counter(rate_key, period)
                
                if current <= limit:
                    # Under limit, proceed with function call
                    return await func(*args, **kwargs)
                
                # We're over the limit
                if retry_count == max_retries:
                    # No more retries, raise exception
                    logger.warning(f"Rate limit exceeded for {func.__name__} after {retry_count} retries")
                    raise RateLimitExceeded(f"Rate limit of {limit} requests per {period}s exceeded")
                
                # Calculate delay with exponential backoff (capped at max_delay)
                delay = min(base_delay * (2 ** retry_count), max_delay)
                logger.info(f"Rate limit hit for {func.__name__}, retrying in {delay}s")
                
                # Wait before retry
                await asyncio.sleep(delay)
                retry_count += 1
        
        return wrapper
    return decorator

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass

# Singleton instance
redis_client_instance = RedisClient()