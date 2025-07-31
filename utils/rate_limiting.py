"""
Rate limiting utilities for the SEO Content Knowledge Graph System.

This module provides rate limiting decorators and utilities for FastAPI endpoints.
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
import structlog

from fastapi import HTTPException, Request, status
from collections import defaultdict, deque

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.last_cleanup = time.time()
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        now = time.time()
        
        # Clean up old buckets periodically
        if now - self.last_cleanup > 60:  # Clean up every minute
            self._cleanup_buckets(now)
            self.last_cleanup = now
        
        # Get or create bucket for identifier
        bucket = self.buckets[identifier]
        
        # Remove old requests outside the window
        cutoff_time = now - self.window_seconds
        while bucket and bucket[0] < cutoff_time:
            bucket.popleft()
        
        # Check if we're within the limit
        current_requests = len(bucket)
        allowed = current_requests < self.max_requests
        
        if allowed:
            bucket.append(now)
        
        # Calculate reset time
        reset_time = int(now + self.window_seconds) if bucket else int(now)
        
        # Rate limit info
        rate_limit_info = {
            'limit': self.max_requests,
            'remaining': max(0, self.max_requests - current_requests - (1 if allowed else 0)),
            'reset': reset_time,
            'retry_after': self.window_seconds if not allowed else 0
        }
        
        return allowed, rate_limit_info
    
    def _cleanup_buckets(self, now: float) -> None:
        """Clean up old buckets."""
        cutoff_time = now - self.window_seconds * 2  # Keep some buffer
        
        for identifier in list(self.buckets.keys()):
            bucket = self.buckets[identifier]
            
            # Remove old requests
            while bucket and bucket[0] < cutoff_time:
                bucket.popleft()
            
            # Remove empty buckets
            if not bucket:
                del self.buckets[identifier]


class DistributedRateLimiter:
    """Distributed rate limiter using Redis."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed using Redis."""
        if not self.redis_client:
            # Fallback to in-memory rate limiter
            return True, {'limit': max_requests, 'remaining': max_requests, 'reset': 0, 'retry_after': 0}
        
        try:
            now = time.time()
            pipeline = self.redis_client.pipeline()
            
            # Use sliding window counter
            key = f"rate_limit:{identifier}"
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, now - window_seconds)
            
            # Count current requests
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(now): now})
            
            # Set expiration
            pipeline.expire(key, window_seconds)
            
            results = await pipeline.execute()
            current_requests = results[1]
            
            # Check if allowed
            allowed = current_requests < max_requests
            
            if not allowed:
                # Remove the request we just added
                await self.redis_client.zrem(key, str(now))
            
            # Calculate reset time
            reset_time = int(now + window_seconds)
            
            rate_limit_info = {
                'limit': max_requests,
                'remaining': max(0, max_requests - current_requests - (1 if allowed else 0)),
                'reset': reset_time,
                'retry_after': window_seconds if not allowed else 0
            }
            
            return allowed, rate_limit_info
            
        except Exception as e:
            logger.error(f"Error in distributed rate limiter: {e}")
            # Fallback to allowing the request
            return True, {'limit': max_requests, 'remaining': max_requests, 'reset': 0, 'retry_after': 0}


# Global rate limiters
_rate_limiters: Dict[str, RateLimiter] = {}
_distributed_rate_limiter: Optional[DistributedRateLimiter] = None


def get_rate_limiter(max_requests: int, window_seconds: int) -> RateLimiter:
    """Get or create rate limiter."""
    key = f"{max_requests}:{window_seconds}"
    
    if key not in _rate_limiters:
        _rate_limiters[key] = RateLimiter(max_requests, window_seconds)
    
    return _rate_limiters[key]


def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get user ID from request
    user_id = getattr(request.state, 'user_id', None)
    if user_id:
        return f"user:{user_id}"
    
    # Fall back to IP address
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"
    
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def rate_limit(
    requests_per_minute: int = 60,
    requests_per_hour: Optional[int] = None,
    per_user: bool = True,
    skip_if_authenticated: bool = False
):
    """
    Rate limiting decorator for FastAPI endpoints.
    
    Args:
        requests_per_minute: Maximum requests per minute
        requests_per_hour: Maximum requests per hour (optional)
        per_user: Whether to apply rate limiting per user or globally
        skip_if_authenticated: Skip rate limiting for authenticated users
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the request object in arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # If no request object found, skip rate limiting
                return await func(*args, **kwargs)
            
            # Skip rate limiting if configured
            if skip_if_authenticated and hasattr(request.state, 'user_id'):
                return await func(*args, **kwargs)
            
            # Get client identifier
            if per_user:
                identifier = get_client_identifier(request)
            else:
                identifier = "global"
            
            # Check minute-based rate limit
            minute_limiter = get_rate_limiter(requests_per_minute, 60)
            allowed, rate_limit_info = minute_limiter.is_allowed(identifier)
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        'X-RateLimit-Limit': str(rate_limit_info['limit']),
                        'X-RateLimit-Remaining': str(rate_limit_info['remaining']),
                        'X-RateLimit-Reset': str(rate_limit_info['reset']),
                        'Retry-After': str(rate_limit_info['retry_after'])
                    }
                )
            
            # Check hour-based rate limit if configured
            if requests_per_hour:
                hour_limiter = get_rate_limiter(requests_per_hour, 3600)
                allowed, hour_rate_limit_info = hour_limiter.is_allowed(identifier)
                
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Hourly rate limit exceeded",
                        headers={
                            'X-RateLimit-Limit': str(hour_rate_limit_info['limit']),
                            'X-RateLimit-Remaining': str(hour_rate_limit_info['remaining']),
                            'X-RateLimit-Reset': str(hour_rate_limit_info['reset']),
                            'Retry-After': str(hour_rate_limit_info['retry_after'])
                        }
                    )
            
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            
            # Add headers if response supports it
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(rate_limit_info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(rate_limit_info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(rate_limit_info['reset'])
            
            return response
        
        return wrapper
    return decorator


def adaptive_rate_limit(
    base_requests_per_minute: int = 60,
    max_requests_per_minute: int = 300,
    error_threshold: float = 0.1,
    success_boost: float = 1.2,
    error_penalty: float = 0.8
):
    """
    Adaptive rate limiting that adjusts based on success/error rates.
    
    Args:
        base_requests_per_minute: Base rate limit
        max_requests_per_minute: Maximum allowed rate limit
        error_threshold: Error rate threshold to trigger penalty
        success_boost: Multiplier for successful requests
        error_penalty: Multiplier for high error rates
    """
    # Track success/error rates per identifier
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'requests': 0,
        'errors': 0,
        'current_limit': base_requests_per_minute,
        'last_adjustment': time.time()
    })
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                return await func(*args, **kwargs)
            
            # Get client identifier
            identifier = get_client_identifier(request)
            client_stats = stats[identifier]
            
            # Adjust rate limit based on recent performance
            now = time.time()
            if now - client_stats['last_adjustment'] > 60:  # Adjust every minute
                if client_stats['requests'] > 0:
                    error_rate = client_stats['errors'] / client_stats['requests']
                    
                    if error_rate > error_threshold:
                        # Increase rate limit for low error rates
                        client_stats['current_limit'] = max(
                            base_requests_per_minute,
                            int(client_stats['current_limit'] * error_penalty)
                        )
                    else:
                        # Decrease rate limit for high error rates
                        client_stats['current_limit'] = min(
                            max_requests_per_minute,
                            int(client_stats['current_limit'] * success_boost)
                        )
                
                # Reset stats
                client_stats['requests'] = 0
                client_stats['errors'] = 0
                client_stats['last_adjustment'] = now
            
            # Apply rate limit
            limiter = get_rate_limiter(client_stats['current_limit'], 60)
            allowed, rate_limit_info = limiter.is_allowed(identifier)
            
            if not allowed:
                client_stats['errors'] += 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        'X-RateLimit-Limit': str(rate_limit_info['limit']),
                        'X-RateLimit-Remaining': str(rate_limit_info['remaining']),
                        'X-RateLimit-Reset': str(rate_limit_info['reset']),
                        'Retry-After': str(rate_limit_info['retry_after'])
                    }
                )
            
            # Execute function
            try:
                response = await func(*args, **kwargs)
                client_stats['requests'] += 1
                return response
            except Exception as e:
                client_stats['requests'] += 1
                client_stats['errors'] += 1
                raise
        
        return wrapper
    return decorator


class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI."""
    
    def __init__(self, app, default_limit: int = 100, default_window: int = 60):
        self.app = app
        self.default_limit = default_limit
        self.default_window = default_window
        self.limiter = get_rate_limiter(default_limit, default_window)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create request object
        from fastapi import Request
        request = Request(scope, receive)
        
        # Get client identifier
        identifier = get_client_identifier(request)
        
        # Check rate limit
        allowed, rate_limit_info = self.limiter.is_allowed(identifier)
        
        if not allowed:
            # Send rate limit exceeded response
            response = {
                "status_code": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"x-ratelimit-limit", str(rate_limit_info['limit']).encode()],
                    [b"x-ratelimit-remaining", str(rate_limit_info['remaining']).encode()],
                    [b"x-ratelimit-reset", str(rate_limit_info['reset']).encode()],
                    [b"retry-after", str(rate_limit_info['retry_after']).encode()]
                ],
                "body": b'{"detail":"Rate limit exceeded"}'
            }
            
            await send({
                "type": "http.response.start",
                "status": response["status_code"],
                "headers": response["headers"]
            })
            await send({
                "type": "http.response.body",
                "body": response["body"]
            })
            return
        
        # Process request normally
        await self.app(scope, receive, send)


def init_distributed_rate_limiter(redis_client=None):
    """Initialize distributed rate limiter."""
    global _distributed_rate_limiter
    _distributed_rate_limiter = DistributedRateLimiter(redis_client)


async def distributed_rate_limit(
    identifier: str,
    max_requests: int,
    window_seconds: int
) -> Tuple[bool, Dict[str, Any]]:
    """Apply distributed rate limiting."""
    if _distributed_rate_limiter:
        return await _distributed_rate_limiter.is_allowed(identifier, max_requests, window_seconds)
    else:
        # Fallback to local rate limiter
        limiter = get_rate_limiter(max_requests, window_seconds)
        return limiter.is_allowed(identifier)


def get_rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiting statistics."""
    stats = {
        'active_limiters': len(_rate_limiters),
        'distributed_enabled': _distributed_rate_limiter is not None,
        'limiters': {}
    }
    
    for key, limiter in _rate_limiters.items():
        stats['limiters'][key] = {
            'max_requests': limiter.max_requests,
            'window_seconds': limiter.window_seconds,
            'active_buckets': len(limiter.buckets)
        }
    
    return stats