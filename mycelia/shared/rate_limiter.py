"""
Rate limiting utilities for API endpoints.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict

from fastapi import HTTPException, Request

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests_per_window: int = 10  # Max requests per window
    window_seconds: int = 60  # Time window in seconds
    burst_size: int = 5  # Max burst requests
    block_duration: int = 300  # Block duration in seconds if limit exceeded


class RateLimiter:
    """
    Token bucket rate limiter with per-client tracking.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.blocked: Dict[str, float] = {}  # client_id → unblock_time
        self.lock = Lock()
    
    def is_allowed(self, client_id: str) -> tuple[bool, str | None]:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Unique identifier for client (e.g., hotkey or IP)
            
        Returns:
            (allowed, reason) - True if allowed, False with reason if not
        """
        with self.lock:
            current_time = time.time()
            
            # Check if client is blocked
            if client_id in self.blocked:
                unblock_time = self.blocked[client_id]
                if current_time < unblock_time:
                    remaining = int(unblock_time - current_time)
                    logger.warning(
                        "Blocked client attempted request",
                        client=client_id[:16] + "...",
                        remaining_seconds=remaining,
                    )
                    return False, f"Rate limit exceeded. Blocked for {remaining}s"
                else:
                    # Unblock
                    del self.blocked[client_id]
                    logger.info(f"Client unblocked: {client_id[:16]}...")
            
            # Get request history for client
            request_times = self.requests[client_id]
            
            # Remove old requests outside window
            window_start = current_time - self.config.window_seconds
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # Check burst limit
            recent_burst = sum(1 for t in request_times if t > current_time - 10)  # Last 10 seconds
            if recent_burst >= self.config.burst_size:
                logger.warning(
                    "Burst limit exceeded",
                    client=client_id[:16] + "...",
                    burst_count=recent_burst,
                )
                # Block client
                self.blocked[client_id] = current_time + self.config.block_duration
                return False, f"Burst limit exceeded ({recent_burst}/{self.config.burst_size})"
            
            # Check window limit
            if len(request_times) >= self.config.requests_per_window:
                logger.warning(
                    "Rate limit exceeded",
                    client=client_id[:16] + "...",
                    count=len(request_times),
                    window=self.config.window_seconds,
                )
                # Block client
                self.blocked[client_id] = current_time + self.config.block_duration
                return False, f"Rate limit exceeded ({len(request_times)}/{self.config.requests_per_window} per {self.config.window_seconds}s)"
            
            # Allow request and record it
            request_times.append(current_time)
            return True, None
    
    def reset(self, client_id: str):
        """Reset rate limit for a client."""
        with self.lock:
            if client_id in self.requests:
                del self.requests[client_id]
            if client_id in self.blocked:
                del self.blocked[client_id]
            logger.info(f"Rate limit reset for client: {client_id[:16]}...")
    
    def get_stats(self, client_id: str) -> Dict:
        """Get current rate limit stats for a client."""
        with self.lock:
            current_time = time.time()
            window_start = current_time - self.config.window_seconds
            
            request_times = self.requests.get(client_id, deque())
            recent_requests = [t for t in request_times if t > window_start]
            
            is_blocked = client_id in self.blocked and current_time < self.blocked[client_id]
            remaining_block_time = (
                int(self.blocked[client_id] - current_time)
                if is_blocked
                else 0
            )
            
            return {
                "requests_in_window": len(recent_requests),
                "max_requests": self.config.requests_per_window,
                "window_seconds": self.config.window_seconds,
                "is_blocked": is_blocked,
                "remaining_block_time": remaining_block_time,
                "requests_remaining": self.config.requests_per_window - len(recent_requests),
            }


# Global rate limiters for different endpoints
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(endpoint: str, config: RateLimitConfig | None = None) -> RateLimiter:
    """
    Get or create rate limiter for an endpoint.
    
    Args:
        endpoint: Endpoint name (e.g., "submit", "download", "ping")
        config: Rate limit configuration (uses default if None)
        
    Returns:
        RateLimiter instance
    """
    if endpoint not in _rate_limiters:
        if config is None:
            # Default configurations per endpoint type
            if endpoint == "submit":
                config = RateLimitConfig(
                    requests_per_window=10,  # 10 submissions per minute
                    window_seconds=60,
                    burst_size=3,
                    block_duration=600,  # 10 minute block
                )
            elif endpoint == "download":
                config = RateLimitConfig(
                    requests_per_window=30,  # 30 downloads per minute
                    window_seconds=60,
                    burst_size=10,
                    block_duration=300,  # 5 minute block
                )
            elif endpoint == "ping":
                config = RateLimitConfig(
                    requests_per_window=60,  # 60 pings per minute
                    window_seconds=60,
                    burst_size=20,
                    block_duration=60,  # 1 minute block
                )
            else:
                # Generic default
                config = RateLimitConfig()
        
        _rate_limiters[endpoint] = RateLimiter(config)
    
    return _rate_limiters[endpoint]


async def rate_limit_dependency(request: Request, endpoint: str, client_id_key: str = "origin_hotkey_ss58"):
    """
    FastAPI dependency for rate limiting.
    
    Usage:
        @app.post("/submit")
        async def submit(rate_limit: None = Depends(lambda req: rate_limit_dependency(req, "submit"))):
            ...
    
    Args:
        request: FastAPI request object
        endpoint: Endpoint name for rate limiter selection
        client_id_key: Key to extract client ID from request (default: origin_hotkey_ss58)
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Try to get client ID from form data or query params
    client_id = None
    
    # Check form data (for POST requests)
    if hasattr(request, "form"):
        try:
            form = await request.form()
            client_id = form.get(client_id_key)
        except:
            pass
    
    # Check query params (for GET requests)
    if not client_id:
        client_id = request.query_params.get(client_id_key)
    
    # Fallback to IP address if no client_id
    if not client_id:
        client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    rate_limiter = get_rate_limiter(endpoint)
    allowed, reason = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)


def rate_limit_middleware(endpoint: str):
    """
    Decorator for rate limiting endpoint functions.
    
    Usage:
        @rate_limit_middleware("submit")
        async def submit_checkpoint(...):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract client_id from kwargs (assumes origin_hotkey_ss58 is passed)
            client_id = kwargs.get("origin_hotkey_ss58") or kwargs.get("miner_hotkey") or "unknown"
            
            # Check rate limit
            rate_limiter = get_rate_limiter(endpoint)
            allowed, reason = rate_limiter.is_allowed(client_id)
            
            if not allowed:
                raise HTTPException(status_code=429, detail=reason)
            
            # Proceed with function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
