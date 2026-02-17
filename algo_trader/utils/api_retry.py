"""
Centralized retry logic and exponential backoff for Alpaca API interactions.
Provides a decorator and a wrapper to handle transient network errors and rate limits.
"""
import time
import random
import functools
from typing import Callable, Any, TypeVar, Tuple
from monitoring.logger import logger

F = TypeVar('F', bound=Callable[..., Any])

def with_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        exceptions: Tuple of exceptions that trigger a retry.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise
                    
                    # Log based on error type
                    error_str = str(e).lower()
                    if "429" in error_str or "too many requests" in error_str:
                        logger.warning(f"Rate limit (429) hit for {func.__name__}. Retry {retries}/{max_retries} after {delay:.1f}s")
                    else:
                        logger.warning(f"Error in {func.__name__}: {e}. Retry {retries}/{max_retries} after {delay:.1f}s")
                    
                    time.sleep(delay + random.uniform(0, 0.1)) # Add jitter
                    delay = min(delay * 2, max_delay)
        return wrapper # type: ignore
    return decorator

def api_call_with_retry(func: Callable, *args, **kwargs) -> Any:
    """Wrapper version of with_retry for direct use."""
    decorated = with_retry()(func)
    return decorated(*args, **kwargs)
