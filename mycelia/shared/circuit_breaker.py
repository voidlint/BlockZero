"""
Circuit breaker pattern for protecting against cascading failures.

Prevents infinite loops from hanging indefinitely when external services fail.
"""

import time
from enum import Enum
from typing import Callable, TypeVar

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: After timeout, allow test request to check recovery

    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)

        result = breaker.call(lambda: risky_operation())
        if result is None:
            # Circuit is open, handle gracefully
            pass
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying again (half-open state)
            expected_exception: Exception type to catch (others will propagate)
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED

    def call(self, func: Callable[[], T]) -> T | None:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute

        Returns:
            Function result, or None if circuit is open
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info(
                    "Circuit breaker transitioning to HALF_OPEN (testing recovery)",
                    timeout=self.timeout,
                )
                self.state = CircuitState.HALF_OPEN
            else:
                # Still in open state, reject immediately
                return None

        try:
            result = func()

            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker CLOSED (service recovered)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            logger.warning(
                "Circuit breaker detected failure",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
                error=str(e),
            )

            if self.failure_count >= self.failure_threshold:
                logger.error(
                    "Circuit breaker OPEN (too many failures)",
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold,
                    timeout=self.timeout,
                )
                self.state = CircuitState.OPEN

            return None

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED
        logger.info("Circuit breaker manually reset to CLOSED")


class RetryWithBackoff:
    """
    Exponential backoff retry logic.

    Example:
        retry = RetryWithBackoff(max_attempts=5, base_delay=1.0, max_delay=60.0)

        result = retry.execute(lambda: unreliable_operation())
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry logic.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff (2.0 = double each time)
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def execute(
        self,
        func: Callable[[], T],
        on_retry: Callable[[int, Exception], None] | None = None,
    ) -> T | None:
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            on_retry: Optional callback on retry (receives attempt number and exception)

        Returns:
            Function result, or None if all attempts failed
        """
        import random

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func()
                if attempt > 1:
                    logger.info(f"Retry succeeded on attempt {attempt}/{self.max_attempts}")
                return result

            except Exception as e:
                if attempt == self.max_attempts:
                    logger.error(
                        f"All {self.max_attempts} retry attempts failed",
                        error=str(e),
                    )
                    return None

                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** (attempt - 1)),
                    self.max_delay,
                )

                # Add jitter (random 0-25% of delay)
                if self.jitter:
                    delay += random.uniform(0, delay * 0.25)

                logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed, retrying in {delay:.1f}s",
                    error=str(e),
                )

                if on_retry:
                    on_retry(attempt, e)

                time.sleep(delay)

        return None


def with_timeout(func: Callable[[], T], timeout_seconds: float) -> T | None:
    """
    Execute function with timeout protection.

    WARNING: This uses threading and may not work with all functions.
    For I/O operations, use asyncio timeouts instead.

    Args:
        func: Function to execute
        timeout_seconds: Timeout in seconds

    Returns:
        Function result, or None if timeout exceeded
    """
    import threading

    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.error(f"Function execution exceeded timeout of {timeout_seconds}s")
        return None

    if exception[0]:
        raise exception[0]

    return result[0]
