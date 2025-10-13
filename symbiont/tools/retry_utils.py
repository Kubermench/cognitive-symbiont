"""Enhanced retry utilities with exponential backoff and fault tolerance.

This module provides a comprehensive retry framework for all outbound calls in Symbiont,
including LLM calls, webhook sends, and daemon poll loops. It supports:
- Exponential backoff with jitter
- Circuit breaker patterns
- Configurable retry policies
- Comprehensive logging and metrics
- Graceful degradation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from functools import wraps
import threading
from collections import defaultdict, deque

from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Circuit breaker states
CIRCUIT_CLOSED = "closed"
CIRCUIT_OPEN = "open"
CIRCUIT_HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    exponential: bool = True
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    
    # Exception handling
    retry_exceptions: tuple[Type[Exception], ...] = (Exception,)
    no_retry_exceptions: tuple[Type[Exception], ...] = ()
    
    # Logging and metrics
    log_attempts: bool = True
    log_success: bool = False
    collect_metrics: bool = True


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker pattern."""
    
    state: str = CIRCUIT_CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    
    # Metrics
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    
    # Recent history for monitoring
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=100))


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: RetryConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState()
        self._lock = threading.RLock()
    
    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self._check_state()
            
            if self.state.state == CIRCUIT_OPEN:
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
            
            self.state.total_calls += 1
            call_start = time.monotonic()
            
            try:
                result = func()
                self._record_success(call_start)
                return result
            except Exception as exc:
                self._record_failure(call_start, exc)
                raise
    
    def _check_state(self) -> None:
        """Check and update circuit breaker state."""
        now = time.monotonic()
        
        if self.state.state == CIRCUIT_OPEN:
            if now - self.state.last_failure_time >= self.config.recovery_timeout:
                self.state.state = CIRCUIT_HALF_OPEN
                self.state.success_count = 0
                logger.info("Circuit breaker '%s' moved to half-open state", self.name)
        
        elif self.state.state == CIRCUIT_HALF_OPEN:
            if self.state.success_count >= self.config.success_threshold:
                self.state.state = CIRCUIT_CLOSED
                self.state.failure_count = 0
                logger.info("Circuit breaker '%s' closed after recovery", self.name)
    
    def _record_success(self, call_start: float) -> None:
        """Record successful call."""
        duration = time.monotonic() - call_start
        self.state.total_successes += 1
        self.state.last_success_time = time.monotonic()
        self.state.recent_calls.append({
            "success": True,
            "duration": duration,
            "timestamp": time.time(),
        })
        
        if self.state.state == CIRCUIT_HALF_OPEN:
            self.state.success_count += 1
        elif self.state.state == CIRCUIT_CLOSED:
            # Reset failure count on success
            self.state.failure_count = max(0, self.state.failure_count - 1)
    
    def _record_failure(self, call_start: float, exc: Exception) -> None:
        """Record failed call."""
        duration = time.monotonic() - call_start
        self.state.total_failures += 1
        self.state.failure_count += 1
        self.state.last_failure_time = time.monotonic()
        self.state.recent_calls.append({
            "success": False,
            "duration": duration,
            "timestamp": time.time(),
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        
        if self.state.failure_count >= self.config.failure_threshold:
            if self.state.state in (CIRCUIT_CLOSED, CIRCUIT_HALF_OPEN):
                self.state.state = CIRCUIT_OPEN
                self.state.success_count = 0
                logger.warning(
                    "Circuit breaker '%s' opened after %d failures",
                    self.name,
                    self.state.failure_count,
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            success_rate = 0.0
            if self.state.total_calls > 0:
                success_rate = self.state.total_successes / self.state.total_calls
            
            recent_success_rate = 0.0
            if self.state.recent_calls:
                recent_successes = sum(1 for call in self.state.recent_calls if call["success"])
                recent_success_rate = recent_successes / len(self.state.recent_calls)
            
            return {
                "name": self.name,
                "state": self.state.state,
                "total_calls": self.state.total_calls,
                "total_successes": self.state.total_successes,
                "total_failures": self.state.total_failures,
                "success_rate": success_rate,
                "recent_success_rate": recent_success_rate,
                "failure_count": self.state.failure_count,
                "last_failure_time": self.state.last_failure_time,
                "last_success_time": self.state.last_success_time,
            }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.RLock()


def get_circuit_breaker(name: str, config: Optional[RetryConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    with _circuit_breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(config or RetryConfig(), name)
        return _circuit_breakers[name]


def list_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """List all circuit breakers and their metrics."""
    with _circuit_breakers_lock:
        return {name: cb.get_metrics() for name, cb in _circuit_breakers.items()}


class RetryableError(Exception):
    """Base class for errors that should trigger retries."""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should not trigger retries."""
    pass


def create_retryer(config: RetryConfig) -> Retrying:
    """Create a tenacity Retrying instance from config."""
    
    # Determine retry condition
    if config.no_retry_exceptions:
        retry_condition = retry_if_not_exception_type(config.no_retry_exceptions)
    else:
        retry_condition = retry_if_exception_type(config.retry_exceptions)
    
    # Determine wait strategy
    if config.exponential:
        if config.jitter:
            wait_strategy = wait_exponential_jitter(
                initial=config.initial_wait,
                max=config.max_wait,
                exp_base=config.multiplier,
            )
        else:
            wait_strategy = wait_exponential(
                multiplier=config.initial_wait,
                exp_base=config.multiplier,
                min=config.initial_wait,
                max=config.max_wait,
            )
    else:
        wait_strategy = wait_fixed(config.initial_wait)
    
    def log_retry_attempt(retry_state: RetryCallState) -> None:
        if not config.log_attempts:
            return
        
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc:
            logger.warning(
                "Retry attempt %d/%d failed: %s",
                retry_state.attempt_number,
                config.attempts,
                exc,
            )
    
    return Retrying(
        retry=retry_condition,
        stop=stop_after_attempt(config.attempts),
        wait=wait_strategy,
        reraise=True,
        before_sleep=log_retry_attempt,
        sleep=time.sleep,
    )


def with_retry(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic and optional circuit breaker to functions."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retry_config = config or RetryConfig()
        retryer = create_retryer(retry_config)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            def attempt():
                return func(*args, **kwargs)
            
            # Apply circuit breaker if specified
            if circuit_breaker:
                cb = get_circuit_breaker(circuit_breaker, retry_config)
                attempt = lambda: cb.call(lambda: func(*args, **kwargs))
            
            start_time = time.monotonic()
            try:
                result = retryer(attempt)
                if retry_config.log_success:
                    duration = time.monotonic() - start_time
                    logger.info(
                        "Function %s succeeded after %.2fs",
                        func.__name__,
                        duration,
                    )
                return result
            except RetryError as exc:
                duration = time.monotonic() - start_time
                last_attempt = exc.last_attempt
                if last_attempt and last_attempt.outcome:
                    original_exc = last_attempt.outcome.exception()
                    if original_exc:
                        logger.error(
                            "Function %s failed after %d attempts in %.2fs: %s",
                            func.__name__,
                            retry_config.attempts,
                            duration,
                            original_exc,
                        )
                        raise original_exc
                raise
        
        return wrapper
    return decorator


def retry_call(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[str] = None,
) -> T:
    """Execute a function with retry logic and optional circuit breaker."""
    retry_config = config or RetryConfig()
    retryer = create_retryer(retry_config)
    
    def attempt():
        return func()
    
    # Apply circuit breaker if specified
    if circuit_breaker:
        cb = get_circuit_breaker(circuit_breaker, retry_config)
        attempt = lambda: cb.call(func)
    
    start_time = time.monotonic()
    try:
        result = retryer(attempt)
        if retry_config.log_success:
            duration = time.monotonic() - start_time
            logger.info("Retry call succeeded after %.2fs", duration)
        return result
    except RetryError as exc:
        duration = time.monotonic() - start_time
        last_attempt = exc.last_attempt
        if last_attempt and last_attempt.outcome:
            original_exc = last_attempt.outcome.exception()
            if original_exc:
                logger.error(
                    "Retry call failed after %d attempts in %.2fs: %s",
                    retry_config.attempts,
                    duration,
                    original_exc,
                )
                raise original_exc
        raise


# Predefined retry configurations for common use cases
LLM_RETRY_CONFIG = RetryConfig(
    attempts=3,
    initial_wait=0.5,
    max_wait=20.0,
    multiplier=2.0,
    jitter=True,
    failure_threshold=5,
    recovery_timeout=60.0,
    retry_exceptions=(Exception,),
    no_retry_exceptions=(KeyboardInterrupt, SystemExit),
)

WEBHOOK_RETRY_CONFIG = RetryConfig(
    attempts=3,
    initial_wait=1.0,
    max_wait=30.0,
    multiplier=2.0,
    jitter=True,
    failure_threshold=3,
    recovery_timeout=120.0,
    retry_exceptions=(Exception,),
    no_retry_exceptions=(KeyboardInterrupt, SystemExit),
)

DAEMON_RETRY_CONFIG = RetryConfig(
    attempts=3,
    initial_wait=1.0,
    max_wait=60.0,
    multiplier=2.0,
    jitter=True,
    failure_threshold=10,
    recovery_timeout=300.0,
    retry_exceptions=(Exception,),
    no_retry_exceptions=(KeyboardInterrupt, SystemExit),
)

EXTERNAL_API_RETRY_CONFIG = RetryConfig(
    attempts=3,
    initial_wait=2.0,
    max_wait=60.0,
    multiplier=2.0,
    jitter=True,
    failure_threshold=5,
    recovery_timeout=180.0,
    retry_exceptions=(Exception,),
    no_retry_exceptions=(KeyboardInterrupt, SystemExit),
)