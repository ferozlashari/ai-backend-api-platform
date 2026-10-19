"""
Token-bucket rate limiter, one bucket per API key. Pure Python (stdlib
`time` only), independently testable without FastAPI/Redis installed.
A production deployment would back this with Redis for multi-instance
consistency; the algorithm itself is unchanged either way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.1f}s")


class TokenBucketRateLimiter:
    def __init__(self):
        self._buckets: dict[str, _Bucket] = {}

    def _get_bucket(self, key: str, capacity: float, now: float) -> _Bucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=capacity, last_refill=now)
            self._buckets[key] = bucket
        return bucket

    def allow(self, key: str, requests_per_minute: int, burst: int, now: float | None = None) -> bool:
        """Returns True if the request is allowed (and consumes a token), False otherwise."""
        now = now if now is not None else time.time()
        refill_rate = requests_per_minute / 60.0  # tokens per second

        bucket = self._get_bucket(key, capacity=burst, now=now)
        elapsed = now - bucket.last_refill
        bucket.tokens = min(burst, bucket.tokens + elapsed * refill_rate)
        bucket.last_refill = now

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return True
        return False

    def time_until_next_token(self, key: str, requests_per_minute: int, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        bucket = self._buckets.get(key)
        if bucket is None or bucket.tokens >= 1:
            return 0.0
        refill_rate = requests_per_minute / 60.0
        tokens_needed = 1 - bucket.tokens
        return tokens_needed / refill_rate

    def check_or_raise(self, key: str, requests_per_minute: int, burst: int, now: float | None = None) -> None:
        if not self.allow(key, requests_per_minute, burst, now):
            retry_after = self.time_until_next_token(key, requests_per_minute, now)
            raise RateLimitExceeded(retry_after)
