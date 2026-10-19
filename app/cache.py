"""
In-memory TTL cache with LRU eviction when full. Pure Python (stdlib
`collections.OrderedDict` only). Used to cache expensive model inference
results keyed by a hash of the request payload.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, max_size: int = 1000, default_ttl_seconds: float = 300):
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: str, now: float | None = None) -> Any | None:
        now = now if now is not None else time.time()
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at < now:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)  # mark as recently used
        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self.max_size:
            self._store.popitem(last=False)  # evict least recently used

        self._store[key] = _Entry(value=value, expires_at=now + ttl)

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
