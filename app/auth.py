"""
API key authentication and tiering. Pure Python (stdlib only), so it's
testable without FastAPI installed. main.py wraps this in a FastAPI
dependency (Depends(get_current_key)) for actual request auth.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field


@dataclass
class APIKey:
    key: str
    owner: str
    tier: str  # "free" | "pro" | "enterprise"
    created_at: float
    revoked: bool = False


TIER_LIMITS = {
    "free": {"requests_per_minute": 10, "burst": 5},
    "pro": {"requests_per_minute": 120, "burst": 30},
    "enterprise": {"requests_per_minute": 1000, "burst": 200},
}


class AuthError(Exception):
    pass


class APIKeyManager:
    """In-memory API key store. Swap for a database-backed implementation in production."""

    def __init__(self):
        self._keys: dict[str, APIKey] = {}  # hashed key -> APIKey

    @staticmethod
    def _hash(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def create_key(self, owner: str, tier: str = "free") -> str:
        if tier not in TIER_LIMITS:
            raise ValueError(f"Unknown tier: {tier}. Must be one of {list(TIER_LIMITS)}")
        raw_key = f"sk-{secrets.token_urlsafe(24)}"
        hashed = self._hash(raw_key)
        self._keys[hashed] = APIKey(key=hashed, owner=owner, tier=tier, created_at=time.time())
        return raw_key  # only returned once, like a real API key issuance flow

    def validate_key(self, raw_key: str) -> APIKey:
        hashed = self._hash(raw_key)
        record = self._keys.get(hashed)
        if record is None:
            raise AuthError("Invalid API key")
        if record.revoked:
            raise AuthError("API key has been revoked")
        return record

    def revoke_key(self, raw_key: str) -> None:
        hashed = self._hash(raw_key)
        record = self._keys.get(hashed)
        if record is None:
            raise AuthError("Invalid API key")
        record.revoked = True

    def get_limits(self, tier: str) -> dict:
        return TIER_LIMITS[tier]
