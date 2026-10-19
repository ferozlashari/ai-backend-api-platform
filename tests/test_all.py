"""
Test suite for app/auth.py, rate_limiter.py, cache.py, model_registry.py, ml_models.py.
Run with: python3 tests/test_all.py
All pure Python / real scikit-learn - no FastAPI/Pydantic required.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.auth import APIKeyManager, AuthError, TIER_LIMITS
from app.rate_limiter import TokenBucketRateLimiter, RateLimitExceeded
from app.cache import TTLCache
from app.model_registry import ModelRegistry, ModelNotFoundError
from app.ml_models import train_churn_models, make_predict_fn

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


print("== auth ==")
mgr = APIKeyManager()
key = mgr.create_key("alice", tier="pro")
check("issued key has expected prefix", key.startswith("sk-"))
record = mgr.validate_key(key)
check("validates correct owner/tier", record.owner == "alice" and record.tier == "pro")

try:
    mgr.validate_key("sk-not-a-real-key")
    check("rejects unknown key", False)
except AuthError:
    check("rejects unknown key", True)

mgr.revoke_key(key)
try:
    mgr.validate_key(key)
    check("rejects revoked key", False)
except AuthError:
    check("rejects revoked key", True)

try:
    mgr.create_key("bob", tier="platinum")
    check("rejects invalid tier", False)
except ValueError:
    check("rejects invalid tier", True)

check("tier limits defined for all standard tiers", set(TIER_LIMITS) == {"free", "pro", "enterprise"})

print("== rate limiter ==")
limiter = TokenBucketRateLimiter()
t0 = 1000.0
results = [limiter.allow("u1", requests_per_minute=60, burst=5, now=t0) for _ in range(7)]
check("allows exactly `burst` requests instantly", results.count(True) == 5)
check("denies requests beyond burst", results.count(False) == 2)

allowed_later = limiter.allow("u1", requests_per_minute=60, burst=5, now=t0 + 5.0)
check("refills over time (1 req/s at 60rpm)", allowed_later is True)

limiter2 = TokenBucketRateLimiter()
for _ in range(3):
    limiter2.allow("u2", requests_per_minute=60, burst=3, now=t0)
try:
    limiter2.check_or_raise("u2", requests_per_minute=60, burst=3, now=t0)
    check("check_or_raise raises when exhausted", False)
except RateLimitExceeded as e:
    check("check_or_raise raises when exhausted", True)
    check("RateLimitExceeded includes retry_after", e.retry_after > 0)

check("separate keys have independent buckets", limiter.allow("u3", requests_per_minute=60, burst=5, now=t0))

print("== cache ==")
cache = TTLCache(max_size=3, default_ttl_seconds=10)
cache.set("a", "val_a", now=0)
check("stores and retrieves a value", cache.get("a", now=1) == "val_a")
check("expired entries return None", cache.get("a", now=20) is None)

cache2 = TTLCache(max_size=2, default_ttl_seconds=100)
cache2.set("a", 1, now=0)
cache2.set("b", 2, now=0)
cache2.get("a", now=0)  # touch a -> b becomes least recently used
cache2.set("c", 3, now=0)  # should evict b
check("LRU eviction removes least-recently-used entry", cache2.get("b", now=0) is None)
check("LRU eviction preserves recently-used entry", cache2.get("a", now=0) == 1)
check("cache respects max_size", len(cache2) == 2)

cache3 = TTLCache()
cache3.get("miss", now=0)
cache3.set("hit", 1, now=0)
cache3.get("hit", now=0)
check("hit_rate tracks hits/misses correctly", abs(cache3.hit_rate - 0.5) < 1e-9)

print("== model registry (real scikit-learn models) ==")
lr, rf = train_churn_models()
registry = ModelRegistry()
registry.register("churn", "v1", make_predict_fn(lr), set_latest=False)
registry.register("churn", "v2", make_predict_fn(rf), set_latest=True)

result = registry.predict("churn", [2.0, 90.0, 4.0])  # short tenure, high support calls
check("latest version resolves correctly", result["model_version"] == "v2")
check("prediction includes probability", 0 <= result["churn_probability"] <= 1)
check("high-risk profile predicts churn", result["prediction"] == "churn")

result_v1 = registry.predict("churn", [2.0, 90.0, 4.0], version="v1")
check("explicit version overrides latest", result_v1["model_version"] == "v1")

low_risk = registry.predict("churn", [70.0, 40.0, 0.0])  # long tenure, low charges, no support calls
check("low-risk profile predicts no_churn", low_risk["prediction"] == "no_churn")

try:
    registry.predict("nonexistent_model", [1, 2, 3])
    check("raises on unknown model", False)
except ModelNotFoundError:
    check("raises on unknown model", True)

models = registry.list_models()
check("list_models returns both versions", len(models) == 2)
check("is_latest flag correct", next(m for m in models if m["version"] == "v2")["is_latest"] is True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
