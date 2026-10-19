# AI Backend API Platform

A production-style backend for serving ML models over a REST API: **API key authentication**, **per-tier token-bucket rate limiting**, **TTL response caching**, and **versioned model serving** — built on **FastAPI**.

## Why this is a different project from #1/#2/#3

Those projects used FastAPI as a thin wrapper around an LLM/agent/RAG pipeline. This project *is* the backend infrastructure: the kind of auth, rate-limiting, and caching layer that sits in front of **any** ML model in production, independent of what the model does.

## Architecture

Every piece of business logic is a dependency-free Python module, independently tested without FastAPI/Pydantic installed. `main.py` is a thin FastAPI layer wiring them into HTTP endpoints:

```
app/
├── auth.py             API key issuance, hashing (SHA-256, keys never stored in plaintext), revocation, tiers
├── rate_limiter.py       Token-bucket algorithm, per-key buckets, burst + steady-state refill
├── cache.py              TTL cache with LRU eviction, hit-rate tracking
├── model_registry.py      Named + versioned model registration and routing
├── ml_models.py           Real scikit-learn models (LogisticRegression v1, RandomForest v2) registered into it
├── main.py                FastAPI: wires the above into REST endpoints
└── schemas.py             Pydantic request/response models
```

## Features

- **Auth**: API keys are hashed (SHA-256) before storage — never kept in plaintext, same principle as password hashing. Revocation is immediate.
- **Rate limiting**: token-bucket per API key, limits scaled by tier (free: 10 req/min, pro: 120, enterprise: 1000), with burst allowance and a `Retry-After` header on 429s.
- **Caching**: identical prediction requests (same features + model version) are served from cache — verified with a real hit/miss/eviction test suite.
- **Model versioning**: `/v1/predict/churn` serves the latest registered version by default, or a specific version on request (`v1` = Logistic Regression baseline, `v2` = Random Forest) — both real, trained scikit-learn models.

## Setup & Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docs at **http://127.0.0.1:8000/docs**.

## Example usage

```bash
# 1. Issue an API key
curl -X POST http://127.0.0.1:8000/auth/keys \
  -H "Content-Type: application/json" \
  -d '{"owner": "alice", "tier": "pro"}'
# -> {"api_key": "sk-...", "owner": "alice", "tier": "pro", ...}

# 2. Use it to get a prediction
curl -X POST http://127.0.0.1:8000/v1/predict/churn \
  -H "X-API-Key: sk-..." \
  -H "Content-Type: application/json" \
  -d '{"features": [2.0, 90.0, 4.0]}'
# -> {"prediction": "churn", "churn_probability": 0.9569, "model_name": "churn", "model_version": "v2", "cached": false}

# 3. Repeat the same request - now served from cache
curl -X POST http://127.0.0.1:8000/v1/predict/churn \
  -H "X-API-Key: sk-..." -H "Content-Type: application/json" \
  -d '{"features": [2.0, 90.0, 4.0]}'
# -> {..., "cached": true}
```

Exceeding your tier's rate limit returns `429` with a `Retry-After` header instead of a generic error.

## Testing

```bash
python3 tests/test_all.py
```

26 tests, all passing: auth (key issuance, hashing, revocation, invalid-tier rejection), rate limiting (burst enforcement, steady-state refill timing, per-key isolation, `Retry-After` correctness), caching (TTL expiry, LRU eviction order, hit-rate tracking), and model registry (version resolution, explicit version override, and — using real trained models — a sanity check that a high-risk customer profile actually predicts churn and a low-risk one doesn't).

## Known limitations / next steps

- In-memory storage (keys, rate-limit buckets, cache) — swap for Redis/Postgres for multi-instance deployments.
- No async model inference / background job queue yet — fine for fast scikit-learn models, but an LLM-backed endpoint would want a job-submission + polling pattern.
- No request logging/observability middleware — add structured logging (e.g. `structlog`) before production use.

## Tech stack

Python · FastAPI · Pydantic · scikit-learn · Uvicorn
