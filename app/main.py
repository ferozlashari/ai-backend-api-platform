"""
Production-style AI Backend API Platform.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs

Combines API key auth, per-tier token-bucket rate limiting, TTL response
caching, and versioned ML model serving into a REST API. The business logic
for each concern lives in its own dependency-free module (auth.py,
rate_limiter.py, cache.py, model_registry.py) - see tests/test_all.py, which
verifies all of it without needing FastAPI installed.
"""

from __future__ import annotations

import hashlib
import json
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.auth import APIKeyManager, AuthError
from app.cache import TTLCache
from app.config import settings
from app.ml_models import make_predict_fn, train_churn_models
from app.model_registry import ModelNotFoundError, ModelRegistry
from app.rate_limiter import RateLimitExceeded, TokenBucketRateLimiter
from app.schemas import (
    CreateKeyRequest,
    CreateKeyResponse,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="AI Backend API Platform",
    description="Production-style backend: API key auth, rate limiting, caching, and versioned ML model serving.",
    version=__version__,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Wiring: instantiate the tested core modules -----------------------

key_manager = APIKeyManager()
rate_limiter = TokenBucketRateLimiter()
cache = TTLCache(max_size=settings.cache_max_size, default_ttl_seconds=settings.cache_ttl_seconds)

registry = ModelRegistry()
_lr, _rf = train_churn_models()
registry.register("churn", "v1", make_predict_fn(_lr), description="Logistic Regression baseline", set_latest=False)
registry.register("churn", "v2", make_predict_fn(_rf), description="Random Forest", set_latest=True)


# --- Auth dependency -----------------------------------------------------

async def get_current_key(x_api_key: str = Header(...)):
    try:
        record = key_manager.validate_key(x_api_key)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))

    limits = key_manager.get_limits(record.tier)
    try:
        rate_limiter.check_or_raise(record.key, limits["requests_per_minute"], limits["burst"])
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(int(e.retry_after) + 1)})

    return record


# --- Routes ----------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, models_registered=len(registry.list_models()))


@app.post("/auth/keys", response_model=CreateKeyResponse)
def create_key(req: CreateKeyRequest) -> CreateKeyResponse:
    try:
        raw_key = key_manager.create_key(req.owner, req.tier)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CreateKeyResponse(api_key=raw_key, owner=req.owner, tier=req.tier)


@app.get("/v1/models", response_model=list[ModelInfo])
def list_models(_=Depends(get_current_key)) -> list[ModelInfo]:
    return [ModelInfo(**m) for m in registry.list_models()]


@app.post("/v1/predict/churn", response_model=PredictResponse)
def predict_churn(req: PredictRequest, key_record=Depends(get_current_key)) -> PredictResponse:
    cache_key = hashlib.sha256(
        json.dumps({"features": req.features, "version": req.version}, sort_keys=True).encode()
    ).hexdigest()

    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return PredictResponse(**cached_result, cached=True)

    try:
        result = registry.predict("churn", req.features, version=req.version)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    cache.set(cache_key, result)
    return PredictResponse(**result, cached=False)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(exc)})
