from pydantic import BaseModel, Field


class CreateKeyRequest(BaseModel):
    owner: str = Field(..., min_length=1)
    tier: str = Field(default="free")


class CreateKeyResponse(BaseModel):
    api_key: str
    owner: str
    tier: str
    warning: str = "Store this key securely - it will not be shown again."


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="[tenure_months, monthly_charges, support_calls]")
    version: str | None = Field(default=None, description="Specific model version, or latest if omitted")


class PredictResponse(BaseModel):
    prediction: str
    churn_probability: float
    model_name: str
    model_version: str
    cached: bool = False


class ModelInfo(BaseModel):
    name: str
    version: str
    description: str
    is_latest: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    models_registered: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
