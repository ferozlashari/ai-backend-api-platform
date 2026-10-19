"""
Trains small, real scikit-learn models on synthetic data and exposes them as
predict_fn callables compatible with ModelRegistry. Two versions of a churn
classifier are provided (v1: Logistic Regression, v2: Random Forest) to
demonstrate genuine model versioning, not just a naming convention.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["tenure_months", "monthly_charges", "support_calls"]


def _make_training_data(seed: int = 42, n: int = 500):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n).astype(float)
    monthly = rng.normal(65, 20, n).clip(15, 150)
    support_calls = rng.poisson(1.5, n).astype(float)

    logits = -2.0 + 0.04 * (72 - tenure) + 0.3 * support_calls
    prob = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, prob)

    X = np.column_stack([tenure, monthly, support_calls])
    return X, y


def train_churn_models() -> tuple[LogisticRegression, RandomForestClassifier]:
    X, y = _make_training_data()
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42).fit(X, y)
    return lr, rf


def make_predict_fn(model) -> callable:
    def predict_fn(features: list[float]) -> dict:
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features {FEATURE_NAMES}, got {len(features)}")
        X = np.array(features).reshape(1, -1)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        return {"prediction": "churn" if pred == 1 else "no_churn", "churn_probability": round(proba, 4)}

    return predict_fn
