"""
Model registry: registers named, versioned prediction functions and routes
inference requests to them. Real scikit-learn models are used here (not
mocked) so registration + routing + prediction are genuinely exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

PredictFn = Callable[[list[float]], dict]


@dataclass
class ModelEntry:
    name: str
    version: str
    predict_fn: PredictFn
    description: str = ""


class ModelNotFoundError(Exception):
    pass


class ModelRegistry:
    def __init__(self):
        self._models: dict[tuple[str, str], ModelEntry] = {}
        self._latest_version: dict[str, str] = {}

    def register(self, name: str, version: str, predict_fn: PredictFn, description: str = "", set_latest: bool = True) -> None:
        self._models[(name, version)] = ModelEntry(name=name, version=version, predict_fn=predict_fn, description=description)
        if set_latest or name not in self._latest_version:
            self._latest_version[name] = version

    def predict(self, name: str, features: list[float], version: str | None = None) -> dict:
        resolved_version = version or self._latest_version.get(name)
        if resolved_version is None or (name, resolved_version) not in self._models:
            raise ModelNotFoundError(f"No model registered as '{name}' version '{resolved_version}'")
        entry = self._models[(name, resolved_version)]
        result = entry.predict_fn(features)
        result["model_name"] = name
        result["model_version"] = resolved_version
        return result

    def list_models(self) -> list[dict]:
        return [
            {"name": e.name, "version": e.version, "description": e.description, "is_latest": self._latest_version.get(e.name) == e.version}
            for e in self._models.values()
        ]
