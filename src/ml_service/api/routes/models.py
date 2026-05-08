"""Model registry endpoints."""

from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError:
    APIRouter = None

from ...models import DEFAULT_MODEL_REGISTRY

router = APIRouter() if APIRouter is not None else None


if router is not None:

    @router.get("/list")
    def list_models() -> dict[str, list[str]]:
        models = sorted(list(DEFAULT_MODEL_REGISTRY.list_models()))
        return {"models": models}

    @router.get("/info/{model_name}")
    def model_info(model_name: str) -> dict[str, object]:
        if model_name not in DEFAULT_MODEL_REGISTRY.list_models():
            return {"error": "unknown model", "model_name": model_name}
        factory = DEFAULT_MODEL_REGISTRY.factories.get(model_name)
        return {"model_name": model_name, "callable": repr(factory)}
