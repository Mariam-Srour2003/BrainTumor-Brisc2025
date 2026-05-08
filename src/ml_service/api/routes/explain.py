"""Explainability API route registration."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any, cast

try:
    from fastapi import Request
except ImportError:
    Request = None  # type: ignore[assignment,misc]

from ...core.config import ServiceConfig, get_settings
from ...inference.predictor import Predictor
from ...explainability import ExplainabilityMethod, ExplainabilityTask
from ...experiments.runner import run_explainability_batch


def explain_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "ready",
        "message": "Use POST /explain/upload with an image file and choose gradcam, shap, or lime.",
        "received": payload,
    }


def _store_upload(upload: Any, config: ServiceConfig) -> Path:
    upload_root = config.outputs_dir / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    target_path = upload_root / upload.filename
    with target_path.open("wb") as target_file:
        shutil.copyfileobj(upload.file, target_file)
    return target_path


def _parse_optional_int(value: Any) -> int | None:
    if value in {None, "", b""}:
        return None

    return int(value)


def register_explain_routes(app: Any) -> None:
    """Register explainability routes on a FastAPI-compatible app."""

    async def explain_upload_endpoint(request: Request) -> dict[str, Any]:
        form = await request.form()
        config = get_settings()
        requested_name = form.get("model_name")
        requested_date = form.get("model_date")
        model_name = None if requested_name in {None, ""} else str(requested_name)
        model_date = None if requested_date in {None, ""} else str(requested_date)

        predictor = getattr(request.app.state, "predictor", None)
        if (
            predictor is None
            or getattr(predictor, "requested_model_name", None) != model_name
            or getattr(predictor, "requested_model_date", None) != model_date
        ):
            predictor = Predictor(config=config, model_name=model_name, model_date=model_date)
            request.app.state.predictor = predictor
        if predictor.model is None:
            return {
                "status": "error",
                "message": "No trained checkpoint found for the selected model/date. Train a model first or omit model_date to use the latest checkpoint.",
            }

        upload = form.get("file")
        if upload is None:
            return {"status": "error", "message": "Missing uploaded file."}

        method = str(form.get("method", "gradcam"))
        target_task = str(form.get("target_task", "auto"))
        target_class = _parse_optional_int(form.get("target_class"))
        target_layer = form.get("target_layer")

        image_path = _store_upload(upload, config)
        artifact = predictor.explain(
            image_path,
            method=cast(ExplainabilityMethod, method),
            target_task=cast(ExplainabilityTask, target_task),
            target_class=target_class,
            target_layer=None if target_layer in {None, ""} else str(target_layer),
        )
        if dataclasses.is_dataclass(artifact) and not isinstance(artifact, type):
            explanation_data: Any = dataclasses.asdict(artifact)
        elif isinstance(artifact, dict):
            explanation_data = {
                k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) and not isinstance(v, type) else v)
                for k, v in artifact.items()
            }
        else:
            explanation_data = artifact
        return {"image_path": str(image_path), "explanation": explanation_data}

    async def explain_batch_endpoint(request: Request) -> dict[str, Any]:
        form = await request.form()
        config = get_settings()
        checkpoint_name = form.get("checkpoint_name")
        checkpoint_path = form.get("checkpoint_path")
        method = str(form.get("method", "gradcam"))
        target_task = str(form.get("target_task", "auto"))
        split = str(form.get("split", "test"))
        limit = int(form.get("limit", 4))

        if checkpoint_path in {None, ""} and checkpoint_name in {None, ""}:
            return {"status": "error", "message": "checkpoint_name or checkpoint_path is required."}

        if checkpoint_path in {None, ""}:
            checkpoint_path = config.models_dir / "checkpoints" / f"{checkpoint_name}.pt"

        try:
            return run_explainability_batch(
                checkpoint_path,
                config,
                method=cast(ExplainabilityMethod, method),
                target_task=cast(ExplainabilityTask, target_task),
                split=split,
                limit=limit,
            )
        except Exception as exc:
            return {"status": "error", "message": str(exc), "checkpoint_path": str(checkpoint_path)}

    app.add_api_route("/explain", explain_endpoint, methods=["POST"], tags=["explain"])
    app.add_api_route("/explain/upload", explain_upload_endpoint, methods=["POST"], tags=["explain"])
    app.add_api_route("/explain/batch", explain_batch_endpoint, methods=["POST"], tags=["explain"])