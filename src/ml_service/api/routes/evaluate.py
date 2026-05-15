"""Evaluation endpoint definitions."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter
except ImportError:
    APIRouter = None

from ...core.config import get_settings
from ...core.runtime import get_logger, log_step
from ...evaluation import Evaluator
from ...experiments.runner import run_checkpoint_evaluation
from ...utils.checkpoint import resolve_checkpoint_path

router = APIRouter() if APIRouter is not None else None
logger = get_logger("api.routes.evaluate")


def evaluate(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    log_step(logger, "Received evaluation request.")

    predictions = payload.get("predictions")
    targets = payload.get("targets")
    if predictions is None or targets is None:
        return {
            "status": "ready",
            "message": "Provide 'predictions' and 'targets' to compute metrics. Supported outputs include accuracy, f1_score, auc, dice_score, iou, and hausdorff_distance.",
        }

    evaluator = Evaluator(model=payload.get("model"))
    metrics = evaluator.evaluate((predictions, targets))
    analysis = evaluator.analyze((predictions, targets))
    return {
        "status": "completed",
        "metrics": metrics,
        "analysis": analysis,
    }


def evaluate_checkpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate a saved checkpoint and export report artifacts."""

    payload = payload or {}
    checkpoint_path = payload.get("checkpoint_path")
    checkpoint_name: str | None = None
    if not checkpoint_path:
        checkpoint_name = payload.get("checkpoint_name")
        if not checkpoint_name:
            return {"status": "error", "message": "checkpoint_path or checkpoint_name is required."}
        settings = get_settings()
        checkpoint_path = resolve_checkpoint_path(settings.models_dir / "checkpoints", str(checkpoint_name))

    if not checkpoint_path:
        return {"status": "error", "message": f"No trained checkpoint found for '{checkpoint_name}'. Train the model first."}

    split = str(payload.get("split", "test"))
    try:
        return run_checkpoint_evaluation(checkpoint_path, get_settings(), split=split)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "checkpoint_path": str(checkpoint_path)}


if router is not None:

    @router.post("/")
    def evaluate_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return evaluate(payload)

    @router.post("/checkpoint")
    def evaluate_checkpoint_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return evaluate_checkpoint(payload)