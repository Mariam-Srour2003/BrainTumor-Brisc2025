"""Training endpoint definitions."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

try:
    from fastapi import APIRouter
    from fastapi.responses import StreamingResponse
except ImportError:
    APIRouter = None
    StreamingResponse = None  # type: ignore[assignment,misc]

from ...core.config import get_settings
from ...core.runtime import get_logger, log_step
from ...core.constants import VIEW_CODES
from ...experiments.runner import prepare_dataset, prepare_divided_dataset, run_all_registered_models_training, run_joint_hyperparameter_tuning, run_joint_training, run_registered_model_training
from ...models import DEFAULT_MODEL_REGISTRY


router = APIRouter() if APIRouter is not None else None
logger = get_logger("api.routes.train")


def train_model(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    action = str(payload.get("action", "train"))
    settings = get_settings()
    log_step(logger, f"Received training request with action={action}.")

    try:
        if action == "prepare":
            summary = prepare_dataset(settings)
            return {
                "status": "completed",
                "action": action,
                "output_root": summary.output_root,
                "processed_count": summary.processed_count,
                "skipped_count": summary.skipped_count,
            }

        if action == "prepare_divided":
            summary = prepare_divided_dataset(settings)
            return {
                "status": "completed",
                "action": action,
                "output_root": summary.output_root,
                "processed_count": summary.processed_count,
                "skipped_count": summary.skipped_count,
            }

        if action == "tune":
            summary = run_joint_hyperparameter_tuning(
                settings,
                trials=int(payload.get("trials", 5)),
                metric_name=str(payload.get("metric_name", "val_loss")),
            )
            return {
                "status": "completed",
                "action": action,
                "best_metric": summary.best_metric,
                "metric_name": summary.metric_name,
                "best_trial": summary.best_trial.__dict__,
                "trial_count": len(summary.trials),
            }

        if action == "train_all":
            return {
                "status": "completed",
                "action": action,
                "result": run_all_registered_models_training(settings),
            }

        if action == "train_model":
            model_name = str(payload.get("model_name", ""))
            if not model_name:
                return {
                    "status": "error",
                    "action": action,
                    "message": "model_name is required when action=train_model.",
                }
            if model_name not in DEFAULT_MODEL_REGISTRY.list_models():
                return {
                    "status": "error",
                    "action": action,
                    "message": f"Unknown model_name: {model_name}.",
                }
            return {
                "status": "completed",
                "action": action,
                "result": run_registered_model_training(model_name, settings),
            }

        result = run_joint_training(settings)
        return {"status": "completed", "action": action, "result": result}
    except ImportError as exc:
        return {"status": "error", "action": action, "message": str(exc)}


if router is not None:

    @router.post("/")
    def train_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return train_model(payload)

    @router.post("/model/{model_name}")
    def train_single_model_endpoint(model_name: str) -> dict[str, Any]:
        settings = get_settings()
        available_models = DEFAULT_MODEL_REGISTRY.list_models()
        if model_name not in available_models:
            return {
                "status": "error",
                "action": "train_model",
                "model_name": model_name,
                "message": "Unknown model name.",
                "available_models": list(available_models),
            }

        try:
            return {
                "status": "completed",
                "action": "train_model",
                "result": run_registered_model_training(model_name, settings),
            }
        except ImportError as exc:
            return {
                "status": "error",
                "action": "train_model",
                "model_name": model_name,
                "message": str(exc),
            }

    @router.get("/stream/{model_name}")
    def train_stream_endpoint(model_name: str) -> Any:
        """SSE endpoint — streams epoch progress events while training model_name.

        Each event is a JSON object:
          {"type": "epoch", "epoch": N, "total": T, "train_loss": ..., "val_loss": ...}
          {"type": "complete", "best_val_loss": ...}
          {"type": "error", "message": ...}
        """
        settings = get_settings()
        available_models = DEFAULT_MODEL_REGISTRY.list_models()

        if model_name not in available_models:
            def _err():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown model: {model_name}'})}\n\n"
            return StreamingResponse(_err(), media_type="text/event-stream")

        # Detect view-specific model (e.g. classification.efficientnet.finetune.ax)
        _name_parts = model_name.rsplit(".", 1)
        _view_filter: str | None = _name_parts[1] if len(_name_parts) == 2 and _name_parts[1] in VIEW_CODES else None
        _base_name = _name_parts[0] if _view_filter else model_name
        _processed_root = settings.processed_divided_data_root if _view_filter else None

        event_q: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def _callback(event: dict[str, Any]) -> None:
            event_q.put(event)

        def _run() -> None:
            try:
                from ...training.trainer import Trainer, TrainingConfig
                # Segmentation models are binary (tumor vs background) — 1 output channel.
                # Joint models use joint_class_names; view_classifier uses 3 view classes;
                # all other classification models use 4 tumor classes.
                _is_seg = _base_name.startswith("segmentation.")
                _is_hybrid = _base_name.startswith("hybrid.")
                _is_view_clf = model_name == "classification.view_classifier"
                if _is_seg:
                    _num_classes = 1
                elif _is_hybrid:
                    _num_classes = len(settings.joint_class_names)
                elif _is_view_clf:
                    _num_classes = len(settings.view_class_names)
                else:
                    _num_classes = len(settings.class_names)
                model = DEFAULT_MODEL_REGISTRY.create(
                    model_name,
                    num_classes=_num_classes,
                    in_channels=3,
                    **({"segmentation_classes": 1} if _is_hybrid else {}),
                )
                tc = TrainingConfig.from_service_config(settings)
                trainer = Trainer(
                    model, config=settings, training_config=tc,
                    view_filter=_view_filter, processed_root=_processed_root,
                    task="view_classification" if _is_view_clf else "classification",
                )

                if _base_name.startswith("classification."):
                    result = trainer.fit_classification(
                        split="train", checkpoint_name=model_name, progress_callback=_callback
                    )
                elif _base_name.startswith("segmentation."):
                    result = trainer.fit_segmentation(
                        split="train", checkpoint_name=model_name, progress_callback=_callback
                    )
                else:
                    result = trainer.fit_joint(
                        split="train", checkpoint_name=model_name, progress_callback=_callback
                    )

                event_q.put({"type": "done", "status": result.status, "metrics": result.metrics})
            except Exception as exc:
                log_step(logger, f"SSE training failed for {model_name}: {exc}")
                event_q.put({"type": "error", "message": str(exc)})
            finally:
                event_q.put(None)  # sentinel

        threading.Thread(target=_run, daemon=True).start()

        def _stream():
            while True:
                item = event_q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )