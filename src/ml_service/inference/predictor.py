"""Prediction wrapper for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import ServiceConfig, get_settings
from ..core.runtime import get_logger, log_step, require_torch_device
from ..data.preprocessing import build_preprocessing_config, preprocess_prediction_image
from ..explainability import ExplainabilityEngine, ExplainabilityMethod, ExplainabilityRequest, ExplainabilityTask
from ..models import DEFAULT_MODEL_REGISTRY
from ..utils import load_checkpoint, model_checkpoint_dir, model_checkpoint_run_dir, sanitize_checkpoint_name


class Predictor:
    """Run the shared preprocessing pipeline before prediction."""

    def __init__(
        self,
        model: Any | None = None,
        config: ServiceConfig | None = None,
        model_name: str | None = None,
        model_date: str | None = None,
    ) -> None:
        self.config = config or get_settings()
        self.preprocessing_config = build_preprocessing_config(self.config)
        self.logger = get_logger("inference.predictor")
        self.requested_model_name = model_name
        self.requested_model_date = model_date
        self.loaded_checkpoint: dict[str, Any] | None = None
        self.model = model if model is not None else self._autoload_trained_model()
        self.torch, self.device = self._init_device()
        model_name = type(getattr(self.model, "model", self.model)).__name__ if self.model is not None else "None"
        checkpoint_path = self.loaded_checkpoint["checkpoint_path"] if self.loaded_checkpoint is not None else "None"
        log_step(self.logger, f"Predictor initialized on device={self.device}, model={model_name}, checkpoint={checkpoint_path}.")

    def _checkpoint_dir(self) -> Path:
        return self.config.models_dir / "checkpoints"

    def _select_checkpoint_path(self) -> Path | None:
        checkpoint_dir = self._checkpoint_dir()
        if not checkpoint_dir.exists():
            return None

        model_name = self.requested_model_name
        model_date = self.requested_model_date

        if model_name:
            safe_model_name = sanitize_checkpoint_name(model_name)
            candidates: list[Path] = []

            if model_date:
                run_dir = model_checkpoint_run_dir(checkpoint_dir, model_name, model_date)
                preferred = run_dir / f"{safe_model_name}.pt"
                if preferred.exists():
                    return preferred
                candidates.extend(run_dir.glob("*.pt"))
            else:
                root_dir = model_checkpoint_dir(checkpoint_dir, model_name)
                if root_dir.exists():
                    run_dirs = sorted((path for path in root_dir.iterdir() if path.is_dir()), reverse=True)
                    for run_dir in run_dirs:
                        preferred = run_dir / f"{safe_model_name}.pt"
                        if preferred.exists():
                            return preferred
                        candidates.extend(run_dir.glob("*.pt"))

            flat_preferred = checkpoint_dir / f"{safe_model_name}.pt"
            if flat_preferred.exists():
                candidates.append(flat_preferred)

            if candidates:
                return max(candidates, key=lambda path: path.stat().st_mtime)
            return None

        candidates: list[Path] = []
        if model_date:
            safe_date = sanitize_checkpoint_name(model_date)
            for model_dir in checkpoint_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                run_dir = model_dir / safe_date
                if run_dir.exists() and run_dir.is_dir():
                    candidates.extend(run_dir.glob("*.pt"))
        else:
            candidates.extend(checkpoint_dir.glob("*/*/*.pt"))
            candidates.extend(checkpoint_dir.glob("*.pt"))

        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    def _build_model_from_name(self, model_name: str) -> Any:
        if model_name == "classification.view_classifier":
            return DEFAULT_MODEL_REGISTRY.create(
                model_name,
                num_classes=len(self.config.view_class_names),
                in_channels=3,
            )

        if model_name.startswith("classification."):
            return DEFAULT_MODEL_REGISTRY.create(
                model_name,
                num_classes=len(self.config.class_names),
                in_channels=3,
            )

        if model_name.startswith("segmentation."):
            return DEFAULT_MODEL_REGISTRY.create(
                model_name,
                num_classes=1,
                in_channels=3,
            )

        if model_name.startswith("hybrid."):
            return DEFAULT_MODEL_REGISTRY.create(
                model_name,
                num_classes=len(self.config.class_names),
                segmentation_classes=1,
                in_channels=3,
                encoder_name="resnet34",
                pretrained=False,
            )

        raise ValueError(f"Unsupported checkpoint model_name: {model_name}")

    def _infer_checkpoint_task(self, metadata: dict[str, Any], checkpoint_path: Path) -> str:
        task = str(metadata.get("task") or "").strip().lower()
        if task in {"classification", "segmentation", "joint", "view_classification"}:
            return task

        name = checkpoint_path.stem.lower()
        if "view_classifier" in name:
            return "view_classification"
        if "segmentation" in name:
            return "segmentation"
        if "classification" in name:
            return "classification"
        return "joint" if "joint" in name or "hybrid" in name else "classification"

    def _candidate_model_names(self, task: str, model_name: str) -> tuple[str, ...]:
        registered = DEFAULT_MODEL_REGISTRY.list_models()
        candidates: list[str] = []

        if task == "view_classification":
            candidates.append("classification.view_classifier")
            return tuple(candidates)

        if model_name in registered:
            candidates.append(model_name)

        if task == "classification":
            candidates.extend([name for name in registered if name.startswith("classification.") and name not in candidates])
        elif task == "segmentation":
            candidates.extend([name for name in registered if name.startswith("segmentation.") and name not in candidates])
        else:
            candidates.extend([name for name in registered if name.startswith("hybrid.") and name not in candidates])

        return tuple(candidates)

    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> tuple[Any | None, dict[str, Any], str, str | None]:
        payload = load_checkpoint(checkpoint_path)
        metadata = payload.get("metadata", {})
        checkpoint = payload.get("checkpoint", {})
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint payload is not a dict.")

        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint payload is missing a state_dict.")

        model_name = str(metadata.get("model_name") or checkpoint_path.stem)
        task = self._infer_checkpoint_task(metadata, checkpoint_path)

        for candidate in self._candidate_model_names(task, model_name):
            try:
                model = self._build_model_from_name(candidate)
                target_model = getattr(model, "model", model)
                target_model.load_state_dict(state_dict)
                return model, metadata, task, candidate
            except Exception:
                continue

        return None, metadata, task, None

    def _autoload_trained_model(self) -> Any | None:
        checkpoint_path = self._select_checkpoint_path()
        if checkpoint_path is None:
            log_step(self.logger, "No checkpoints found for automatic predictor loading.")
            return None

        try:
            payload = load_checkpoint(checkpoint_path)
        except Exception as exc:
            log_step(self.logger, f"Failed to load checkpoint payload: {checkpoint_path} ({exc})")
            return None

        try:
            model, metadata, task, resolved_model_name = self._load_model_from_checkpoint(checkpoint_path)
            if model is None:
                raise ValueError(f"Could not match checkpoint {checkpoint_path.name} to a registered model.")
            self.loaded_checkpoint = {
                "checkpoint_path": str(checkpoint_path),
                "metadata": metadata,
                "resolved_model_name": resolved_model_name,
                "resolved_task": task,
                "requested_model_name": self.requested_model_name,
                "requested_model_date": self.requested_model_date,
            }
            log_step(self.logger, f"Loaded trained checkpoint for prediction: {checkpoint_path} (model_name={resolved_model_name}, task={task}).")
            return model
        except Exception as exc:
            log_step(self.logger, f"Failed to reconstruct model from checkpoint {checkpoint_path}: {exc}")
            return None

    def _init_device(self) -> tuple[Any | None, Any | None]:
        try:
            torch, device = require_torch_device()
        except ImportError:
            return None, None

        if self.model is not None:
            target_model = getattr(self.model, "model", self.model)
            if hasattr(target_model, "to"):
                target_model.to(device)
        return torch, device

    def preprocess(self, input_data: str | Path | np.ndarray | Any) -> np.ndarray:
        log_step(self.logger, f"Preprocessing input of type {type(input_data).__name__}.")
        if isinstance(input_data, np.ndarray):
            return input_data

        prepared = preprocess_prediction_image(input_data, self.preprocessing_config)
        log_step(self.logger, f"Preprocessing complete with shape={prepared.shape}.")
        return prepared

    def _to_torch_input(self, prepared_input: np.ndarray) -> Any:
        if self.torch is None or self.device is None:
            raise ImportError("Install torch to run predictions on CPU or GPU.")

        tensor = self.torch.tensor(prepared_input[None, ...], dtype=self.torch.float32, device=self.device)
        log_step(self.logger, f"Converted input to tensor on device={self.device} with shape={tuple(tensor.shape)}.")
        return tensor

    def _forward_model(self, prepared_input: np.ndarray) -> Any:
        if self.model is None:
            return {
                "status": "ready",
                "message": "Prediction input preprocessed successfully.",
                "shape": tuple(prepared_input.shape),
            }

        target_model = getattr(self.model, "model", self.model)
        if hasattr(self.model, "predict"):
            log_step(self.logger, "Using model.predict fallback.")
            return self.model.predict(prepared_input)

        if self.torch is not None and isinstance(target_model, self.torch.nn.Module):
            if hasattr(target_model, "to"):
                target_model.to(self.device)
            target_model.eval()
            tensor_input = self._to_torch_input(prepared_input)
            with self.torch.no_grad():
                log_step(self.logger, f"Running forward pass on device={self.device}.")
                return target_model(tensor_input)

        if callable(self.model):
            log_step(self.logger, "Using callable model fallback.")
            return self.model(prepared_input)

        raise TypeError("Model must be callable or expose a predict method.")

    def predict(self, input_data: str | Path | np.ndarray | Any = None) -> Any:
        log_step(self.logger, "Starting prediction pipeline.")
        prepared_input = self.preprocess(input_data)
        result = self._forward_model(prepared_input)
        log_step(self.logger, f"Prediction pipeline complete. Result type={type(result).__name__}.")
        return result

    def predict_tta(
        self,
        input_data: str | Path | np.ndarray | Any = None,
        n_augments: int = 4,
    ) -> Any:
        """Test-Time Augmentation: average predictions over flipped variants.

        Augments: original, h-flip, v-flip, h+v-flip.  Falls back to a single
        forward pass when torch is unavailable or no model is loaded.
        """
        if self.torch is None or self.model is None:
            return self.predict(input_data)

        prepared = self.preprocess(input_data)  # (C, H, W) float32

        variants = [
            prepared,
            prepared[:, :, ::-1].copy(),
            prepared[:, ::-1, :].copy(),
            prepared[:, ::-1, ::-1].copy(),
        ][:max(1, min(n_augments, 4))]

        batch = self.torch.tensor(np.stack(variants), dtype=self.torch.float32)
        if self.device is not None:
            batch = batch.to(self.device)

        target_model = getattr(self.model, "model", self.model)
        target_model.eval()

        with self.torch.no_grad():
            outputs = target_model(batch)

        if isinstance(outputs, dict):
            averaged: dict[str, Any] = {}
            cls_logits = outputs.get("classification")
            if cls_logits is not None:
                averaged["classification"] = self.torch.softmax(cls_logits, dim=1).mean(0, keepdim=True)
            seg_logits = outputs.get("segmentation")
            if seg_logits is not None:
                averaged["segmentation"] = self.torch.sigmoid(seg_logits[0:1])
            log_step(self.logger, f"TTA complete: {len(variants)} augments averaged.")
            return averaged

        probs = self.torch.softmax(outputs, dim=1).mean(0, keepdim=True)
        log_step(self.logger, f"TTA complete: {len(variants)} augments averaged.")
        return probs

    def explain(
        self,
        input_data: str | Path,
        *,
        method: ExplainabilityMethod = "gradcam",
        target_task: ExplainabilityTask = "auto",
        target_class: int | None = None,
        target_layer: str | None = None,
        output_root: str | Path | None = None,
    ) -> Any:
        """Generate an explanation artifact for the current model and image."""

        if self.model is None:
            raise ValueError("An attached model is required for explainability.")

        if target_task == "auto":
            resolved_task = None
            if self.loaded_checkpoint is not None:
                resolved_task = str(self.loaded_checkpoint.get("resolved_task") or self.loaded_checkpoint.get("metadata", {}).get("task") or "").strip().lower()
            if resolved_task in {"classification", "segmentation", "both"}:
                target_task = resolved_task  # type: ignore[assignment]

        engine = ExplainabilityEngine(self.model, self.config)
        log_step(self.logger, f"Generating explanation with method={method}, target_task={target_task}.")
        request = ExplainabilityRequest(
            image_path=input_data,
            method=method,
            target_task=target_task,
            target_class=target_class,
            target_layer=target_layer,
            output_root=output_root,
        )
        return engine.explain(request)