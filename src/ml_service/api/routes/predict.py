"""Prediction endpoint definitions."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
except ImportError:
    APIRouter = None

from ...core.config import ServiceConfig, get_settings
from ...inference.predictor import Predictor

router = APIRouter() if APIRouter is not None else None


def _store_upload(upload: UploadFile, config: ServiceConfig) -> Path:
    upload_root = config.outputs_dir / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    target_path = upload_root / upload.filename
    with target_path.open("wb") as target_file:
        shutil.copyfileobj(upload.file, target_file)
    return target_path


def predict(payload: dict[str, Any] | None = None) -> dict[str, str]:
    return {
        "status": "ready",
        "message": "Use POST /predict/upload with an image file to run prediction, and set explain=true to return an explanation.",
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {str(k): _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}

    try:
        import torch  # type: ignore
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim == 0:
                return tensor.item()
            return tensor.tolist()
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64).ravel()
    if values.size == 0:
        return values

    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)
    if total <= 0:
        return np.full(values.shape, 1.0 / float(values.size), dtype=np.float64)
    return exp_values / total


def _as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _image_to_base64(image: Any) -> str:
    import base64, io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _save_segmentation_artifact(image_path: Path, mask: np.ndarray, config: ServiceConfig) -> dict[str, str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install pillow to save segmentation prediction images.") from exc

    predictions_root = config.outputs_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    mask_array = np.asarray(mask, dtype=np.uint8)
    if mask_array.ndim == 3:
        mask_array = mask_array[0]
    mask_array = (mask_array > 0).astype(np.uint8) * 255

    base_image = Image.open(image_path).convert("RGB")
    mask_image = Image.fromarray(mask_array, mode="L").resize(base_image.size)
    overlay = Image.new("RGBA", base_image.size, (255, 0, 0, 0))
    overlay.putalpha(mask_image.point(lambda pixel: 130 if pixel > 0 else 0))
    composed = Image.alpha_composite(base_image.convert("RGBA"), overlay)

    mask_path = predictions_root / f"{image_path.stem}_segmentation_mask.png"
    overlay_path = predictions_root / f"{image_path.stem}_segmentation_overlay.png"
    mask_image.save(mask_path)
    composed.save(overlay_path)

    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "mask_base64": _image_to_base64(mask_image),
        "overlay_base64": _image_to_base64(composed.convert("RGB")),
    }


def _select_class_scores(prediction: Any) -> np.ndarray | None:
    if isinstance(prediction, dict):
        for key in ("classification", "logits", "scores", "probabilities", "prediction"):
            if key in prediction:
                candidate = prediction[key]
                try:
                    array = _as_numpy(candidate).astype(np.float64)
                except Exception:
                    continue
                if array.size > 0:
                    return array

    try:
        array = _as_numpy(prediction).astype(np.float64)
    except Exception:
        return None

    return array if array.size > 0 else None


def _decode_classification(prediction: Any, class_names: tuple[str, ...]) -> dict[str, Any]:
    decoded: dict[str, Any] = {
        "class_index": None,
        "class_name": "",
        "confidence": None,
        "probabilities": {},
    }

    if isinstance(prediction, dict):
        class_index = prediction.get("class_index")
        if isinstance(class_index, (int, np.integer)):
            decoded["class_index"] = int(class_index)

        class_name = prediction.get("class_name") or prediction.get("class") or prediction.get("predicted_class")
        if isinstance(class_name, str) and class_name.strip():
            decoded["class_name"] = class_name.strip()
            if decoded["class_index"] is None and class_name.strip().isdigit():
                decoded["class_index"] = int(class_name.strip())

        confidence = prediction.get("confidence") or prediction.get("score") or prediction.get("probability")
        if isinstance(confidence, (int, float, np.integer, np.floating)):
            decoded["confidence"] = float(confidence)

        probabilities = prediction.get("probabilities") or prediction.get("classificationProbabilities")
        if isinstance(probabilities, dict):
            decoded["probabilities"] = {
                str(key): float(value)
                for key, value in probabilities.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }

    scores = _select_class_scores(prediction)
    if scores is None:
        return decoded

    if scores.ndim > 1:
        scores = scores[0]

    if scores.ndim == 0:
        scores = scores.reshape(1)

    if scores.size == 1:
        confidence = float(scores.ravel()[0])
        decoded["confidence"] = confidence if decoded["confidence"] is None else decoded["confidence"]
        if not decoded["class_name"]:
            decoded["class_name"] = class_names[0] if class_names else "class_0"
        decoded["class_index"] = 0 if decoded["class_index"] is None else decoded["class_index"]
        decoded["probabilities"] = {decoded["class_name"]: float(confidence)}
        return decoded

    if np.all((scores >= 0.0) & (scores <= 1.0)):
        total = float(np.sum(scores))
        probabilities = scores if 0.95 <= total <= 1.05 else _softmax(scores)
    else:
        probabilities = _softmax(scores)

    class_index = int(np.argmax(probabilities))
    class_name = class_names[class_index] if class_index < len(class_names) else f"class_{class_index}"

    decoded["class_index"] = class_index if decoded["class_index"] is None else decoded["class_index"]
    decoded["class_name"] = class_name if not decoded["class_name"] else decoded["class_name"]
    decoded["confidence"] = float(probabilities[class_index]) if decoded["confidence"] is None else decoded["confidence"]
    decoded["probabilities"] = {
        class_names[index] if index < len(class_names) else f"class_{index}": float(probability)
        for index, probability in enumerate(probabilities.tolist())
    }

    return decoded


def _decode_task_output(prediction: Any, class_names: tuple[str, ...], task: str | None = None) -> dict[str, Any]:
    if task == "segmentation":
        array = _as_numpy(prediction)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        mask = (1.0 / (1.0 + np.exp(-array))) >= 0.5 if array.dtype.kind in {"f", "i"} else array >= 0.5
        return {
            "task": "segmentation",
            "class_index": None,
            "class_name": "segmentation",
            "confidence": None,
            "probabilities": {},
            "segmentation_mask": mask.astype(np.uint8),
        }

    if isinstance(prediction, dict) and "segmentation" in prediction and "classification" in prediction:
        decoded = _decode_classification(prediction, class_names)
        decoded["task"] = "joint"
        segmentation_array = _as_numpy(prediction["segmentation"])
        if segmentation_array.ndim == 4:
            segmentation_array = segmentation_array[0]
        if segmentation_array.ndim == 3 and segmentation_array.shape[0] == 1:
            segmentation_array = segmentation_array[0]
        if segmentation_array.ndim == 3 and segmentation_array.shape[-1] == 1:
            segmentation_array = segmentation_array[..., 0]
        decoded["segmentation_mask"] = ((1.0 / (1.0 + np.exp(-segmentation_array))) >= 0.5).astype(np.uint8)
        return decoded

    decoded = _decode_classification(prediction, class_names)
    decoded["task"] = task or "classification"
    return decoded


if router is not None:

    @router.post("/upload")
    def predict_endpoint(
        request: Request,
        file: UploadFile = File(...),
        explain: bool = Form(False),
        model_name: str | None = Form(None),
        model_date: str | None = Form(None),
        method: str = Form("gradcam"),
        target_task: str = Form("auto"),
        target_class: int | None = Form(None),
        target_layer: str | None = Form(None),
    ) -> dict[str, Any]:
        config = get_settings()
        predictor = getattr(request.app.state, "predictor", None)
        requested_name = None if model_name in {None, ""} else model_name
        requested_date = None if model_date in {None, ""} else model_date

        if (
            predictor is None
            or getattr(predictor, "requested_model_name", None) != requested_name
            or getattr(predictor, "requested_model_date", None) != requested_date
        ):
            predictor = Predictor(config=config, model_name=requested_name, model_date=requested_date)
            request.app.state.predictor = predictor
        if predictor.model is None:
            raise HTTPException(
                status_code=400,
                detail="No trained checkpoint found for the selected model/date. Train a model first or omit model_date to use the latest checkpoint.",
            )

        image_path = _store_upload(file, config)
        prediction = predictor.predict(image_path)
        checkpoint_metadata = getattr(predictor, "loaded_checkpoint", {}) or {}
        task = str(checkpoint_metadata.get("metadata", {}).get("task") or checkpoint_metadata.get("resolved_task") or "").lower() or None
        _output_class_names = config.view_class_names if task == "view_classification" else config.class_names
        decoded = _decode_task_output(prediction, _output_class_names, task)
        response: dict[str, Any] = {
            "class": decoded["class_name"],
            "class_name": decoded["class_name"],
            "predicted_class": decoded["class_name"],
            "class_index": decoded["class_index"],
            "confidence": decoded["confidence"],
            "probabilities": decoded["probabilities"],
            "prediction": _to_jsonable(prediction),
            "task": decoded.get("task", task or "classification"),
            "image_path": str(image_path),
        }

        if "segmentation_mask" in decoded:
            artifacts = _save_segmentation_artifact(image_path, decoded["segmentation_mask"], config)
            response.update({
                "segmentation_mask": _to_jsonable(decoded["segmentation_mask"]),
                "segmentation_mask_path": artifacts["mask_path"],
                "segmentation_overlay_path": artifacts["overlay_path"],
                "segmentation_mask_base64": artifacts.get("mask_base64"),
                "segmentation_overlay_base64": artifacts.get("overlay_base64"),
            })

        if explain:
            artifact = predictor.explain(
                image_path,
                method=method,  # type: ignore[arg-type]
                target_task=target_task,
                target_class=target_class,
                target_layer=target_layer,
            )
            response["explanation"] = _to_jsonable(artifact)

        return response

    @router.post("/")
    def predict_root_endpoint(payload: dict[str, Any] | None = None) -> dict[str, str]:
        return predict(payload)