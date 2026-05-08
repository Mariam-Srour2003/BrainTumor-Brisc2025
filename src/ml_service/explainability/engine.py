"""Explainable AI utilities for Grad-CAM, SHAP, and LIME."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from ..core.config import ServiceConfig, get_settings
from ..core.runtime import get_logger, log_step, require_torch_device
from ..data.preprocessing import ImagePreprocessingConfig, load_image, preprocess_prediction_image

ExplainabilityMethod = Literal["gradcam", "shap", "lime"]
ExplainabilityTask = Literal["auto", "classification", "segmentation", "both"]


@dataclass(frozen=True)
class ExplainabilityRequest:
    """Inputs for an explainability run."""

    image_path: str | Path
    method: ExplainabilityMethod = "gradcam"
    target_task: ExplainabilityTask = "auto"
    target_class: int | None = None
    target_layer: str | None = None
    output_root: str | Path | None = None


@dataclass(frozen=True)
class ExplainabilityArtifact:
    """Artifact metadata produced by an explanation run."""

    method: ExplainabilityMethod
    task: str
    target_class: int | None
    original_image_path: str
    overlay_path: str | None = None
    heatmap_path: str | None = None
    raw_path: str | None = None
    overlay_base64: str | None = None
    heatmap_base64: str | None = None
    raw_base64: str | None = None
    prediction: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Install torch to use Grad-CAM explainability.") from exc

    return torch, F


def _ensure_output_root(output_root: str | Path | None, config: ServiceConfig) -> Path:
    root = Path(output_root) if output_root is not None else config.outputs_dir / "explanations"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tensor_to_pil(array: np.ndarray) -> Image.Image:
    array = np.clip(array, 0.0, 1.0)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray((array * 255).astype(np.uint8))


def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_rgb = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    heatmap_rgb[..., 0] = np.clip(255 * heatmap, 0, 255).astype(np.uint8)
    heatmap_rgba = Image.fromarray(heatmap_rgb).convert("RGBA")
    heatmap_rgba.putalpha(int(255 * alpha))
    return Image.alpha_composite(base, heatmap_rgba)


def _save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None

    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_model(model: Any) -> Any:
    return getattr(model, "model", model)


# Submodule root names that belong to the segmentation path, not the classification path.
# Used to avoid hooking into layers that have no gradient flow from classification loss.
_SEGMENTATION_ROOT_NAMES: frozenset[str] = frozenset({
    "dec1", "dec2", "dec3", "dec4", "dec5",
    "decoder", "segmentation_head", "seg_head",
    "bottleneck",  # shared-encoder joint models only use bottleneck for the decoder path
    "up1", "up2", "up3", "up4", "up5",
    "outc", "out_conv", "final_conv",
})

# Top-level prefixes that indicate a dedicated backbone/encoder submodule.
_ENCODER_PREFIXES: tuple[str, ...] = ("encoder.", "backbone.", "base_model.", "features.")


def _select_default_target_layer(model: Any) -> tuple[str, Any]:
    named_modules = list(model.named_modules()) if hasattr(model, "named_modules") else []
    for name, module in reversed(named_modules):
        if module.__class__.__name__.lower().startswith("conv") or module.__class__.__name__.lower().endswith("conv2d"):
            return name, module
    for name, module in reversed(named_modules):
        if hasattr(module, "weight") and getattr(module, "weight", None) is not None and len(getattr(module.weight, "shape", [])) == 4:
            return name, module
    raise ValueError("Could not auto-select a convolutional target layer for Grad-CAM.")


def _select_classification_target_layer(model: Any) -> tuple[str, Any]:
    """Select the last Conv2d that is in the classification gradient path.

    Strategy:
      1. Prefer layers inside an explicit encoder/backbone submodule (e.g. encoder.layer4.*).
         These are always in the classification forward/backward path.
      2. Fall back to any Conv2d whose root module name is not a known segmentation component
         (decoder blocks, segmentation head, bottleneck used only by the decoder, etc.).
      3. Last resort: accept any conv layer.
    """
    named_modules = list(model.named_modules()) if hasattr(model, "named_modules") else []

    # Pass 1 — inside a dedicated encoder/backbone submodule
    for name, module in reversed(named_modules):
        cls = module.__class__.__name__.lower()
        if not (cls.startswith("conv") or cls.endswith("conv2d")):
            continue
        if any(name.startswith(pfx) for pfx in _ENCODER_PREFIXES):
            return name, module

    # Pass 2 — exclude segmentation-only root modules
    for name, module in reversed(named_modules):
        cls = module.__class__.__name__.lower()
        if not (cls.startswith("conv") or cls.endswith("conv2d")):
            continue
        root = name.split(".")[0].lower()
        if root in _SEGMENTATION_ROOT_NAMES:
            continue
        return name, module

    # Fallback — take any conv layer
    return _select_default_target_layer(model)


def _resolve_target_layer(
    model: Any,
    target_layer: str | None,
    *,
    prefer_classification: bool = False,
) -> tuple[str, Any]:
    if target_layer:
        modules = dict(model.named_modules()) if hasattr(model, "named_modules") else {}
        if target_layer not in modules:
            raise ValueError(f"Unknown target_layer: {target_layer}")
        return target_layer, modules[target_layer]

    if prefer_classification:
        return _select_classification_target_layer(model)
    return _select_default_target_layer(model)


def _classification_logits(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        if "classification" in outputs:
            return outputs["classification"]
        if "logits" in outputs:
            return outputs["logits"]
    return outputs


def _resolve_task(outputs: Any, target_task: ExplainabilityTask) -> ExplainabilityTask:
    if target_task != "auto":
        return target_task

    if isinstance(outputs, dict) and {"classification", "segmentation"}.issubset(outputs):
        return "both"
    if isinstance(outputs, dict) and "segmentation" in outputs:
        return "segmentation"
    # Raw 4-D tensor → segmentation map (U-Net etc.)
    if hasattr(outputs, "ndim") and outputs.ndim == 4:
        return "segmentation"
    return "classification"


def _normalize_mask_map(mask: Any) -> Any:
    mask = mask.detach().float() if hasattr(mask, "detach") else np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask.mean(dim=0) if hasattr(mask, "mean") else mask.mean(axis=0)
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)
    return mask


class ExplainabilityEngine:
    """Run model explanations using Grad-CAM, SHAP, or LIME."""

    def __init__(self, model: Any, config: ServiceConfig | None = None) -> None:
        self.model = _extract_model(model)
        self.config = config or get_settings()
        self.logger = get_logger("explainability.engine")
        self.torch, self.device = self._init_device()
        self.preprocessing_config = ImagePreprocessingConfig(
            image_size=self.config.image_size,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
            augment_training_data=False,
        )
        log_step(self.logger, f"Explainability engine initialized on device={self.device}.")

    def _init_device(self) -> tuple[Any | None, Any | None]:
        try:
            torch, device = require_torch_device()
        except ImportError:
            return None, None

        if hasattr(self.model, "to"):
            self.model.to(device)
        return torch, device

    def explain(self, request: ExplainabilityRequest) -> ExplainabilityArtifact:
        log_step(self.logger, f"Explain request received: method={request.method}, task={request.target_task}, image={request.image_path}.")
        try:
            if request.method == "gradcam":
                return self._gradcam(request)
            if request.method == "shap":
                return self._shap(request)
            if request.method == "lime":
                return self._lime(request)
            raise ValueError(f"Unsupported explainability method: {request.method}")
        except ImportError as exc:
            log_step(self.logger, f"Optional dependency missing for method={request.method}: {exc}")
            return ExplainabilityArtifact(
                method=request.method,
                task=request.target_task if request.target_task != "auto" else "classification",
                target_class=request.target_class,
                original_image_path=str(request.image_path),
                metadata={"error": str(exc), "missing_dependency": True},
            )

    def _prepare_input(self, image_path: str | Path) -> tuple[Image.Image, np.ndarray, Any]:
        image = load_image(image_path)
        tensor = preprocess_prediction_image(image, self.preprocessing_config)
        return image, tensor, np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    def _predict(self, prepared: np.ndarray) -> Any:
        torch, _ = _require_torch()
        self.model.eval()
        input_tensor = torch.tensor(prepared[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_step(self.logger, f"Running model forward pass for explainability on device={self.device}.")
            return self.model(input_tensor)

    def _classification_artifact(
        self,
        image: Image.Image,
        outputs: Any,
        request: ExplainabilityRequest,
        target_layer: str | None = None,
    ) -> ExplainabilityArtifact:
        torch, F = _require_torch()
        model = self.model
        target_layer_name, target_layer_module = _resolve_target_layer(
            model, target_layer or request.target_layer, prefer_classification=True
        )
        log_step(self.logger, f"Using target layer {target_layer_name} for classification explanation.")
        activations: list[Any] = []
        gradients: list[Any] = []

        def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations.append(output.detach())

        def backward_hook(_module: Any, _grad_inputs: Any, grad_outputs: Any) -> None:
            gradients.append(grad_outputs[0].detach())

        forward_handle = target_layer_module.register_forward_hook(forward_hook)
        backward_handle = target_layer_module.register_full_backward_hook(backward_hook)

        try:
            prepared = preprocess_prediction_image(image, self.preprocessing_config)
            input_tensor = torch.tensor(prepared[None, ...], dtype=torch.float32, device=self.device, requires_grad=True)
            outputs = model(input_tensor)
            logits = _classification_logits(outputs)
            if logits.ndim != 2:
                raise ValueError("Classification explanations expect logits with shape [batch, classes].")

            target_index = request.target_class if request.target_class is not None else int(torch.argmax(logits[0]).item())
            score = logits[0, target_index]
            model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

            if not activations or not gradients:
                raise RuntimeError("Unable to capture activations/gradients for Grad-CAM.")

            activation = activations[-1][0]
            gradient = gradients[-1][0]
            weights = gradient.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=0))
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = F.interpolate(cam[None, None, ...], size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
            heatmap = cam.detach().cpu().numpy()

            output_root = _ensure_output_root(request.output_root, self.config)
            overlay = _overlay_heatmap(image, heatmap)
            overlay_path = _save_image(overlay, output_root / f"gradcam_{target_layer_name.replace('.', '_')}_{target_index}.png")
            heatmap_path = _save_image(_tensor_to_pil(heatmap), output_root / f"gradcam_{target_layer_name.replace('.', '_')}_{target_index}_heatmap.png")
            log_step(self.logger, f"Saved classification explainability overlay to {overlay_path}.")

            prediction = {
                "class_index": target_index,
                "score": float(score.detach().cpu().item()),
            }
            return ExplainabilityArtifact(
                method="gradcam",
                task="classification",
                target_class=target_index,
                original_image_path=str(request.image_path),
                overlay_path=str(overlay_path),
                heatmap_path=str(heatmap_path),
                overlay_base64=_encode_image(overlay),
                heatmap_base64=_encode_file(heatmap_path),
                prediction=prediction,
                metadata={
                    "prediction_score": prediction["score"],
                    "target_layer": target_layer_name,
                },
            )
        finally:
            forward_handle.remove()
            backward_handle.remove()

    def _segmentation_artifact(self, image: Image.Image, outputs: Any, request: ExplainabilityRequest) -> ExplainabilityArtifact:
        torch, F = _require_torch()

        segmentation_logits = outputs.get("segmentation") if isinstance(outputs, dict) else outputs
        if segmentation_logits is None:
            raise ValueError("No segmentation output available for explanation.")

        if segmentation_logits.ndim != 4:
            raise ValueError("Segmentation explanations expect logits with shape [batch, channels, height, width].")

        if segmentation_logits.shape[1] == 1:
            mask = torch.sigmoid(segmentation_logits[0, 0])
        else:
            channel_index = request.target_class if request.target_class is not None else int(torch.argmax(segmentation_logits[0].mean(dim=(1, 2))).item())
            mask = torch.softmax(segmentation_logits[0], dim=0)[channel_index]

        mask = _normalize_mask_map(mask)
        mask = F.interpolate(mask[None, None, ...], size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        heatmap = mask.detach().cpu().numpy()

        output_root = _ensure_output_root(request.output_root, self.config)
        overlay = _overlay_heatmap(image, heatmap)
        overlay_path = _save_image(overlay, output_root / "segmentation_overlay.png")
        heatmap_path = _save_image(_tensor_to_pil(heatmap), output_root / "segmentation_heatmap.png")
        raw_path = _save_image(_tensor_to_pil(heatmap > 0.5), output_root / "segmentation_mask.png")
        log_step(self.logger, f"Saved segmentation explanation artifacts to {output_root}.")

        prediction = {
            "mask_mean": float(np.asarray(heatmap).mean()),
            "mask_max": float(np.asarray(heatmap).max()),
        }
        return ExplainabilityArtifact(
            method="gradcam",
            task="segmentation",
            target_class=request.target_class,
            original_image_path=str(request.image_path),
            overlay_path=str(overlay_path),
            heatmap_path=str(heatmap_path),
            raw_path=str(raw_path),
            overlay_base64=_encode_image(overlay),
            heatmap_base64=_encode_file(heatmap_path),
            raw_base64=_encode_file(raw_path),
            prediction=prediction,
            metadata={"task": "segmentation"},
        )

    def _joint_artifact(self, request: ExplainabilityRequest) -> dict[str, Any]:
        image, prepared, _ = self._prepare_input(request.image_path)
        outputs = self._predict(prepared)
        log_step(self.logger, f"Joint explanation requested with target_task={request.target_task}.")

        results: dict[str, Any] = {"task": "joint"}

        if request.target_task in {"auto", "classification", "both"} and (not isinstance(outputs, dict) or "classification" in outputs):
            results["classification"] = self._classification_artifact(image, outputs, request)

        if request.target_task in {"auto", "segmentation", "both"} and (not isinstance(outputs, dict) or "segmentation" in outputs):
            results["segmentation"] = self._segmentation_artifact(image, outputs, request)

        if len(results) == 1:
            raise ValueError("Joint model produced no explanation artifacts.")

        return results

    def _gradcam(self, request: ExplainabilityRequest) -> ExplainabilityArtifact:
        image, prepared, _ = self._prepare_input(request.image_path)
        outputs = self._predict(prepared)

        task = _resolve_task(outputs, request.target_task)
        if task == "both":
            return self._joint_artifact(request)
        if task == "segmentation":
            return self._segmentation_artifact(image, outputs, request)

        return self._classification_artifact(image, outputs, request)

    def _shap(self, request: ExplainabilityRequest) -> ExplainabilityArtifact:
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install shap to use SHAP explanations.") from exc

        image, _, image_array = self._prepare_input(request.image_path)

        def predict_fn(batch: np.ndarray) -> np.ndarray:
            torch, _ = _require_torch()
            self.model.eval()
            batch_tensor = torch.tensor(batch.transpose(0, 3, 1, 2), dtype=torch.float32, device=self.device)
            outputs = self.model(batch_tensor)
            logits = _classification_logits(outputs)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.detach().cpu().numpy()

        masker = shap.maskers.Image("blur(32,32)", image_array.shape)
        explainer = shap.Explainer(predict_fn, masker)
        shap_values = explainer(image_array[None, ...])

        output_root = _ensure_output_root(request.output_root, self.config)
        raw_path = output_root / "shap_values.npy"
        np.save(raw_path, shap_values.values)
        log_step(self.logger, f"Saved SHAP values to {raw_path}.")

        # Build a visual heatmap from SHAP values so the browser can display something.
        # shap_values.values shape: (1, H, W, channels) or (1, H, W, channels, num_classes)
        shap_arr = np.asarray(shap_values.values[0], dtype=np.float32)
        if shap_arr.ndim == 4:          # (H, W, channels, num_classes)
            shap_map = np.abs(shap_arr).sum(axis=(2, 3))
        elif shap_arr.ndim == 3:        # (H, W, channels)
            shap_map = np.abs(shap_arr).sum(axis=2)
        else:
            shap_map = np.abs(shap_arr)
        shap_map -= shap_map.min()
        shap_map /= shap_map.max() + 1e-8

        heatmap_img = _tensor_to_pil(shap_map)
        overlay_img = _overlay_heatmap(image, shap_map)
        heatmap_path = _save_image(heatmap_img, output_root / "shap_heatmap.png")
        overlay_path = _save_image(overlay_img, output_root / "shap_overlay.png")
        log_step(self.logger, f"Saved SHAP visual heatmap to {overlay_path}.")

        return ExplainabilityArtifact(
            method="shap",
            task="classification",
            target_class=request.target_class,
            original_image_path=str(request.image_path),
            overlay_path=str(overlay_path),
            heatmap_path=str(heatmap_path),
            raw_path=str(raw_path),
            overlay_base64=_encode_image(overlay_img),
            heatmap_base64=_encode_image(heatmap_img),
            metadata={"shap_base_values": np.asarray(shap_values.base_values).tolist()},
        )

    def _lime(self, request: ExplainabilityRequest) -> ExplainabilityArtifact:
        try:
            from lime import lime_image
        except ImportError as exc:
            raise ImportError("Install lime to use LIME explanations.") from exc

        image, _, image_array = self._prepare_input(request.image_path)

        def predict_fn(batch: np.ndarray) -> np.ndarray:
            torch, _ = _require_torch()
            self.model.eval()
            batch_tensor = torch.tensor(batch.transpose(0, 3, 1, 2), dtype=torch.float32, device=self.device)
            outputs = self.model(batch_tensor)
            logits = _classification_logits(outputs)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.detach().cpu().numpy()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            (image_array * 255).astype(np.uint8),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
        )

        label = request.target_class if request.target_class is not None else int(explanation.top_labels[0])
        temp_image, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)

        output_root = _ensure_output_root(request.output_root, self.config)

        overlay = Image.fromarray(temp_image.astype(np.uint8))
        overlay_path = _save_image(overlay, output_root / f"lime_{label}.png")

        # Convert the binary superpixel mask to a PNG so the browser can display it.
        mask_img = _tensor_to_pil(mask.astype(np.float32))
        mask_path = _save_image(mask_img, output_root / "lime_mask.png")
        np.save(output_root / "lime_mask.npy", mask)

        log_step(self.logger, f"Saved LIME overlay to {overlay_path}.")

        return ExplainabilityArtifact(
            method="lime",
            task="classification",
            target_class=label,
            original_image_path=str(request.image_path),
            overlay_path=str(overlay_path),
            raw_path=str(mask_path),
            overlay_base64=_encode_image(overlay),
            raw_base64=_encode_image(mask_img),
            metadata={"top_labels": list(explanation.top_labels)},
        )