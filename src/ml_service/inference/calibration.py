"""Post-hoc temperature scaling for neural network calibration.

Reference: Guo et al. 2017 "On Calibration of Modern Neural Networks".
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Install torch to use TemperatureScaler.") from exc
    return torch, nn, F


class TemperatureScaler:
    """Single-parameter post-hoc calibration via temperature scaling.

    Divides logits by a learnable scalar T before softmax.
    T > 1  → softer (less confident) predictions.
    T < 1  → sharper (more confident) predictions.
    T = 1  → no change.

    Typical usage:
        scaler = TemperatureScaler()
        scaler.fit_from_arrays(logits_np, labels_np)
        calibrated = scaler.calibrate(logits_np)   # → (N, C) float32 probabilities
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        self.temperature: float = init_temperature

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_from_arrays(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "TemperatureScaler":
        """Find T that minimises NLL on (logits, labels).

        Args:
            logits: (N, C) raw pre-softmax logits as float32 numpy.
            labels: (N,) integer class indices as int64 numpy.
        """
        torch, nn, F = _require_torch()

        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temperature = nn.Parameter(torch.tensor([float(self.temperature)]))
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def closure() -> Any:
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_t / temperature.clamp(min=1e-4), labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(temperature.item())
        return self

    def fit(
        self,
        model: Any,
        records: list[Any],
        label_to_index: dict[str, int],
        preprocessing_config: Any,
        batch_size: int = 16,
        device: Any = None,
    ) -> "TemperatureScaler":
        """Collect logits from a model + record list then fit temperature.

        Args:
            model: trained model (or wrapper with .model attribute).
            records: list of DatasetRecord with .image_path and .label.
            label_to_index: mapping class name -> integer index.
            preprocessing_config: ImagePreprocessingConfig (augment=False).
            batch_size: how many samples per forward pass.
            device: torch device; None = auto-detect CPU/CUDA.
        """
        torch, nn, F = _require_torch()
        from ..data.preprocessing import image_to_array, preprocess_image

        target_model = getattr(model, "model", model)
        if device is None:
            device = next(target_model.parameters(), torch.tensor(0.0)).device
        target_model.eval()

        all_logits: list[Any] = []
        all_labels: list[int] = []

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            imgs: list[Any] = []
            lbs: list[int] = []
            for r in batch:
                img = preprocess_image(r.image_path, preprocessing_config, augment=False)
                arr = image_to_array(img, preprocessing_config)
                imgs.append(torch.tensor(arr, dtype=torch.float32))
                idx = label_to_index.get(getattr(r, "label", None), -1)  # type: ignore[arg-type]
                lbs.append(idx)

            if not imgs:
                continue
            batch_t = torch.stack(imgs).to(device)
            with torch.no_grad():
                outputs = target_model(batch_t)
                logits = outputs["classification"] if isinstance(outputs, dict) else outputs
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(lbs)

        if not all_logits:
            return self

        logits_np = np.concatenate(all_logits, axis=0).astype(np.float32)
        labels_np = np.array(all_labels, dtype=np.int64)
        valid = labels_np >= 0
        return self.fit_from_arrays(logits_np[valid], labels_np[valid])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: (N, C) or (C,) float32 raw logits.

        Returns:
            (N, C) or (C,) float32 probabilities.
        """
        torch, nn, F = _require_torch()
        t = max(self.temperature, 1e-6)
        logits_t = torch.tensor(logits, dtype=torch.float32)
        return F.softmax(logits_t / t, dim=-1).numpy()

    def calibrate_tensor(self, logits: Any) -> Any:
        """Apply temperature scaling to a torch tensor, returning a tensor."""
        t = max(self.temperature, 1e-6)
        import torch.nn.functional as F
        return F.softmax(logits / t, dim=-1)
