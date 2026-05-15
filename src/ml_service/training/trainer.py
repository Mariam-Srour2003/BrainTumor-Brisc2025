"""Training interface for models."""

from __future__ import annotations

import contextlib
import copy
import math
import random as _random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.config import ServiceConfig, get_settings
from ..core.runtime import get_logger, log_step, require_torch_device
from ..data.dataset import (
    DatasetRecord,
    JointDatasetRecord,
    discover_classification_samples,
    discover_joint_samples,
    discover_segmentation_pairs,
    discover_view_classification_samples,
)
from ..data.preprocessing import (
    ImagePreprocessingConfig,
    apply_augmentation_only,
    image_to_array,
    preprocess_image,
    preprocess_mask,
    preprocess_mask_augmented,
)
from ..evaluation.metrics import compute_analysis, compute_metrics
from ..utils import (
    format_checkpoint_run_date,
    load_checkpoint,
    sanitize_checkpoint_name,
    save_checkpoint,
    versioned_checkpoint_path,
)


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Install torch to run the trainer.") from exc
    return torch, F


def _label_to_index_map(class_names: tuple[str, ...]) -> dict[str, int]:
    return {name: index for index, name in enumerate(class_names)}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _dice_loss(logits: Any, targets: Any, smooth: float = 1.0) -> Any:
    """Binary Dice loss — handles foreground/background imbalance better than BCE alone."""
    import torch
    pred = torch.sigmoid(logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = targets.view(targets.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    return (1.0 - (2.0 * intersection + smooth) / (union + smooth)).mean()


def _tversky_loss(
    logits: Any,
    targets: Any,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> Any:
    """Tversky loss: α weights FP, β weights FN (β>0.5 penalises missed tumors more)."""
    import torch
    pred = torch.sigmoid(logits)
    pred_f = pred.view(pred.size(0), -1)
    tgt_f = targets.view(targets.size(0), -1)
    tp = (pred_f * tgt_f).sum(1)
    fp = (pred_f * (1.0 - tgt_f)).sum(1)
    fn = ((1.0 - pred_f) * tgt_f).sum(1)
    return (1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


def _focal_tversky_loss(
    logits: Any,
    targets: Any,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
) -> Any:
    """Focal Tversky: focuses on hard, small-foreground segmentation regions."""
    return _tversky_loss(logits, targets, alpha, beta) ** gamma


def _combined_seg_loss(logits: Any, targets: Any) -> Any:
    """0.4 BCE + 0.3 Dice + 0.3 Focal-Tversky — optimised for small tumor segmentation."""
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = _dice_loss(logits, targets)
    ft = _focal_tversky_loss(logits, targets)
    return 0.4 * bce + 0.3 * dice + 0.3 * ft


def _focal_loss(
    logits: Any,
    targets: Any,
    gamma: float = 2.0,
    weight: Any = None,
    label_smoothing: float = 0.1,
) -> Any:
    """Focal loss with label smoothing: down-weights easy examples, improves on hard ones."""
    import torch
    import torch.nn.functional as F
    # Label-smoothed CE for the actual loss magnitude
    ce_smooth = F.cross_entropy(
        logits, targets, weight=weight, label_smoothing=label_smoothing, reduction="none"
    )
    # Raw probability used only for the focal weighting factor (no grad needed)
    with torch.no_grad():
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
    return ((1.0 - pt) ** gamma * ce_smooth).mean()


def _compute_class_weights(
    records: list[Any], label_to_index: dict[str, int]
) -> list[float]:
    """Inverse-frequency class weights to counteract label imbalance."""
    counts: dict[int, int] = {}
    for r in records:
        label = getattr(r, "label", None)
        idx = label_to_index.get(label, -1)  # type: ignore[arg-type]
        if idx >= 0:
            counts[idx] = counts.get(idx, 0) + 1
    total = sum(counts.values()) or 1
    n = len(label_to_index)
    return [total / (n * max(counts.get(i, 1), 1)) for i in range(n)]


def _apply_cutmix(
    images: Any, labels: Any, alpha: float = 1.0
) -> tuple[Any, Any, Any, float]:
    """CutMix: paste a rectangular crop from a shuffled sample into each image."""
    import torch
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)
    cut_h = int(H * math.sqrt(1.0 - lam))
    cut_w = int(W * math.sqrt(1.0 - lam))
    cy, cx = _random.randint(0, max(H - 1, 1)), _random.randint(0, max(W - 1, 1))
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(max(H * W, 1))
    return mixed, labels, labels[idx], lam


def _r_drop_loss(logits1: Any, logits2: Any) -> Any:
    """Bidirectional KL divergence between two stochastic forward passes (R-Drop)."""
    import torch.nn.functional as F
    kl1 = F.kl_div(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1), reduction="batchmean")
    kl2 = F.kl_div(F.log_softmax(logits2, dim=1), F.softmax(logits1, dim=1), reduction="batchmean")
    return 0.5 * (kl1 + kl2)


def _apply_mixup(
    images: Any, labels: Any, alpha: float = 0.2
) -> tuple[Any, Any, Any, float]:
    """Returns (mixed_images, labels_a, labels_b, lambda)."""
    import torch
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def _extract_boundary(mask: Any, kernel_size: int = 3) -> Any:
    """Extract boundary pixels from a binary mask via morphological erosion."""
    import torch.nn.functional as _F
    pad = kernel_size // 2
    eroded = -_F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (mask - eroded).clamp(0.0, 1.0)


def _lr_scale(epoch: int, warmup: int, total: int, min_factor: float) -> float:
    """Linear warmup then cosine decay to min_factor × base_lr."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def _oversample_balanced(records: list[Any], label_to_index: dict[str, int]) -> list[Any]:
    """Oversample minority classes so every class appears equally in each epoch."""
    by_class: dict[int, list[Any]] = {}
    unlabeled: list[Any] = []
    for r in records:
        idx = label_to_index.get(getattr(r, "label", None), -1)  # type: ignore[arg-type]
        if idx >= 0:
            by_class.setdefault(idx, []).append(r)
        else:
            unlabeled.append(r)
    if not by_class:
        return records
    max_n = max(len(v) for v in by_class.values())
    balanced: list[Any] = []
    for samples in by_class.values():
        balanced.extend(_random.choices(samples, k=max_n))
    balanced.extend(unlabeled)
    _random.shuffle(balanced)
    return balanced


def _freeze_backbone(model: Any, frozen: bool) -> None:
    """Freeze or unfreeze all backbone (non-head) parameters."""
    head_kws = {
        "fc",
        "head",
        "classifier",
        "classification_head",
        "segmentation_head",
        "dec",
        "projection",
        "router",
        "fusion",
        "context",
        "gate",
    }
    for name, p in model.named_parameters():
        if not any(kw in name for kw in head_kws):
            p.requires_grad = not frozen


class _LookaheadOptimizer:
    """Lookahead wrapper: maintains 'slow' weights updated every k fast steps.

    Every k optimizer steps, interpolates slow weights toward fast weights
    by alpha, then resets fast weights to slow.  Consistently improves
    generalisation with no change to hyperparameter tuning of the base optimizer.
    """

    def __init__(self, base_optimizer: Any, k: int = 5, alpha: float = 0.5) -> None:
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow: list[list[Any]] = [
            [p.clone().detach() for p in pg["params"]]
            for pg in base_optimizer.param_groups
        ]

    def step(self, closure: Any = None) -> Any:
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for pg, slow_g in zip(self.base_optimizer.param_groups, self._slow):
                for fast_p, slow_p in zip(pg["params"], slow_g):
                    if fast_p.requires_grad:
                        slow_p.add_(fast_p.data - slow_p, alpha=self.alpha)
                        fast_p.data.copy_(slow_p)
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Any:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state: Any) -> None:
        self.base_optimizer.load_state_dict(state)

    @property
    def param_groups(self) -> Any:
        return self.base_optimizer.param_groups

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_optimizer, name)


class _EMAWrapper:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of all parameters updated as:
        shadow = decay * shadow + (1 - decay) * param
    The `applied` context manager temporarily swaps in EMA weights for
    validation/inference then restores originals on exit.
    """

    def __init__(self, model: Any, decay: float = 0.9998) -> None:
        self.decay = decay
        self.shadow: dict[str, Any] = {
            n: p.data.clone().float() for n, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model: Any) -> None:
        d = self.decay
        with contextlib.suppress(Exception):
            for name, param in model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    self.shadow[name].mul_(d).add_(param.data.float(), alpha=1.0 - d)

    @contextlib.contextmanager
    def applied(self, model: Any) -> Any:
        """Temporarily replace model weights with EMA weights."""
        original: dict[str, Any] = {
            n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow
        }
        try:
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.device))
            yield
        finally:
            for name, param in model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Tunable hyperparameters for multitask training."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    segmentation_loss_weight: float
    classification_loss_weight: float
    optimizer_name: str
    # Advanced training options (all have defaults for backward compat)
    warmup_epochs: int = 3
    min_lr_factor: float = 0.01
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    use_amp: bool = True
    clip_grad_norm: float = 1.0
    patience: int = 10
    val_split_ratio: float = 0.15
    backbone_lr_factor: float = 0.1
    focal_gamma: float = 2.0
    # Gradient accumulation — effective batch = batch_size × accumulation_steps
    accumulation_steps: int = 4
    # EMA — shadow copy of weights, used for validation and final model
    use_ema: bool = True
    ema_decay: float = 0.9998
    # SWA — average weights over the last swa_start_ratio fraction of epochs
    use_swa: bool = True
    swa_start_ratio: float = 0.75
    # Balanced oversampling — equalise class counts each epoch
    balanced_sampling: bool = True
    # Gradual unfreezing — backbone frozen for first freeze_epochs, then released
    gradual_unfreeze: bool = True
    freeze_epochs: int = 5
    # CutMix — alternated with Mixup per batch (each active with 50 % probability)
    use_cutmix: bool = True
    # R-Drop — bidirectional KL between two dropout forward passes on clean batches
    use_r_drop: bool = True
    r_drop_alpha: float = 0.3
    # Lookahead — wraps base optimizer, syncs slow weights every lookahead_k steps
    use_lookahead: bool = True
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    # Tversky loss weights (β > α penalises false negatives more — good for tumors)
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7

    @classmethod
    def from_service_config(cls, config: ServiceConfig) -> "TrainingConfig":
        return cls(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            segmentation_loss_weight=config.segmentation_loss_weight,
            classification_loss_weight=config.classification_loss_weight,
            optimizer_name=config.optimizer_name,
        )


@dataclass(frozen=True)
class TrainingResult:
    """Minimal training output container."""

    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trainer with:
    - Focal loss + label smoothing (classification)
    - Dice + BCE combined loss (segmentation)
    - Mixup augmentation
    - Cosine-annealing LR with linear warmup
    - Automatic mixed precision (AMP, CUDA only)
    - Gradient clipping
    - Differential learning rates (backbone vs head)
    - In-memory validation split + early stopping
    - Paired geometric augmentation for image/mask
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        config: ServiceConfig | None = None,
        training_config: TrainingConfig | None = None,
        view_filter: str | None = None,
        processed_root: "Path | None" = None,
        task: str = "classification",
    ) -> None:
        self.model = model
        self.config = config or get_settings()
        self.training_config = training_config or TrainingConfig.from_service_config(self.config)
        self.view_filter = view_filter
        self.processed_root = processed_root
        self.task = task
        self.logger = get_logger("training.trainer")
        self.training_run_date = format_checkpoint_run_date()
        self.torch, self.device = self._init_device()
        log_step(self.logger, f"Trainer initialized on device={self.device}.")

    def _get_classification_records(self, split: str) -> list[DatasetRecord]:
        if self.task == "view_classification":
            return discover_view_classification_samples(split, processed_root=self.processed_root)
        return discover_classification_samples(split, processed_root=self.processed_root, view_filter=self.view_filter)

    def _get_class_names(self) -> tuple[str, ...]:
        if self.task == "view_classification":
            return self.config.view_class_names
        return self.config.class_names

    def _init_device(self) -> tuple[Any | None, Any | None]:
        try:
            torch, device = require_torch_device()
        except ImportError:
            return None, None

        if self.model is not None:
            target = getattr(self.model, "model", self.model)
            if hasattr(target, "to"):
                target.to(device)
        return torch, device

    def _use_amp(self) -> bool:
        return (
            self.training_config.use_amp
            and self.device is not None
            and str(self.device).startswith("cuda")
        )

    # ------------------------------------------------------------------
    # Optimizer & scheduler helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self, model: Any) -> Any:
        """AdamW/Adam/SGD with differential LR: backbone at backbone_lr_factor × head LR."""
        torch, _ = _require_torch()
        head_kws = {"fc", "head", "classifier", "classification_head", "segmentation_head", "dec"}
        backbone_params: list[Any] = []
        head_params: list[Any] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in head_kws):
                head_params.append(p)
            else:
                backbone_params.append(p)

        backbone_lr = self.training_config.learning_rate * self.training_config.backbone_lr_factor
        head_lr = self.training_config.learning_rate
        groups: list[dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr, "base_lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "base_lr": head_lr})
        if not groups:
            groups = [{"params": list(model.parameters()), "lr": head_lr, "base_lr": head_lr}]

        wd = self.training_config.weight_decay
        name = self.training_config.optimizer_name.lower()
        if name == "adam":
            base_opt = torch.optim.Adam(groups, weight_decay=wd)
        elif name == "sgd":
            base_opt = torch.optim.SGD(groups, weight_decay=wd, momentum=0.9)
        else:
            base_opt = torch.optim.AdamW(groups, weight_decay=wd)

        if self.training_config.use_lookahead:
            return _LookaheadOptimizer(
                base_opt,
                k=self.training_config.lookahead_k,
                alpha=self.training_config.lookahead_alpha,
            )
        return base_opt

    def _update_lr(self, optimizer: Any, epoch: int, total_epochs: int) -> None:
        scale = _lr_scale(epoch, self.training_config.warmup_epochs, total_epochs, self.training_config.min_lr_factor)
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("base_lr", self.training_config.learning_rate) * scale

    # ------------------------------------------------------------------
    # Public fit / evaluate entry points
    # ------------------------------------------------------------------

    def fit(self, data: Any | None = None) -> TrainingResult:
        if data is None:
            return self.fit_joint()
        return TrainingResult(status="completed", metadata={"note": "Custom data path not implemented."})

    # ----- classification -----

    def fit_classification(
        self,
        split: str = "train",
        *,
        checkpoint_name: str | None = None,
        resume: bool = True,
        progress_callback: Any | None = None,
    ) -> TrainingResult:
        torch, F = _require_torch()
        tc = self.training_config
        log_step(self.logger, f"Classification training: split={split}, epochs={tc.epochs}, batch={tc.batch_size}.")

        if self.model is None:
            return TrainingResult(status="failed", metadata={"reason": "No model provided."})

        all_records = self._get_classification_records(split)
        if not all_records:
            return TrainingResult(status="failed", metadata={"reason": f"No classification samples for split={split}."})

        records = list(all_records)
        _random.shuffle(records)
        val_n = max(1, int(len(records) * tc.val_split_ratio))
        val_records = records[:val_n]
        base_train = records[val_n:]
        log_step(self.logger, f"Classification: {len(base_train)} base train, {len(val_records)} val.")

        log_step(self.logger, f"🗂️ Pre-loading {len(all_records)} images into RAM (CLAHE+resize cached, runs once)...")
        _base_prep = self._make_preprocessing(augment=False)
        _img_cache = self._build_image_cache(all_records, _base_prep)
        log_step(self.logger, f"✅ Image cache ready: {len(_img_cache)} images in RAM.")

        model = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(model, "to") and self.device is not None:
            model.to(self.device, non_blocking=True)

        # Phase-1: freeze backbone, only train head
        if tc.gradual_unfreeze:
            _freeze_backbone(model, frozen=True)
            log_step(self.logger, f"Backbone frozen for first {tc.freeze_epochs} epochs.")

        optimizer = self._build_optimizer(model)
        ckpt_path = self._training_checkpoint_path(checkpoint_name or "classification_model", split)
        start_epoch, resume_meta = (
            self._try_resume_training_state(model, optimizer, ckpt_path) if resume else (0, {})
        )

        preprocessing = self._make_preprocessing(augment=True)
        val_prep = self._make_preprocessing(augment=False)
        label_to_index = _label_to_index_map(self._get_class_names())

        raw_weights = _compute_class_weights(base_train, label_to_index)
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
        if self.device is not None:
            class_weights = class_weights.to(self.device, non_blocking=True)

        use_amp = self._use_amp()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # EMA
        ema = _EMAWrapper(model, decay=tc.ema_decay) if tc.use_ema else None

        # SWA — AveragedModel wraps the model; SWALR is created lazily at swa_start
        # to always use the current (unfrozen) optimizer and avoid Lookahead incompatibility.
        swa_model: Any = None
        swa_scheduler: Any = None
        swa_start = int(tc.epochs * tc.swa_start_ratio)
        if tc.use_swa:
            try:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
                log_step(self.logger, f"SWA enabled — will average from epoch {swa_start}.")
            except Exception as exc:
                log_step(self.logger, f"SWA setup failed ({exc}); disabled.")
                swa_model = None

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_ctr = 0
        last_ckpt: dict[str, str] | None = None
        epoch_metrics: list[dict[str, float]] = []
        avg_train_loss = 0.0
        avg_val_loss = float("inf")
        swa_active = False

        if start_epoch >= tc.epochs:
            log_step(self.logger, f"Already at epoch {start_epoch}; skipping.")

        for epoch in range(start_epoch, tc.epochs):
            # Phase-2: unfreeze backbone after freeze_epochs
            if tc.gradual_unfreeze and epoch == tc.freeze_epochs:
                _freeze_backbone(model, frozen=False)
                optimizer = self._build_optimizer(model)
                scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
                swa_scheduler = None  # will be re-created lazily with the new optimizer
                log_step(self.logger, "Backbone unfrozen — full network now training.")

            # Standard cosine-warmup LR (skip if SWA has taken over)
            if not swa_active:
                self._update_lr(optimizer, epoch, tc.epochs)

            model.train()
            # Balanced oversampling each epoch
            train_records = _oversample_balanced(base_train, label_to_index) if tc.balanced_sampling else list(base_train)
            _random.shuffle(train_records)

            n_samples_epoch = len(train_records)
            n_batches_total = max(1, math.ceil(n_samples_epoch / tc.batch_size))
            log_interval = max(1, n_batches_total // 10)
            log_step(self.logger, f"🔄 Epoch {epoch+1}/{tc.epochs} — {n_samples_epoch} samples, ~{n_batches_total} batches | lr={optimizer.param_groups[-1]['lr']:.2e}")

            epoch_loss, n_batches = 0.0, 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(self._batch_records(train_records, tc.batch_size)):
                images, labels = self._prepare_classification_batch(batch, preprocessing, label_to_index, image_cache=_img_cache)
                if self.device is not None:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                # Randomly pick CutMix OR Mixup (never both in the same batch)
                aug_mode: str | None = None
                if tc.use_cutmix and tc.mixup_alpha > 0 and _random.random() < 0.5:
                    aug_mode = "cutmix" if _random.random() < 0.5 else "mixup"
                elif tc.mixup_alpha > 0 and _random.random() < 0.5:
                    aug_mode = "mixup"

                if aug_mode == "cutmix":
                    images, la, lb, lam = _apply_cutmix(images, labels, tc.mixup_alpha)
                elif aug_mode == "mixup":
                    images, la, lb, lam = _apply_mixup(images, labels, tc.mixup_alpha)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    if aug_mode is not None:
                        outputs = model(images)
                        logits = outputs["classification"] if isinstance(outputs, dict) else outputs
                        raw_loss = (
                            lam * _focal_loss(logits, la, tc.focal_gamma, class_weights, tc.label_smoothing)
                            + (1.0 - lam) * _focal_loss(logits, lb, tc.focal_gamma, class_weights, tc.label_smoothing)
                        )
                    else:
                        outputs = model(images)
                        logits = outputs["classification"] if isinstance(outputs, dict) else outputs
                        raw_loss = _focal_loss(logits, labels, tc.focal_gamma, class_weights, tc.label_smoothing)
                        # R-Drop: second forward with different dropout, add consistency loss
                        if tc.use_r_drop and model.training and _random.random() < 0.5:
                            outputs2 = model(images)
                            logits2 = outputs2["classification"] if isinstance(outputs2, dict) else outputs2
                            raw_loss = raw_loss + tc.r_drop_alpha * _r_drop_loss(logits, logits2)
                    loss = raw_loss / tc.accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % tc.accumulation_steps == 0:
                    if tc.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

                epoch_loss += float(raw_loss.item())
                n_batches += 1
                if batch_idx == 0 or (batch_idx + 1) % log_interval == 0:
                    log_step(self.logger, f"   📦 Batch {batch_idx+1}/{n_batches_total} | loss={float(raw_loss.item()):.4f}")

            # Flush remaining accumulated gradients
            if n_batches % tc.accumulation_steps != 0:
                if tc.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            avg_train_loss = epoch_loss / max(1, n_batches)

            # SWA accumulation — create SWALR lazily so it always uses the current optimizer
            if swa_model is not None and epoch >= swa_start:
                if swa_scheduler is None:
                    _base_opt = optimizer.base_optimizer if isinstance(optimizer, _LookaheadOptimizer) else optimizer
                    with contextlib.suppress(Exception):
                        swa_scheduler = torch.optim.swa_utils.SWALR(
                            _base_opt, swa_lr=tc.learning_rate * tc.min_lr_factor, anneal_epochs=5,
                        )
                swa_model.update_parameters(model)
                if swa_scheduler is not None:
                    swa_scheduler.step()
                swa_active = True

            # Validation — use EMA weights if available
            model.eval()
            val_loss, val_batches = 0.0, 0
            eval_ctx = ema.applied(model) if ema is not None else contextlib.nullcontext()
            with eval_ctx, torch.no_grad():
                for batch in self._batch_records(val_records, tc.batch_size):
                    images, labels = self._prepare_classification_batch(batch, val_prep, label_to_index, image_cache=_img_cache)
                    if self.device is not None:
                        images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(images)
                        logits = outputs["classification"] if isinstance(outputs, dict) else outputs
                        v_loss = _focal_loss(logits, labels, tc.focal_gamma, class_weights, tc.label_smoothing)
                    val_loss += float(v_loss.item())
                    val_batches += 1
            avg_val_loss = val_loss / max(1, val_batches)

            is_best = avg_val_loss < best_val_loss
            epoch_metrics.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            log_step(self.logger, (
                f"{'✅' if is_best else '📊'} Epoch {epoch+1}/{tc.epochs}: "
                f"train={avg_train_loss:.4f} | val={avg_val_loss:.4f}"
                + (" 🎯 new best!" if is_best else f" (patience {patience_ctr+1}/{tc.patience})")
            ))
            if progress_callback is not None:
                progress_callback({"epoch": epoch + 1, "total": tc.epochs, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "type": "epoch"})

            last_ckpt = self._save_training_checkpoint(
                checkpoint_path=ckpt_path, model=model, optimizer=optimizer, epoch=epoch,
                task=self.task, split=split,
                metrics={"loss": avg_train_loss, "val_loss": avg_val_loss},
                extra_metadata={"best_val_loss": best_val_loss},
            )

            if is_best:
                best_val_loss = avg_val_loss
                if ema is not None:
                    with ema.applied(model):
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= tc.patience:
                    log_step(self.logger, f"⏹️ Early stopping at epoch {epoch+1} (patience={tc.patience} exhausted).")
                    break

        # Apply SWA final weights (update BN stats via a forward pass)
        if swa_model is not None and swa_active:
            log_step(self.logger, "Updating BN stats for SWA model.")
            swa_model.eval()
            _bn_prep = self._make_preprocessing(augment=False)
            with torch.no_grad():
                for batch in self._batch_records(base_train[:min(len(base_train), 256)], tc.batch_size):
                    imgs, _ = self._prepare_classification_batch(batch, _bn_prep, label_to_index, image_cache=_img_cache)
                    if self.device is not None:
                        imgs = imgs.to(self.device, non_blocking=True)
                    swa_model(imgs)
            model.load_state_dict(swa_model.module.state_dict())
            log_step(self.logger, "SWA weights applied to model.")
        elif best_state is not None:
            model.load_state_dict(best_state)
            log_step(self.logger, f"Restored best model (val_loss={best_val_loss:.4f}).")

        if progress_callback is not None:
            progress_callback({"type": "complete", "best_val_loss": best_val_loss})

        return TrainingResult(
            status="completed",
            metrics={"loss": avg_train_loss, "val_loss": best_val_loss},
            metadata={
                "split": split,
                "best_val_loss": best_val_loss,
                "epoch_metrics": epoch_metrics,
                "checkpoint": last_ckpt,
                "swa_applied": swa_active,
                "ema_used": ema is not None,
                **resume_meta,
            },
        )

    def evaluate_classification(self, split: str = "test") -> TrainingResult:
        torch, _ = _require_torch()
        log_step(self.logger, f"Classification evaluation: split={split}.")

        if self.model is None:
            return TrainingResult(status="failed", metadata={"reason": "No model provided."})

        records = self._get_classification_records(split)
        if not records:
            return TrainingResult(status="failed", metadata={"reason": f"No classification samples for split={split}."})

        model = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(model, "to") and self.device is not None:
            model.to(self.device, non_blocking=True)
        model.eval()

        preprocessing = self._make_preprocessing(augment=False)
        label_to_index = _label_to_index_map(self._get_class_names())
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        with torch.no_grad():
            for batch in self._batch_records(records, self.training_config.batch_size):
                images, labels = self._prepare_classification_batch(batch, preprocessing, label_to_index)
                if self.device is not None:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = model(images)
                logits = outputs["classification"] if isinstance(outputs, dict) else outputs
                preds.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                targets.append(labels.detach().cpu().numpy())

        predictions = np.concatenate(preds, axis=0)
        target_arr = np.concatenate(targets, axis=0)
        metrics = compute_metrics(predictions=predictions, targets=target_arr)
        analysis = compute_analysis(predictions=predictions, targets=target_arr)
        return TrainingResult(status="completed", metrics=metrics, metadata={"split": split, "analysis": analysis})

    # ----- segmentation -----

    def fit_segmentation(
        self,
        split: str = "train",
        *,
        checkpoint_name: str | None = None,
        resume: bool = True,
        progress_callback: Any | None = None,
    ) -> TrainingResult:
        torch, F = _require_torch()
        tc = self.training_config
        log_step(self.logger, f"Segmentation training: split={split}, epochs={tc.epochs}.")

        if self.model is None:
            return TrainingResult(status="failed", metadata={"reason": "No model provided."})

        all_records = [r for r in discover_segmentation_pairs(split, processed_root=self.processed_root, view_filter=self.view_filter) if r.mask_path is not None]
        if not all_records:
            return TrainingResult(status="failed", metadata={"reason": f"No segmentation samples for split={split}."})

        records = list(all_records)
        _random.shuffle(records)
        val_n = max(1, int(len(records) * tc.val_split_ratio))
        val_records = records[:val_n]
        base_train = records[val_n:]
        log_step(self.logger, f"Segmentation split: {len(base_train)} train, {len(val_records)} val.")

        log_step(self.logger, f"🗂️ Pre-loading {len(all_records)} images and masks into RAM (CLAHE+resize cached, runs once)...")
        _base_prep = self._make_preprocessing(augment=False)
        _img_cache = self._build_image_cache(all_records, _base_prep)
        _mask_cache = self._build_mask_cache(all_records, _base_prep)
        log_step(self.logger, f"✅ Cache ready: {len(_img_cache)} images, {len(_mask_cache)} masks in RAM.")

        model = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(model, "to") and self.device is not None:
            model.to(self.device, non_blocking=True)

        if tc.gradual_unfreeze:
            _freeze_backbone(model, frozen=True)
            log_step(self.logger, f"Backbone frozen for first {tc.freeze_epochs} epochs.")

        optimizer = self._build_optimizer(model)
        ckpt_path = self._training_checkpoint_path(checkpoint_name or "segmentation_model", split)
        start_epoch, resume_meta = (
            self._try_resume_training_state(model, optimizer, ckpt_path) if resume else (0, {})
        )

        use_amp = self._use_amp()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        ema = _EMAWrapper(model, decay=tc.ema_decay) if tc.use_ema else None

        swa_model: Any = None
        swa_scheduler: Any = None
        swa_start = int(tc.epochs * tc.swa_start_ratio)
        if tc.use_swa:
            try:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
                log_step(self.logger, f"SWA enabled — will average from epoch {swa_start}.")
            except Exception as exc:
                log_step(self.logger, f"SWA setup failed ({exc}); disabled.")
                swa_model = None

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_ctr = 0
        last_ckpt: dict[str, str] | None = None
        epoch_metrics: list[dict[str, float]] = []
        avg_train_loss = 0.0
        avg_val_loss = float("inf")
        swa_active = False

        if start_epoch >= tc.epochs:
            log_step(self.logger, f"Already at epoch {start_epoch}; skipping segmentation training.")

        for epoch in range(start_epoch, tc.epochs):
            if tc.gradual_unfreeze and epoch == tc.freeze_epochs:
                _freeze_backbone(model, frozen=False)
                optimizer = self._build_optimizer(model)
                scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
                swa_scheduler = None
                log_step(self.logger, "Backbone unfrozen — full network now training.")

            if not swa_active:
                self._update_lr(optimizer, epoch, tc.epochs)

            model.train()
            train_records = list(base_train)
            _random.shuffle(train_records)

            n_samples_epoch = len(train_records)
            n_batches_total = max(1, math.ceil(n_samples_epoch / tc.batch_size))
            log_interval = max(1, n_batches_total // 10)
            log_step(self.logger, f"🔄 Epoch {epoch+1}/{tc.epochs} — {n_samples_epoch} samples, ~{n_batches_total} batches | lr={optimizer.param_groups[-1]['lr']:.2e}")

            epoch_loss, n_batches = 0.0, 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(self._batch_records(train_records, tc.batch_size)):
                images, masks = self._prepare_segmentation_batch(batch, augment=True, image_cache=_img_cache, mask_cache=_mask_cache)
                if self.device is not None:
                    images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    logits = outputs["segmentation"] if isinstance(outputs, dict) else outputs
                    raw_loss = _combined_seg_loss(logits, masks)
                    loss = raw_loss / tc.accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % tc.accumulation_steps == 0:
                    if tc.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

                epoch_loss += float(raw_loss.item())
                n_batches += 1
                if batch_idx == 0 or (batch_idx + 1) % log_interval == 0:
                    log_step(self.logger, f"   📦 Batch {batch_idx+1}/{n_batches_total} | loss={float(raw_loss.item()):.4f}")

            if n_batches % tc.accumulation_steps != 0:
                if tc.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            avg_train_loss = epoch_loss / max(1, n_batches)

            if swa_model is not None and epoch >= swa_start:
                if swa_scheduler is None:
                    _base_opt = optimizer.base_optimizer if isinstance(optimizer, _LookaheadOptimizer) else optimizer
                    with contextlib.suppress(Exception):
                        swa_scheduler = torch.optim.swa_utils.SWALR(
                            _base_opt, swa_lr=tc.learning_rate * tc.min_lr_factor, anneal_epochs=5,
                        )
                swa_model.update_parameters(model)
                if swa_scheduler is not None:
                    swa_scheduler.step()
                swa_active = True

            model.eval()
            val_loss, val_batches = 0.0, 0
            eval_ctx = ema.applied(model) if ema is not None else contextlib.nullcontext()
            with eval_ctx, torch.no_grad():
                for batch in self._batch_records(val_records, tc.batch_size):
                    images, masks = self._prepare_segmentation_batch(batch, augment=False, image_cache=_img_cache, mask_cache=_mask_cache)
                    if self.device is not None:
                        images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(images)
                        logits = outputs["segmentation"] if isinstance(outputs, dict) else outputs
                        v_loss = _combined_seg_loss(logits, masks)
                    val_loss += float(v_loss.item())
                    val_batches += 1
            avg_val_loss = val_loss / max(1, val_batches)

            is_best = avg_val_loss < best_val_loss
            epoch_metrics.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            log_step(self.logger, (
                f"{'✅' if is_best else '📊'} Epoch {epoch+1}/{tc.epochs}: "
                f"train={avg_train_loss:.4f} | val={avg_val_loss:.4f}"
                + (" 🎯 new best!" if is_best else f" (patience {patience_ctr+1}/{tc.patience})")
            ))
            if progress_callback is not None:
                progress_callback({"epoch": epoch + 1, "total": tc.epochs, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "type": "epoch"})

            last_ckpt = self._save_training_checkpoint(
                checkpoint_path=ckpt_path, model=model, optimizer=optimizer, epoch=epoch,
                task="segmentation", split=split,
                metrics={"loss": avg_train_loss, "val_loss": avg_val_loss},
                extra_metadata={"best_val_loss": best_val_loss},
            )

            if is_best:
                best_val_loss = avg_val_loss
                if ema is not None:
                    with ema.applied(model):
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= tc.patience:
                    log_step(self.logger, f"⏹️ Early stopping at epoch {epoch+1} (patience={tc.patience} exhausted).")
                    break

        if swa_model is not None and swa_active:
            log_step(self.logger, "Updating BN stats for SWA model.")
            swa_model.eval()
            with torch.no_grad():
                for batch in self._batch_records(base_train[:min(len(base_train), 256)], tc.batch_size):
                    imgs, _ = self._prepare_segmentation_batch(batch, augment=False, image_cache=_img_cache, mask_cache=_mask_cache)
                    if self.device is not None:
                        imgs = imgs.to(self.device, non_blocking=True)
                    swa_model(imgs)
            model.load_state_dict(swa_model.module.state_dict())
            log_step(self.logger, "SWA weights applied to segmentation model.")
        elif best_state is not None:
            model.load_state_dict(best_state)
            log_step(self.logger, f"Restored best segmentation model (val_loss={best_val_loss:.4f}).")

        if progress_callback is not None:
            progress_callback({"type": "complete", "best_val_loss": best_val_loss})

        return TrainingResult(
            status="completed",
            metrics={"loss": avg_train_loss, "val_loss": best_val_loss},
            metadata={
                "split": split,
                "best_val_loss": best_val_loss,
                "epoch_metrics": epoch_metrics,
                "checkpoint": last_ckpt,
                "swa_applied": swa_active,
                "ema_used": ema is not None,
                **resume_meta,
            },
        )

    def evaluate_segmentation(self, split: str = "test") -> TrainingResult:
        torch, _ = _require_torch()
        log_step(self.logger, f"Segmentation evaluation: split={split}.")

        if self.model is None:
            return TrainingResult(status="failed", metadata={"reason": "No model provided."})

        records = [r for r in discover_segmentation_pairs(split, processed_root=self.processed_root, view_filter=self.view_filter) if r.mask_path is not None]
        if not records:
            return TrainingResult(status="failed", metadata={"reason": f"No segmentation samples for split={split}."})

        model = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(model, "to") and self.device is not None:
            model.to(self.device, non_blocking=True)
        model.eval()

        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        with torch.no_grad():
            for batch in self._batch_records(records, self.training_config.batch_size):
                images, masks = self._prepare_segmentation_batch(batch, augment=False)
                if self.device is not None:
                    images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
                outputs = model(images)
                logits = outputs["segmentation"] if isinstance(outputs, dict) else outputs
                preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                targets.append(masks.detach().cpu().numpy())

        predictions = np.concatenate(preds, axis=0)
        target_arr = np.concatenate(targets, axis=0)
        metrics = compute_metrics(predictions=predictions, targets=target_arr)
        return TrainingResult(status="completed", metrics=metrics, metadata={"split": split})

    # ----- joint -----

    def fit_joint(
        self,
        split: str = "train",
        *,
        checkpoint_name: str | None = None,
        resume: bool = True,
        progress_callback: Any | None = None,
    ) -> TrainingResult:
        torch, F = _require_torch()
        tc = self.training_config
        log_step(self.logger, f"Joint training: split={split}, epochs={tc.epochs}.")

        if self.model is None:
            return TrainingResult(status="failed", metadata={"reason": "No model provided."})

        all_records = discover_joint_samples(split, processed_root=self.processed_root, view_filter=self.view_filter)
        if not all_records:
            return TrainingResult(status="failed", metadata={"reason": f"No joint samples for split={split}."})

        records = list(all_records)
        _random.shuffle(records)
        val_n = max(1, int(len(records) * tc.val_split_ratio))
        val_records = records[:val_n]
        base_train = records[val_n:]
        log_step(self.logger, f"Joint split: {len(base_train)} train, {len(val_records)} val.")

        log_step(self.logger, f"🗂️ Pre-loading {len(all_records)} images and masks into RAM (CLAHE+resize cached, runs once)...")
        _base_prep = self._make_preprocessing(augment=False)
        _img_cache = self._build_image_cache(all_records, _base_prep)
        _mask_cache = self._build_mask_cache(all_records, _base_prep)
        log_step(self.logger, f"✅ Cache ready: {len(_img_cache)} images, {len(_mask_cache)} masks in RAM.")

        model = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(model, "to") and self.device is not None:
            model.to(self.device, non_blocking=True)

        if tc.gradual_unfreeze:
            _freeze_backbone(model, frozen=True)
            log_step(self.logger, f"Backbone frozen for first {tc.freeze_epochs} epochs.")

        optimizer = self._build_optimizer(model)
        ckpt_path = self._training_checkpoint_path(checkpoint_name or getattr(self.config, "joint_model_name", "hybrid.adpt_net"), split)
        start_epoch, resume_meta = (
            self._try_resume_training_state(model, optimizer, ckpt_path) if resume else (0, {})
        )

        label_to_index = _label_to_index_map(self.config.joint_class_names)
        raw_weights = _compute_class_weights(base_train, label_to_index)
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
        if self.device is not None:
            class_weights = class_weights.to(self.device, non_blocking=True)

        use_amp = self._use_amp()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        ema = _EMAWrapper(model, decay=tc.ema_decay) if tc.use_ema else None

        swa_model: Any = None
        swa_scheduler: Any = None
        swa_start = int(tc.epochs * tc.swa_start_ratio)
        if tc.use_swa:
            try:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
                log_step(self.logger, f"SWA enabled — will average from epoch {swa_start}.")
            except Exception as exc:
                log_step(self.logger, f"SWA setup failed ({exc}); disabled.")
                swa_model = None

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_ctr = 0
        last_ckpt: dict[str, str] | None = None
        epoch_metrics: list[dict[str, float]] = []
        avg_train_loss = avg_seg_loss = avg_cls_loss = 0.0
        avg_val_loss = float("inf")
        swa_active = False

        if start_epoch >= tc.epochs:
            log_step(self.logger, f"Already at epoch {start_epoch}; skipping joint training.")

        # Must match _freeze_backbone's keyword set exactly — these params stay trainable
        # during the frozen phase and are already registered in the optimizer.  Only params
        # whose names contain NONE of these keywords were actually frozen and need to be added.
        _freeze_head_kws = {
            "fc", "head", "classifier", "classification_head", "segmentation_head",
            "dec", "projection", "router", "fusion", "context", "gate",
        }

        for epoch in range(start_epoch, tc.epochs):
            if tc.gradual_unfreeze and epoch == tc.freeze_epochs:
                _freeze_backbone(model, frozen=False)
                # Add only the newly-unfrozen backbone params (those absent from the optimizer).
                # Using _freeze_head_kws (= _freeze_backbone's set) avoids adding params that
                # were already trainable and present in an existing optimizer group.
                backbone_params = [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and not any(kw in n for kw in _freeze_head_kws)
                ]
                if backbone_params:
                    current_scale = _lr_scale(epoch, tc.warmup_epochs, tc.epochs, tc.min_lr_factor)
                    new_pg = {
                        "params": backbone_params,
                        "lr": tc.learning_rate * tc.backbone_lr_factor * current_scale,
                        "base_lr": tc.learning_rate * tc.backbone_lr_factor,
                    }
                    base_opt = optimizer.base_optimizer if isinstance(optimizer, _LookaheadOptimizer) else optimizer
                    base_opt.add_param_group(new_pg)
                    if isinstance(optimizer, _LookaheadOptimizer):
                        optimizer._slow.append([p.clone().detach() for p in backbone_params])
                swa_scheduler = None
                log_step(self.logger, "Backbone unfrozen — backbone params added to optimizer, head state preserved.")

            if not swa_active:
                self._update_lr(optimizer, epoch, tc.epochs)

            model.train()
            train_records = _oversample_balanced(base_train, label_to_index) if tc.balanced_sampling else list(base_train)
            _random.shuffle(train_records)

            n_samples_epoch = len(train_records)
            n_batches_total = max(1, math.ceil(n_samples_epoch / tc.batch_size))
            log_interval = max(1, n_batches_total // 10)
            log_step(self.logger, f"🔄 Epoch {epoch+1}/{tc.epochs} — {n_samples_epoch} samples, ~{n_batches_total} batches | lr={optimizer.param_groups[-1]['lr']:.2e}")

            e_total = e_seg = e_cls = 0.0
            n_batches = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(self._batch_records(train_records, tc.batch_size)):
                images, masks, labels = self._prepare_batch(batch, label_to_index, augment=True, image_cache=_img_cache, mask_cache=_mask_cache)
                if self.device is not None:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                # CutMix / Mixup selection
                aug_mode_j: str | None = None
                if tc.use_cutmix and tc.mixup_alpha > 0 and _random.random() < 0.5:
                    aug_mode_j = "cutmix" if _random.random() < 0.5 else "mixup"
                elif tc.mixup_alpha > 0 and _random.random() < 0.5:
                    aug_mode_j = "mixup"

                if aug_mode_j == "cutmix":
                    images, la, lb, lam = _apply_cutmix(images, labels, tc.mixup_alpha)
                elif aug_mode_j == "mixup":
                    images, la, lb, lam = _apply_mixup(images, labels, tc.mixup_alpha)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    seg_loss = _combined_seg_loss(outputs["segmentation"], masks)
                    if aug_mode_j is not None:
                        cls_loss = (
                            lam * F.cross_entropy(outputs["classification"], la, weight=class_weights, label_smoothing=tc.label_smoothing)
                            + (1.0 - lam) * F.cross_entropy(outputs["classification"], lb, weight=class_weights, label_smoothing=tc.label_smoothing)
                        )
                    else:
                        cls_loss = F.cross_entropy(outputs["classification"], labels, weight=class_weights, label_smoothing=tc.label_smoothing)
                        if tc.use_r_drop and model.training and _random.random() < 0.4:
                            outputs2 = model(images)
                            cls2 = outputs2["classification"]
                            cls_loss = cls_loss + tc.r_drop_alpha * _r_drop_loss(outputs["classification"], cls2)
                    # ── Uncertainty-aware task weighting (Kendall & Gal 2018) ──
                    # If the model exposes log_var_seg / log_var_cls parameters,
                    # use learned weighting; otherwise fall back to fixed config weights.
                    if hasattr(model, "log_var_seg") and hasattr(model, "log_var_cls"):
                        seg_w = torch.exp(-model.log_var_seg)
                        cls_w = torch.exp(-model.log_var_cls)
                        raw_loss = (
                            seg_w * seg_loss + cls_w * cls_loss
                            + model.log_var_seg + model.log_var_cls
                        )
                    else:
                        raw_loss = tc.segmentation_loss_weight * seg_loss + tc.classification_loss_weight * cls_loss
                    # ── Auxiliary losses (TGD-ADPT-Net initial predictions) ──
                    if "initial_segmentation" in outputs:
                        raw_loss = raw_loss + 0.4 * tc.segmentation_loss_weight * _combined_seg_loss(outputs["initial_segmentation"], masks)
                    if "initial_classification" in outputs:
                        if aug_mode_j is not None:
                            aux_cls = (
                                lam * F.cross_entropy(outputs["initial_classification"], la, weight=class_weights, label_smoothing=tc.label_smoothing)
                                + (1.0 - lam) * F.cross_entropy(outputs["initial_classification"], lb, weight=class_weights, label_smoothing=tc.label_smoothing)
                            )
                        else:
                            aux_cls = F.cross_entropy(outputs["initial_classification"], labels, weight=class_weights, label_smoothing=tc.label_smoothing)
                        raw_loss = raw_loss + 0.4 * tc.classification_loss_weight * aux_cls
                    # ── Cross-task consistency ─────────────────────────────
                    # Predicted mask presence must agree with the class label:
                    # no_tumor → mask should be blank; tumor → mask non-zero.
                    # Use raw logits + BCEWithLogits (AMP-safe; sigmoid is monotone
                    # so max(logits) ↔ max(sigmoid(logits))).
                    _no_tumor_idx = label_to_index.get("no_tumor", -1)
                    if _no_tumor_idx >= 0:
                        mask_max_logit = outputs["segmentation"].flatten(1).max(dim=1).values
                        if aug_mode_j is not None:
                            consistency_loss = (
                                lam * F.binary_cross_entropy_with_logits(mask_max_logit, (la != _no_tumor_idx).float())
                                + (1.0 - lam) * F.binary_cross_entropy_with_logits(mask_max_logit, (lb != _no_tumor_idx).float())
                            )
                        else:
                            consistency_loss = F.binary_cross_entropy_with_logits(
                                mask_max_logit, (labels != _no_tumor_idx).float()
                            )
                        raw_loss = raw_loss + 0.25 * consistency_loss
                    # ── Boundary supervision ───────────────────────────────
                    # Penalises contour inaccuracy directly → lowers Hausdorff.
                    if "boundary" in outputs:
                        raw_loss = raw_loss + 0.2 * F.binary_cross_entropy_with_logits(
                            outputs["boundary"], _extract_boundary(masks)
                        )
                    loss = raw_loss / tc.accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % tc.accumulation_steps == 0:
                    if tc.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

                e_total += float(raw_loss.item())
                e_seg += float(seg_loss.item())
                e_cls += float(cls_loss.item())
                n_batches += 1
                if batch_idx == 0 or (batch_idx + 1) % log_interval == 0:
                    log_step(self.logger, f"   📦 Batch {batch_idx+1}/{n_batches_total} | total={float(raw_loss.item()):.4f} seg={float(seg_loss.item()):.4f} cls={float(cls_loss.item()):.4f}")

            if n_batches % tc.accumulation_steps != 0:
                if tc.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            avg_train_loss = e_total / max(1, n_batches)
            avg_seg_loss = e_seg / max(1, n_batches)
            avg_cls_loss = e_cls / max(1, n_batches)

            if swa_model is not None and epoch >= swa_start:
                if swa_scheduler is None:
                    _base_opt = optimizer.base_optimizer if isinstance(optimizer, _LookaheadOptimizer) else optimizer
                    with contextlib.suppress(Exception):
                        swa_scheduler = torch.optim.swa_utils.SWALR(
                            _base_opt, swa_lr=tc.learning_rate * tc.min_lr_factor, anneal_epochs=5,
                        )
                swa_model.update_parameters(model)
                if swa_scheduler is not None:
                    swa_scheduler.step()
                swa_active = True

            model.eval()
            val_loss, val_batches = 0.0, 0
            eval_ctx = ema.applied(model) if ema is not None else contextlib.nullcontext()
            with eval_ctx, torch.no_grad():
                for batch in self._batch_records(val_records, tc.batch_size):
                    images, masks, labels = self._prepare_batch(batch, label_to_index, augment=False, image_cache=_img_cache, mask_cache=_mask_cache)
                    if self.device is not None:
                        images = images.to(self.device, non_blocking=True)
                        masks = masks.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(images)
                        v_seg = _combined_seg_loss(outputs["segmentation"], masks)
                        v_cls = F.cross_entropy(outputs["classification"], labels, weight=class_weights, label_smoothing=tc.label_smoothing)
                        v_loss = tc.segmentation_loss_weight * v_seg + tc.classification_loss_weight * v_cls
                    val_loss += float(v_loss.item())
                    val_batches += 1
            avg_val_loss = val_loss / max(1, val_batches)

            is_best = avg_val_loss < best_val_loss
            epoch_metrics.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "seg_loss": avg_seg_loss,
                "cls_loss": avg_cls_loss,
                "val_loss": avg_val_loss,
            })
            log_step(
                self.logger,
                (
                    f"{'✅' if is_best else '📊'} Epoch {epoch+1}/{tc.epochs}: "
                    f"total={avg_train_loss:.4f} | seg={avg_seg_loss:.4f} | cls={avg_cls_loss:.4f} | val={avg_val_loss:.4f}"
                    + (" 🎯 new best!" if is_best else f" (patience {patience_ctr+1}/{tc.patience})")
                ),
            )
            if progress_callback is not None:
                progress_callback({
                    "epoch": epoch + 1, "total": tc.epochs,
                    "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                    "seg_loss": avg_seg_loss, "cls_loss": avg_cls_loss,
                    "type": "epoch",
                })

            last_ckpt = self._save_training_checkpoint(
                checkpoint_path=ckpt_path, model=model, optimizer=optimizer, epoch=epoch,
                task="joint", split=split,
                metrics={"loss": avg_train_loss, "seg_loss": avg_seg_loss, "cls_loss": avg_cls_loss, "val_loss": avg_val_loss},
                extra_metadata={"best_val_loss": best_val_loss},
            )

            if is_best:
                best_val_loss = avg_val_loss
                if ema is not None:
                    with ema.applied(model):
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= tc.patience:
                    log_step(self.logger, f"⏹️ Early stopping at epoch {epoch+1} (patience={tc.patience} exhausted).")
                    break

        if swa_model is not None and swa_active:
            log_step(self.logger, "Updating BN stats for SWA model.")
            swa_model.eval()
            with torch.no_grad():
                for batch in self._batch_records(base_train[:min(len(base_train), 256)], tc.batch_size):
                    imgs, _, _ = self._prepare_batch(batch, label_to_index, augment=False, image_cache=_img_cache, mask_cache=_mask_cache)
                    if self.device is not None:
                        imgs = imgs.to(self.device, non_blocking=True)
                    swa_model(imgs)
            model.load_state_dict(swa_model.module.state_dict())
            log_step(self.logger, "SWA weights applied to joint model.")
        elif best_state is not None:
            model.load_state_dict(best_state)
            log_step(self.logger, f"Restored best joint model (val_loss={best_val_loss:.4f}).")

        if progress_callback is not None:
            progress_callback({"type": "complete", "best_val_loss": best_val_loss})

        return TrainingResult(
            status="completed",
            metrics={
                "loss": avg_train_loss,
                "segmentation_loss": avg_seg_loss,
                "classification_loss": avg_cls_loss,
                "val_loss": best_val_loss,
            },
            metadata={
                "split": split,
                "best_val_loss": best_val_loss,
                "epoch_metrics": epoch_metrics,
                "checkpoint": last_ckpt,
                "swa_applied": swa_active,
                "ema_used": ema is not None,
                **resume_meta,
            },
        )

    def evaluate_joint(self, split: str = "test") -> TrainingResult:
        torch, F = _require_torch()
        log_step(self.logger, f"Joint evaluation: split={split}.")

        if self.model is None:
            return TrainingResult(status="failed", metadata={"reason": "No model provided."})

        records = discover_joint_samples(split, processed_root=self.processed_root, view_filter=self.view_filter)
        if not records:
            return TrainingResult(status="failed", metadata={"reason": f"No joint samples for split={split}."})

        model = self.model.model if hasattr(self.model, "model") else self.model
        if hasattr(model, "to") and self.device is not None:
            model.to(self.device, non_blocking=True)
        model.eval()

        label_to_index = _label_to_index_map(self.config.joint_class_names)
        total_loss = total_seg = total_cls = 0.0
        n_batches = 0
        cls_preds: list[np.ndarray] = []
        cls_targets: list[np.ndarray] = []
        seg_preds: list[np.ndarray] = []
        seg_targets: list[np.ndarray] = []

        with torch.no_grad():
            for batch in self._batch_records(records, self.training_config.batch_size):
                images, masks, labels = self._prepare_batch(batch, label_to_index, augment=False)
                if self.device is not None:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                outputs = model(images)
                seg_loss = _combined_seg_loss(outputs["segmentation"], masks)
                cls_loss = F.cross_entropy(outputs["classification"], labels)
                loss = (
                    self.training_config.segmentation_loss_weight * seg_loss
                    + self.training_config.classification_loss_weight * cls_loss
                )
                seg_preds.append(torch.sigmoid(outputs["segmentation"]).detach().cpu().numpy())
                seg_targets.append(masks.detach().cpu().numpy())
                cls_preds.append(torch.softmax(outputs["classification"], dim=1).detach().cpu().numpy())
                cls_targets.append(labels.detach().cpu().numpy())
                total_loss += float(loss.item())
                total_seg += float(seg_loss.item())
                total_cls += float(cls_loss.item())
                n_batches += 1

        metrics: dict[str, float] = {
            "val_loss": total_loss / max(1, n_batches),
            "val_segmentation_loss": total_seg / max(1, n_batches),
            "val_classification_loss": total_cls / max(1, n_batches),
        }
        analysis: dict[str, Any] = {}
        if cls_preds and seg_preds:
            all_preds = {
                "classification": np.concatenate(cls_preds, axis=0),
                "segmentation": np.concatenate(seg_preds, axis=0),
            }
            all_tgt = {
                "classification": np.concatenate(cls_targets, axis=0),
                "segmentation": np.concatenate(seg_targets, axis=0),
            }
            metrics.update(compute_metrics(predictions=all_preds, targets=all_tgt))
            analysis = compute_analysis(predictions=all_preds, targets=all_tgt)

        return TrainingResult(
            status="completed",
            metrics=metrics,
            metadata={"split": split, "analysis": analysis},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_preprocessing(self, *, augment: bool) -> ImagePreprocessingConfig:
        return ImagePreprocessingConfig(
            image_size=self.config.image_size,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
            augment_training_data=augment and self.config.augment_training_data,
        )

    def _training_checkpoint_path(self, checkpoint_name: str, split: str) -> Path:
        safe_name = sanitize_checkpoint_name(checkpoint_name)
        return self.config.models_dir / "checkpoints" / f"{safe_name}.{split}.pt"

    def _try_resume_training_state(
        self, model: Any, optimizer: Any, checkpoint_path: Path
    ) -> tuple[int, dict[str, Any]]:
        if not checkpoint_path.exists():
            log_step(self.logger, f"No checkpoint at {checkpoint_path}; starting fresh.")
            return 0, {}
        try:
            payload = load_checkpoint(checkpoint_path)
            checkpoint = payload.get("checkpoint", {})
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint payload is not a dict.")
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, dict):
                raise ValueError("Missing state_dict in checkpoint.")
            getattr(model, "model", model).load_state_dict(state_dict)
            opt_state = checkpoint.get("optimizer_state")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
            resumed = int(checkpoint.get("epoch", -1)) + 1
            log_step(self.logger, f"Resumed from epoch {resumed} at {checkpoint_path}.")
            return resumed, {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_metadata": payload.get("metadata", {}),
            }
        except Exception as exc:
            log_step(self.logger, f"Resume failed ({exc}); starting fresh.")
            return 0, {"resume_warning": str(exc)}

    def _save_training_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        model: Any,
        optimizer: Any,
        epoch: int,
        task: str,
        split: str,
        metrics: dict[str, float],
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        target = getattr(model, "model", model)
        payload = {
            "state_dict": target.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "task": task,
            "split": split,
            "metrics": metrics,
        }
        if extra_metadata:
            payload.update({k: v for k, v in extra_metadata.items() if v is not None})
        meta = {
            "model_name": checkpoint_path.stem.rsplit(".", 1)[0],
            "task": task,
            "split": split,
            "epoch": epoch,
            "metrics": metrics,
            "run_date": self.training_run_date,
            **({} if not extra_metadata else {k: v for k, v in extra_metadata.items() if v is not None}),
        }
        flat_info = save_checkpoint(checkpoint_path, payload, metadata={**meta, "storage": "flat"})
        vpath = versioned_checkpoint_path(
            self.config.models_dir / "checkpoints",
            meta["model_name"],
            self.training_run_date,
            checkpoint_path.name,
        )
        v_info = save_checkpoint(vpath, payload, metadata={**meta, "storage": "versioned"})
        log_step(self.logger, f"Checkpoint saved: epoch={epoch + 1}, task={task}, split={split}.")
        return {
            **flat_info,
            "flat_checkpoint_path": flat_info["checkpoint_path"],
            "flat_metadata_path": flat_info["metadata_path"],
            "versioned_checkpoint_path": v_info["checkpoint_path"],
            "versioned_metadata_path": v_info["metadata_path"],
            "run_date": self.training_run_date,
        }

    @staticmethod
    def _batch_records(records: list[Any], batch_size: int) -> Iterable[list[Any]]:
        for start in range(0, len(records), batch_size):
            yield records[start: start + batch_size]

    @staticmethod
    def _build_image_cache(
        records: list[Any],
        preprocessing: ImagePreprocessingConfig | None = None,
    ) -> dict[str, Any]:
        """Parallel-load and deterministically preprocess all images into RAM.

        Stores images after CLAHE + resize (but before stochastic augmentation), so the
        training loop only needs to run the fast augmentation step — not the expensive
        CLAHE or disk I/O — on each batch.
        """
        cache: dict[str, Any] = {}

        def _load(r: Any) -> None:
            p = getattr(r, "image_path", None)
            if p is not None:
                key = str(p)
                if key not in cache:
                    try:
                        cache[key] = preprocess_image(p, preprocessing, augment=False)
                    except Exception:
                        pass

        n_workers = min(8, max(1, len(records)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_load, records))
        return cache

    @staticmethod
    def _build_mask_cache(
        records: list[Any],
        preprocessing: ImagePreprocessingConfig | None = None,
    ) -> dict[str, Any]:
        """Parallel-load and resize all segmentation masks into RAM."""
        cache: dict[str, Any] = {}

        def _load(r: Any) -> None:
            p = getattr(r, "mask_path", None)
            if p is not None:
                key = str(p)
                if key not in cache:
                    try:
                        cache[key] = preprocess_mask(p, preprocessing)
                    except Exception:
                        pass

        n_workers = min(8, max(1, len(records)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_load, records))
        return cache

    # Keep old name for any direct callers
    @staticmethod
    def _batch_classification_records(
        records: list[DatasetRecord], batch_size: int
    ) -> Iterable[list[DatasetRecord]]:
        for start in range(0, len(records), batch_size):
            yield records[start: start + batch_size]

    def _prepare_batch(
        self,
        records: list[JointDatasetRecord],
        label_to_index: dict[str, int],
        *,
        augment: bool = False,
        image_cache: dict[str, Any] | None = None,
        mask_cache: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any]:
        torch, _ = _require_torch()
        preprocessing = self._make_preprocessing(augment=augment)
        imgs: list[Any] = []
        masks_t: list[Any] = []
        labels_t: list[int] = []

        for record in records:
            seed = _random.randint(0, 2**31) if augment else None
            img_cached = image_cache.get(str(record.image_path)) if image_cache else None
            mask_src = (mask_cache.get(str(record.mask_path)) if mask_cache else None) or record.mask_path
            if img_cached is not None:
                img = apply_augmentation_only(img_cached.copy(), preprocessing, seed=seed)
            else:
                img = preprocess_image(record.image_path, preprocessing, augment=augment, seed=seed)
            if augment and seed is not None:
                mask = preprocess_mask_augmented(mask_src, preprocessing, seed=seed)
            else:
                mask = preprocess_mask(mask_src, preprocessing)
            imgs.append(torch.tensor(image_to_array(img, preprocessing), dtype=torch.float32))
            masks_t.append(torch.tensor(
                (np.asarray(mask, dtype=np.float32) / 255.0)[None, ...], dtype=torch.float32
            ))
            if record.label not in label_to_index:
                raise ValueError(f"Unknown label '{record.label}' for {record.image_path}.")
            labels_t.append(label_to_index[record.label])

        if not imgs:
            raise ValueError("Empty joint batch.")
        return torch.stack(imgs), torch.stack(masks_t), torch.tensor(labels_t, dtype=torch.long)

    def _prepare_classification_batch(
        self,
        records: list[DatasetRecord],
        preprocessing: ImagePreprocessingConfig,
        label_to_index: dict[str, int],
        *,
        image_cache: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        torch, _ = _require_torch()
        imgs: list[Any] = []
        labels_t: list[int] = []

        for record in records:
            cached = image_cache.get(str(record.image_path)) if image_cache else None
            if cached is not None:
                img = apply_augmentation_only(cached.copy(), preprocessing)
            else:
                img = preprocess_image(record.image_path, preprocessing, augment=preprocessing.augment_training_data)
            imgs.append(torch.tensor(image_to_array(img, preprocessing), dtype=torch.float32))
            if record.label not in label_to_index:
                raise ValueError(f"Unknown label '{record.label}' for {record.image_path}.")
            labels_t.append(label_to_index[record.label])

        if not imgs:
            raise ValueError("Empty classification batch.")
        return torch.stack(imgs), torch.tensor(labels_t, dtype=torch.long)

    def _prepare_segmentation_batch(
        self,
        records: list[DatasetRecord],
        augment: bool = False,
        *,
        image_cache: dict[str, Any] | None = None,
        mask_cache: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        torch, _ = _require_torch()
        preprocessing = self._make_preprocessing(augment=augment)
        imgs: list[Any] = []
        masks_t: list[Any] = []

        for record in records:
            if record.mask_path is None:
                continue
            seed = _random.randint(0, 2**31) if augment else None
            img_cached = image_cache.get(str(record.image_path)) if image_cache else None
            mask_src = (mask_cache.get(str(record.mask_path)) if mask_cache else None) or record.mask_path
            if img_cached is not None:
                img = apply_augmentation_only(img_cached.copy(), preprocessing, seed=seed)
            else:
                img = preprocess_image(record.image_path, preprocessing, augment=augment, seed=seed)
            if augment and seed is not None:
                mask = preprocess_mask_augmented(mask_src, preprocessing, seed=seed)
            else:
                mask = preprocess_mask(mask_src, preprocessing)
            imgs.append(torch.tensor(image_to_array(img, preprocessing), dtype=torch.float32))
            masks_t.append(torch.tensor(
                (np.asarray(mask, dtype=np.float32) / 255.0)[None, ...], dtype=torch.float32
            ))

        if not imgs:
            raise ValueError("Empty segmentation batch.")
        return torch.stack(imgs), torch.stack(masks_t)
