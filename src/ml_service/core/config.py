"""Configuration helpers for the ML service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import (
    CLASS_NAMES,
    CLASSIFICATION_DATA_ROOT,
    DATA_ROOT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NORMALIZE_MEAN,
    DEFAULT_NORMALIZE_STD,
    DIVIDED_CLASSIFICATION_DATA_ROOT,
    DIVIDED_PROCESSED_DATA_ROOT,
    DIVIDED_SEGMENTATION_DATA_ROOT,
    JOINT_CLASS_NAMES,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_ROOT,
    PROJECT_ROOT,
    SEGMENTATION_DATA_ROOT,
    TRAIN_SPLITS,
    VIEW_CLASS_NAMES,
)


@dataclass(frozen=True)
class ServiceConfig:
    """Filesystem-oriented defaults for the service scaffold."""

    project_root: Path = PROJECT_ROOT
    data_root: Path = DATA_ROOT
    classification_data_root: Path = CLASSIFICATION_DATA_ROOT
    segmentation_data_root: Path = SEGMENTATION_DATA_ROOT
    divided_classification_data_root: Path = DIVIDED_CLASSIFICATION_DATA_ROOT
    divided_segmentation_data_root: Path = DIVIDED_SEGMENTATION_DATA_ROOT
    models_dir: Path = MODELS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    processed_data_root: Path = PROCESSED_DATA_ROOT
    processed_divided_data_root: Path = DIVIDED_PROCESSED_DATA_ROOT
    notebooks_dir: Path = NOTEBOOKS_DIR
    class_names: tuple[str, ...] = CLASS_NAMES
    joint_class_names: tuple[str, ...] = JOINT_CLASS_NAMES
    view_class_names: tuple[str, ...] = VIEW_CLASS_NAMES
    train_splits: tuple[str, ...] = TRAIN_SPLITS
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    normalize_mean: tuple[float, float, float] = DEFAULT_NORMALIZE_MEAN
    normalize_std: tuple[float, float, float] = DEFAULT_NORMALIZE_STD
    augment_training_data: bool = True
    materialize_augmentations: bool = False
    save_format: str = "png"
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 50
    segmentation_loss_weight: float = 1.0
    classification_loss_weight: float = 2.0
    optimizer_name: str = "adamw"
    joint_model_name: str = "hybrid.adpt_net"


def get_settings() -> ServiceConfig:
    """Return a default service configuration object."""

    return ServiceConfig()