"""Core settings and constants."""

from .config import ServiceConfig, get_settings
from .constants import (
    CLASS_NAMES,
    CLASSIFICATION_DATA_ROOT,
    DATA_ROOT,
    IMAGE_EXTENSIONS,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SEGMENTATION_DATA_ROOT,
    SRC_ROOT,
)

__all__ = ["ServiceConfig", "get_settings"]