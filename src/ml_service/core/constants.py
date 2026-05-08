"""Project-wide constants and filesystem paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "ml_service"
DATA_ROOT = PROJECT_ROOT / "data" / "brisc2025"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "outputs" / "processed" / "brisc2025"
CLASSIFICATION_DATA_ROOT = DATA_ROOT / "classification_task"
SEGMENTATION_DATA_ROOT = DATA_ROOT / "segmentation_task"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")
TRAIN_SPLITS = ("train", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZE_STD = (0.229, 0.224, 0.225)