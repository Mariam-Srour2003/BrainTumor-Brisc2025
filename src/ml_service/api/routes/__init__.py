"""Route modules for the ML service API."""

from .evaluate import router as evaluate_router
from .explain import register_explain_routes
from .predict import router as predict_router
from .train import router as train_router

__all__ = ["evaluate_router", "predict_router", "register_explain_routes", "train_router"]