"""Application entrypoint for the ML service API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..core.config import get_settings
from ..core.runtime import configure_logging, get_logger, log_step

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend" / "simple"


def create_app(model: Any | None = None) -> Any:
    """Create the FastAPI application when FastAPI is installed."""

    settings = get_settings()
    logger = configure_logging(logging.INFO)
    log_step(logger, "Booting ML service API.")

    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, RedirectResponse
        from fastapi.staticfiles import StaticFiles
        from .routes.evaluate import router as evaluate_router
        from .routes.explain import register_explain_routes
        from .routes.predict import router as predict_router
        from .routes.train import router as train_router
        from .routes.models import router as models_router
        from ..inference.predictor import Predictor
    except ImportError:
        return {
            "service": "ml_service",
            "status": "fastapi-not-installed",
            "data_root": str(settings.data_root),
        }

    app = FastAPI(
        title="Brain Tumor ML Service",
        version="0.1.0",
        description=(
            "Segmentation and classification API for the BRISC 2025 brain tumor dataset. "
            "Supports U-Net, ResNet, EfficientNet, ViT, and joint multitask models with Grad-CAM/SHAP/LIME explainability."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.predictor = Predictor(model=model, config=settings)
    log_step(get_logger("api.main"), f"API predictor initialized with model={type(model).__name__ if model is not None else 'None'}.")

    outputs_dir = settings.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    try:
        app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")
    except RuntimeError:
        log_step(get_logger("api.main"), "Outputs directory is empty; static file mount skipped until files are written.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def root() -> Any:
        index = _FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index), media_type="text/html")
        return RedirectResponse("/docs")

    if _FRONTEND_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="ui")
        log_step(get_logger("api.main"), f"Frontend dashboard served at /  and  /ui  ({_FRONTEND_DIR})")

    if train_router is not None:
        app.include_router(train_router, prefix="/train", tags=["train"])
    if predict_router is not None:
        app.include_router(predict_router, prefix="/predict", tags=["predict"])
    if models_router is not None:
        app.include_router(models_router, prefix="/models", tags=["models"])
    register_explain_routes(app)
    if evaluate_router is not None:
        app.include_router(evaluate_router, prefix="/evaluate", tags=["evaluate"])

    return app


def main() -> None:
    """Console entrypoint used by the project script."""

    app = create_app()
    if isinstance(app, dict):
        print("ML Service scaffold is ready. Install fastapi to run the API.")
        print(f"Data root: {app['data_root']}")
        return

    print("ML Service app created. Run it with:")
    print("  uvicorn ml_service.api.main:create_app --factory --reload")
