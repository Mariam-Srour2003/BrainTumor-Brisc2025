"""Runtime helpers for logging and device selection."""

from __future__ import annotations

import logging
import sys
from typing import Any


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a simple console logger once for the application."""

    root_logger = logging.getLogger("ml_service")
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger.propagate = False
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ml_service hierarchy."""

    configure_logging()
    return logging.getLogger(f"ml_service.{name}")


def log_step(logger: logging.Logger, message: str) -> None:
    """Emit a visible progress message to the terminal."""

    logger.info(message)


def require_torch_device() -> tuple[Any, Any]:
    """Return torch and the best available compute device."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install torch to use GPU or CPU-backed model execution.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch, device


def describe_device(torch_module: Any | None = None) -> str:
    """Return a readable device description for logs."""

    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            return "torch-not-installed"

    return "cuda" if torch_module.cuda.is_available() else "cpu"