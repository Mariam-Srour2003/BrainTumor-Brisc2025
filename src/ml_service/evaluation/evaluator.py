"""Evaluation driver for trained models."""

from __future__ import annotations

from typing import Any

from .metrics import compute_analysis, compute_metrics
from ..core.runtime import get_logger, log_step


class Evaluator:
    """Evaluator that computes medical metrics from predictions and targets."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model
        self.logger = get_logger("evaluation.evaluator")

    def evaluate(self, data: Any | None = None) -> dict[str, float]:
        if data is None:
            log_step(self.logger, "No evaluation data supplied.")
            return {}

        if isinstance(data, dict) and "predictions" in data and "targets" in data:
            log_step(self.logger, "Computing evaluation metrics from explicit predictions and targets.")
            return compute_metrics(predictions=data["predictions"], targets=data["targets"])

        if isinstance(data, tuple) and len(data) == 2:
            log_step(self.logger, "Computing evaluation metrics from a prediction-target tuple.")
            return compute_metrics(predictions=data[0], targets=data[1])

        raise ValueError("Evaluation data must provide predictions and targets.")

    def analyze(self, data: Any | None = None) -> dict[str, Any]:
        if data is None:
            log_step(self.logger, "No analysis data supplied.")
            return {}

        if isinstance(data, dict) and "predictions" in data and "targets" in data:
            log_step(self.logger, "Computing confusion matrix and class-wise analysis.")
            return compute_analysis(predictions=data["predictions"], targets=data["targets"])

        if isinstance(data, tuple) and len(data) == 2:
            log_step(self.logger, "Computing confusion matrix and class-wise analysis from tuple input.")
            return compute_analysis(predictions=data[0], targets=data[1])

        raise ValueError("Analysis data must provide predictions and targets.")