"""Training execution engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .hyperparameters import HyperparameterSearchSpace, HyperparameterTrial, sample_trial
from .trainer import Trainer, TrainingConfig, TrainingResult


@dataclass(frozen=True)
class TuningTrialResult:
    """Result of one hyperparameter tuning trial."""

    trial: HyperparameterTrial
    train_result: TrainingResult
    validation_result: TrainingResult


@dataclass(frozen=True)
class TuningSummary:
    """Aggregate result for random-search tuning."""

    best_trial: HyperparameterTrial
    best_metric: float
    metric_name: str
    trials: tuple[TuningTrialResult, ...]


def tune_hyperparameters(
    trainer: Trainer,
    *,
    search_space: HyperparameterSearchSpace | None = None,
    trials: int = 5,
    metric_name: str = "val_loss",
    seed: int = 42,
) -> TuningSummary:
    """Run random-search hyperparameter tuning for joint training."""

    space = search_space or HyperparameterSearchSpace()
    results: list[TuningTrialResult] = []

    best_metric = float("inf")
    best_trial = sample_trial(space, seed=seed)

    for index in range(trials):
        sampled = sample_trial(space, seed=seed + index)
        tuned_config = TrainingConfig(
            batch_size=sampled.batch_size,
            learning_rate=sampled.learning_rate,
            weight_decay=sampled.weight_decay,
            epochs=sampled.epochs,
            segmentation_loss_weight=sampled.segmentation_loss_weight,
            classification_loss_weight=sampled.classification_loss_weight,
            optimizer_name=sampled.optimizer_name,
        )

        trainer.training_config = tuned_config
        train_result = trainer.fit_joint(split="train")
        validation_result = trainer.evaluate_joint(split="test")

        current_metric = validation_result.metrics.get(metric_name, float("inf"))
        if current_metric < best_metric:
            best_metric = current_metric
            best_trial = sampled

        results.append(
            TuningTrialResult(
                trial=sampled,
                train_result=train_result,
                validation_result=validation_result,
            )
        )

    return TuningSummary(
        best_trial=best_trial,
        best_metric=best_metric,
        metric_name=metric_name,
        trials=tuple(results),
    )


def run_training_engine(trainer: Any, data: Any | None = None) -> Any:
    """Delegate training to a trainer object."""

    return trainer.fit(data)