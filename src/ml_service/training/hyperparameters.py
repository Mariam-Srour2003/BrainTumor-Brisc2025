"""Hyperparameter space definitions and random-search utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class HyperparameterTrial:
    """Concrete hyperparameter values for one trial."""

    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    segmentation_loss_weight: float
    classification_loss_weight: float
    optimizer_name: str
    encoder_name: str


@dataclass(frozen=True)
class HyperparameterSearchSpace:
    """Search ranges for random tuning — biased toward proven high-accuracy settings."""

    learning_rates: tuple[float, ...] = (5e-4, 3e-4, 1e-4, 5e-5)
    weight_decays: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)
    batch_sizes: tuple[int, ...] = (8, 16)
    epochs: tuple[int, ...] = (40, 50, 75)
    segmentation_loss_weights: tuple[float, ...] = (0.5, 1.0, 1.5)
    classification_loss_weights: tuple[float, ...] = (0.5, 1.0, 1.5)
    optimizers: tuple[str, ...] = ("adamw", "adamw", "adam")  # adamw weighted 2× heavier
    encoders: tuple[str, ...] = ("resnet50", "efficientnet_b3", "resnet34")


def sample_trial(space: HyperparameterSearchSpace, seed: int | None = None) -> HyperparameterTrial:
    """Sample one random hyperparameter trial from a space."""

    rng = random.Random(seed)
    return HyperparameterTrial(
        learning_rate=rng.choice(space.learning_rates),
        weight_decay=rng.choice(space.weight_decays),
        batch_size=rng.choice(space.batch_sizes),
        epochs=rng.choice(space.epochs),
        segmentation_loss_weight=rng.choice(space.segmentation_loss_weights),
        classification_loss_weight=rng.choice(space.classification_loss_weights),
        optimizer_name=rng.choice(space.optimizers),
        encoder_name=rng.choice(space.encoders),
    )