"""Top-level experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.config import ServiceConfig, get_settings
from ..core.runtime import get_logger, log_step
from ..data.preprocessing import ProcessingSummary, materialize_clean_dataset
from ..data.dataset import discover_classification_samples, discover_segmentation_pairs
from ..models import DEFAULT_MODEL_REGISTRY
from ..training import HyperparameterSearchSpace, Trainer, TrainingConfig, TuningSummary, tune_hyperparameters
from ..evaluation import export_evaluation_report
from ..explainability import ExplainabilityMethod, ExplainabilityRequest, ExplainabilityTask, ExplainabilityEngine
from ..utils import clear_model_checkpoints, load_checkpoint, save_checkpoint, versioned_checkpoint_path


logger = get_logger("experiments.runner")


def _clear_existing_model_artifacts(settings: ServiceConfig, model_name: str) -> list[str]:
    """Remove previous checkpoints for a model before retraining."""

    removed = clear_model_checkpoints(settings.models_dir / "checkpoints", model_name)
    if removed:
        log_step(logger, f"Removed {len(removed)} previous checkpoint artifact(s) for model={model_name}.")
    else:
        log_step(logger, f"No previous checkpoint artifacts found for model={model_name}.")
    return removed


def _save_model_checkpoint(
    *,
    trainer: Trainer,
    settings: ServiceConfig,
    checkpoint_stem: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Persist a torch model checkpoint and JSON metadata sidecar."""

    model_obj = getattr(trainer.model, "model", trainer.model)
    checkpoint_path = settings.models_dir / "checkpoints" / f"{checkpoint_stem}.pt"
    run_date = trainer.training_run_date
    common_metadata = {
        **metadata,
        "model_name": checkpoint_stem,
        "run_date": run_date,
    }

    flat_info = save_checkpoint(checkpoint_path, model_obj, {**common_metadata, "storage": "flat"})
    versioned_path = versioned_checkpoint_path(
        settings.models_dir / "checkpoints",
        checkpoint_stem,
        run_date,
        checkpoint_path.name,
    )
    versioned_info = save_checkpoint(versioned_path, model_obj, {**common_metadata, "storage": "versioned"})

    eval_summary_path = versioned_path.with_name("evaluation_summary.json")
    eval_summary_path.write_text(json.dumps(common_metadata, indent=2, sort_keys=True), encoding="utf-8")

    log_step(
        logger,
        f"Saved model checkpoints: flat={flat_info['checkpoint_path']}, versioned={versioned_info['checkpoint_path']}",
    )
    return {
        **flat_info,
        "flat_checkpoint_path": flat_info["checkpoint_path"],
        "flat_metadata_path": flat_info["metadata_path"],
        "versioned_checkpoint_path": versioned_info["checkpoint_path"],
        "versioned_metadata_path": versioned_info["metadata_path"],
        "evaluation_summary_path": str(eval_summary_path),
        "run_date": run_date,
    }


def run_experiment(config: Any | None = None) -> dict[str, str]:
    """Run the dataset preprocessing workflow."""

    summary = prepare_dataset(config if isinstance(config, ServiceConfig) else None)
    return {
        "status": "completed",
        "output_root": summary.output_root,
        "processed_count": str(summary.processed_count),
        "skipped_count": str(summary.skipped_count),
    }


def prepare_dataset(config: ServiceConfig | None = None, output_root: str | Path | None = None) -> ProcessingSummary:
    """Materialize the cleaned BRISC dataset to disk."""

    settings = config or get_settings()
    return materialize_clean_dataset(settings, output_root=output_root)


def build_joint_trainer(config: ServiceConfig | None = None) -> Trainer:
    """Build trainer configured for the joint segmentation-classification model."""

    settings = config or get_settings()
    model = DEFAULT_MODEL_REGISTRY.create(
        settings.joint_model_name,
        num_classes=len(settings.class_names),
        segmentation_classes=1,
        in_channels=3,
        encoder_name="resnet34",
        pretrained=True,
    )
    training_config = TrainingConfig.from_service_config(settings)
    return Trainer(model=model, config=settings, training_config=training_config)


def run_joint_training(config: ServiceConfig | None = None) -> dict[str, Any]:
    """Run one joint multitask training + evaluation cycle."""

    try:
        settings = config or get_settings()
        _clear_existing_model_artifacts(settings, settings.joint_model_name)
        trainer = build_joint_trainer(settings)
        train_result = trainer.fit_joint(split="train", checkpoint_name=settings.joint_model_name)
        if train_result.status == "failed":
            return {
                "status": "failed",
                "model_name": settings.joint_model_name,
                "message": train_result.metadata.get("reason", "Training failed — no data found or model error."),
            }
        validation_result = trainer.evaluate_joint(split="test")
        checkpoint_info = _save_model_checkpoint(
            trainer=trainer,
            settings=settings,
            checkpoint_stem=settings.joint_model_name,
            metadata={
                "model_name": settings.joint_model_name,
                "task": "joint",
                "train_metrics": train_result.metrics,
                "validation_metrics": validation_result.metrics,
            },
        )
    except ImportError as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

    return {
        "status": "completed",
        "train": {
            "status": train_result.status,
            "metrics": train_result.metrics,
            "metadata": train_result.metadata,
        },
        "validation": {
            "status": validation_result.status,
            "metrics": validation_result.metrics,
            "metadata": validation_result.metadata,
        },
        "checkpoint": checkpoint_info,
    }


def run_joint_hyperparameter_tuning(
    config: ServiceConfig | None = None,
    *,
    trials: int = 5,
    metric_name: str = "val_loss",
) -> TuningSummary:
    """Run random-search tuning on the joint multitask pipeline."""

    trainer = build_joint_trainer(config)
    return tune_hyperparameters(
        trainer,
        search_space=HyperparameterSearchSpace(),
        trials=trials,
        metric_name=metric_name,
    )


def run_all_registered_models_training(config: ServiceConfig | None = None) -> dict[str, Any]:
    """Train and evaluate every registered model family with the appropriate task-specific loop."""

    settings = config or get_settings()
    results: dict[str, Any] = {}

    for model_name in DEFAULT_MODEL_REGISTRY.list_models():
        try:
            results[model_name] = run_registered_model_training(model_name, settings)
        except Exception as exc:
            results[model_name] = {"status": "error", "message": str(exc)}

    return {
        "status": "completed",
        "model_count": len(results),
        "results": results,
    }


def run_registered_model_training(model_name: str, config: ServiceConfig | None = None) -> dict[str, Any]:
    """Train and evaluate one registered model by name."""

    settings = config or get_settings()

    try:
        removed_artifacts = _clear_existing_model_artifacts(settings, model_name)
        if model_name.startswith("classification."):
            model = DEFAULT_MODEL_REGISTRY.create(
                model_name,
                num_classes=len(settings.class_names),
                in_channels=3,
            )
            trainer = Trainer(model=model, config=settings, training_config=TrainingConfig.from_service_config(settings))
            train_result = trainer.fit_classification(split="train", checkpoint_name=model_name)
            if train_result.status == "failed":
                return {
                    "status": "failed",
                    "model_name": model_name,
                    "message": train_result.metadata.get("reason", "Training failed — no data found or model error."),
                    "removed_artifacts": removed_artifacts,
                }
            eval_result = trainer.evaluate_classification(split="test")
            task = "classification"
        elif model_name.startswith("segmentation."):
            model = DEFAULT_MODEL_REGISTRY.create(
                model_name,
                num_classes=1,
                in_channels=3,
            )
            trainer = Trainer(model=model, config=settings, training_config=TrainingConfig.from_service_config(settings))
            train_result = trainer.fit_segmentation(split="train", checkpoint_name=model_name)
            if train_result.status == "failed":
                return {
                    "status": "failed",
                    "model_name": model_name,
                    "message": train_result.metadata.get("reason", "Training failed — no data found or model error."),
                    "removed_artifacts": removed_artifacts,
                }
            eval_result = trainer.evaluate_segmentation(split="test")
            task = "segmentation"
        elif model_name.startswith("hybrid."):
            model = DEFAULT_MODEL_REGISTRY.create(
                model_name,
                num_classes=len(settings.class_names),
                segmentation_classes=1,
                in_channels=3,
            )
            trainer = Trainer(model=model, config=settings, training_config=TrainingConfig.from_service_config(settings))
            train_result = trainer.fit_joint(split="train", checkpoint_name=model_name)
            if train_result.status == "failed":
                return {
                    "status": "failed",
                    "model_name": model_name,
                    "message": train_result.metadata.get("reason", "Training failed — no data found or model error."),
                    "removed_artifacts": removed_artifacts,
                }
            eval_result = trainer.evaluate_joint(split="test")
            task = "joint"
        else:
            return {"status": "skipped", "reason": "No supported trainer available."}

        checkpoint_info = _save_model_checkpoint(
            trainer=trainer,
            settings=settings,
            checkpoint_stem=model_name,
            metadata={
                "model_name": model_name,
                "task": task,
                "train_metrics": train_result.metrics,
                "evaluation_metrics": eval_result.metrics,
            },
        )
    except ImportError as exc:
        return {
            "status": "error",
            "model_name": model_name,
            "message": str(exc),
        }

    return {
        "status": "completed",
        "model_name": model_name,
        "removed_artifacts": removed_artifacts,
        "train": {
            "status": train_result.status,
            "metrics": train_result.metrics,
            "metadata": train_result.metadata,
        },
        "evaluation": {
            "status": eval_result.status,
            "metrics": eval_result.metrics,
            "metadata": eval_result.metadata,
        },
        "checkpoint": checkpoint_info,
    }


def _infer_checkpoint_task(metadata: dict[str, Any], checkpoint_path: Path) -> str:
    task = str(metadata.get("task") or "").strip().lower()
    if task in {"classification", "segmentation", "joint"}:
        return task

    name = checkpoint_path.stem.lower()
    if "segmentation" in name:
        return "segmentation"
    if "classification" in name:
        return "classification"
    return "joint" if "joint" in name or "hybrid" in name else "classification"


def _candidate_model_names(task: str, model_name: str) -> tuple[str, ...]:
    registered = DEFAULT_MODEL_REGISTRY.list_models()
    candidates: list[str] = []

    if model_name in registered:
        candidates.append(model_name)

    if task == "classification":
        candidates.extend([name for name in registered if name.startswith("classification.") and name not in candidates])
    elif task == "segmentation":
        candidates.extend([name for name in registered if name.startswith("segmentation.") and name not in candidates])
    else:
        candidates.extend([name for name in registered if name.startswith("hybrid.") and name not in candidates])

    return tuple(candidates)


def _load_model_for_checkpoint(checkpoint_path: Path, settings: ServiceConfig) -> tuple[Any, dict[str, Any], str, str]:
    payload = load_checkpoint(checkpoint_path)
    metadata = payload.get("metadata", {})
    checkpoint = payload.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload is not a dict.")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint payload missing state_dict.")

    model_name = str(metadata.get("model_name") or checkpoint_path.stem)
    task = _infer_checkpoint_task(metadata, checkpoint_path)

    for candidate in _candidate_model_names(task, model_name):
        try:
            if candidate.startswith("classification."):
                model = DEFAULT_MODEL_REGISTRY.create(
                    candidate,
                    num_classes=len(settings.class_names),
                    in_channels=3,
                )
            elif candidate.startswith("segmentation."):
                model = DEFAULT_MODEL_REGISTRY.create(
                    candidate,
                    num_classes=1,
                    in_channels=3,
                )
            else:
                model = DEFAULT_MODEL_REGISTRY.create(
                    candidate,
                    num_classes=len(settings.class_names),
                    segmentation_classes=1,
                    in_channels=3,
                    encoder_name="resnet34",
                    pretrained=False,
                )

            target_model = getattr(model, "model", model)
            target_model.load_state_dict(state_dict)
            return model, metadata, task, candidate
        except Exception:
            continue

    raise ValueError(f"Unable to reconstruct a model for checkpoint {checkpoint_path}.")


def run_checkpoint_evaluation(
    checkpoint_path: str | Path,
    config: ServiceConfig | None = None,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a saved checkpoint and export report artifacts."""

    settings = config or get_settings()
    checkpoint_path = Path(checkpoint_path)
    model, metadata, task, model_name = _load_model_for_checkpoint(checkpoint_path, settings)
    trainer = Trainer(model=model, config=settings, training_config=TrainingConfig.from_service_config(settings))

    if task == "classification":
        evaluation = trainer.evaluate_classification(split=split)
    elif task == "segmentation":
        evaluation = trainer.evaluate_segmentation(split=split)
    else:
        evaluation = trainer.evaluate_joint(split=split)

    report = export_evaluation_report(
        settings.outputs_dir / "evaluations",
        model_name=model_name,
        task=task,
        split=split,
        metrics=evaluation.metrics,
        analysis=evaluation.metadata.get("analysis", {}),
        checkpoint_path=checkpoint_path,
        class_names=settings.class_names,
    )

    return {
        "status": evaluation.status,
        "model_name": model_name,
        "task": task,
        "checkpoint_path": str(checkpoint_path),
        "metrics": evaluation.metrics,
        "metadata": evaluation.metadata,
        "report": report,
    }


def run_explainability_batch(
    checkpoint_path: str | Path,
    config: ServiceConfig | None = None,
    *,
    method: ExplainabilityMethod = "gradcam",
    target_task: ExplainabilityTask = "auto",
    split: str = "test",
    limit: int = 4,
) -> dict[str, Any]:
    """Generate explainability artifacts for a small batch of samples."""

    settings = config or get_settings()
    checkpoint_path = Path(checkpoint_path)
    model, metadata, task, model_name = _load_model_for_checkpoint(checkpoint_path, settings)
    if model is None:
        raise ValueError(f"Unable to reconstruct a model for checkpoint {checkpoint_path}.")

    engine = ExplainabilityEngine(model, settings)
    resolved_target_task = target_task
    if resolved_target_task == "auto":
        resolved_target_task = "segmentation" if task == "segmentation" else "classification"
    if task == "classification":
        records = discover_classification_samples(split)[:limit]
    elif task == "segmentation":
        records = [record for record in discover_segmentation_pairs(split) if record.mask_path is not None][:limit]
    else:
        records = discover_classification_samples(split)[:limit]

    output_root = settings.outputs_dir / "explanations" / model_name / split
    artifacts: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        request = ExplainabilityRequest(
            image_path=record.image_path,
            method=method,
            target_task=resolved_target_task,
            output_root=output_root / f"sample_{index:03d}",
        )
        artifact = engine.explain(request)
        artifacts.append(
            {
                "image_path": str(record.image_path),
                "label": getattr(record, "label", None),
                "artifact": artifact,
            }
        )

    summary_path = output_root / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "model_name": model_name,
                "task": task,
                "split": split,
                "method": method,
                "count": len(artifacts),
                "artifacts": [
                    {
                        "image_path": item["image_path"],
                        "label": item["label"],
                        "artifact": item["artifact"].__dict__,
                    }
                    for item in artifacts
                ],
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "status": "completed",
        "checkpoint_path": str(checkpoint_path),
        "model_name": model_name,
        "task": task,
        "split": split,
        "method": method,
        "count": len(artifacts),
        "summary_path": str(summary_path),
        "artifacts": [
            {
                "image_path": item["image_path"],
                "label": item["label"],
                "artifact": item["artifact"].__dict__,
            }
            for item in artifacts
        ],
    }