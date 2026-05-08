"""Metrics for classification and segmentation evaluation."""

from __future__ import annotations

from math import inf
from typing import Any

import numpy as np


def _to_numpy(values: Any) -> np.ndarray:
    if values is None:
        raise ValueError("Predictions and targets must not be None.")

    if isinstance(values, np.ndarray):
        return values

    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()

    return np.asarray(values)


def _binary_auc_score(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = targets.astype(np.int32).ravel()
    scores = scores.astype(np.float64).ravel()

    positives = targets == 1
    negatives = targets == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())

    if positive_count == 0 or negative_count == 0:
        return float("nan")

    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    positive_rank_sum = float(ranks[positives].sum())
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def _confusion_matrix(targets: np.ndarray, predictions: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_index = {int(label): index for index, label in enumerate(classes.tolist())}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, predicted in zip(targets, predictions):
        if int(target) in class_to_index and int(predicted) in class_to_index:
            matrix[class_to_index[int(target)], class_to_index[int(predicted)]] += 1
    return matrix


def _classwise_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    classes: np.ndarray,
) -> tuple[list[dict[str, float | int]], dict[str, float], list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for index, class_label in enumerate(classes.tolist()):
        true_positive = float(confusion_matrix[index, index])
        false_positive = float(confusion_matrix[:, index].sum() - true_positive)
        false_negative = float(confusion_matrix[index, :].sum() - true_positive)
        support = int(confusion_matrix[index, :].sum())

        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        f1_score = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)

        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1_score))

        row = {
            "class": int(class_label),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "support": support,
        }
        rows.append(row)
        summary_rows.append(row)

    macro = {
        "precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "f1_score": float(np.mean(f1_values)) if f1_values else 0.0,
    }
    summary_rows.append(
        {
            "class": "macro_avg",
            "precision": macro["precision"],
            "recall": macro["recall"],
            "f1_score": macro["f1_score"],
            "support": int(confusion_matrix.sum()),
        }
    )
    return rows, macro, summary_rows


def _classification_analysis(predictions: Any, targets: Any) -> dict[str, Any]:
    prediction_array = _to_numpy(predictions)
    target_array = _to_numpy(targets).astype(np.int64).ravel()

    if prediction_array.ndim == 1:
        predicted_labels = (prediction_array >= 0.5).astype(np.int64)
    elif prediction_array.ndim == 2:
        if prediction_array.shape[1] == 1:
            predicted_labels = (prediction_array[:, 0] >= 0.5).astype(np.int64)
        else:
            predicted_labels = np.argmax(prediction_array, axis=1)
    else:
        raise ValueError("Classification predictions must have shape [samples], [samples, 1], or [samples, classes].")

    classes = np.unique(np.concatenate([target_array, predicted_labels])) if target_array.size else np.array([], dtype=np.int64)
    confusion = _confusion_matrix(target_array, predicted_labels, classes)
    classwise_rows, macro, summary_table = _classwise_from_confusion_matrix(confusion, classes)
    return {
        "confusion_matrix": confusion.tolist(),
        "classes": [int(label) for label in classes.tolist()],
        "classwise": classwise_rows,
        "summary_table": summary_table,
        "macro_avg": macro,
    }


def _classification_metrics(predictions: Any, targets: Any) -> dict[str, float]:
    prediction_array = _to_numpy(predictions)
    target_array = _to_numpy(targets).astype(np.int64).ravel()
    binary_vector = False

    if prediction_array.ndim == 1:
        prediction_array = prediction_array[:, None]
        binary_vector = True

    if prediction_array.ndim != 2:
        raise ValueError("Classification predictions must have shape [samples, classes].")

    if prediction_array.shape[1] == 1 or binary_vector:
        positive_scores = prediction_array[:, 0]
        predicted_labels = (positive_scores >= 0.5).astype(np.int64)
        binary_targets = target_array.astype(np.int64)
        if binary_targets.size and not np.isin(binary_targets, [0, 1]).all():
            raise ValueError("Binary classification targets must be encoded as 0/1.")

        accuracy = float((predicted_labels == binary_targets).mean()) if binary_targets.size else 0.0
        precision = np.sum((predicted_labels == 1) & (binary_targets == 1)) / max(1, np.sum(predicted_labels == 1))
        recall = np.sum((predicted_labels == 1) & (binary_targets == 1)) / max(1, np.sum(binary_targets == 1))
        f1_score = 0.0 if precision + recall == 0 else float(2.0 * precision * recall / (precision + recall))
        auc = _binary_auc_score(binary_targets, positive_scores)
        return {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "auc": float(auc) if np.isfinite(auc) else 0.0,
        }

    predicted_labels = np.argmax(prediction_array, axis=1)

    accuracy = float((predicted_labels == target_array).mean()) if target_array.size else 0.0

    classes = np.unique(np.concatenate([target_array, predicted_labels])) if target_array.size else np.array([], dtype=np.int64)
    f1_scores: list[float] = []
    auc_scores: list[float] = []

    for class_index in classes:
        true_positive = np.sum((predicted_labels == class_index) & (target_array == class_index))
        false_positive = np.sum((predicted_labels == class_index) & (target_array != class_index))
        false_negative = np.sum((predicted_labels != class_index) & (target_array == class_index))

        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(float(2.0 * precision * recall / (precision + recall)))

        class_scores = prediction_array[:, class_index]
        auc_scores.append(_binary_auc_score((target_array == class_index).astype(np.int32), class_scores))

    finite_auc_scores = [score for score in auc_scores if not np.isnan(score)]
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    macro_auc = float(np.mean(finite_auc_scores)) if finite_auc_scores else 0.0

    return {
        "accuracy": accuracy,
        "f1_score": macro_f1,
        "auc": macro_auc,
    }


def _as_segmentation_mask(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        if array.shape[1] == 1:
            return (array[:, 0] >= 0.5).astype(np.uint8)
        return np.argmax(array, axis=1).astype(np.uint8)

    if array.ndim == 3:
        return (array >= 0.5).astype(np.uint8)

    raise ValueError("Segmentation predictions must have shape [samples, channels, height, width] or [samples, height, width].")


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 2), dtype=np.float64)

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = padded[1:-1, 1:-1]

    neighbor_stack = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    ]
    boundary = interior & ~np.logical_and.reduce(neighbor_stack)
    return np.argwhere(boundary).astype(np.float64)


def _directed_hausdorff(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.size == 0 and points_b.size == 0:
        return 0.0
    if points_a.size == 0 or points_b.size == 0:
        return float(inf)

    distances = np.sqrt(((points_a[:, None, :] - points_b[None, :, :]) ** 2).sum(axis=2))
    return float(distances.min(axis=1).max())


def _segmentation_metrics(predictions: Any, targets: Any) -> dict[str, float]:
    prediction_array = _to_numpy(predictions)
    target_array = _to_numpy(targets)

    predicted_masks = _as_segmentation_mask(prediction_array)
    target_masks = _as_segmentation_mask(target_array)

    if predicted_masks.shape != target_masks.shape:
        raise ValueError("Segmentation predictions and targets must have matching shapes.")

    dice_scores: list[float] = []
    iou_scores: list[float] = []
    hausdorff_scores: list[float] = []

    for predicted_mask, target_mask in zip(predicted_masks, target_masks):
        predicted_mask = predicted_mask.astype(bool)
        target_mask = target_mask.astype(bool)

        intersection = float(np.logical_and(predicted_mask, target_mask).sum())
        predicted_total = float(predicted_mask.sum())
        target_total = float(target_mask.sum())
        union = float(np.logical_or(predicted_mask, target_mask).sum())

        if predicted_total == 0.0 and target_total == 0.0:
            dice_scores.append(1.0)
            iou_scores.append(1.0)
        else:
            dice_scores.append(float((2.0 * intersection) / max(1.0, predicted_total + target_total)))
            iou_scores.append(float(intersection / max(1.0, union)))

        boundary_predicted = _mask_boundary(predicted_mask)
        boundary_target = _mask_boundary(target_mask)
        hausdorff_scores.append(max(_directed_hausdorff(boundary_predicted, boundary_target), _directed_hausdorff(boundary_target, boundary_predicted)))

    finite_hausdorff_scores = [score for score in hausdorff_scores if np.isfinite(score)]

    return {
        "dice_score": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "hausdorff_distance": float(np.mean(finite_hausdorff_scores)) if finite_hausdorff_scores else float(inf),
    }


def compute_metrics(predictions: Any, targets: Any) -> dict[str, float]:
    """Compute classification, segmentation, or combined medical metrics."""

    if isinstance(predictions, dict) and isinstance(targets, dict):
        metrics: dict[str, float] = {}

        if "classification" in predictions and "classification" in targets:
            classification_metrics = _classification_metrics(predictions["classification"], targets["classification"])
            metrics.update({f"classification_{name}": value for name, value in classification_metrics.items()})

        if "segmentation" in predictions and "segmentation" in targets:
            segmentation_metrics = _segmentation_metrics(predictions["segmentation"], targets["segmentation"])
            metrics.update({f"segmentation_{name}": value for name, value in segmentation_metrics.items()})

        return metrics

    if isinstance(predictions, dict) or isinstance(targets, dict):
        raise ValueError("Predictions and targets must both be dicts for multi-task metrics.")

    prediction_array = _to_numpy(predictions)
    target_array = _to_numpy(targets)

    if prediction_array.ndim >= 3 or target_array.ndim >= 3:
        return _segmentation_metrics(prediction_array, target_array)

    return _classification_metrics(prediction_array, target_array)


def compute_analysis(predictions: Any, targets: Any) -> dict[str, Any]:
    """Compute confusion matrix and class-wise analysis for classification outputs."""

    if isinstance(predictions, dict) and isinstance(targets, dict):
        result: dict[str, Any] = {}
        if "classification" in predictions and "classification" in targets:
            result["classification"] = _classification_analysis(predictions["classification"], targets["classification"])
        return result

    if isinstance(predictions, dict) or isinstance(targets, dict):
        raise ValueError("Predictions and targets must both be dicts for multi-task analysis.")

    return _classification_analysis(predictions, targets)