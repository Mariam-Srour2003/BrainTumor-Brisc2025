"""Utilities for exporting evaluation artifacts to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _colorize_cell(intensity: float) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, intensity))
    red = int(245 - 155 * clamped)
    green = int(247 - 190 * clamped)
    blue = int(250 - 220 * clamped)
    return red, green, blue


def save_confusion_matrix_image(
    confusion_matrix: list[list[int]] | np.ndarray,
    class_names: list[str] | tuple[str, ...],
    output_path: str | Path,
    *,
    title: str = "Confusion Matrix",
) -> str:
    """Render a simple confusion matrix image using Pillow only."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("Install pillow to export confusion matrix images.") from exc

    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    labels = [str(label) for label in class_names]
    size = max(1, matrix.shape[0])

    cell_size = 88
    margin_left = 140
    margin_top = 110
    width = margin_left + cell_size * size + 30
    height = margin_top + cell_size * size + 70

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((20, 20), title, fill="black", font=font)
    draw.text((margin_left + (cell_size * size) // 2 - 50, 70), "Predicted", fill="black", font=font)
    draw.text((20, margin_top + (cell_size * size) // 2), "Actual", fill="black", font=font)

    max_value = float(matrix.max()) if matrix.size else 0.0
    for row_index in range(size):
        for col_index in range(size):
            value = float(matrix[row_index, col_index]) if row_index < matrix.shape[0] and col_index < matrix.shape[1] else 0.0
            norm = 0.0 if max_value <= 0.0 else value / max_value
            fill = _colorize_cell(norm)
            x0 = margin_left + col_index * cell_size
            y0 = margin_top + row_index * cell_size
            draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=fill, outline="black")
            text = str(int(value))
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x0 + (cell_size - text_width) / 2, y0 + (cell_size - text_height) / 2), text, fill="black", font=font)

    for index, label in enumerate(labels):
        x = margin_left + index * cell_size + 10
        draw.text((x, margin_top - 25), label, fill="black", font=font)
        y = margin_top + index * cell_size + 10
        draw.text((25, y), label, fill="black", font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def export_evaluation_report(
    output_root: str | Path,
    *,
    model_name: str,
    task: str,
    split: str,
    metrics: dict[str, Any],
    analysis: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Persist evaluation metrics, analysis, and a confusion matrix image when available."""

    output_root = Path(output_root)
    safe_model_name = model_name.replace("/", "_")
    report_dir = output_root / safe_model_name / split
    report_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "model_name": model_name,
        "task": task,
        "split": split,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "metrics": _to_jsonable(metrics),
        "analysis": _to_jsonable(analysis or {}),
    }

    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    confusion_matrix_path = report_dir / "confusion_matrix.png"
    classification_analysis: dict[str, Any] | None = None
    if analysis:
        if "classification" in analysis and isinstance(analysis["classification"], dict):
            classification_analysis = analysis["classification"]
        elif "confusion_matrix" in analysis:
            classification_analysis = analysis

    if classification_analysis is not None:
        confusion = classification_analysis.get("confusion_matrix")
        classes = classification_analysis.get("classes") or []
        labels = [str(label) for label in (class_names or classes)]
        if confusion is not None and labels:
            save_confusion_matrix_image(confusion, labels, confusion_matrix_path, title=f"{model_name} - {split}")

    analysis_path = report_dir / "analysis.json"
    analysis_path.write_text(json.dumps(_to_jsonable(analysis or {}), indent=2, sort_keys=True), encoding="utf-8")

    summary_path = report_dir / "summary.txt"
    summary_lines = [
        f"model_name: {model_name}",
        f"task: {task}",
        f"split: {split}",
        f"checkpoint_path: {checkpoint_path}",
        f"metrics_path: {metrics_path}",
        f"analysis_path: {analysis_path}",
    ]
    if confusion_matrix_path.exists():
        summary_lines.append(f"confusion_matrix_path: {confusion_matrix_path}")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "report_dir": str(report_dir),
        "metrics_path": str(metrics_path),
        "analysis_path": str(analysis_path),
        "summary_path": str(summary_path),
        "confusion_matrix_path": str(confusion_matrix_path) if confusion_matrix_path.exists() else "",
    }