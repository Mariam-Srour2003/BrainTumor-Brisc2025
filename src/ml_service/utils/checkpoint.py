"""Checkpoint helpers for saving and loading model state."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _extract_state_dict(model_or_state: Any) -> dict[str, Any]:
    """Extract a PyTorch-compatible state_dict from a model wrapper or raw module."""

    target = getattr(model_or_state, "model", model_or_state)

    if hasattr(target, "state_dict"):
        return target.state_dict()

    if isinstance(model_or_state, dict):
        if "state_dict" in model_or_state and isinstance(model_or_state["state_dict"], dict):
            return model_or_state["state_dict"]
        if "model_state_dict" in model_or_state and isinstance(model_or_state["model_state_dict"], dict):
            return model_or_state["model_state_dict"]

    raise TypeError("Checkpoint save expects a torch model, wrapper with .model, or a dict containing state_dict.")


def _build_checkpoint_payload(model_or_state: Any) -> dict[str, Any]:
    """Build a checkpoint payload while preserving extra fields when already provided."""

    if isinstance(model_or_state, dict) and "state_dict" in model_or_state:
        return dict(model_or_state)

    return {"state_dict": _extract_state_dict(model_or_state)}


def sanitize_checkpoint_name(name: str) -> str:
    """Normalize model/checkpoint names to filesystem-safe path segments."""

    safe = []
    for char in name.strip():
        if char.isalnum() or char in {".", "_", "-"}:
            safe.append(char)
        else:
            safe.append("_")

    collapsed = "".join(safe).strip("._-")
    return collapsed or "model"


def format_checkpoint_run_date(now: datetime | None = None) -> str:
    """Return a sortable timestamp token used as checkpoint run folder name."""

    return (now or datetime.utcnow()).strftime("%Y-%m-%d_%H-%M-%S")


def model_checkpoint_dir(checkpoint_root: Path, model_name: str) -> Path:
    """Return the root checkpoint directory for one model family."""

    return checkpoint_root / sanitize_checkpoint_name(model_name)


def model_checkpoint_run_dir(checkpoint_root: Path, model_name: str, run_date: str) -> Path:
    """Return the model/date checkpoint folder path."""

    return model_checkpoint_dir(checkpoint_root, model_name) / sanitize_checkpoint_name(run_date)


def versioned_checkpoint_path(checkpoint_root: Path, model_name: str, run_date: str, filename: str) -> Path:
    """Build a versioned checkpoint path under models/checkpoints/<model>/<date>/..."""

    return model_checkpoint_run_dir(checkpoint_root, model_name, run_date) / filename


def resolve_checkpoint_path(
    checkpoint_root: Path,
    checkpoint_name: str,
    model_date: str | None = None,
) -> Path | None:
    """Resolve a checkpoint name to an existing artifact on disk."""

    safe_model_name = sanitize_checkpoint_name(checkpoint_name)

    direct_candidates = [
        checkpoint_root / f"{safe_model_name}.pt",
        checkpoint_root / f"{safe_model_name}.train.pt",
    ]
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    model_dir = checkpoint_root / safe_model_name
    if model_date:
        safe_date = sanitize_checkpoint_name(model_date)
        dated_dir = model_dir / safe_date
        if dated_dir.exists() and dated_dir.is_dir():
            dated_candidates = sorted(
                (path for path in dated_dir.rglob("*.pt") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if dated_candidates:
                return dated_candidates[0]

    if model_dir.exists() and model_dir.is_dir():
        model_candidates = sorted(
            (path for path in model_dir.rglob("*.pt") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if model_candidates:
            return model_candidates[0]

    loose_candidates = sorted(
        (path for path in checkpoint_root.glob(f"{safe_model_name}*.pt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if loose_candidates:
        return loose_candidates[0]

    # Partial-match fallback: find the checkpoint whose filename contains the most
    # tokens from the requested name. Useful when e.g. hybrid.adpt_net.ax is looked up
    # but only hybrid.tgd_adpt_net.ax exists on disk.
    tokens = [t for t in safe_model_name.replace("-", "_").split(".") if len(t) > 1]
    if tokens:
        all_pts = sorted(
            (p for p in checkpoint_root.rglob("*.pt") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        best_path: Path | None = None
        best_score = 0
        for pt in all_pts:
            stem = pt.stem.replace("-", "_")
            score = sum(1 for t in tokens if t in stem)
            if score > best_score:
                best_score = score
                best_path = pt
        if best_score >= max(1, len(tokens) - 1):
            return best_path

    return None


def clear_model_checkpoints(checkpoint_root: Path, model_name: str) -> list[str]:
    """Delete all known checkpoint artifacts for one model and return removed paths."""

    removed: list[str] = []
    safe_model_name = sanitize_checkpoint_name(model_name)

    model_dir = checkpoint_root / safe_model_name
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)
        removed.append(str(model_dir))

    for pattern in (f"{safe_model_name}*.pt", f"{safe_model_name}*.json"):
        for artifact in checkpoint_root.glob(pattern):
            if artifact.is_file():
                artifact.unlink(missing_ok=True)
                removed.append(str(artifact))

    return removed


def save_checkpoint(path: Path, model_or_state: Any, metadata: dict[str, Any] | None = None) -> dict[str, str]:
    """Write model weights as a torch checkpoint and save metadata in adjacent JSON."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install torch to save model checkpoints.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = _build_checkpoint_payload(model_or_state)
    torch.save(checkpoint_payload, path)

    metadata_path = path.with_suffix(".json")
    metadata_payload = {
        "checkpoint_path": str(path),
        "saved_at_utc": datetime.utcnow().isoformat() + "Z",
        **(metadata or {}),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "checkpoint_path": str(path),
        "metadata_path": str(metadata_path),
    }


def load_checkpoint(path: Path, map_location: str | None = "cpu") -> dict[str, Any]:
    """Load torch checkpoint payload and optional metadata JSON sidecar."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install torch to load model checkpoints.") from exc

    checkpoint_payload = torch.load(path, map_location=map_location)
    metadata_path = path.with_suffix(".json")
    metadata_payload: dict[str, Any] = {}
    if metadata_path.exists():
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "checkpoint": checkpoint_payload,
        "metadata": metadata_payload,
    }