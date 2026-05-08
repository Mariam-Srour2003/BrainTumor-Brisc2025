"""Training callbacks and hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Callback:
    """Minimal callback hook container for training lifecycle events."""

    name: str

    def on_train_start(self) -> None:
        return None

    def on_epoch_start(self, epoch: int) -> None:
        return None

    def on_epoch_end(self, epoch: int) -> None:
        return None

    def on_batch_end(self, batch_index: int, metrics: dict[str, float] | None = None) -> None:
        return None

    def on_train_end(self, metrics: dict[str, float] | None = None) -> None:
        return None


def run_callbacks(callbacks: list[Callback], hook_name: str, *args: Any, **kwargs: Any) -> None:
    """Invoke a lifecycle hook on every callback that exposes it."""

    for callback in callbacks:
        hook = getattr(callback, hook_name, None)
        if callable(hook):
            hook(*args, **kwargs)


class MetricTracker:
    """Accumulates per-epoch metrics for post-training analysis.

    Compatible with the ``progress_callback`` signature used by Trainer.fit_*.
    """

    def __init__(self) -> None:
        self.epoch_metrics: list[dict[str, Any]] = []
        self.best_val_loss: float = float("inf")

    def __call__(self, event: dict[str, Any]) -> None:
        if event.get("type") == "epoch":
            self.epoch_metrics.append(dict(event))
            val = float(event.get("val_loss", float("inf")))
            if val < self.best_val_loss:
                self.best_val_loss = val
        elif event.get("type") == "complete":
            self.best_val_loss = float(event.get("best_val_loss", self.best_val_loss))

    def summary(self) -> dict[str, Any]:
        if not self.epoch_metrics:
            return {"epochs": 0, "best_val_loss": self.best_val_loss}
        last = self.epoch_metrics[-1]
        return {
            "epochs": len(self.epoch_metrics),
            "best_val_loss": self.best_val_loss,
            "final_train_loss": last.get("train_loss"),
            "final_val_loss": last.get("val_loss"),
        }


class EpochProgressCallback:
    """Logs epoch progress to a callable (e.g. ``print`` or a logger).

    Compatible with the ``progress_callback`` signature used by Trainer.fit_*.
    """

    def __init__(self, log_fn: Callable[[str], None] | None = None) -> None:
        self._log = log_fn or print

    def __call__(self, event: dict[str, Any]) -> None:
        if event.get("type") == "epoch":
            ep, tot = event.get("epoch", "?"), event.get("total", "?")
            tl = event.get("train_loss")
            vl = event.get("val_loss")
            parts = [f"Epoch {ep}/{tot}"]
            if tl is not None:
                parts.append(f"train={tl:.4f}")
            if vl is not None:
                parts.append(f"val={vl:.4f}")
            self._log("  ".join(parts))
        elif event.get("type") == "complete":
            bv = event.get("best_val_loss")
            self._log(f"Training complete — best_val_loss={bv:.4f}" if bv is not None else "Training complete.")