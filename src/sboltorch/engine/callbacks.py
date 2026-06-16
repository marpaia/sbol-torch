"""Training callbacks.

Raw PyTorch gives us no early stopping or checkpointing, so we provide small,
explicit callbacks instead of a heavyweight framework. Each is independently
testable and the Trainer simply invokes them in order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback:
    def on_train_start(self, trainer: "Trainer") -> None:
        return None

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        return None

    def on_train_end(self, trainer: "Trainer") -> None:
        return None


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


class EarlyStopping(Callback):
    def __init__(self, monitor: str, mode: str = "min", patience: int = 5, min_delta: float = 0.0) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf") if mode == "min" else float("-inf")
        self._waited = 0

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        if _is_improvement(current, self._best, self.mode, self.min_delta):
            self._best = current
            self._waited = 0
        else:
            self._waited += 1
            if self._waited >= self.patience:
                trainer.should_stop = True


class ModelCheckpoint(Callback):
    """Saves the model state dict whenever the monitored metric improves."""

    def __init__(self, output_dir: str | Path, monitor: str, mode: str = "min") -> None:
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self._best = float("inf") if mode == "min" else float("-inf")
        self.best_path: Path | None = None

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        if not _is_improvement(current, self._best, self.mode, min_delta=0.0):
            return
        self._best = current
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.output_dir / "best.pt"
        trainer.save_checkpoint(self.best_path, epoch=epoch, metrics=metrics)


class MetricLogger(Callback):
    """Prints per-epoch metrics and appends them to ``metrics.jsonl``."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._handle: Any = None

    def on_train_start(self, trainer: "Trainer") -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handle = (self.output_dir / "metrics.jsonl").open("a")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        record = {"epoch": epoch, **metrics}
        line = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
        print(line)
        if self._handle is not None:
            self._handle.write(json.dumps(record) + "\n")
            self._handle.flush()

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
