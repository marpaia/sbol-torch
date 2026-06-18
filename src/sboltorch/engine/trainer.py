"""A minimal, explicit raw-PyTorch training loop.

This is the only place the library re-implements training infrastructure.
It is deliberately small and boring: AMP, gradient accumulation, gradient
clipping, a linear warmup/decay schedule, and a list of callbacks for early
stopping / checkpointing / logging. Everything model- or task-specific lives
behind the Task and model abstractions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from sboltorch.config import TrainConfig
from sboltorch.engine.batch import BatchAdapter, TensorBatchAdapter
from sboltorch.tasks.base import Task


class Callback:
    """The hooks the training loop invokes. Concrete callbacks live in
    ``sboltorch.engine.callbacks``."""

    def on_train_start(self, trainer: Trainer) -> None:
        return None

    def on_step_end(self, trainer: Trainer, step: int, logs: dict[str, float]) -> None:
        return None

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        return None

    def on_train_end(self, trainer: Trainer) -> None:
        return None


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _linear_schedule(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        config: TrainConfig,
        *,
        callbacks: Sequence[Callback] | None = None,
        device: torch.device | None = None,
        batch_adapter: BatchAdapter | None = None,
    ) -> None:
        self.model = model
        self.task = task
        self.config = config
        self.callbacks = list(callbacks or [])
        self.device = device or select_device()
        self.adapter = batch_adapter or TensorBatchAdapter()
        self.should_stop = False
        self.global_step = 0
        self.model.to(self.device)

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, float]:
        optimizer = torch.optim.AdamW(
            self._trainable_params(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        steps_per_epoch = max(1, len(train_loader) // self.config.grad_accum)
        total_steps = steps_per_epoch * self.config.epochs
        scheduler = _linear_schedule(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
        use_amp = self.config.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for cb in self.callbacks:
            cb.on_train_start(self)

        last_metrics: dict[str, float] = {}
        try:
            for epoch in range(self.config.epochs):
                train_loss = self._train_epoch(train_loader, optimizer, scheduler, scaler, use_amp)
                metrics = {"train_loss": train_loss}
                if val_loader is not None:
                    metrics.update(self._validate(val_loader))
                last_metrics = metrics
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, metrics)
                if self.should_stop:
                    break
        finally:
            # Always run teardown — closes log handles and finishes the W&B run
            # even if an epoch raises.
            for cb in self.callbacks:
                cb.on_train_end(self)
        return last_metrics

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        scaler: torch.amp.GradScaler,
        use_amp: bool,
    ) -> float:
        self.model.train()
        optimizer.zero_grad()
        total = 0.0
        count = 0
        for step, batch in enumerate(loader):
            batch = self.adapter.to_device(batch, self.device)
            labels = self.adapter.labels(batch)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                logits = self.adapter.forward(self.model, batch)
                loss = self.task.loss(logits, labels) / self.config.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % self.config.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._trainable_params(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1
                logs: dict[str, float] = {
                    "step_loss": float(loss.item() * self.config.grad_accum),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                for cb in self.callbacks:
                    cb.on_step_end(self, self.global_step, logs)
            total += loss.item() * self.config.grad_accum
            count += 1
        return total / max(1, count)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        preds: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        for batch in loader:
            batch = self.adapter.to_device(batch, self.device)
            labels = self.adapter.labels(batch)
            logits = self.adapter.forward(self.model, batch)
            losses.append(self.task.loss(logits, labels).item())
            # Ravel to 1-D so batches of differing sequence length (MLM) still
            # concatenate; regression/classification predictions are already 1-D.
            preds.append(self.task.predict(logits).detach().cpu().numpy().ravel())
            labels_all.append(labels.detach().cpu().numpy().ravel())
        metrics = {"val_loss": float(np.mean(losses)) if losses else 0.0}
        if preds:
            metrics.update(self.task.epoch_metrics(np.concatenate(preds), np.concatenate(labels_all)))
        return {f"val_{k}" if not k.startswith("val_") else k: v for k, v in metrics.items()}

    def save_checkpoint(self, path: str | Path, *, epoch: int, metrics: dict[str, float]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "epoch": epoch, "metrics": metrics},
            path,
        )
