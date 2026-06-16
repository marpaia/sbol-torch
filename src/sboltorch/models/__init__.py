"""Model construction: assemble a backbone + head (or an MLM model) from config."""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig, TaskConfig
from .backbone import build_from_scratch_encoder, load_backbone
from .heads import ClassificationHead, RegressionHead
from .mlm import MaskedLMModel, build_mlm_model
from .sequence_model import SequenceModel

__all__ = [
    "build_model",
    "SequenceModel",
    "MaskedLMModel",
    "RegressionHead",
    "ClassificationHead",
    "load_backbone",
]


def build_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    *,
    vocab_size: int | None = None,
    pad_token_id: int | None = None,
) -> nn.Module:
    """Build the model for the given task.

    - ``mlm`` → a MaskedLMModel (from-scratch needs ``vocab_size``/``pad_token_id``).
    - ``frozen`` → a SequenceModel with a frozen backbone and a trainable head.
    - ``supervised`` → a SequenceModel fine-tuned end to end.
    """
    if task_config.kind == "mlm":
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required to build an MLM model")
        return build_mlm_model(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)

    if model_config.from_scratch:
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required for a from-scratch model")
        backbone, hidden_size = build_from_scratch_encoder(
            model_config, vocab_size=vocab_size, pad_token_id=pad_token_id
        )
    else:
        backbone, hidden_size = load_backbone(model_config.backbone)
    head: nn.Module
    if task_config.objective == "classification":
        assert task_config.num_classes is not None  # enforced by TaskConfig validation
        head = ClassificationHead(hidden_size, task_config.num_classes, model_config.dropout)
    else:
        head = RegressionHead(hidden_size, model_config.dropout)
    return SequenceModel(backbone, head, freeze_backbone=task_config.kind == "frozen")
