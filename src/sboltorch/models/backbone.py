"""Construct transformer encoder backbones — pretrained or from scratch."""

from __future__ import annotations

import torch.nn as nn
from transformers import AutoConfig, AutoModel

from sboltorch.config import ModelConfig
from sboltorch.exceptions import ConfigError


def load_backbone(model_name: str) -> tuple[nn.Module, int]:
    """Return ``(backbone_module, hidden_size)`` for a HuggingFace encoder model.

    ``model_name`` may be a hub id or a local directory (e.g. a backbone written
    out by an MLM pretraining run).
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    if hidden_size is None:
        raise ConfigError(f"could not determine hidden size for backbone {model_name}")
    return model, int(hidden_size)


def build_from_scratch_encoder(
    model_config: ModelConfig, *, vocab_size: int, pad_token_id: int
) -> tuple[nn.Module, int]:
    """Instantiate an untrained encoder sized to a given vocab (e.g. our k-mer vocab)."""
    arch = model_config.arch
    config = AutoConfig.for_model(
        arch.model_type,
        vocab_size=vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        num_attention_heads=arch.num_attention_heads,
        intermediate_size=arch.intermediate_size,
        max_position_embeddings=arch.max_position_embeddings,
        pad_token_id=pad_token_id,
    )
    return AutoModel.from_config(config), model_config.hidden_size
