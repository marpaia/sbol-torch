"""Encoders: the input-modality plug point."""

from __future__ import annotations

from .base import Encoder, EncoderSpec, ModelInput, build_encoder
from .graph import GraphEncoder, GraphSpec
from .sequence import SequenceEncoder
from .structure import StructureAwareEncoder

__all__ = [
    "Encoder",
    "EncoderSpec",
    "ModelInput",
    "build_encoder",
    "SequenceEncoder",
    "StructureAwareEncoder",
    "GraphEncoder",
    "GraphSpec",
]
