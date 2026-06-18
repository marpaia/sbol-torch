"""Encoders: the input-modality plug point."""

from __future__ import annotations

from sboltorch.encoders.base import Encoder, EncoderSpec, ModelInput, build_encoder
from sboltorch.encoders.graph import GraphEncoder, GraphSpec
from sboltorch.encoders.sequence import SequenceEncoder
from sboltorch.encoders.structure import StructureAwareEncoder

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
