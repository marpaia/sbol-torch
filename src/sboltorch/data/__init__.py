"""Data layer: corpus sources and reproducible materialization."""

from __future__ import annotations

from sboltorch.data.corpus import Corpus, build_corpus
from sboltorch.data.local import LocalFileCorpus
from sboltorch.data.materialize import MaterializedCorpus, materialize
from sboltorch.data.sbol_db import SbolDbClient
from sboltorch.data.synthetic import SyntheticCorpus, generate_components, write_sbol_turtle

__all__ = [
    "Corpus",
    "build_corpus",
    "LocalFileCorpus",
    "SbolDbClient",
    "MaterializedCorpus",
    "materialize",
    "SyntheticCorpus",
    "generate_components",
    "write_sbol_turtle",
]
