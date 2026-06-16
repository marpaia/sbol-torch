"""Data layer: corpus sources and reproducible materialization."""

from __future__ import annotations

from .corpus import Corpus, build_corpus
from .local import LocalFileCorpus
from .materialize import MaterializedCorpus, materialize
from .sbol_db import SbolDbClient
from .synthetic import SyntheticCorpus, generate_components, write_sbol_turtle

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
