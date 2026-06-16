"""The Corpus protocol — the single interface training code reads data through."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from ..config import CorpusConfig
from ..types import SbolObject


@runtime_checkable
class Corpus(Protocol):
    """A stream of normalized SbolObject records from any source."""

    def __iter__(self) -> Iterator[SbolObject]: ...

    def fingerprint(self) -> str:
        """A stable content hash identifying this corpus for caching/reproducibility."""
        ...


def build_corpus(config: CorpusConfig) -> Corpus:
    """Construct the corpus implementation named by ``config.source``."""
    if config.source == "sbol_db":
        from .sbol_db import SbolDbClient

        return SbolDbClient.from_config(config)
    if config.source == "local":
        from .local import LocalFileCorpus

        return LocalFileCorpus.from_config(config)
    if config.source == "synthetic":
        from .synthetic import SyntheticCorpus

        return SyntheticCorpus(config.n, seed=config.synthetic_seed, with_labels=config.label_key is not None)
    raise ValueError(f"unknown corpus source: {config.source}")  # pragma: no cover
