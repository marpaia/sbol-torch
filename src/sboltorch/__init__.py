"""sboltorch — a PyTorch library for synthetic biology and biodesign automation.

Installed as ``sbol-torch``; imported as ``sboltorch``, commonly::

    import sboltorch as st

    config = st.RunConfig.from_yaml("run.yaml")
    metrics = st.run_training(config)
"""

from __future__ import annotations

from .config import RunConfig
from .data.corpus import build_corpus
from .data.materialize import materialize
from .pipeline import prepare_data, run_training
from .types import Alphabet, Feature, SbolObject, SbolSequence

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "SbolObject",
    "SbolSequence",
    "Feature",
    "Alphabet",
    "run_training",
    "prepare_data",
    "build_corpus",
    "materialize",
    "__version__",
]
