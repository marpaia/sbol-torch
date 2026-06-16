"""End-to-end orchestration: config -> corpus -> dataset -> model -> training.

This wires the layers together for the supervised / frozen-backbone sequence
path. It is intentionally a straight-line function so the flow is readable; each
step delegates to a swappable component built from config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import DataLoader

from .config import RunConfig
from .data.corpus import build_corpus
from .data.materialize import MaterializedCorpus, materialize
from .datasets.dataset import Collator, EncodedDataset
from .datasets.mlm_collator import MlmCollator
from .datasets.splits import Split, make_split
from .encoders.base import build_encoder
from .engine.callbacks import Callback, EarlyStopping, MetricLogger, ModelCheckpoint, WandbLogger
from .engine.trainer import Trainer
from .models import build_model
from .models.mlm import MaskedLMModel
from .reproducibility import set_seed
from .tasks.base import Task, build_task
from .tokenize.base import build_tokenizer
from .types import SbolObject


@dataclass
class PreparedData:
    corpus: MaterializedCorpus
    objects: list[SbolObject]
    split: Split


def prepare_data(config: RunConfig) -> PreparedData:
    """Materialize the corpus and compute the seeded split."""
    corpus = build_corpus(config.corpus)
    materialized = materialize(corpus, config.corpus.cache_dir)
    objects = materialized.read_all()

    supervised = config.task.kind in ("supervised", "frozen")
    labels = [o.label for o in objects] if supervised and config.splits.strategy == "stratified" else None
    if labels is not None and any(label is None for label in labels):
        labels = None  # cannot stratify on partially-unlabeled data
    split = make_split(
        len(objects),
        ratios=config.splits.ratios,
        seed=config.seed,
        labels=labels,  # type: ignore[arg-type]
        strategy=config.splits.strategy,
    )
    return PreparedData(corpus=materialized, objects=objects, split=split)


def _loader(
    objects: list[SbolObject],
    indices: tuple[int, ...],
    encoder: object,
    collator: Callable[[list[Any]], Any],
    config: RunConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = EncodedDataset([objects[i] for i in indices], encoder)  # type: ignore[arg-type]
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        collate_fn=collator,
    )


def _build_sequence_run(config: RunConfig, data: PreparedData, task: Task) -> tuple:
    """Build (model, train_loader, val_loader, adapter) for the sequence/MLM path."""
    tokenizer = build_tokenizer(config.tokenizer)
    encoder = build_encoder(config.encoder, tokenizer)
    spec = encoder.output_spec
    model = build_model(config.model, config.task, vocab_size=spec.vocab_size, pad_token_id=spec.pad_token_id)
    collator: Callable[[list[Any]], Any]
    if config.task.kind == "mlm":
        collator = MlmCollator(tokenizer, mlm_probability=config.task.mlm_probability)
    else:
        collator = Collator(tokenizer.pad_token_id, with_labels=True, label_dtype=task.label_dtype)
    train_loader = _loader(data.objects, data.split.train, encoder, collator, config, shuffle=True)
    val_loader = _loader(data.objects, data.split.val, encoder, collator, config, shuffle=False)
    return model, train_loader, val_loader, None


def _build_graph_run(config: RunConfig, data: PreparedData) -> tuple:
    """Build (model, train_loader, val_loader, adapter) for the graph path."""
    from torch_geometric.loader import DataLoader as GeoLoader

    from .encoders.graph import GraphEncoder
    from .encoders.structure import DEFAULT_ROLES
    from .engine.batch import GraphBatchAdapter
    from .models.graph import build_graph_model

    encoder = GraphEncoder(roles=config.encoder.roles or DEFAULT_ROLES)
    model = build_graph_model(config.model, config.task, encoder.spec)

    def loader(indices: tuple[int, ...], *, shuffle: bool) -> GeoLoader:
        dataset = EncodedDataset([data.objects[i] for i in indices], encoder)
        return GeoLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=shuffle,
            num_workers=config.train.num_workers,
        )

    return model, loader(data.split.train, shuffle=True), loader(data.split.val, shuffle=False), GraphBatchAdapter()


def run_training(config: RunConfig) -> dict[str, float]:
    """Run the full training pipeline and return the final epoch's metrics."""
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(config.to_yaml())

    data = prepare_data(config)
    task = build_task(config.task)

    if config.encoder.kind == "graph":
        model, train_loader, val_loader, adapter = _build_graph_run(config, data)
    else:
        model, train_loader, val_loader, adapter = _build_sequence_run(config, data, task)

    metric_name, mode = task.primary_metric
    monitored = f"val_{metric_name}"
    callbacks: list[Callback] = [
        MetricLogger(output_dir),
        ModelCheckpoint(output_dir, monitor=monitored, mode=mode),
    ]
    if config.train.early_stop is not None:
        es = config.train.early_stop
        callbacks.append(EarlyStopping(monitor=es.monitor, mode=es.mode, patience=es.patience, min_delta=es.min_delta))
    if config.wandb.enabled:
        callbacks.append(WandbLogger(config, data.corpus, data.split, output_dir))

    trainer = Trainer(model, task, config.train, callbacks=callbacks, batch_adapter=adapter)
    metrics = trainer.fit(train_loader, val_loader)
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Write the pretrained model in HF format so a later supervised run can set
    # `model.backbone` to this directory and load it as a plain encoder.
    if isinstance(model, MaskedLMModel):
        backbone_dir = output_dir / "backbone"
        model.save_pretrained(backbone_dir)

    return metrics
