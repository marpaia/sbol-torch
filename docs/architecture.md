# Architecture

sbol-torch turns SBOL designs into trained transformer models. Three ideas shape
it: every source normalizes to one record type, the parts that vary plug in
behind protocols, and the training engine stays small and explicit.

## One record type

Every data source — the sbol-db REST API, local SBOL/FASTA files, or the synthetic
generator — is normalized into `SbolObject` (`sboltorch.types`):

```
SbolObject(iri, sbol_class, roles, types, sequence, features, neighbors, label, raw)
```

Training code consumes only `SbolObject`s and never branches on provenance. A
source is anything satisfying the `Corpus` protocol (`__iter__` yielding
`SbolObject`s, plus a `fingerprint()` for caching).

## Swappable plug points

Three independent axes are each a `Protocol` with interchangeable implementations,
selected by configuration:

- **Tokenizer** — how a sequence becomes tokens (`hf`, `kmer`, `char`).
- **Encoder** — the input modality, turning an `SbolObject` into model input
  (`sequence`, `structure_aware`, `graph`).
- **Task** — the training objective, owning loss and metrics (`supervised`,
  `frozen`, `mlm`).

Adding an implementation and registering it in the matching `build_*` factory
extends a capability without touching the engine. See [extending.md](extending.md).

## The training engine

The training loop (`sboltorch.engine`) is plain PyTorch: AMP, gradient
accumulation and clipping, a linear warmup/decay schedule, and a list of
callbacks (`EarlyStopping`, `ModelCheckpoint`, `MetricLogger`). It learns the
batch shape only through a `BatchAdapter`, so the same loop trains a sequence
model (tensor-dict batches) or a graph model (PyG `Batch` objects).

## Data flow

A `RunConfig` drives the whole pipeline. The configured `Corpus` source is
materialized to Parquet and split into train/val/test (seeded). The `Encoder`
turns each `SbolObject` into model input for its modality, using the `Tokenizer`,
and a `DataLoader` batches the result through a collator. The `Trainer` then runs
the loop under the `Task` and `BatchAdapter`, writing a checkpoint and metrics.

## Layers

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config | `sboltorch.config` | One Pydantic `RunConfig` per run; validated, serialized. |
| Data | `sboltorch.data` | Corpus sources (`SbolDbClient`, `LocalFileCorpus`, synthetic) and Parquet materialization. |
| Tokenize | `sboltorch.tokenize` | `hf` / `kmer` / `char` behind one protocol. |
| Encoders | `sboltorch.encoders` | Turn an `SbolObject` into model input, per modality. |
| Datasets | `sboltorch.datasets` | Torch `Dataset`, padding collator, MLM collator, seeded splits. |
| Models | `sboltorch.models` | Backbone (pretrained or from-scratch) + pooling + head; MLM and graph models. |
| Tasks | `sboltorch.tasks` | Loss, metrics, label dtype, target transform. |
| Engine | `sboltorch.engine` | Training loop, callbacks, batch adapters. |
| Pipeline | `sboltorch.pipeline` | Wires the layers from a `RunConfig`. |

## Key protocols

- `Corpus`: `__iter__() -> Iterator[SbolObject]`, `fingerprint() -> str`.
- `Tokenizer`: `encode`, `tokenize_content`, `vocab_size`, `pad_token_id`,
  `mask_token_id`, `special_token_ids`, `max_length`.
- `Encoder`: `encode(SbolObject) -> ModelInput`, `output_spec -> EncoderSpec`.
- `Task`: `loss`, `predict`, `epoch_metrics`, `primary_metric`, `label_dtype`.
- `BatchAdapter`: `to_device`, `forward(model, batch)`, `labels(batch)`.
- `Callback`: `on_train_start`, `on_epoch_end`, `on_train_end`.

## Reproducibility

- A run is fully specified by its `RunConfig`; the resolved config is written to
  `<output_dir>/config.resolved.yaml`.
- `seed` seeds Python, NumPy, and torch, and the train/val/test split is a pure
  function of `(n, ratios, seed, strategy)`.
- Corpora are materialized to content-fingerprinted Parquet, so a run is offline
  and byte-for-byte comparable across executions. See [data.md](data.md).

## Consuming sbol-db

`SbolDbClient` reads designs over the sbol-db REST API: keyset-paginated object
listing, single/bulk IRI resolution, bounded neighborhood traversal, sequence
search, and ontology descendant expansion. Sequence elements are read from each
object's JSON-LD slice by predicate local-name. Details in [data.md](data.md).
