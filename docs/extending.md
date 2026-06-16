# Extending sboltorch

sbol-torch is a small set of `Protocol`s with interchangeable implementations.
Adding a capability means writing one implementation and registering it in the
relevant `build_*` factory — the training engine and pipeline are untouched.

## Add a tokenizer

Implement the `Tokenizer` protocol (`sboltorch.tokenize.base`):

```python
class MyTokenizer:
    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def mask_token_id(self) -> int | None: ...
    @property
    def special_token_ids(self) -> frozenset[int]: ...
    @property
    def max_length(self) -> int: ...
    def tokenize_content(self, sequence: str) -> list[int]: ...   # no special wrapping
    def encode(self, sequence: str) -> Encoded: ...               # with <cls>/<sep>
```

Register it in `build_tokenizer` and add a `kind` to `TokenizerConfig`.

## Add an encoder (input modality)

A tensor encoder implements `Encoder` (`sboltorch.encoders.base`):

```python
class MyEncoder:
    def encode(self, obj: SbolObject) -> ModelInput: ...   # input_ids, attention_mask, label
    @property
    def output_spec(self) -> EncoderSpec: ...              # vocab_size, pad/mask ids, max_length
```

Register it in `build_encoder` and add a `kind` to `EncoderConfig`. Encoders that
produce a non-tensor batch (like the graph encoder, which returns a PyG `Data`)
are wired in the pipeline alongside a matching `BatchAdapter` and loader.

## Add an objective (task)

Implement the `Task` protocol (`sboltorch.tasks.base`):

```python
class MyTask:
    label_dtype: str  # "float" or "long"
    def loss(self, logits, labels): ...
    def predict(self, logits): ...
    def epoch_metrics(self, preds, labels) -> dict[str, float]: ...
    @property
    def primary_metric(self) -> tuple[str, str]: ...   # (name, "min"|"max")
```

Register it in `build_task` and add a `kind` to `TaskConfig`. If the objective
needs a different head/model, extend `build_model`.

## Add a callback

Subclass `Callback` (`sboltorch.engine.callbacks`) and override
`on_train_start` / `on_epoch_end` / `on_train_end`. The bundled callbacks are
`EarlyStopping`, `ModelCheckpoint`, and `MetricLogger`; the pipeline assembles
them from config.

## Add a batch modality

To train on a batch shape the engine doesn't yet understand, implement a
`BatchAdapter` (`sboltorch.engine.batch`):

```python
class MyBatchAdapter:
    def to_device(self, batch, device): ...
    def forward(self, model, batch) -> torch.Tensor: ...   # logits
    def labels(self, batch) -> torch.Tensor: ...
```

Pass it to `Trainer(..., batch_adapter=...)`. `TensorBatchAdapter` handles
`dict[str, Tensor]` batches; `GraphBatchAdapter` handles PyG `Batch` objects.

## Add a data source

Implement the `Corpus` protocol (`sboltorch.data.corpus`) — `__iter__` yielding
`SbolObject`s and a `fingerprint()` for caching — and register it in
`build_corpus` with a new `CorpusConfig.source`. Downstream code (materialization,
splitting, encoding, training) works unchanged.
