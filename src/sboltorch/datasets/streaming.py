"""Streaming, split-aware, multi-worker iteration over a corpus of SbolObjects.

For a corpus too large to hold in memory: iterate shard by shard, keep only the
records whose hashed IRI lands in the requested split, and encode lazily. Under a
multi-worker DataLoader each worker takes a disjoint slice of the stream, so the
union across workers is exactly the split with no duplication.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator

from torch.utils.data import IterableDataset, get_worker_info

from sboltorch.datasets.splits import split_of
from sboltorch.encoders.base import SupportsEncode
from sboltorch.types import SbolObject


def iter_split_records(
    source: Iterable[SbolObject],
    which: str | None,
    ratios: tuple[float, float, float],
    seed: int,
) -> Iterator[SbolObject]:
    """Yield the records of ``source`` for this worker that fall in ``which`` split.

    Worker partitioning prefers whole-shard assignment when the source exposes
    ``iter_for_worker`` (the materialized corpus does), avoiding every worker
    re-reading every shard; otherwise it falls back to round-robin by record.
    ``which=None`` keeps every record (no split filtering).
    """
    info = get_worker_info()
    num_workers = info.num_workers if info is not None else 1
    worker_id = info.id if info is not None else 0

    iter_for_worker = getattr(source, "iter_for_worker", None)
    if callable(iter_for_worker):
        records: Iterable[SbolObject] = iter_for_worker(worker_id, num_workers)
    else:
        records = (obj for i, obj in enumerate(source) if i % num_workers == worker_id)

    for obj in records:
        if which is None or split_of(obj.iri, ratios, seed) == which:
            yield obj


def _shuffle_buffered(stream: Iterator[SbolObject], buffer_size: int, seed: int) -> Iterator[SbolObject]:
    """Approximate shuffle: emit from a fixed-size reservoir as it fills."""
    rng = random.Random(seed)
    buffer: list[SbolObject] = []
    for item in stream:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            yield buffer.pop(rng.randrange(len(buffer)))
    rng.shuffle(buffer)
    yield from buffer


class StreamingEncodedDataset(IterableDataset):
    """Lazily encodes the records of one split, streamed from a corpus."""

    def __init__(
        self,
        source: Iterable[SbolObject],
        encoder: SupportsEncode,
        *,
        which: str,
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        shuffle_buffer: int = 0,
    ) -> None:
        self._source = source
        self._encoder = encoder
        self._which = which
        self._ratios = ratios
        self._seed = seed
        self._shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[object]:
        stream = iter_split_records(self._source, self._which, self._ratios, self._seed)
        if self._shuffle_buffer > 1:
            info = get_worker_info()
            worker_id = info.id if info is not None else 0
            stream = _shuffle_buffered(stream, self._shuffle_buffer, self._seed + worker_id)
        for obj in stream:
            yield self._encoder.encode(obj)
