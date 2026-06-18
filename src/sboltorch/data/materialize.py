"""Materialize a corpus to a versioned Parquet cache.

A long training run should not depend on a live database, and two runs over the
"same data" must be byte-for-byte comparable. Materialization streams a corpus
once into Parquet, hashing the contents into a fingerprint. Re-materializing the
same data is a no-op that returns the cached shard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from sboltorch.data.corpus import Corpus
from sboltorch.types import (
    Alphabet,
    SbolObject,
    SbolSequence,
    feature_from_dict,
    feature_to_dict,
    graph_from_dict,
    graph_to_dict,
)

_SCHEMA = pa.schema(
    [
        ("iri", pa.string()),
        ("sbol_class", pa.string()),
        ("display_id", pa.string()),
        ("name", pa.string()),
        ("roles", pa.list_(pa.string())),
        ("types", pa.list_(pa.string())),
        ("elements", pa.string()),
        ("alphabet", pa.string()),
        ("encoding_iri", pa.string()),
        ("label", pa.float64()),
        ("features_json", pa.string()),
        ("graph_json", pa.string()),
        ("raw_json", pa.string()),
    ]
)


@dataclass(frozen=True)
class MaterializedCorpus:
    """A Parquet-backed corpus: random-access, reproducible, offline."""

    path: Path
    fingerprint: str
    count: int

    def __len__(self) -> int:
        return self.count

    def labels(self) -> list[float | int | None]:
        table = pq.read_table(self.path / "data.parquet", columns=["label"])
        return [None if v is None else v for v in table.column("label").to_pylist()]

    def __iter__(self) -> Iterator[SbolObject]:
        table = pq.read_table(self.path / "data.parquet")
        for row in table.to_pylist():
            yield _row_to_object(row)

    def read_all(self) -> list[SbolObject]:
        return list(self)


def materialize(corpus: Corpus, cache_dir: str | Path, *, force: bool = False) -> MaterializedCorpus:
    """Stream ``corpus`` to Parquet under ``cache_dir``, keyed by a content hash."""
    cache_root = Path(cache_dir)
    namespace = corpus.fingerprint()
    staging = cache_root / namespace / "staging"

    # If a completed manifest already exists for this source identity, reuse it.
    existing = _find_complete(cache_root / namespace)
    if existing is not None and not force:
        return existing

    staging.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    count = 0
    rows: list[dict[str, object]] = []
    for obj in corpus:
        row = _object_to_row(obj)
        rows.append(row)
        hasher.update(_hash_payload(obj))
        count += 1

    fingerprint = f"{namespace}-{hasher.hexdigest()[:16]}"
    target = cache_root / namespace / fingerprint
    target.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows, schema=_SCHEMA) if rows else _SCHEMA.empty_table()
    pq.write_table(table, target / "data.parquet")
    manifest = {"fingerprint": fingerprint, "count": count, "namespace": namespace}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return MaterializedCorpus(path=target, fingerprint=fingerprint, count=count)


def _find_complete(namespace_dir: Path) -> MaterializedCorpus | None:
    if not namespace_dir.exists():
        return None
    for child in sorted(namespace_dir.iterdir()):
        manifest = child / "manifest.json"
        if manifest.exists() and (child / "data.parquet").exists():
            meta = json.loads(manifest.read_text())
            return MaterializedCorpus(path=child, fingerprint=meta["fingerprint"], count=meta["count"])
    return None


def _hash_payload(obj: SbolObject) -> bytes:
    seq = obj.sequence.elements if obj.sequence else ""
    return f"{obj.iri}\x00{seq}\x00{obj.label}".encode()


def _object_to_row(obj: SbolObject) -> dict[str, Any]:
    seq = obj.sequence
    return {
        "iri": obj.iri,
        "sbol_class": obj.sbol_class,
        "display_id": obj.display_id,
        "name": obj.name,
        "roles": list(obj.roles),
        "types": list(obj.types),
        "elements": seq.elements if seq else None,
        "alphabet": seq.alphabet.value if seq else None,
        "encoding_iri": seq.encoding_iri if seq else None,
        "label": float(obj.label) if obj.label is not None else None,
        "features_json": json.dumps([feature_to_dict(f) for f in obj.features]) if obj.features else None,
        "graph_json": json.dumps(graph_to_dict(obj.neighbors)) if obj.neighbors is not None else None,
        "raw_json": json.dumps(obj.raw, default=str),
    }


def _row_to_object(row: dict[str, Any]) -> SbolObject:
    elements = row.get("elements")
    sequence = None
    if elements:
        sequence = SbolSequence(
            elements=str(elements),
            alphabet=Alphabet(row["alphabet"]) if row.get("alphabet") else Alphabet.DNA,
            encoding_iri=row.get("encoding_iri"),  # type: ignore[arg-type]
        )
    raw_json = row.get("raw_json")
    features_json = row.get("features_json")
    graph_json = row.get("graph_json")
    return SbolObject(
        iri=str(row["iri"]),
        sbol_class=str(row.get("sbol_class") or ""),
        display_id=row.get("display_id"),  # type: ignore[arg-type]
        name=row.get("name"),  # type: ignore[arg-type]
        roles=tuple(row.get("roles") or ()),
        types=tuple(row.get("types") or ()),
        sequence=sequence,
        features=tuple(feature_from_dict(f) for f in json.loads(features_json)) if features_json else (),
        neighbors=graph_from_dict(json.loads(graph_json)) if graph_json else None,
        label=row.get("label"),  # type: ignore[arg-type]
        raw=json.loads(raw_json) if raw_json else {},
    )
