"""Source-agnostic record types.

Every data source — the sbol-db REST API or a local file — is normalized into
``SbolObject`` instances. Training code consumes only these types and never
branches on where the data came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Matches the SBOL elements/encoding predicates regardless of compaction, e.g.
# "http://sbols.org/v3#elements", "sbol:elements", or a bare "elements" key.
_LOCAL_NAME = re.compile(r"[#/:]")


def local_name(iri: str) -> str:
    """Return the local name of an IRI/CURIE (the part after the last #, / or :)."""
    return _LOCAL_NAME.split(iri.rstrip("#/"))[-1]


class Alphabet(str, Enum):
    """Sequence alphabet, mirroring sbol-db's ``sbol_sequences.alphabet``."""

    DNA = "DNA"
    RNA = "RNA"
    PROTEIN = "PROTEIN"
    OTHER = "OTHER"

    @classmethod
    def from_encoding(cls, encoding_iri: str | None) -> "Alphabet":
        """Infer an alphabet from an SBOL encoding IRI; default to DNA."""
        if not encoding_iri:
            return cls.DNA
        name = local_name(encoding_iri).lower()
        if "protein" in name or "aminoacid" in name:
            return cls.PROTEIN
        if "rna" in name:
            return cls.RNA
        if "dna" in name:
            return cls.DNA
        return cls.OTHER


@dataclass(frozen=True)
class SbolSequence:
    """A biological sequence: the raw elements plus its alphabet."""

    elements: str
    alphabet: Alphabet = Alphabet.DNA
    encoding_iri: str | None = None

    def __len__(self) -> int:
        return len(self.elements)


@dataclass(frozen=True)
class Location:
    """A position within a sequence (Range or Cut), with optional orientation."""

    start: int | None = None
    end: int | None = None
    orientation: str | None = None


@dataclass(frozen=True)
class Feature:
    """A feature within a component (typically a SubComponent)."""

    iri: str
    kind: str | None = None
    instance_of: str | None = None
    roles: tuple[str, ...] = ()
    locations: tuple[Location, ...] = ()


@dataclass(frozen=True)
class GraphNode:
    iri: str
    depth: int
    sbol_class: str | None = None
    display_id: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    subject: str
    predicate: str
    object: str
    depth: int


@dataclass(frozen=True)
class GraphSlice:
    """A bounded neighborhood of an object, used by the structure/graph encoders."""

    root_iri: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class SbolObject:
    """The canonical unit of data flowing through the library."""

    iri: str
    sbol_class: str
    display_id: str | None = None
    name: str | None = None
    roles: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    sequence: SbolSequence | None = None
    features: tuple[Feature, ...] = ()
    neighbors: GraphSlice | None = None
    label: float | int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any], label: float | int | None = None) -> "SbolObject":
        """Build an SbolObject from an sbol-db ``SbolObjectRecord`` payload.

        The sequence elements live inside the lossless JSON-LD ``data`` slice
        under the ``sbol:elements`` predicate; we extract them by local name so
        the code is robust to IRI compaction.
        """
        data = record.get("data") or {}
        sequence = _sequence_from_data(data)
        return cls(
            iri=record["iri"],
            sbol_class=record.get("sbol_class", ""),
            display_id=record.get("display_id"),
            name=record.get("name"),
            roles=tuple(record.get("roles") or ()),
            types=tuple(record.get("types") or ()),
            sequence=sequence,
            label=label,
            raw=record,
        )


def feature_to_dict(feature: Feature) -> dict[str, Any]:
    return {
        "iri": feature.iri,
        "kind": feature.kind,
        "instance_of": feature.instance_of,
        "roles": list(feature.roles),
        "locations": [
            {"start": loc.start, "end": loc.end, "orientation": loc.orientation} for loc in feature.locations
        ],
    }


def feature_from_dict(data: dict[str, Any]) -> Feature:
    return Feature(
        iri=data["iri"],
        kind=data.get("kind"),
        instance_of=data.get("instance_of"),
        roles=tuple(data.get("roles") or ()),
        locations=tuple(
            Location(start=loc.get("start"), end=loc.get("end"), orientation=loc.get("orientation"))
            for loc in data.get("locations") or ()
        ),
    )


def graph_to_dict(graph: GraphSlice) -> dict[str, Any]:
    return {
        "root_iri": graph.root_iri,
        "truncated": graph.truncated,
        "nodes": [
            {"iri": n.iri, "depth": n.depth, "sbol_class": n.sbol_class, "display_id": n.display_id}
            for n in graph.nodes
        ],
        "edges": [
            {"subject": e.subject, "predicate": e.predicate, "object": e.object, "depth": e.depth} for e in graph.edges
        ],
    }


def graph_from_dict(data: dict[str, Any]) -> GraphSlice:
    return GraphSlice(
        root_iri=data["root_iri"],
        truncated=bool(data.get("truncated", False)),
        nodes=tuple(
            GraphNode(iri=n["iri"], depth=n["depth"], sbol_class=n.get("sbol_class"), display_id=n.get("display_id"))
            for n in data.get("nodes") or ()
        ),
        edges=tuple(
            GraphEdge(subject=e["subject"], predicate=e["predicate"], object=e["object"], depth=e["depth"])
            for e in data.get("edges") or ()
        ),
    )


def _scalar(value: Any) -> Any:
    """Unwrap JSON-LD value shapes: {"@value": x}, [x], or x -> x."""
    if isinstance(value, list):
        return _scalar(value[0]) if value else None
    if isinstance(value, dict):
        return value.get("@value", value.get("value"))
    return value


def _sequence_from_data(data: dict[str, Any]) -> SbolSequence | None:
    """Extract elements + encoding from a JSON-LD object slice, by local name."""
    elements: str | None = None
    encoding: str | None = None
    for key, value in data.items():
        name = local_name(key)
        if name == "elements" and elements is None:
            elements = _scalar(value)
        elif name == "encoding" and encoding is None:
            enc = _scalar(value)
            encoding = enc if isinstance(enc, str) else None
    if not elements:
        return None
    return SbolSequence(
        elements=str(elements),
        alphabet=Alphabet.from_encoding(encoding),
        encoding_iri=encoding,
    )
