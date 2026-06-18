"""Synthetic SBOL fixture generator.

Produces deterministic synthetic transcriptional units — a promoter, RBS, CDS,
and terminator composed into a parent Component — as rich ``SbolObject`` records:
sequence, features (sub-components with Range locations, roles, orientation), and
a composition ``GraphSlice``. Parts are drawn from a shared catalog so the same
part is reused across components, giving the composition graphs real structure.

Used to develop and test the structure-aware and graph encoders without a
populated sbol-db. ``write_sbol_turtle`` serializes to SBOL3 RDF for file-based
corpus and ingestion round-trips.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterator

from sboltorch.types import Alphabet, Feature, GraphEdge, GraphNode, GraphSlice, Location, SbolObject, SbolSequence

SBOL3 = "http://sbols.org/v3#"
NS = "https://sboltorch.test/"

# Sequence Ontology roles.
ROLE = {
    "promoter": "https://identifiers.org/SO:0000167",
    "rbs": "https://identifiers.org/SO:0000139",
    "cds": "https://identifiers.org/SO:0000316",
    "terminator": "https://identifiers.org/SO:0000141",
}
ENGINEERED_REGION = "https://identifiers.org/SO:0000804"
ORIENTATION_INLINE = f"{SBOL3}inline"
ORIENTATION_RC = f"{SBOL3}reverseComplement"

# Part catalog: role -> list of part display ids. Lengths are role-typical.
_CATALOG = {
    "promoter": (["J23100", "J23101", "J23106", "J23118"], 35),
    "rbs": (["B0034", "B0032", "B0030"], 20),
    "cds": (["GFP", "RFP", "BFP", "YFP"], 90),
    "terminator": (["B0015", "L3S2P21"], 40),
}
_ROLE_ORDER = ["promoter", "rbs", "cds", "terminator"]
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


class PartCatalog:
    """A fixed set of named parts with stable sequences (seeded once)."""

    def __init__(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.sequences: dict[str, str] = {}
        self.roles: dict[str, str] = {}
        for role, (names, length) in _CATALOG.items():
            for name in names:
                self.sequences[name] = "".join(rng.choice("ACGT") for _ in range(length))
                self.roles[name] = role

    def part_iri(self, name: str) -> str:
        return f"{NS}part/{name}"


def generate_components(n: int, *, seed: int = 0, with_labels: bool = True) -> list[SbolObject]:
    """Generate ``n`` synthetic composite Components."""
    catalog = PartCatalog(seed)
    rng = random.Random(seed + 1)
    # A per-promoter "strength" gives a learnable supervised signal.
    strengths = {name: rng.uniform(1.0, 10.0) for name in _CATALOG["promoter"][0]}

    components: list[SbolObject] = []
    for i in range(n):
        chosen = {role: rng.choice(names) for role, (names, _) in _CATALOG.items()}
        comp_iri = f"{NS}component/tu{i}"

        elements_parts: list[str] = []
        features: list[Feature] = []
        nodes: list[GraphNode] = [GraphNode(iri=comp_iri, depth=0, sbol_class=f"{SBOL3}Component", display_id=f"tu{i}")]
        edges: list[GraphEdge] = []
        cursor = 0
        for role in _ROLE_ORDER:
            part = chosen[role]
            part_seq = catalog.sequences[part]
            reverse = rng.random() < 0.2
            placed = _reverse_complement(part_seq) if reverse else part_seq
            start = cursor + 1  # SBOL Ranges are 1-based, inclusive.
            end = cursor + len(part_seq)
            cursor = end
            elements_parts.append(placed)

            feature_iri = f"{comp_iri}/{role}"
            part_iri = catalog.part_iri(part)
            features.append(
                Feature(
                    iri=feature_iri,
                    kind="SubComponent",
                    instance_of=part_iri,
                    roles=(ROLE[role],),
                    locations=(
                        Location(
                            start=start,
                            end=end,
                            orientation=ORIENTATION_RC if reverse else ORIENTATION_INLINE,
                        ),
                    ),
                )
            )
            nodes.append(GraphNode(iri=feature_iri, depth=1, sbol_class=f"{SBOL3}SubComponent", display_id=role))
            nodes.append(GraphNode(iri=part_iri, depth=2, sbol_class=f"{SBOL3}Component", display_id=part))
            edges.append(GraphEdge(subject=comp_iri, predicate=f"{SBOL3}hasFeature", object=feature_iri, depth=1))
            edges.append(GraphEdge(subject=feature_iri, predicate=f"{SBOL3}instanceOf", object=part_iri, depth=2))

        sequence = "".join(elements_parts)
        seq_iri = f"{comp_iri}/sequence"
        nodes.append(GraphNode(iri=seq_iri, depth=1, sbol_class=f"{SBOL3}Sequence", display_id="sequence"))
        edges.append(GraphEdge(subject=comp_iri, predicate=f"{SBOL3}hasSequence", object=seq_iri, depth=1))

        label = strengths[chosen["promoter"]] if with_labels else None
        components.append(
            SbolObject(
                iri=comp_iri,
                sbol_class=f"{SBOL3}Component",
                display_id=f"tu{i}",
                roles=(ENGINEERED_REGION,),
                types=(f"{SBOL3}DNA",),
                sequence=SbolSequence(elements=sequence, alphabet=Alphabet.DNA),
                features=tuple(features),
                neighbors=GraphSlice(root_iri=comp_iri, nodes=tuple(nodes), edges=tuple(edges), truncated=False),
                label=label,
            )
        )
    return components


class SyntheticCorpus:
    """An in-memory Corpus of synthetic components, for tests and local development."""

    def __init__(self, n: int = 64, *, seed: int = 0, with_labels: bool = True) -> None:
        self.n = n
        self.seed = seed
        self.with_labels = with_labels

    def __iter__(self) -> Iterator[SbolObject]:
        return iter(generate_components(self.n, seed=self.seed, with_labels=self.with_labels))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"synthetic:{self.n}:{self.seed}:{self.with_labels}".encode())
        return h.hexdigest()[:16]


def write_sbol_turtle(components: list[SbolObject], path: str | Path) -> Path:
    """Serialize synthetic components to an SBOL3 Turtle document."""
    from rdflib import RDF, Graph, Literal, Namespace, URIRef

    sbol = Namespace(SBOL3)
    graph = Graph()
    graph.bind("sbol", sbol)

    for comp in components:
        comp_ref = URIRef(comp.iri)
        graph.add((comp_ref, RDF.type, sbol.Component))
        for role in comp.roles:
            graph.add((comp_ref, sbol.role, URIRef(role)))
        if comp.sequence is not None:
            seq_ref = URIRef(f"{comp.iri}/sequence")
            graph.add((comp_ref, sbol.hasSequence, seq_ref))
            graph.add((seq_ref, RDF.type, sbol.Sequence))
            graph.add((seq_ref, sbol.elements, Literal(comp.sequence.elements)))
        for feature in comp.features:
            feat_ref = URIRef(feature.iri)
            graph.add((comp_ref, sbol.hasFeature, feat_ref))
            graph.add((feat_ref, RDF.type, sbol.SubComponent))
            if feature.instance_of:
                graph.add((feat_ref, sbol.instanceOf, URIRef(feature.instance_of)))
            for role in feature.roles:
                graph.add((feat_ref, sbol.role, URIRef(role)))
            for loc in feature.locations:
                loc_ref = URIRef(f"{feature.iri}/loc")
                graph.add((feat_ref, sbol.hasLocation, loc_ref))
                graph.add((loc_ref, RDF.type, sbol.Range))
                if loc.start is not None:
                    graph.add((loc_ref, sbol.start, Literal(loc.start)))
                if loc.end is not None:
                    graph.add((loc_ref, sbol.end, Literal(loc.end)))
                if loc.orientation:
                    graph.add((loc_ref, sbol.orientation, URIRef(loc.orientation)))

    out = Path(path)
    graph.serialize(destination=str(out), format="turtle")
    return out
