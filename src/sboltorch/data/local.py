"""Local-file corpus: FASTA and SBOL RDF, normalized to SbolObject.

This is the offline fallback to the sbol-db client. It produces the exact same
SbolObject records, so downstream code is identical regardless of source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from rdflib import Graph, URIRef
from rdflib.term import Node

from ..config import CorpusConfig
from ..exceptions import ParseError
from ..types import Alphabet, SbolObject, SbolSequence, local_name

_SBOL_ELEMENTS = "http://sbols.org/v2#elements"
_SBOL_ELEMENTS_V3 = "http://sbols.org/v3#elements"


class LocalFileCorpus:
    """Reads SbolObject records from a FASTA or SBOL RDF file (or a directory of them)."""

    def __init__(self, path: str | Path, *, fmt: str = "auto", label_key: str | None = None) -> None:
        self.path = Path(path)
        self.fmt = fmt
        self.label_key = label_key

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "LocalFileCorpus":
        assert config.path is not None  # guaranteed by CorpusConfig validation
        return cls(config.path, fmt=config.fmt, label_key=config.label_key)

    def _files(self) -> list[Path]:
        if self.path.is_dir():
            return sorted(p for p in self.path.rglob("*") if p.is_file())
        return [self.path]

    def _format_for(self, file: Path) -> str:
        if self.fmt != "auto":
            return self.fmt
        suffix = file.suffix.lower()
        if suffix in {".fa", ".fasta", ".fna"}:
            return "fasta"
        if suffix in {".xml", ".rdf", ".ttl", ".nt", ".sbol"}:
            return "sbol"
        raise ParseError(f"cannot infer format for {file}; set corpus.fmt explicitly")

    def __iter__(self) -> Iterator[SbolObject]:
        for file in self._files():
            fmt = self._format_for(file)
            if fmt == "fasta":
                yield from _parse_fasta(file, self.label_key)
            else:
                yield from _parse_sbol(file, self.label_key)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for file in self._files():
            stat = file.stat()
            h.update(str(file).encode())
            h.update(str(stat.st_size).encode())
            h.update(str(int(stat.st_mtime)).encode())
        h.update(repr(self.label_key).encode())
        return h.hexdigest()[:16]


def _parse_fasta(file: Path, label_key: str | None) -> Iterator[SbolObject]:
    """Parse FASTA. Labels are read from ``key=value`` tokens in the header."""
    header: str | None = None
    chunks: list[str] = []

    def flush() -> SbolObject | None:
        if header is None:
            return None
        seq_id = header.split()[0] if header.split() else header
        label = _label_from_header(header, label_key) if label_key else None
        return SbolObject(
            iri=seq_id,
            sbol_class="http://sbols.org/v3#Sequence",
            display_id=seq_id,
            sequence=SbolSequence(elements="".join(chunks), alphabet=Alphabet.DNA),
            label=label,
            raw={"header": header},
        )

    with file.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                obj = flush()
                if obj is not None:
                    yield obj
                header = line[1:].strip()
                chunks = []
            elif line:
                chunks.append(line.strip())
    obj = flush()
    if obj is not None:
        yield obj


def _label_from_header(header: str, label_key: str) -> float | int | None:
    for token in header.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key == label_key:
                try:
                    num = float(value)
                except ValueError:
                    return None
                return int(num) if num.is_integer() else num
    return None


def _parse_sbol(file: Path, label_key: str | None) -> Iterator[SbolObject]:
    """Parse an SBOL RDF document, yielding one SbolObject per sbol:Sequence subject."""
    graph = Graph()
    try:
        graph.parse(str(file))
    except Exception as exc:  # rdflib raises a variety of parser errors
        raise ParseError(f"failed to parse SBOL file {file}: {exc}") from exc

    for predicate in (URIRef(_SBOL_ELEMENTS_V3), URIRef(_SBOL_ELEMENTS)):
        for subject, _, elements in graph.triples((None, predicate, None)):
            iri = str(subject)
            label = _label_from_graph(graph, subject, label_key) if label_key else None
            yield SbolObject(
                iri=iri,
                sbol_class="http://sbols.org/v3#Sequence",
                display_id=local_name(iri),
                sequence=SbolSequence(elements=str(elements), alphabet=Alphabet.DNA),
                label=label,
                raw={"file": str(file)},
            )


def _label_from_graph(graph: Graph, subject: Node, label_key: str) -> float | int | None:
    for _, predicate, value in graph.triples((subject, None, None)):
        if local_name(str(predicate)) != label_key:
            continue
        try:
            num = float(str(value))
        except ValueError:
            return None
        return int(num) if num.is_integer() else num
    return None
