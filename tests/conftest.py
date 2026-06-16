"""Shared fixtures: tiny synthetic SBOL data and sbol-db record payloads."""

from __future__ import annotations

from pathlib import Path

import pytest

FASTA = """\
>seq1 measure=12.5
ACGTACGTACGTACGT
>seq2 measure=3.0
TTTTGGGGCCCCAAAA
>seq3 measure=8.25
ACACACACGTGTGTGT
"""

SBOL_TTL = """\
@prefix sbol: <http://sbols.org/v3#> .
@prefix ex:   <https://example.org/> .

ex:seqA a sbol:Sequence ;
    sbol:elements "ttgacggctagctcagtcctaggtacagtgctagc" ;
    ex:measure "5.0" .

ex:seqB a sbol:Sequence ;
    sbol:elements "aaagaggagaaatactagatgcgtaaaggcgaa" ;
    ex:measure "9.5" .
"""


@pytest.fixture
def fasta_file(tmp_path: Path) -> Path:
    path = tmp_path / "seqs.fasta"
    path.write_text(FASTA)
    return path


@pytest.fixture
def sbol_file(tmp_path: Path) -> Path:
    path = tmp_path / "design.ttl"
    path.write_text(SBOL_TTL)
    return path


@pytest.fixture
def object_records() -> list[dict]:
    """SbolObjectRecord payloads as returned by sbol-db /objects/list."""
    return [
        {
            "iri": "https://example.org/seqA",
            "sbol_class": "http://sbols.org/v3#Sequence",
            "display_id": "seqA",
            "roles": ["SO:0000167"],
            "types": [],
            "data": {
                "http://sbols.org/v3#elements": [{"@value": "ACGTACGTACGT"}],
                "http://sbols.org/v3#encoding": [{"@value": "https://identifiers.org/edam:format_1207"}],
                "https://example.org/measure": [{"@value": "12.5"}],
            },
        },
        {
            "iri": "https://example.org/seqB",
            "sbol_class": "http://sbols.org/v3#Sequence",
            "display_id": "seqB",
            "roles": ["SO:0000167"],
            "types": [],
            "data": {
                "http://sbols.org/v3#elements": [{"@value": "TTTTGGGGCCCC"}],
                "https://example.org/measure": [{"@value": "3.0"}],
            },
        },
    ]
