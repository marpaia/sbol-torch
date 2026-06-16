"""Parse real-world SBOL files (SynBioDex SBOLTestSuite) through LocalFileCorpus.

These complement the synthetic fixtures with genuine SBOL2/SBOL3 documents,
covering RDF/XML, Turtle, and N-Triples serializations and both sequence-bearing
and abstract (sequence-free) designs. See fixtures/sbol/PROVENANCE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sboltorch.data.local import LocalFileCorpus

FIXTURES = Path(__file__).parent / "fixtures" / "sbol"
ALL_FILES = sorted(p for p in FIXTURES.rglob("*") if p.suffix in {".ttl", ".nt", ".xml"})

# These vendored fixtures may be absent (e.g. not committed due to licensing);
# skip rather than fail when they are not present.
pytestmark = pytest.mark.skipif(not ALL_FILES, reason="SBOL fixtures not present (see fixtures/sbol/PROVENANCE.md)")


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(FIXTURES)))
def test_every_fixture_parses_without_error(path):
    # Robustness: real files (including abstract designs) parse cleanly.
    objects = list(LocalFileCorpus(path, fmt="sbol"))
    assert isinstance(objects, list)


def test_sbol2_plasmid_sequence():
    objs = list(LocalFileCorpus(FIXTURES / "sbol2" / "pICH44179.xml", fmt="sbol"))
    seqs = [o for o in objs if o.sequence and o.sequence.elements]
    assert len(seqs) == 1
    assert len(seqs[0].sequence.elements) == 2307
    assert set(seqs[0].sequence.elements.upper()) <= set("ACGTN")


def test_sbol3_device_has_multiple_sequences():
    objs = list(LocalFileCorpus(FIXTURES / "sbol3" / "BBa_F2620_PoPSReceiver.ttl", fmt="sbol"))
    seqs = [o for o in objs if o.sequence and o.sequence.elements]
    assert len(seqs) == 10


def test_abstract_design_has_no_sequences():
    # toggle_switch is composition-only (components/interactions, no elements);
    # the parser returns no sequences rather than failing.
    objs = list(LocalFileCorpus(FIXTURES / "sbol3" / "toggle_switch.ttl", fmt="sbol"))
    assert all(o.sequence is None or not o.sequence.elements for o in objs)
