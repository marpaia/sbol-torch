from __future__ import annotations

from sboltorch.data.local import LocalFileCorpus


def test_fasta_parsing_with_labels(fasta_file):
    corpus = LocalFileCorpus(fasta_file, fmt="fasta", label_key="measure")
    objects = list(corpus)
    assert len(objects) == 3
    assert objects[0].display_id == "seq1"
    assert objects[0].sequence.elements == "ACGTACGTACGTACGT"
    assert objects[0].label == 12.5
    assert objects[2].label == 8.25


def test_fasta_without_label_key_is_unlabeled(fasta_file):
    corpus = LocalFileCorpus(fasta_file, fmt="fasta")
    objects = list(corpus)
    assert all(o.label is None for o in objects)


def test_sbol_parsing(sbol_file):
    corpus = LocalFileCorpus(sbol_file, fmt="sbol", label_key="measure")
    objects = sorted(corpus, key=lambda o: o.iri)
    assert len(objects) == 2
    assert objects[0].sequence.elements.startswith("ttgacg")
    assert objects[0].label == 5.0


def test_fingerprint_is_stable(fasta_file):
    a = LocalFileCorpus(fasta_file, fmt="fasta").fingerprint()
    b = LocalFileCorpus(fasta_file, fmt="fasta").fingerprint()
    assert a == b
