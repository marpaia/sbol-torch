"""Overlapping k-mer tokenizer for nucleotide sequences.

Every k-mer over the {A,C,G,T} alphabet is a vocabulary entry. Ambiguous IUPAC
bases (N, R, Y, ...) fall back to the ``<unk>`` token rather than being treated
as a real base, which SeqTrainer's one-hot path got wrong.
"""

from __future__ import annotations

from itertools import product

from sboltorch.tokenize.base import Encoded

_BASES = "ACGT"
# Reserved ids: 0 pad, 1 unk, 2 cls, 3 sep, 4 mask. Real k-mers follow.
_SPECIAL = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]


class KmerTokenizer:
    def __init__(self, k: int = 6, stride: int = 1, max_length: int = 512) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.stride = stride
        self._max_length = max_length
        kmers = ["".join(p) for p in product(_BASES, repeat=k)]
        self._vocab = {tok: i for i, tok in enumerate(_SPECIAL + kmers)}
        self._unk = self._vocab["<unk>"]
        self._special_ids = frozenset(self._vocab[tok] for tok in _SPECIAL)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_token_id(self) -> int:
        return self._vocab["<pad>"]

    @property
    def mask_token_id(self) -> int | None:
        return self._vocab["<mask>"]

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def max_length(self) -> int:
        return self._max_length

    def tokenize_content(self, sequence: str) -> list[int]:
        seq = sequence.upper()
        return [
            self._vocab.get(seq[start : start + self.k], self._unk)
            for start in range(0, max(0, len(seq) - self.k + 1), self.stride)
        ]

    def encode(self, sequence: str) -> Encoded:
        content = self.tokenize_content(sequence)[: self._max_length - 2]
        ids = [self._vocab["<cls>"], *content, self._vocab["<sep>"]]
        return Encoded(input_ids=ids, attention_mask=[1] * len(ids))
