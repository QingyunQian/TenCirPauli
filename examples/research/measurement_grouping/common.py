"""Shared deterministic workloads for the measurement-grouping study."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def make_workload(
    nqubits: int = 64,
    term_count: int = 1024,
    support_width: int = 4,
    seed: int = 20260810,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build unique short-range Pauli words and real coefficients."""
    if nqubits < 1:
        raise ValueError("nqubits must be positive")
    if term_count < 1:
        raise ValueError("term_count must be positive")
    if support_width < 1 or support_width > nqubits:
        raise ValueError("support_width must be in [1, nqubits]")
    available_words = sum(
        math.comb(nqubits, width) * (3**width) for width in range(1, support_width + 1)
    )
    if term_count > available_words:
        raise ValueError(
            f"term_count exceeds the {available_words} unique words available"
        )

    rng = np.random.default_rng(seed)
    words = set()
    rows = []
    while len(rows) < term_count:
        start = int(rng.integers(0, nqubits))
        width = int(rng.integers(1, support_width + 1))
        row = np.zeros(nqubits, dtype=np.uint8)
        positions = (start + np.arange(width)) % nqubits
        row[positions] = rng.integers(1, 4, size=width, dtype=np.uint8)
        word = tuple(int(code) for code in row)
        if word in words:
            continue
        words.add(word)
        rows.append(row)

    codes = np.asarray(rows, dtype=np.uint8)
    coefficients = rng.normal(loc=0.0, scale=1.0, size=term_count).astype(np.complex128)
    return np.ascontiguousarray(codes), np.ascontiguousarray(coefficients)


def pauli_labels(codes: np.ndarray) -> Tuple[str, ...]:
    """Convert code rows to public IXYZ labels for external frameworks."""
    alphabet = "IXYZ"
    return tuple("".join(alphabet[int(code)] for code in row) for row in codes)
