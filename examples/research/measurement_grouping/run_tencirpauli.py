"""Benchmark TenCirPauli QWC grouping and its measurement reconstruction path."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from common import make_workload

import tencirpauli as tcp


def _median_seconds(function: Callable[[], object], repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _qwc_compatibility(codes: np.ndarray) -> np.ndarray:
    """Build the independent QWC compatibility matrix."""
    term_count = len(codes)
    compatibility = np.ones((term_count, term_count), dtype=np.bool_)
    for index in range(term_count):
        left = codes[index]
        compatibility[index, index + 1 :] = np.all(
            (left == 0) | (codes[index + 1 :] == 0) | (left == codes[index + 1 :]),
            axis=1,
        )
        compatibility[index + 1 :, index] = compatibility[index, index + 1 :]
    return compatibility


def python_largest_first(codes: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    """Independent NumPy/Python largest-first QWC coloring baseline."""
    compatibility = _qwc_compatibility(codes)
    degrees = np.count_nonzero(~compatibility, axis=1)
    order = sorted(range(len(codes)), key=lambda index: (-int(degrees[index]), index))
    groups: List[List[int]] = []
    for term_index in order:
        for group in groups:
            if bool(np.all(compatibility[term_index, group])):
                group.append(term_index)
                break
        else:
            groups.append([term_index])
    normalized = tuple(tuple(sorted(group)) for group in groups)
    return tuple(sorted(normalized, key=lambda group: group[0]))


def _canonical_codes(operator: tcp.PauliOperator) -> np.ndarray:
    return np.asarray([term.word.to_codes() for term in operator.terms], dtype=np.uint8)


def _python_reconstruct(
    groups: Sequence[Sequence[int]], codes: np.ndarray, bitstrings: np.ndarray
) -> Tuple[np.ndarray, ...]:
    values = []
    for group in groups:
        result = np.empty((len(bitstrings), len(group)), dtype=np.int8)
        for column, term_index in enumerate(group):
            support = codes[term_index] != 0
            parity = np.sum(bitstrings[:, support], axis=1) & 1
            result[:, column] = np.where(parity == 0, 1, -1)
        values.append(result)
    return tuple(values)


def _native_reconstruct(
    grouping: tcp.QWCGroupingResult, bitstrings: np.ndarray
) -> Tuple[np.ndarray, ...]:
    return tuple(
        grouping.reconstruct(group_index, bitstrings)
        for group_index in range(grouping.group_count)
    )


def _same_arrays(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(left_value, right_value)
        for left_value, right_value in zip(left, right)
    )


def run(
    nqubits: int,
    term_count: int,
    support_width: int,
    shots: int,
    seed: int,
    repetitions: int,
) -> Dict[str, object]:
    codes, coefficients = make_workload(
        nqubits=nqubits,
        term_count=term_count,
        support_width=support_width,
        seed=seed,
    )
    operator = tcp.PauliOperator.from_code_arrays(codes, coefficients)
    canonical_codes = _canonical_codes(operator)
    native_grouping = operator.group_commuting(
        mode="qubit_wise", algorithm="largest_first"
    )
    python_groups = python_largest_first(canonical_codes)
    bits_rng = np.random.default_rng(seed + 1)
    bitstrings = np.ascontiguousarray(
        bits_rng.integers(0, 2, size=(shots, nqubits), dtype=np.int8)
    )
    native_values = _native_reconstruct(native_grouping, bitstrings)
    python_values = _python_reconstruct(python_groups, canonical_codes, bitstrings)

    native_group_seconds = _median_seconds(
        lambda: operator.group_commuting(mode="qubit_wise", algorithm="largest_first"),
        repetitions,
    )
    python_group_seconds = _median_seconds(
        lambda: python_largest_first(canonical_codes), repetitions
    )
    native_end_to_end_seconds = _median_seconds(
        lambda: tcp.PauliOperator.from_code_arrays(codes, coefficients).group_commuting(
            mode="qubit_wise", algorithm="largest_first"
        ),
        repetitions,
    )
    native_reconstruct_seconds = _median_seconds(
        lambda: _native_reconstruct(native_grouping, bitstrings), repetitions
    )
    python_reconstruct_seconds = _median_seconds(
        lambda: _python_reconstruct(python_groups, canonical_codes, bitstrings),
        repetitions,
    )

    return {
        "implementation": "tencirpauli",
        "python": sys.executable,
        "package": getattr(tcp, "__version__", "unknown"),
        "nqubits": nqubits,
        "requested_term_count": term_count,
        "term_count": operator.term_count,
        "support_width": support_width,
        "shots": shots,
        "seed": seed,
        "repetitions": repetitions,
        "native_group_count": native_grouping.group_count,
        "python_group_count": len(python_groups),
        "largest_group": max(
            (len(group) for group in native_grouping.groups), default=0
        ),
        "qwc_correct": all(
            bool(
                np.all(
                    (canonical_codes[left] == 0)
                    | (canonical_codes[right] == 0)
                    | (canonical_codes[left] == canonical_codes[right])
                )
            )
            for group in native_grouping.groups
            for position, left in enumerate(group)
            for right in group[position + 1 :]
        ),
        "python_grouping_matches_native": native_grouping.groups == python_groups,
        "reconstruction_correct": _same_arrays(native_values, python_values),
        "timings_seconds": {
            "native_grouping": native_group_seconds,
            "python_grouping": python_group_seconds,
            "native_end_to_end": native_end_to_end_seconds,
            "native_reconstruction_all_groups": native_reconstruct_seconds,
            "python_reconstruction_all_groups": python_reconstruct_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=64)
    parser.add_argument("--term-count", type=int, default=1024)
    parser.add_argument("--support-width", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.shots < 1 or args.repetitions < 1:
        raise SystemExit("shots and repetitions must be positive")
    print(
        json.dumps(
            run(
                args.nqubits,
                args.term_count,
                args.support_width,
                args.shots,
                args.seed,
                args.repetitions,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
