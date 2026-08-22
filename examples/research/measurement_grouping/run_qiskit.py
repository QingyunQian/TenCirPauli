"""Optional Qiskit comparison runner for QWC measurement grouping."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
from common import make_workload, pauli_labels
from qiskit.quantum_info import SparsePauliOp


def _median_seconds(function: Callable[[], object], repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _build_operator(labels: Sequence[str], coefficients: np.ndarray) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [
            (label, complex(coefficient))
            for label, coefficient in zip(labels, coefficients, strict=True)
        ]
    )


def _group(operator: SparsePauliOp):
    return operator.group_commuting(qubit_wise=True)


def _valid_groups(groups: Iterable[SparsePauliOp], labels: Sequence[str]) -> bool:
    expected = sorted(labels)
    observed = sorted(label for group in groups for label in group.paulis.to_labels())
    if observed != expected:
        return False
    for group in groups:
        group_labels = group.paulis.to_labels()
        for position, left in enumerate(group_labels):
            for right in group_labels[position + 1 :]:
                if any(
                    left_code != "I" and right_code != "I" and left_code != right_code
                    for left_code, right_code in zip(left, right, strict=True)
                ):
                    return False
    return True


def run(
    nqubits: int,
    term_count: int,
    support_width: int,
    shots: int,
    seed: int,
    repetitions: int,
) -> Dict[str, object]:
    del shots
    codes, coefficients = make_workload(
        nqubits=nqubits,
        term_count=term_count,
        support_width=support_width,
        seed=seed,
    )
    labels = pauli_labels(codes)
    operator = _build_operator(labels, coefficients)
    groups = _group(operator)
    construction_seconds = _median_seconds(
        lambda: _build_operator(labels, coefficients), repetitions
    )
    grouping_seconds = _median_seconds(lambda: _group(operator), repetitions)
    end_to_end_seconds = _median_seconds(
        lambda: _group(_build_operator(labels, coefficients)), repetitions
    )
    return {
        "implementation": "qiskit",
        "python": sys.executable,
        "qiskit": getattr(__import__("qiskit"), "__version__", "unknown"),
        "nqubits": nqubits,
        "term_count": term_count,
        "support_width": support_width,
        "seed": seed,
        "repetitions": repetitions,
        "group_count": len(groups),
        "largest_group": max((len(group) for group in groups), default=0),
        "qwc_correct": _valid_groups(groups, labels),
        "timings_seconds": {
            "construction": construction_seconds,
            "grouping": grouping_seconds,
            "end_to_end": end_to_end_seconds,
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
