"""Validate the static GF(2) structure of paper-listed BB workloads."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from workloads import _gf2_rank, available_workloads, get_workload


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("workload list must not be empty")
    return names


def _inspect(name: str) -> dict[str, Any]:
    workload = get_workload(name)
    if workload.family != "bivariate_bicycle":
        raise ValueError(f"{name!r} is not a BB workload")
    x_checks = workload.x_checks
    z_checks = workload.z_checks
    x_rank = _gf2_rank(x_checks)
    z_rank = _gf2_rank(z_checks)
    commutator = (x_checks @ z_checks.T) % 2
    row_weights = np.concatenate((x_checks.sum(axis=1), z_checks.sum(axis=1)))
    qubit_degrees = x_checks.sum(axis=0) + z_checks.sum(axis=0)
    nqubits = workload.nqubits
    encoded_qubits = nqubits - x_rank - z_rank
    if workload.paper_l is None or workload.paper_m is None:
        raise AssertionError("BB workload is missing its paper dimensions")
    if nqubits != 2 * workload.paper_l * workload.paper_m:
        raise AssertionError(f"{name}: wrong physical-qubit count")
    if np.any(commutator):
        raise AssertionError(f"{name}: H_X H_Z^T is not zero over GF(2)")
    if not np.all(row_weights == 6):
        raise AssertionError(f"{name}: expected all checks to have weight six")
    if not np.all(qubit_degrees == 6):
        raise AssertionError(f"{name}: expected every qubit degree to be six")
    if encoded_qubits != workload.paper_k:
        raise AssertionError(f"{name}: rank-derived k disagrees with the paper")
    return {
        "workload": name,
        "paper": {
            "l": workload.paper_l,
            "m": workload.paper_m,
            "k": workload.paper_k,
            "distance": workload.paper_distance,
            "distance_exact": workload.paper_distance_exact,
            "a": list(workload.paper_a),
            "b": list(workload.paper_b),
        },
        "nqubits": nqubits,
        "x_check_rows": int(x_checks.shape[0]),
        "z_check_rows": int(z_checks.shape[0]),
        "x_rank": x_rank,
        "z_rank": z_rank,
        "nlogical": encoded_qubits,
        "check_weight_min": int(row_weights.min()),
        "check_weight_max": int(row_weights.max()),
        "qubit_degree_min": int(qubit_degrees.min()),
        "qubit_degree_max": int(qubit_degrees.max()),
        "commuting": True,
        "matrix_hash": workload.matrix_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        type=_parse_names,
        default=tuple(name for name in available_workloads() if name.startswith("bb_")),
        help="comma-separated BB workload names",
    )
    arguments = parser.parse_args()
    records = [_inspect(name) for name in arguments.workloads]
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
