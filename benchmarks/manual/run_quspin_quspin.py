"""Run one frozen QuSpin-capability workload with QuSpin only."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from _provenance import worktree_fingerprint
from quspin.basis import (
    boson_basis_1d,
    spin_basis_1d,
    spinless_fermion_basis_1d,
    tensor_basis,
)
from quspin.operators import hamiltonian, quantum_LinearOperator
from quspin_capabilities import (
    ROOT,
    canonical_state,
    exact_command,
    load_manifest,
    peak_rss_bytes,
    save_action_artifact,
    save_action_vector_artifact,
    thread_environment,
)


StaticTerms = list[list[object]]
StateNode = tuple[Any, ...]
_STATIC_STARTS: dict[int, float] = {}


def _add(static: StaticTerms, opstr: str, coefficient: complex, *indices: int) -> None:
    """Append one QuSpin static operator-string term."""
    if not static:
        _STATIC_STARTS[id(static)] = time.perf_counter()
    static.append([opstr, [[coefficient, *indices]]])


def _spin_basis(length: int, particles: int | None = None) -> Any:
    """Build a QuSpin spin basis using TenCirPauli's level-one convention."""
    if particles is None:
        return spin_basis_1d(length, pauli=-1)
    return spin_basis_1d(length, Nup=length - particles, pauli=-1)


def _single_states(basis: Any, node: StateNode) -> list[tuple[int, ...]]:
    """Decode QuSpin integer basis states into public local occupations."""
    domain = str(node[0])
    length = int(node[1])
    values = np.asarray(basis.states)
    if domain == "boson":
        local_dimension = int(node[2])
        return [
            tuple(
                (int(value) // local_dimension ** (length - 1 - index))
                % local_dimension
                for index in range(length)
            )
            for value in values
        ]
    occupations: list[tuple[int, ...]] = []
    for value in values:
        bits = tuple(
            (int(value) >> (length - 1 - index)) & 1 for index in range(length)
        )
        occupations.append(tuple(1 - bit for bit in bits) if domain == "spin" else bits)
    return occupations


def _basis_states(basis: Any, node: StateNode) -> list[tuple[int, ...]]:
    """Decode a tensor basis recursively in QuSpin's documented factor order."""
    if node[0] != "tensor":
        return _single_states(basis, node)
    left = _basis_states(basis.basis_left, node[1])
    right = _basis_states(basis.basis_right, node[2])
    return [first + second for first, second in itertools.product(left, right)]


def _basis_static_and_node(capability: str) -> tuple[Any, StaticTerms, StateNode]:
    """Build the QuSpin basis and exact operator-string counterpart."""
    static: StaticTerms = []
    if capability == "spin_full":
        basis = _spin_basis(6)
        _add(static, "x", 0.41, 0)
        _add(static, "z", -0.23, 2)
        _add(static, "zz", 0.37, 0, 1)
        _add(static, "+-", 0.29, 1, 4)
        _add(static, "-+", 0.29, 1, 4)
        return basis, static, ("spin", 6)
    if capability == "spin_full_throughput":
        qubits = 18
        basis = _spin_basis(qubits)
        _add(static, "x", 0.13, 0)
        _add(static, "z", -0.19, 0)
        for index in range(qubits - 1):
            _add(static, "zz", 0.07, index, index + 1)
            _add(static, "+-", 0.23, index, index + 1)
            _add(static, "-+", 0.23, index, index + 1)
        return basis, static, ("spin", qubits)
    if capability == "spin_fixed_hamming":
        basis = _spin_basis(6, particles=2)
        _add(static, "+-", 0.67, 0, 1)
        _add(static, "-+", 0.67, 0, 1)
        _add(static, "+-", -0.29, 2, 5)
        _add(static, "-+", -0.29, 2, 5)
        _add(static, "zz", 0.22, 0, 4)
        _add(static, "z", 0.11, 3)
        return basis, static, ("spin", 6)
    if capability == "fermion_full":
        basis = spinless_fermion_basis_1d(4)
        _add(static, "+-", -0.73, 0, 1)
        _add(static, "+-", -0.73, 1, 0)
        _add(static, "+-", 0.19j, 2, 3)
        _add(static, "+-", -0.19j, 3, 2)
        _add(static, "n", 0.35, 2)
        _add(static, "nn", 0.17, 0, 3)
        return basis, static, ("fermion", 4)
    if capability == "boson_full":
        basis = boson_basis_1d(3, sps=3)
        _add(static, "+-", -0.43, 0, 1)
        _add(static, "+-", -0.43, 1, 0)
        _add(static, "n", 0.31, 2)
        _add(static, "nn", 0.13, 0, 1)
        _add(static, "+", 0.17, 0)
        _add(static, "-", 0.17, 0)
        return basis, static, ("boson", 3, 3)
    if capability == "fermion_boson_full":
        basis = tensor_basis(spinless_fermion_basis_1d(3), boson_basis_1d(2, sps=3))
        _add(static, "+-|I", -0.51, 0, 1, 0)
        _add(static, "+-|I", -0.51, 1, 0, 0)
        _add(static, "I|+-", 0.23, 0, 0, 1)
        _add(static, "I|+-", 0.23, 0, 1, 0)
        _add(static, "n|+", 0.37, 0, 0)
        _add(static, "n|-", 0.37, 0, 0)
        _add(static, "n|n", 0.19, 1, 1)
        return basis, static, ("tensor", ("fermion", 3), ("boson", 2, 3))
    if capability == "fermion_boson_full_throughput":
        fermions = 10
        bosons = 4
        basis = tensor_basis(
            spinless_fermion_basis_1d(fermions), boson_basis_1d(bosons, sps=4)
        )
        _add(static, "I|n", 0.11, 0, 0)
        for index in range(fermions - 1):
            _add(static, "+-|I", -0.41, index, index + 1, 0)
            _add(static, "+-|I", -0.41, index + 1, index, 0)
            _add(static, "n|+", 0.04, index, index % bosons)
            _add(static, "n|-", 0.04, index, index % bosons)
        for index in range(bosons - 1):
            _add(static, "I|+-", 0.17, 0, index, index + 1)
            _add(static, "I|+-", 0.17, 0, index + 1, index)
        return basis, static, ("tensor", ("fermion", fermions), ("boson", bosons, 4))
    if capability == "fermion_fixed_boson_spectators":
        basis = tensor_basis(
            spinless_fermion_basis_1d(4, Nf=2), boson_basis_1d(2, sps=3)
        )
        _add(static, "+-|I", -0.61, 0, 1, 0)
        _add(static, "+-|I", -0.61, 1, 0, 0)
        _add(static, "+-|I", -0.61, 2, 3, 0)
        _add(static, "+-|I", -0.61, 3, 2, 0)
        _add(static, "n|+", 0.28, 0, 0)
        _add(static, "n|-", 0.28, 0, 0)
        _add(static, "I|n", 0.16, 0, 1)
        return basis, static, ("tensor", ("fermion", 4), ("boson", 2, 3))
    if capability == "boson_fixed_fermion_spectators":
        basis = tensor_basis(
            spinless_fermion_basis_1d(3),
            boson_basis_1d(3, Nb=2, sps=4),
        )
        _add(static, "I|+-", -0.55, 0, 0, 1)
        _add(static, "I|+-", -0.55, 0, 1, 0)
        _add(static, "I|+-", -0.55, 0, 1, 2)
        _add(static, "I|+-", -0.55, 0, 2, 1)
        _add(static, "n|I", 0.21, 0, 0)
        _add(static, "+-|I", -0.14, 1, 2, 0)
        _add(static, "+-|I", -0.14, 2, 1, 0)
        _add(static, "n|n", 0.09, 2, 0)
        return basis, static, ("tensor", ("fermion", 3), ("boson", 3, 4))
    if capability == "independent_spin_factors":
        basis = tensor_basis(_spin_basis(3, particles=1), _spin_basis(4, particles=2))
        _add(static, "+-|I", 0.52, 0, 2, 0)
        _add(static, "-+|I", 0.52, 0, 2, 0)
        _add(static, "I|+-", -0.34, 0, 0, 3)
        _add(static, "I|-+", -0.34, 0, 0, 3)
        _add(static, "z|z", 0.18, 1, 2)
        _add(static, "I|z", -0.07, 0, 1)
        return basis, static, ("tensor", ("spin", 3), ("spin", 4))
    if capability == "independent_spin_factors_throughput":
        factor_size = 11
        basis = tensor_basis(
            _spin_basis(factor_size, particles=5),
            _spin_basis(factor_size, particles=5),
        )
        _add(static, "z|z", 0.09, 0, 0)
        for index in range(factor_size - 1):
            _add(static, "+-|I", 0.21, index, index + 1, 0)
            _add(static, "-+|I", 0.21, index, index + 1, 0)
            _add(static, "I|+-", -0.17, 0, index, index + 1)
            _add(static, "I|-+", -0.17, 0, index, index + 1)
            _add(static, "z|z", 0.03, index, index)
        return basis, static, ("tensor", ("spin", factor_size), ("spin", factor_size))
    if capability == "reversed_boson_spin_factors":
        basis = tensor_basis(
            boson_basis_1d(3, Nb=2, sps=4), _spin_basis(5, particles=2)
        )
        _add(static, "+-|I", -0.47, 0, 1, 0)
        _add(static, "+-|I", -0.47, 1, 0, 0)
        _add(static, "I|+-", 0.39, 0, 1, 4)
        _add(static, "I|-+", 0.39, 0, 1, 4)
        _add(static, "n|z", 0.16, 2, 0)
        _add(static, "I|z", -0.12, 0, 3)
        return basis, static, ("tensor", ("boson", 3, 4), ("spin", 5))
    raise ValueError(f"QuSpin has no direct optimized runner for {capability!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate and emit the canonical action artifact without timing",
    )
    return parser.parse_args()


def main() -> None:
    """Construct, validate, and time one QuSpin capability operator."""
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    if manifest["comparison"] != "directly_comparable":
        raise SystemExit("this capability has no fair optimized QuSpin baseline")

    basis_start = time.perf_counter()
    basis, static, node = _basis_static_and_node(str(manifest["capability"]))
    total_basis_and_static_seconds = time.perf_counter() - basis_start
    static_start = _STATIC_STARTS.pop(id(static), time.perf_counter())
    static_build_seconds = time.perf_counter() - static_start
    basis_build_seconds = total_basis_and_static_seconds - static_build_seconds
    operator_start = time.perf_counter()
    if manifest["quspin_mode"] == "matrix_free":
        operator = quantum_LinearOperator(
            static,
            basis=basis,
            dtype=np.complex128,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
    else:
        operator = hamiltonian(
            static,
            [],
            basis=basis,
            dtype=np.complex128,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
            static_fmt="csr",
        )
    operator_build_seconds = time.perf_counter() - operator_start

    validation_start = time.perf_counter()
    states = np.asarray(_basis_states(basis, node), dtype=np.uint64)
    state = canonical_state(states, int(manifest["seed"]))
    validation_output = operator.dot(state)
    validation_mode = str(manifest.get("action_validation", "dense"))
    if validation_mode == "dense":
        explicit = hamiltonian(
            static,
            [],
            basis=basis,
            dtype=np.complex128,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
            static_fmt="csr",
        )
        dense_action = np.asarray(explicit.toarray(), dtype=np.complex128)
        action_error = float(np.max(np.abs(validation_output - dense_action @ state)))
        tolerance = float(manifest["action_tolerance"])
        if action_error > tolerance:
            raise RuntimeError(
                f"QuSpin dense-action mismatch {action_error} exceeds {tolerance}"
            )
        save_action_artifact(args.artifact, dense_action, states)
    else:
        action_error = 0.0
        save_action_vector_artifact(args.artifact, validation_output, states)
    validation_seconds = time.perf_counter() - validation_start

    if args.preflight:
        print(
            json.dumps(
                {
                    "stage": "preflight",
                    "library": "quspin",
                    "capability": manifest["capability"],
                    "dimension": int(basis.Ns),
                    "action_error": action_error,
                    "validation_seconds": validation_seconds,
                    "thread_environment": thread_environment(),
                    "workload_manifest": manifest["workload_manifest"],
                    "exact_command": exact_command(),
                    **worktree_fingerprint(ROOT),
                },
                sort_keys=True,
            )
        )
        return

    warmups = int(manifest["warmups"])
    repeats = int(manifest["repeats"])
    for _ in range(warmups):
        operator.dot(state)
    first_public_apply_start = time.perf_counter()
    output = operator.dot(state)
    first_public_apply_seconds = time.perf_counter() - first_public_apply_start
    public_apply_seconds: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = operator.dot(state)
        public_apply_seconds.append(time.perf_counter() - start)

    record: dict[str, Any] = {
        "library": "quspin",
        "capability": manifest["capability"],
        "comparison": manifest["comparison"],
        "operator_family": manifest["operator_family"],
        "dtype": manifest["dtype"],
        "quspin_mode": manifest["quspin_mode"],
        "dimension": int(basis.Ns),
        "term_count": len(static),
        "basis": str(type(basis).__name__),
        "basis_build_seconds": basis_build_seconds,
        "static_build_seconds": static_build_seconds,
        "operator_build_seconds": operator_build_seconds,
        "validation_seconds": validation_seconds,
        "action_error": action_error,
        "warmups": warmups,
        "sample_count": repeats,
        "first_public_apply_seconds": first_public_apply_seconds,
        "public_apply_seconds": public_apply_seconds,
        "public_apply_seconds_median": float(np.median(public_apply_seconds)),
        "caller_owned_apply_supported": False,
        "output_norm": float(np.linalg.norm(output)),
        "peak_rss_bytes": peak_rss_bytes(),
        "thread_environment": thread_environment(),
        "workload_manifest": manifest["workload_manifest"],
        "exact_command": exact_command(),
        **worktree_fingerprint(ROOT),
    }
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
