"""Compare the public TenCirPauli and qLDPC static-code workflows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from workloads import (
    available_workloads,
    correction_batch,
    digest_arrays,
    error_batch,
    get_workload,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _record(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON "):
            value = json.loads(line.removeprefix("RESULT_JSON "))
            if isinstance(value, dict):
                return value
    raise RuntimeError("runner did not emit RESULT_JSON")


def _run(
    executable: str,
    script: Path,
    workload: str,
    batch_size: int,
    repetitions: int,
    correctness_only: bool,
    environment: dict[str, str],
) -> dict[str, Any]:
    command = [
        executable,
        str(script),
        "--workload",
        workload,
        "--batch-size",
        str(batch_size),
        "--repetitions",
        str(repetitions),
    ]
    if correctness_only:
        command.append("--correctness-only")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{script.name} failed with exit code {completed.returncode}:\n{completed.stderr.strip()}"
        )
    return _record(completed.stdout)


def _unpack_bits(value: dict[str, Any]) -> np.ndarray[Any, Any]:
    rows, columns = (int(item) for item in value["shape"])
    if rows == 0:
        return np.empty((0, columns), dtype=np.uint8)
    packed = np.frombuffer(bytes.fromhex(value["hex"]), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little").reshape((rows, -1))[:, :columns]


def _symplectic(
    left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    nqubits = left.shape[1] // 2
    return (
        left[:, :nqubits] @ right[:, nqubits:].T
        + left[:, nqubits:] @ right[:, :nqubits].T
    ) % 2


def _check_ops(workload: Any) -> np.ndarray[Any, Any]:
    if workload.kind == "css":
        return np.vstack(
            (
                np.hstack((workload.x_checks, np.zeros_like(workload.x_checks))),
                np.hstack((np.zeros_like(workload.z_checks), workload.z_checks)),
            )
        )
    return np.hstack((workload.x_checks, workload.z_checks))


def _symplectic_coordinates(
    vectors: np.ndarray[Any, Any], logical_ops: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    k = logical_ops.shape[0] // 2
    logical_x = _symplectic(vectors, logical_ops[k:])
    logical_z = _symplectic(vectors, logical_ops[:k])
    return np.hstack((logical_x, logical_z)).astype(np.uint8)


def _logical_coordinates(
    errors: np.ndarray[Any, Any], logical_ops: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    error_x = (errors == 1) | (errors == 2)
    error_z = (errors == 2) | (errors == 3)
    vectors = np.hstack((error_x, error_z)).astype(np.uint8)
    coordinates = _symplectic_coordinates(vectors, logical_ops)
    k = logical_ops.shape[0] // 2
    return coordinates[:, :k], coordinates[:, k:]


def _gf2_rank(rows: np.ndarray[Any, Any]) -> int:
    basis: dict[int, int] = {}
    for row in np.asarray(rows, dtype=np.uint8):
        packed = np.packbits(row, bitorder="little")
        value = int.from_bytes(packed.tobytes(), "little")
        while value:
            pivot = value.bit_length() - 1
            previous = basis.get(pivot)
            if previous is None:
                basis[pivot] = value
                break
            value ^= previous
    return len(basis)


def _validate_logicals(
    workload: Any,
    tencirpauli_ops: np.ndarray[Any, Any],
    qldpc_ops: np.ndarray[Any, Any],
) -> dict[str, Any]:
    if tencirpauli_ops.shape != qldpc_ops.shape:
        raise RuntimeError(
            f"logical shape mismatch: {tencirpauli_ops.shape} vs {qldpc_ops.shape}"
        )
    k = tencirpauli_ops.shape[0] // 2
    expected_pairing = np.zeros((2 * k, 2 * k), dtype=np.uint8)
    expected_pairing[:k, k:] = np.eye(k, dtype=np.uint8)
    expected_pairing[k:, :k] = np.eye(k, dtype=np.uint8)
    for name, ops in (("tencirpauli", tencirpauli_ops), ("qldpc", qldpc_ops)):
        checks = _check_ops(workload)
        if np.any(_symplectic(ops, checks)):
            raise RuntimeError(
                f"{name} logical operators do not commute with the checks"
            )
        if not np.array_equal(_symplectic(ops, ops), expected_pairing):
            raise RuntimeError(
                f"{name} logical operators are not a canonical symplectic basis"
            )
    checks = _check_ops(workload)
    if _gf2_rank(np.vstack((checks, tencirpauli_ops, qldpc_ops))) != _gf2_rank(
        np.vstack((checks, tencirpauli_ops))
    ):
        raise RuntimeError("logical quotient spans do not agree")
    transform_matrix = _symplectic_coordinates(qldpc_ops, tencirpauli_ops)
    if _gf2_rank(transform_matrix) != 2 * k:
        raise RuntimeError("logical basis transformation is singular")
    return {
        "shape": [2 * k, 2 * k],
        "digest": digest_arrays(transform_matrix),
        "matrix": transform_matrix,
    }


def _compare_payloads(
    workload: Any,
    errors: np.ndarray[Any, Any],
    corrections: np.ndarray[Any, Any],
    tencirpauli: dict[str, Any],
    qldpc: dict[str, Any],
) -> dict[str, Any]:
    if tencirpauli["matrix_hash"] != qldpc["matrix_hash"]:
        raise RuntimeError("matrix hash mismatch")
    if tencirpauli["workload_hash"] != qldpc["workload_hash"]:
        raise RuntimeError("workload hash mismatch")
    if tencirpauli["metadata"] != qldpc["metadata"]:
        raise RuntimeError(
            f"metadata mismatch: {tencirpauli['metadata']} vs {qldpc['metadata']}"
        )
    t_ops = _unpack_bits(tencirpauli["logical_ops"]["packed"])
    q_ops = _unpack_bits(qldpc["logical_ops"]["packed"])
    transformation = _validate_logicals(workload, t_ops, q_ops)

    for operation in ("syndrome", "classification", "correction"):
        left = tencirpauli["correctness"][operation]
        right = qldpc["correctness"][operation]
        if operation == "syndrome":
            if left != right:
                raise RuntimeError("syndrome mismatch")
            continue
        if left["syndromes"] != right["syndromes"]:
            raise RuntimeError(f"{operation} syndrome mismatch")
        if left["kinds"] != right["kinds"]:
            raise RuntimeError(f"{operation} kind mismatch")
        if operation != "syndrome":
            if left["zero_syndrome_indices"] != right["zero_syndrome_indices"]:
                raise RuntimeError(f"{operation} zero-syndrome index mismatch")
            left_x = _unpack_bits(left["logical_x"])
            left_z = _unpack_bits(left["logical_z"])
            right_x = _unpack_bits(right["logical_x"])
            right_z = _unpack_bits(right["logical_z"])
            right_coordinates = np.hstack((right_x, right_z))
            transformed = (right_coordinates @ transformation["matrix"]) % 2
            expected = np.hstack((left_x, left_z))
            if not np.array_equal(transformed, expected):
                raise RuntimeError(f"{operation} logical coordinates mismatch")

    independent_errors = _logical_coordinates(errors, t_ops)
    independent_residuals = _logical_coordinates(errors ^ corrections, t_ops)
    return {
        "correct": True,
        "logical_basis_transform": {
            key: value for key, value in transformation.items() if key != "matrix"
        },
        "independent_error_logical_digest": digest_arrays(*independent_errors),
        "independent_residual_logical_digest": digest_arrays(*independent_residuals),
    }


def _thread_environment() -> dict[str, str]:
    names = (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return {name: "1" for name in names}


def _run_one(
    args: argparse.Namespace, workload_name: str, batch_size: int
) -> dict[str, Any]:
    workload = get_workload(workload_name)
    errors = error_batch(workload, batch_size)
    corrections = correction_batch(workload, errors)
    environment = {**os.environ, **_thread_environment()}
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), environment.get("PYTHONPATH", "")]
    )
    tencirpauli = _run(
        args.tencirpauli_python,
        HERE / "run_tencirpauli.py",
        workload_name,
        batch_size,
        args.repetitions,
        True,
        environment,
    )
    qldpc = _run(
        args.qldpc_python,
        HERE / "run_qldpc.py",
        workload_name,
        batch_size,
        args.repetitions,
        True,
        environment,
    )
    correctness = _compare_payloads(workload, errors, corrections, tencirpauli, qldpc)
    tencirpauli = _run(
        args.tencirpauli_python,
        HERE / "run_tencirpauli.py",
        workload_name,
        batch_size,
        args.repetitions,
        False,
        environment,
    )
    qldpc = _run(
        args.qldpc_python,
        HERE / "run_qldpc.py",
        workload_name,
        batch_size,
        args.repetitions,
        False,
        environment,
    )
    timed_correctness = _compare_payloads(
        workload, errors, corrections, tencirpauli, qldpc
    )
    if not timed_correctness["correct"]:
        raise RuntimeError("timed result failed correctness validation")
    return {
        "workload": workload_name,
        "batch_size": batch_size,
        "correctness": correctness,
        "tencirpauli": tencirpauli,
        "qldpc": qldpc,
        "speedups_qldpc_over_tencirpauli": {
            operation: qldpc["timings"][operation]["steady_seconds_median"]
            / tencirpauli["timings"][operation]["steady_seconds_median"]
            for operation in (
                "construction_to_logicals",
                "syndrome",
                "classification",
                "correction",
            )
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", default="repetition_64,hgp_8,five_qubit_16")
    parser.add_argument("--batch-sizes", default="1,256,4096")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tencirpauli-python", default=sys.executable)
    parser.add_argument("--qldpc-python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    workload_names = (
        available_workloads()
        if args.workloads == "all"
        else tuple(item.strip() for item in args.workloads.split(",") if item.strip())
    )
    batch_sizes = tuple(
        int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()
    )
    if args.repetitions < 1 or not batch_sizes:
        raise SystemExit("repetitions and batch sizes must be positive")
    results = [
        _run_one(args, workload, batch_size)
        for workload in workload_names
        for batch_size in batch_sizes
    ]
    table = []
    for result in results:
        table.append(
            {
                "workload": result["workload"],
                "batch_size": result["batch_size"],
                "nqubits": result["tencirpauli"]["metadata"]["nqubits"],
                "nchecks": result["tencirpauli"]["metadata"]["nchecks"],
                "nlogical": result["tencirpauli"]["metadata"]["nlogical"],
                "tencirpauli_peak_rss_bytes": result["tencirpauli"]["peak_rss_bytes"],
                "qldpc_peak_rss_bytes": result["qldpc"]["peak_rss_bytes"],
                **result["speedups_qldpc_over_tencirpauli"],
            }
        )
    print(
        "RESULT_JSON "
        + json.dumps(
            {"schema_version": 1, "correct": True, "results": results, "table": table},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
