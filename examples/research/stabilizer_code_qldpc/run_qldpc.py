"""Run the documented qLDPC 0.3.3 static-code path for one workload."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import resource
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np
import qldpc
from qldpc import codes
from workloads import (
    Workload,
    correction_batch,
    digest_arrays,
    error_batch,
    get_workload,
    workload_hash,
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _build(workload: Workload) -> Any:
    if workload.kind == "css":
        return codes.CSSCode(workload.x_checks, workload.z_checks)
    matrix = np.hstack((workload.x_checks, workload.z_checks))
    return codes.QuditCode(matrix, field=2)


def _family_generation(workload: Workload) -> float:
    if workload.family != "hgp_repetition":
        return 0.0
    length = int(workload.name.removeprefix("hgp_"))
    start = time.perf_counter()
    generated = codes.HGPCode(
        codes.RepetitionCode(length), codes.RepetitionCode(length)
    )
    generated_x = np.asarray(generated.matrix_x, dtype=np.uint8) % 2
    generated_z = np.asarray(generated.matrix_z, dtype=np.uint8) % 2
    if not np.array_equal(generated_x, workload.x_checks) or not np.array_equal(
        generated_z, workload.z_checks
    ):
        raise RuntimeError(
            "independent HGP workload does not match qLDPC's family constructor"
        )
    return time.perf_counter() - start


def _logical_ops(code: Any) -> np.ndarray[Any, Any]:
    return np.ascontiguousarray(np.asarray(code.get_logical_ops(), dtype=np.uint8) % 2)


def _pack_bits(matrix: np.ndarray[Any, Any]) -> dict[str, Any]:
    array = np.ascontiguousarray(matrix, dtype=np.uint8)
    packed = np.packbits(array, axis=1, bitorder="little") if array.ndim == 2 else array
    return {
        "shape": list(array.shape),
        "hex": np.ascontiguousarray(packed).tobytes().hex(),
    }


def _payload(array: np.ndarray[Any, Any]) -> dict[str, Any]:
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "digest": digest_arrays(value)}


def _syndromes(
    workload: Workload, errors: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    error_x = (errors == 1) | (errors == 2)
    error_z = (errors == 2) | (errors == 3)
    if workload.kind == "css":
        return np.ascontiguousarray(
            np.hstack((error_z @ workload.x_checks.T, error_x @ workload.z_checks.T))
            % 2,
            dtype=np.uint8,
        )
    return np.ascontiguousarray(
        (error_z @ workload.x_checks.T + error_x @ workload.z_checks.T) % 2,
        dtype=np.uint8,
    )


def _classify(
    code: Any,
    workload: Workload,
    logical_ops: np.ndarray[Any, Any],
    errors: np.ndarray[Any, Any],
) -> dict[str, Any]:
    nqubits = workload.nqubits
    error_x = (errors == 1) | (errors == 2)
    error_z = (errors == 2) | (errors == 3)
    if workload.kind == "css":
        syndromes = (
            np.hstack((error_z @ workload.x_checks.T, error_x @ workload.z_checks.T))
            % 2
        )
    else:
        syndromes = (error_z @ workload.x_checks.T + error_x @ workload.z_checks.T) % 2
    logical_x_ops = logical_ops[: code.dimension]
    logical_z_ops = logical_ops[code.dimension :]
    logical_x = (
        error_z @ logical_z_ops[:, :nqubits].T + error_x @ logical_z_ops[:, nqubits:].T
    ) % 2
    logical_z = (
        error_z @ logical_x_ops[:, :nqubits].T + error_x @ logical_x_ops[:, nqubits:].T
    ) % 2
    zero = ~np.any(syndromes, axis=1)
    zero_indices = np.flatnonzero(zero).astype(np.uintp, copy=False)
    zero_logical_x = np.ascontiguousarray(logical_x[zero], dtype=np.uint8)
    zero_logical_z = np.ascontiguousarray(logical_z[zero], dtype=np.uint8)
    kinds = np.zeros(errors.shape[0], dtype=np.uint8)
    kinds[zero] = np.where(
        np.any(zero_logical_x | zero_logical_z, axis=1), 2, 1
    ).astype(np.uint8)
    return {
        "syndromes": _payload(syndromes),
        "kinds": _payload(kinds),
        "zero_syndrome_indices": _payload(zero_indices),
        "logical_x": _pack_bits(zero_logical_x),
        "logical_z": _pack_bits(zero_logical_z),
        "digest": digest_arrays(
            syndromes, kinds, zero_indices, zero_logical_x, zero_logical_z
        ),
    }


def _measure(
    function: Callable[[], Any], repetitions: int
) -> tuple[dict[str, Any], Any]:
    start = time.perf_counter()
    first_result = function()
    first_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    mean = statistics.fmean(samples)
    return (
        {
            "first_seconds": first_seconds,
            "steady_seconds_median": statistics.median(samples),
            "steady_seconds_min": min(samples),
            "steady_seconds_max": max(samples),
            "steady_seconds_cv": statistics.pstdev(samples) / mean if mean else 0.0,
            "steady_rounds": repetitions,
        },
        first_result,
    )


def _metadata(code: Any) -> dict[str, int]:
    return {
        "nqubits": int(code.num_qudits),
        "nchecks": int(code.num_checks),
        "rank": int(code.rank),
        "nlogical": int(code.dimension),
        "nrelations": int(code.num_checks - code.rank),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="repetition_64")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--correctness-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    workload = get_workload(args.workload)
    errors = error_batch(workload, args.batch_size)
    corrections = correction_batch(workload, errors)
    family_generation_seconds = _family_generation(workload)

    if args.correctness_only:
        warmup_seconds = 0.0
        code = _build(workload)
        warmup_logicals = _logical_ops(code)
        logical_ops = warmup_logicals
        metadata_result = _metadata(code)
        logical_result = logical_ops
        syndrome_result = _payload(_syndromes(workload, errors))
        classification_result = _classify(code, workload, logical_ops, errors)
        correction_result = _classify(code, workload, logical_ops, errors ^ corrections)
        logical_timing = {}
        syndrome_timing = {}
        classification_timing = {}
        correction_timing = {}
    else:
        warmup_start = time.perf_counter()
        warmup_code = _build(workload)
        warmup_logicals = _logical_ops(warmup_code)
        _classify(warmup_code, workload, warmup_logicals, errors[:1])
        warmup_seconds = time.perf_counter() - warmup_start

        metadata_result = _metadata(warmup_code)
        logical_timing, logical_result = _measure(
            lambda: _logical_ops(_build(workload)), args.repetitions
        )

        code = _build(workload)
        logical_ops = _logical_ops(code)
        syndrome_timing, syndrome_result = _measure(
            lambda: _payload(_syndromes(workload, errors)),
            args.repetitions,
        )
        classification_timing, classification_result = _measure(
            lambda: _classify(code, workload, logical_ops, errors), args.repetitions
        )
        correction_timing, correction_result = _measure(
            lambda: _classify(code, workload, logical_ops, errors ^ corrections),
            args.repetitions,
        )

    output = {
        "schema_version": 1,
        "implementation": "qldpc",
        "backend": "public_qldpc_code_plus_documented_symplectic_adapter",
        "package_version": importlib.metadata.version("qldpc"),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "galois", "stim")
        },
        "qldpc_path": qldpc.__file__,
        "python": sys.version,
        "workload": workload.name,
        "family": workload.family,
        "kind": workload.kind,
        "matrix_hash": workload.matrix_hash,
        "workload_hash": workload_hash(workload, errors, corrections),
        "batch_size": args.batch_size,
        "input_bytes": int(errors.nbytes + corrections.nbytes),
        "output_bytes": {
            "syndrome": int(args.batch_size * metadata_result["nchecks"]),
            "classification": int(
                args.batch_size * metadata_result["nchecks"]
                + args.batch_size
                + classification_result["logical_x"]["shape"][0]
                * (np.dtype(np.uintp).itemsize + metadata_result["nlogical"] * 2)
            ),
            "correction": int(
                args.batch_size * metadata_result["nchecks"]
                + args.batch_size
                + correction_result["logical_x"]["shape"][0]
                * (np.dtype(np.uintp).itemsize + metadata_result["nlogical"] * 2)
            ),
        },
        "metadata": metadata_result,
        "logical_ops": {
            "shape": list(logical_result.shape),
            "packed": _pack_bits(logical_result),
            "digest": digest_arrays(logical_result),
        },
        "correctness": {
            "syndrome": syndrome_result,
            "classification": classification_result,
            "correction": correction_result,
        },
        "timings": {
            "family_generation_seconds": family_generation_seconds,
            "warmup_seconds": warmup_seconds,
            "construction_to_logicals": logical_timing,
            "syndrome": syndrome_timing,
            "classification": classification_timing,
            "correction": correction_timing,
        },
        "peak_rss_bytes": _peak_rss_bytes(),
        "gc_enabled": gc.isenabled(),
        "thread_environment": {
            name: os.environ.get(name, "unset")
            for name in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "reference_logicals_digest": digest_arrays(workload.reference_logicals),
        "warmup_logicals_digest": digest_arrays(warmup_logicals),
    }
    print("RESULT_JSON " + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
