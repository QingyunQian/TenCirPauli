"""Run the public TenCirPauli stabilizer-code path for one workload."""

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
from workloads import (
    Workload,
    correction_batch,
    digest_arrays,
    error_batch,
    get_workload,
    workload_hash,
)

import tencirpauli as tcp
import tencirpauli._native as native


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _build(workload: Workload) -> tcp.StabilizerCode:
    if workload.kind == "css":
        return tcp.StabilizerCode.from_css(
            workload.x_checks, workload.z_checks, max_bytes=None
        )
    return tcp.StabilizerCode.from_symplectic(
        workload.x_checks, workload.z_checks, max_bytes=None
    )


def _unpack_words(words: np.ndarray[Any, Any], nqubits: int) -> np.ndarray[Any, Any]:
    word_count = (nqubits + 63) // 64
    packed = np.asarray(words, dtype=np.uint64).reshape((-1, word_count))
    shifts = np.arange(64, dtype=np.uint64)
    bits = ((packed[:, :, None] >> shifts) & 1).astype(np.uint8)
    return bits.reshape((packed.shape[0], word_count * 64))[:, :nqubits]


def _logical_ops(code: tcp.StabilizerCode) -> np.ndarray[Any, Any]:
    arrays = code.logical_operator_arrays()
    logical_x_x = _unpack_words(arrays.logical_x_x_words, code.nqubits)
    logical_x_z = _unpack_words(arrays.logical_x_z_words, code.nqubits)
    logical_z_x = _unpack_words(arrays.logical_z_x_words, code.nqubits)
    logical_z_z = _unpack_words(arrays.logical_z_z_words, code.nqubits)
    return np.vstack(
        (
            np.hstack((logical_x_x, logical_x_z)),
            np.hstack((logical_z_x, logical_z_z)),
        )
    ).astype(np.uint8, copy=False)


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


def _classification_payload(result: Any) -> dict[str, Any]:
    syndromes = (
        result.residual_syndromes
        if hasattr(result, "residual_syndromes")
        else result.syndromes
    )
    return {
        "syndromes": _payload(syndromes),
        "kinds": _payload(result.kinds),
        "zero_syndrome_indices": _payload(result.zero_syndrome_indices),
        "logical_x": _pack_bits(result.logical_x),
        "logical_z": _pack_bits(result.logical_z),
        "digest": digest_arrays(
            syndromes,
            result.kinds,
            result.zero_syndrome_indices,
            result.logical_x,
            result.logical_z,
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


def _metadata(code: tcp.StabilizerCode) -> dict[str, int]:
    return {
        "nqubits": code.nqubits,
        "nchecks": code.nchecks,
        "rank": code.rank,
        "nlogical": code.nlogical,
        "nrelations": code.nrelations,
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

    if args.correctness_only:
        warmup_seconds = 0.0
        code = _build(workload)
        warmup_logicals = _logical_ops(code)
        metadata_result = _metadata(code)
        logical_result = warmup_logicals
        syndrome_result = _payload(code.syndrome_many(errors))
        classification_result = _classification_payload(code.classify_many(errors))
        correction_result = _classification_payload(
            code.analyze_corrections(errors, corrections)
        )
        logical_timing = {}
        syndrome_timing = {}
        classification_timing = {}
        correction_timing = {}
    else:
        warmup_start = time.perf_counter()
        warmup_code = _build(workload)
        warmup_logicals = _logical_ops(warmup_code)
        warmup_code.classify_many(errors[:1])
        warmup_seconds = time.perf_counter() - warmup_start

        metadata_result = _metadata(warmup_code)
        logical_timing, logical_result = _measure(
            lambda: _logical_ops(_build(workload)), args.repetitions
        )

        code = _build(workload)
        _logical_ops(code)
        syndrome_timing, syndrome_result = _measure(
            lambda: _payload(code.syndrome_many(errors)), args.repetitions
        )
        classification_timing, classification_result = _measure(
            lambda: _classification_payload(code.classify_many(errors)),
            args.repetitions,
        )
        correction_timing, correction_result = _measure(
            lambda: _classification_payload(
                code.analyze_corrections(errors, corrections)
            ),
            args.repetitions,
        )

    output = {
        "schema_version": 1,
        "implementation": "tencirpauli",
        "backend": "public_stabilizer_code",
        "package_version": str(tcp.__version__),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "tensorcircuit-ng")
        },
        "python": sys.version,
        "module_path": tcp.__file__,
        "native_module_path": native.__file__,
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
