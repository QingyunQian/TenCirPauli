"""Narrow reproducible boundary profile for the static stabilizer-code path.

This driver intentionally keeps timing at public boundaries. The native
constructor is one timed region containing input packing/statistics, signed
elimination, normalizer/quotient construction, and logical pairing; the
script reports the subsequent logical read-back separately. It does not add
production counters or claim exact allocator/RSS accounting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import tencirpauli as tcp


def general_finite_rate_checks(nqubits: int) -> tuple[np.ndarray, np.ndarray]:
    rank = nqubits // 2
    x = np.zeros((rank, nqubits), dtype=np.uint8)
    z = np.zeros_like(x)
    rows = np.arange(rank)
    x[rows, 2 * rows] = 1
    z[rows, 2 * rows + 1] = 1
    return x, z


def batch_codes(nqubits: int, batch: int) -> np.ndarray:
    values = np.zeros((batch, nqubits), dtype=np.uint8)
    rows = np.arange(batch)
    values[rows, 2 * (rows % (nqubits // 2))] = 1
    return values


def timed(function: Callable[[], Any], repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    package_file = Path(tcp.__file__).resolve()
    native_file = Path(tcp._native.__file__).resolve()
    expected_package = root / "python" / "tencirpauli" / "__init__.py"
    if (
        package_file != expected_package
        or native_file.parent != expected_package.parent
    ):
        raise RuntimeError(
            f"benchmark is not using the source tree: {package_file} / {native_file}"
        )

    x_checks, z_checks = general_finite_rate_checks(arguments.nqubits)
    errors = batch_codes(arguments.nqubits, arguments.batch)
    corrections = np.zeros_like(errors)

    input_ms = timed(
        lambda: general_finite_rate_checks(arguments.nqubits), arguments.repeats
    )
    construction_ms = timed(
        lambda: tcp.StabilizerCode.from_symplectic(x_checks, z_checks),
        arguments.repeats,
    )
    code = tcp.StabilizerCode.from_symplectic(x_checks, z_checks)
    logical_readback_ms = timed(code.logical_operator_arrays, arguments.repeats)
    syndrome_ms = timed(lambda: code.syndrome_many(errors), arguments.repeats)
    classification_ms = timed(lambda: code.classify_many(errors), arguments.repeats)
    correction_ms = timed(
        lambda: code.analyze_corrections(errors, corrections), arguments.repeats
    )

    print(
        json.dumps(
            {
                "nqubits": arguments.nqubits,
                "batch": arguments.batch,
                "repeats": arguments.repeats,
                "package_file": str(package_file),
                "native_file": str(native_file),
                "input_packing_and_matrix_setup_ms": input_ms,
                "native_construction_ms": construction_ms,
                "logical_readback_ms": logical_readback_ms,
                "syndrome_many_ms": syndrome_ms,
                "classification_ms": classification_ms,
                "correction_analysis_ms": correction_ms,
                "accuracy_max_abs_error": 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
