"""Run and compare the TenCirPauli and PauliPropagation.jl VQE cases."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=ROOT.parents[2], text=True, capture_output=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise RuntimeError(f"command did not emit RESULT_JSON: {completed.stdout}")


def _python_result(
    mode: str, *, nqubits: int | None = None, layers: int | None = None
) -> dict[str, Any]:
    command = [sys.executable, str(ROOT / "run_tencirpauli.py"), "--mode", mode]
    if nqubits is not None:
        command.extend(["--nqubits", str(nqubits), "--layers", str(layers)])
    return _run(command)


def _julia_result(
    julia: str, mode: str, *, nqubits: int | None = None, layers: int | None = None
) -> dict[str, Any]:
    command = [
        julia,
        "--threads=auto",
        "--startup-file=no",
        "--history-file=no",
        str(ROOT / "run_julia.jl"),
        "--mode",
        mode,
    ]
    if nqubits is not None:
        command.extend(["--nqubits", str(nqubits), "--layers", str(layers)])
    return _run(command)


def _correctness(julia: str) -> None:
    python = _python_result("correctness")
    julia_result = _julia_result(julia, "correctness")
    reference_value = float(python["dense_value"])
    reference_gradient = np.asarray(python["dense_gradient"], dtype=np.float64)
    python_value = float(python["native_value"])
    python_gradient = np.asarray(python["native_gradient"], dtype=np.float64)
    julia_value = float(julia_result["julia_value"])
    julia_gradient = np.asarray(julia_result["julia_gradient"], dtype=np.float64)
    python_truncated_gradient = np.asarray(
        python["truncated_gradient"], dtype=np.float64
    )
    julia_truncated_gradient = np.asarray(
        julia_result["truncated_gradient"], dtype=np.float64
    )
    print("Correctness: 8 qubits, 2 layers, max_weight=8")
    print(
        f"  TenCirPauli vs dense value error: {abs(python_value - reference_value):.3e}"
    )
    print(
        f"  TenCirPauli vs dense gradient max error: {np.max(np.abs(python_gradient - reference_gradient)):.3e}"
    )
    print(f"  Julia vs dense value error: {abs(julia_value - reference_value):.3e}")
    print(
        f"  Julia vs dense gradient max error: {np.max(np.abs(julia_gradient - reference_gradient)):.3e}"
    )
    print(f"  TenCirPauli vs Julia value error: {abs(python_value - julia_value):.3e}")
    print(
        f"  TenCirPauli vs Julia gradient max error: {np.max(np.abs(python_gradient - julia_gradient)):.3e}"
    )
    print("Correctness: 8 qubits, 2 layers, max_weight=5")
    print(
        f"  TenCirPauli vs its finite-difference gradient: {python['truncated_fd_gradient_max_abs_error']:.3e}"
    )
    print(
        f"  Julia rewindgradient vs its finite-difference gradient: {julia_result['truncated_fd_gradient_max_abs_error']:.3e}"
    )
    print(
        f"  TenCirPauli vs Julia truncated value error: {abs(float(python['truncated_value']) - float(julia_result['truncated_value'])):.3e}"
    )
    print(
        f"  TenCirPauli vs Julia truncated gradient max error: {np.max(np.abs(python_truncated_gradient - julia_truncated_gradient)):.3e}"
    )
    if abs(python_value - reference_value) > 1e-10:
        raise AssertionError("TenCirPauli dense correctness check failed")
    if np.max(np.abs(python_gradient - reference_gradient)) > 1e-9:
        raise AssertionError("TenCirPauli dense gradient check failed")
    if abs(julia_value - reference_value) > 1e-10:
        raise AssertionError("PauliPropagation.jl dense correctness check failed")
    if np.max(np.abs(julia_gradient - reference_gradient)) > 1e-8:
        raise AssertionError("PauliPropagation.jl dense gradient check failed")
    if float(python["truncated_fd_gradient_max_abs_error"]) > 1e-7:
        raise AssertionError(
            "TenCirPauli truncated gradient finite-difference check failed"
        )
    if float(julia_result["truncated_fd_gradient_max_abs_error"]) > 1e-7:
        raise AssertionError(
            "PauliPropagation.jl truncated gradient finite-difference check failed"
        )


def _benchmark(julia: str, repeats: int, iterations: int, learning_rate: float) -> None:
    python = _run(
        [
            sys.executable,
            str(ROOT / "run_tencirpauli.py"),
            "--mode",
            "benchmark",
            "--nqubits",
            "28",
            "--layers",
            "6",
            "--max-weight",
            "5",
            "--repeats",
            str(repeats),
            "--iterations",
            str(iterations),
            "--learning-rate",
            str(learning_rate),
        ]
    )
    julia_result = _run(
        [
            julia,
            "--threads=auto",
            "--startup-file=no",
            "--history-file=no",
            str(ROOT / "run_julia.jl"),
            "--mode",
            "benchmark",
            "--nqubits",
            "28",
            "--layers",
            "6",
            "--max-weight",
            "5",
            "--repeats",
            str(repeats),
            "--iterations",
            str(iterations),
            "--learning-rate",
            str(learning_rate),
        ]
    )
    print("Benchmark: 28 qubits, 6 layers, max_weight=5, 168 parameters")
    print(
        "  implementation                 setup       first       steady      VQE(steps)   final energy"
    )
    for result in (python, julia_result):
        print(
            f"  {result['implementation']:<30} "
            f"{result['setup_seconds']:>8.4f}s "
            f"{result['first_value_and_grad_seconds']:>8.4f}s "
            f"{result['steady_value_and_grad_seconds_median']:>8.4f}s "
            f"{result['vqe_seconds']:>10.4f}s "
            f"{result['vqe_final_value']:>12.8f}"
        )
    print(f"  Julia threads: {julia_result['julia_threads']}")
    python_first_gradient = np.asarray(python["first_gradient"], dtype=np.float64)
    julia_first_gradient = np.asarray(julia_result["first_gradient"], dtype=np.float64)
    print(
        "  truncated initial-point cross-result: value delta="
        f"{abs(float(python['first_value']) - float(julia_result['first_value'])):.3e}, "
        "gradient max delta="
        f"{np.max(np.abs(python_first_gradient - julia_first_gradient)):.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--julia", default=shutil.which("julia") or "julia")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    _correctness(args.julia)
    _benchmark(args.julia, args.repeats, args.iterations, args.learning_rate)


if __name__ == "__main__":
    main()
