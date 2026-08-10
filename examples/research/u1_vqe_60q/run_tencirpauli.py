"""Native TenCirPauli U(1) VQE value-and-gradient comparison workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import sys
import time
from typing import Any

import numpy as np

import tencirpauli as tcp


NQUBITS = 60
PARTICLES = 2
LAYERS = 10
PARAMETERS_PER_LAYER = NQUBITS // 2 + NQUBITS
PARAMETER_COUNT = LAYERS * PARAMETERS_PER_LAYER


def _peak_rss_bytes() -> int:
    """Return the process high-water RSS in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _pairs(layer: int) -> tuple[tuple[int, int], ...]:
    """Return one periodic brickwork matching layer."""
    if layer % 2 == 0:
        return tuple((index, index + 1) for index in range(0, NQUBITS, 2))
    return tuple((index, (index + 1) % NQUBITS) for index in range(1, NQUBITS, 2))


def _angle_vector(seed: int) -> np.ndarray[Any, Any]:
    """Return deterministic small angles for every gate occurrence."""
    generator = np.random.default_rng(seed)
    values = generator.uniform(-0.18, 0.18, size=PARAMETER_COUNT)
    return np.asarray(values, dtype=np.float64)


def _hamiltonian() -> tcp.PauliOperator:
    """Build the periodic nearest-neighbour XX+YY Hamiltonian."""
    terms: list[tuple[str, float]] = []
    for index in range(NQUBITS):
        neighbour = (index + 1) % NQUBITS
        xx = ["I"] * NQUBITS
        yy = ["I"] * NQUBITS
        xx[index] = xx[neighbour] = "X"
        yy[index] = yy[neighbour] = "Y"
        terms.append(("".join(xx), 1.0))
        terms.append(("".join(yy), 1.0))
    return tcp.PauliOperator.from_terms(NQUBITS, terms)


def _circuit(parameters: np.ndarray[Any, Any]) -> tcp.U1Circuit:
    """Build the native circuit with one angle per gate occurrence."""
    if parameters.shape != (PARAMETER_COUNT,):
        raise ValueError(
            f"expected {PARAMETER_COUNT} parameters, got {parameters.shape}"
        )
    circuit = tcp.U1Circuit(NQUBITS, particle_number=PARTICLES, occupied=(0, 1))
    cursor = 0
    for layer in range(LAYERS):
        for left, right in _pairs(layer):
            circuit.iswap(left, right, theta=float(parameters[cursor]))
            cursor += 1
        for qubit in range(NQUBITS):
            circuit.rz(qubit, theta=float(parameters[cursor]))
            cursor += 1
    if cursor != PARAMETER_COUNT:
        raise AssertionError(f"constructed {cursor} angles, expected {PARAMETER_COUNT}")
    return circuit


def _measure(function: Any, repeats: int) -> float:
    """Return the synchronized median of repeated native calls."""
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    """Validate and measure the native U(1) value-and-gradient path."""
    args = _parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    parameters = _angle_vector(args.seed)
    hamiltonian = _hamiltonian()

    build_start = time.perf_counter()
    circuit = _circuit(parameters)
    circuit_build_seconds = time.perf_counter() - build_start

    first_start = time.perf_counter()
    first = circuit.value_and_grad(hamiltonian)
    first_value_and_grad_seconds = time.perf_counter() - first_start

    steady_seconds = _measure(lambda: circuit.value_and_grad(hamiltonian), args.repeats)

    end_to_end_start = time.perf_counter()
    end_to_end_result = _circuit(parameters).value_and_grad(hamiltonian)
    end_to_end_first_seconds = time.perf_counter() - end_to_end_start

    if not np.isfinite(first.value) or not np.isfinite(first.gradient).all():
        raise RuntimeError("native value-and-gradient result is not finite")
    if not np.allclose(first.value, end_to_end_result.value, atol=1e-12, rtol=1e-12):
        raise RuntimeError("native repeated construction changed the energy")
    if not np.allclose(
        first.gradient, end_to_end_result.gradient, atol=1e-12, rtol=1e-12
    ):
        raise RuntimeError("native repeated construction changed the gradient")

    output = {
        "library": "tencirpauli",
        "backend": "native_u1_value_and_grad",
        "nqubits": NQUBITS,
        "particle_number": PARTICLES,
        "layers": LAYERS,
        "parameter_count": PARAMETER_COUNT,
        "dimension": math.comb(NQUBITS, PARTICLES),
        "hamiltonian": "periodic_sum_xx_plus_yy",
        "hamiltonian_terms": 2 * NQUBITS,
        "seed": args.seed,
        "circuit_build_seconds": circuit_build_seconds,
        "first_value_and_grad_seconds": first_value_and_grad_seconds,
        "steady_value_and_grad_seconds_median": steady_seconds,
        "end_to_end_first_seconds": end_to_end_first_seconds,
        "energy": float(first.value),
        "gradient_values": [float(value) for value in first.gradient],
        "gradient_l2_norm": float(np.linalg.norm(first.gradient)),
        "gradient_max_abs": float(np.max(np.abs(first.gradient))),
        "state_bytes": math.comb(NQUBITS, PARTICLES) * np.dtype(np.complex128).itemsize,
        "gate_count": LAYERS * (NQUBITS // 2 + NQUBITS),
        "angle_count": circuit.angle_count,
        "peak_rss_bytes": _peak_rss_bytes(),
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
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
