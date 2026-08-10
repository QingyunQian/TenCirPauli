"""TenCirPauli side of the Pauli-propagation Julia VQE comparison."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Sequence

import numpy as np

import tencirpauli as tcp
from tencirpauli import advanced


def _parameters(nqubits: int, layers: int) -> np.ndarray:
    return np.linspace(-0.17, 0.23, nqubits * layers, dtype=np.float64)


def _brickwork_pairs(nqubits: int, layer: int) -> list[tuple[int, int]]:
    start = layer % 2
    return [(wire, wire + 1) for wire in range(start, nqubits - 1, 2)]


def _pauli_string(nqubits: int, updates: Sequence[tuple[int, str]]) -> str:
    word = ["I"] * nqubits
    for wire, code in updates:
        word[wire] = code
    return "".join(word)


def _observable(nqubits: int) -> tcp.PauliOperator:
    terms: list[tuple[str, float]] = []
    for wire in range(nqubits):
        terms.append((_pauli_string(nqubits, [(wire, "X")]), -0.5))
    for wire in range(nqubits - 1):
        terms.append((_pauli_string(nqubits, [(wire, "Z"), (wire + 1, "Z")]), -1.0))
    return tcp.PauliOperator.from_terms(nqubits, terms)


def _engine(
    nqubits: int, layers: int, max_weight: int | None
) -> advanced.PropagationEngine:
    tape = advanced.GateTape(nqubits)
    parameter = 0
    for layer in range(layers):
        for wire in range(nqubits):
            tape.ry(wire, parameter=parameter)
            parameter += 1
        for wire0, wire1 in _brickwork_pairs(nqubits, layer):
            tape.cnot(wire0, wire1)
    return advanced.PropagationEngine(
        tape,
        _observable(nqubits),
        max_weight=max_weight,
    )


PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _kron_word(word: str) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for code in word:
        result = np.kron(result, PAULI_MATRICES[code])
    return result


def _single_qubit_gate(nqubits: int, wire: int, matrix: np.ndarray) -> np.ndarray:
    factors = [PAULI_MATRICES["I"] for _ in range(nqubits)]
    factors[wire] = matrix
    result = np.array([[1.0 + 0.0j]])
    for factor in factors:
        result = np.kron(result, factor)
    return result


def _cnot_gate(nqubits: int, control: int, target: int) -> np.ndarray:
    dimension = 1 << nqubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for source in range(dimension):
        control_bit = (source >> (nqubits - 1 - control)) & 1
        destination = source
        if control_bit:
            destination ^= 1 << (nqubits - 1 - target)
        matrix[destination, source] = 1.0
    return matrix


def _dense_energy(nqubits: int, layers: int, parameters: np.ndarray) -> float:
    hamiltonian = np.zeros((1 << nqubits, 1 << nqubits), dtype=np.complex128)
    for wire in range(nqubits):
        hamiltonian += -0.5 * _kron_word(_pauli_string(nqubits, [(wire, "X")]))
    for wire in range(nqubits - 1):
        hamiltonian += -1.0 * _kron_word(
            _pauli_string(nqubits, [(wire, "Z"), (wire + 1, "Z")])
        )

    state = np.zeros(1 << nqubits, dtype=np.complex128)
    state[0] = 1.0
    parameter = 0
    for layer in range(layers):
        for wire in range(nqubits):
            angle = parameters[parameter]
            parameter += 1
            rotation = (
                math.cos(angle / 2.0) * PAULI_MATRICES["I"]
                - 1j * math.sin(angle / 2.0) * PAULI_MATRICES["Y"]
            )
            state = _single_qubit_gate(nqubits, wire, rotation) @ state
        for wire0, wire1 in _brickwork_pairs(nqubits, layer):
            state = _cnot_gate(nqubits, wire0, wire1) @ state
    return float(np.real(np.vdot(state, hamiltonian @ state)))


def _dense_value_and_grad(
    nqubits: int, layers: int, parameters: np.ndarray
) -> tuple[float, np.ndarray]:
    value = _dense_energy(nqubits, layers, parameters)
    gradient = np.empty_like(parameters)
    shift = math.pi / 2.0
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += shift
        minus[index] -= shift
        gradient[index] = 0.5 * (
            _dense_energy(nqubits, layers, plus) - _dense_energy(nqubits, layers, minus)
        )
    return value, gradient


def _call(
    engine: advanced.PropagationEngine, parameters: np.ndarray
) -> tuple[float, np.ndarray]:
    result = engine.value_and_grad(parameters)
    return result.value, np.asarray(result.gradient, dtype=np.float64)


def _finite_difference(
    engine: advanced.PropagationEngine, parameters: np.ndarray, step: float = 1.0e-6
) -> np.ndarray:
    gradient = np.empty_like(parameters)
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            0.5 * (engine.expectation(plus) - engine.expectation(minus)) / step
        )
    return gradient


def run_correctness() -> dict[str, Any]:
    nqubits, layers, max_weight = 8, 2, 8
    parameters = _parameters(nqubits, layers)
    engine = _engine(nqubits, layers, max_weight)
    native_value, native_gradient = _call(engine, parameters)
    dense_value, dense_gradient = _dense_value_and_grad(nqubits, layers, parameters)
    truncated_engine = _engine(nqubits, layers, 5)
    truncated_value, truncated_gradient = _call(truncated_engine, parameters)
    truncated_fd_gradient = _finite_difference(truncated_engine, parameters)
    return {
        "nqubits": nqubits,
        "layers": layers,
        "max_weight": max_weight,
        "nparameters": int(parameters.size),
        "native_value": native_value,
        "dense_value": dense_value,
        "native_dense_value_abs_error": abs(native_value - dense_value),
        "native_dense_gradient_max_abs_error": float(
            np.max(np.abs(native_gradient - dense_gradient))
        ),
        "truncated_max_weight": 5,
        "truncated_value": truncated_value,
        "truncated_gradient": truncated_gradient.tolist(),
        "truncated_fd_gradient": truncated_fd_gradient.tolist(),
        "truncated_fd_gradient_max_abs_error": float(
            np.max(np.abs(truncated_gradient - truncated_fd_gradient))
        ),
        "parameters": parameters.tolist(),
        "native_gradient": native_gradient.tolist(),
        "dense_gradient": dense_gradient.tolist(),
    }


def run_benchmark(
    nqubits: int,
    layers: int,
    max_weight: int,
    repeats: int,
    iterations: int,
    learning_rate: float,
) -> dict[str, Any]:
    parameters = _parameters(nqubits, layers)
    setup_start = time.perf_counter()
    engine = _engine(nqubits, layers, max_weight)
    setup_seconds = time.perf_counter() - setup_start

    first_start = time.perf_counter()
    first_value, first_gradient = _call(engine, parameters)
    first_seconds = time.perf_counter() - first_start

    steady_seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        value, gradient = _call(engine, parameters)
        steady_seconds.append(time.perf_counter() - start)
    steady_parameters = parameters.copy()
    vqe_start = time.perf_counter()
    for _ in range(iterations):
        value, gradient = _call(engine, steady_parameters)
        steady_parameters -= learning_rate * gradient
    vqe_seconds = time.perf_counter() - vqe_start

    return {
        "implementation": "TenCirPauli",
        "nqubits": nqubits,
        "layers": layers,
        "max_weight": max_weight,
        "nparameters": int(parameters.size),
        "term_count": engine.term_count,
        "first_value_and_grad_seconds": first_seconds,
        "steady_value_and_grad_seconds_median": float(np.median(steady_seconds)),
        "steady_value": value,
        "first_value": first_value,
        "first_gradient": first_gradient.tolist(),
        "first_gradient_l2": float(np.linalg.norm(first_gradient)),
        "vqe_iterations": iterations,
        "vqe_seconds": vqe_seconds,
        "vqe_final_value": float(value),
        "setup_seconds": setup_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("correctness", "benchmark"), default="benchmark"
    )
    parser.add_argument("--nqubits", type=int, default=28)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--max-weight", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    if args.mode == "correctness":
        result = run_correctness()
    else:
        result = run_benchmark(
            args.nqubits,
            args.layers,
            args.max_weight,
            args.repeats,
            args.iterations,
            args.learning_rate,
        )
    print("RESULT_JSON=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
