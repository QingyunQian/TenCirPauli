"""TensorCircuit/JAX U(1) VQE value-and-gradient comparison workload."""

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


def _pauli_hamiltonian() -> tuple[tuple[tuple[int, ...], ...], np.ndarray[Any, Any]]:
    """Return the periodic nearest-neighbour XX+YY Pauli sum."""
    words: list[tuple[int, ...]] = []
    for index in range(NQUBITS):
        neighbour = (index + 1) % NQUBITS
        xx = [0] * NQUBITS
        yy = [0] * NQUBITS
        xx[index] = xx[neighbour] = 1
        yy[index] = yy[neighbour] = 2
        words.append(tuple(xx))
        words.append(tuple(yy))
    return tuple(words), np.ones(2 * NQUBITS, dtype=np.float64)


def _synchronize(jax: Any, result: Any) -> Any:
    """Block until a scalar and gradient leaves are available on the host."""
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


def _measure(jax: Any, function: Any, repeats: int) -> float:
    """Return the synchronized median of repeated JAX calls."""
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _synchronize(jax, function())
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    """Validate and measure TensorCircuit's JAX-jitted U(1) path."""
    args = _parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    import jax
    import tensorcircuit as tc

    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    import jax.numpy as jnp

    parameters = _angle_vector(args.seed)
    words, coefficients = _pauli_hamiltonian()
    jax_parameters = jnp.asarray(parameters, dtype=jnp.float64)

    even_pairs = _pairs(0)
    odd_pairs = _pairs(1)

    def apply_layer(
        state: Any, layer_values: Any, pairs: tuple[tuple[int, int], ...]
    ) -> Any:
        circuit = tc.U1Circuit(NQUBITS, k=PARTICLES, filled=(0, 1), inputs=state)
        cursor = 0
        for left, right in pairs:
            circuit.iswap(left, right, theta=layer_values[cursor])
            cursor += 1
        for qubit in range(NQUBITS):
            circuit.rz(qubit, theta=layer_values[cursor])
            cursor += 1
        return circuit.state()

    def scan_body(state: Any, inputs: tuple[Any, Any]) -> tuple[Any, Any]:
        layer_values, parity = inputs
        next_state = jax.lax.cond(
            parity == 0,
            lambda value: apply_layer(value, layer_values, even_pairs),
            lambda value: apply_layer(value, layer_values, odd_pairs),
            state,
        )
        return next_state, jnp.zeros((), dtype=jnp.float64)

    def objective(values: Any) -> Any:
        initial = tc.U1Circuit(NQUBITS, k=PARTICLES, filled=(0, 1)).state()
        layer_values = values.reshape((LAYERS, PARAMETERS_PER_LAYER))
        parity = jnp.arange(LAYERS, dtype=jnp.int32) % 2
        final_state, _ = jax.lax.scan(scan_body, initial, (layer_values, parity))
        circuit = tc.U1Circuit(NQUBITS, k=PARTICLES, filled=(0, 1), inputs=final_state)
        return tc.backend.real(circuit.expectation_pss(words, coefficients))

    setup_start = time.perf_counter()
    runner = jax.jit(jax.value_and_grad(objective))
    setup_seconds = time.perf_counter() - setup_start

    first_start = time.perf_counter()
    first = _synchronize(jax, runner(jax_parameters))
    first_value_and_grad_seconds = time.perf_counter() - first_start

    steady_seconds = _measure(jax, lambda: runner(jax_parameters), args.repeats)

    if (
        not np.isfinite(np.asarray(first[0]))
        or not np.isfinite(np.asarray(first[1])).all()
    ):
        raise RuntimeError("TensorCircuit value-and-gradient result is not finite")

    output = {
        "library": "tensorcircuit",
        "backend": "jax_cpu_jit_value_and_grad",
        "nqubits": NQUBITS,
        "particle_number": PARTICLES,
        "layers": LAYERS,
        "parameter_count": PARAMETER_COUNT,
        "dimension": math.comb(NQUBITS, PARTICLES),
        "hamiltonian": "periodic_sum_xx_plus_yy",
        "hamiltonian_terms": 2 * NQUBITS,
        "seed": args.seed,
        "jax_version": jax.__version__,
        "tensorcircuit_backend": "jax",
        "jit_setup_seconds": setup_seconds,
        "first_value_and_grad_seconds": first_value_and_grad_seconds,
        "steady_value_and_grad_seconds_median": steady_seconds,
        "end_to_end_first_seconds": first_value_and_grad_seconds,
        "energy": float(np.asarray(first[0])),
        "gradient_values": [float(value) for value in np.asarray(first[1])],
        "gradient_l2_norm": float(np.linalg.norm(np.asarray(first[1]))),
        "gradient_max_abs": float(np.max(np.abs(np.asarray(first[1])))),
        "state_bytes": math.comb(NQUBITS, PARTICLES) * np.dtype(np.complex128).itemsize,
        "gate_count": LAYERS * (NQUBITS // 2 + NQUBITS),
        "angle_count": PARAMETER_COUNT,
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
