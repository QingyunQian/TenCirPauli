"""Compare Pauli shadows and report standalone Clifford-shadow timings."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import tencirpauli as tcp


MAX_BENCHMARK_BYTES = 16 * 1024**3


def _non_clifford_state(nqubits: int) -> np.ndarray[Any, Any]:
    """Return ``|0...0>`` with one deterministic non-Clifford rotation."""
    theta = math.pi / 7.0
    phase = math.pi / 11.0
    state = np.zeros(1 << nqubits, dtype=np.complex128)
    state[0] = math.cos(theta)
    state[1 << (nqubits - 1)] = np.exp(1j * phase) * math.sin(theta)
    return np.ascontiguousarray(state, dtype=np.complex128)


def _dense_non_clifford_target(nqubits: int, seed: int) -> np.ndarray[Any, Any]:
    """Return a deterministic dense non-stabilizer target statevector."""
    rng = np.random.default_rng(seed)
    target = rng.normal(size=1 << nqubits) + 1j * rng.normal(size=1 << nqubits)
    target = target.astype(np.complex128, copy=False)
    target /= np.linalg.norm(target)
    return np.ascontiguousarray(target, dtype=np.complex128)


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _benchmark(
    function: Callable[[], Any], repetitions: int
) -> Tuple[Dict[str, Any], Any]:
    """Warm up once and return steady-state timing statistics plus the last result."""
    warmup_start = time.perf_counter()
    function()
    warmup_seconds = time.perf_counter() - warmup_start
    durations = []
    result: Any = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    values = np.asarray(durations, dtype=np.float64)
    return (
        {
            "median_seconds": float(np.median(values)),
            "minimum_seconds": float(np.min(values)),
            "maximum_seconds": float(np.max(values)),
            "repetitions": repetitions,
            "warmup_seconds": warmup_seconds,
        },
        result,
    )


def _per_shot(timing: Dict[str, Any], shots: int) -> Dict[str, Any]:
    result = dict(timing)
    result["microseconds_per_shot"] = 1.0e6 * timing["median_seconds"] / shots
    return result


def _load_tensorcircuit(backend_name: str) -> Tuple[Any, Any]:
    """Load TensorCircuit after selecting its backend in this fresh process."""
    import tensorcircuit as tc

    tc.set_backend(backend_name)
    import tensorcircuit.shadows as tc_shadows

    return tc, tc_shadows


def _pauli_codes(nqubits: int) -> Tuple[int, ...]:
    support = min(2, nqubits)
    return tuple(3 if index < support else 0 for index in range(nqubits))


def _run_pauli_comparison(
    state: np.ndarray[Any, Any],
    nqubits: int,
    nsettings: int,
    repeats: int,
    repetitions: int,
    seed: int,
    tensorcircuit_backend: str,
    tensorcircuit_batch_size: Optional[int],
) -> Dict[str, Any]:
    codes = _pauli_codes(nqubits)
    observable = tcp.PauliWord.from_codes(codes)
    shots = nsettings * repeats

    native_sampling, native_snapshots = _benchmark(
        lambda: tcp.Snapshots.sample(
            state,
            nsettings=nsettings,
            repeats=repeats,
            protocol="pauli",
            seed=seed,
            max_bytes=MAX_BENCHMARK_BYTES,
        ),
        repetitions,
    )
    native_estimation, native_estimate = _benchmark(
        lambda: native_snapshots.expectation(observable), repetitions
    )
    native_arrays = native_snapshots.to_arrays()
    native_rdm = native_snapshots.rdm(subsystem_to_keep=list(range(min(2, nqubits))))

    tc, tc_shadows = _load_tensorcircuit(tensorcircuit_backend)
    settings_rng = np.random.default_rng(seed ^ 0x51A0)
    bases = settings_rng.integers(1, 4, size=(nsettings, nqubits), dtype=np.int32)
    statuses = settings_rng.random((nsettings, repeats), dtype=np.float32)

    batch_size = tensorcircuit_batch_size
    if batch_size is None:
        batch_size = min(64, nsettings) if tensorcircuit_backend == "jax" else nsettings

    def tensorcircuit_sample_batch(
        input_state: Any, batch_bases: Any, batch_statuses: Any
    ) -> Any:
        return tc_shadows.shadow_snapshots(
            input_state,
            batch_bases,
            status=batch_statuses,
            measurement_only=True,
        )

    def tensorcircuit_estimate_batch(batch_outcomes: Any, batch_bases: Any) -> Any:
        return tc_shadows.expectation_ps_shadow(
            batch_outcomes,
            pauli_strings=batch_bases,
            ps=codes,
        )[0]

    if tensorcircuit_backend == "jax":
        tensorcircuit_sample_batch = tc.backend.jit(tensorcircuit_sample_batch)
        tensorcircuit_estimate_batch = tc.backend.jit(tensorcircuit_estimate_batch)

    def tensorcircuit_sample() -> np.ndarray[Any, Any]:
        chunks = []
        for start in range(0, nsettings, batch_size):
            stop = min(start + batch_size, nsettings)
            samples = tensorcircuit_sample_batch(
                state,
                bases[start:stop],
                statuses[start:stop],
            )
            chunks.append(np.asarray(samples))
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    tc_sampling, tc_outcomes = _benchmark(tensorcircuit_sample, repetitions)
    tc_estimation, tc_estimate = _benchmark(
        lambda: float(np.asarray(tensorcircuit_estimate_batch(tc_outcomes, bases))),
        repetitions,
    )

    imported = tcp.Snapshots.from_pauli_measurements(bases, tc_outcomes)
    imported_estimation, imported_estimate = _benchmark(
        lambda: imported.expectation(observable), repetitions
    )
    bound_settings, bound_blocks = tcp.shadow_bound(observable, epsilon=0.25)
    true_expectation = math.cos(2.0 * math.pi / 7.0)
    pauli_standard_error = (
        math.sqrt((9.0 - true_expectation**2) / nsettings) if repeats == 1 else None
    )
    return {
        "observable_codes": list(codes),
        "exact_input_is_non_clifford": True,
        "true_expectation": true_expectation,
        "one_repeat_normal_standard_error": pauli_standard_error,
        "one_repeat_normal_95_percent_half_width": (
            None if pauli_standard_error is None else 1.96 * pauli_standard_error
        ),
        "shadow_bound_epsilon_0_25": {
            "settings": bound_settings,
            "blocks": bound_blocks,
        },
        "tencirpauli": {
            "sampling": _per_shot(native_sampling, shots),
            "estimation": native_estimation,
            "estimate": float(native_estimate),
            "imported_tensorcircuit_estimation": imported_estimation,
            "imported_tensorcircuit_estimate": float(imported_estimate),
            "snapshot_shape": list(native_arrays.outcomes.shape),
            "rdm_shape": list(native_rdm.shape),
        },
        "tensorcircuit": {
            "backend": tensorcircuit_backend,
            "batch_size": batch_size,
            "jit": tensorcircuit_backend == "jax",
            "sampling": _per_shot(tc_sampling, shots),
            "estimation": tc_estimation,
            "estimate": float(tc_estimate),
            "outcome_shape": list(tc_outcomes.shape),
        },
        "speedup_ratios": {
            "tensorcircuit_over_tencirpauli_sampling": tc_sampling["median_seconds"]
            / native_sampling["median_seconds"],
            "tensorcircuit_over_tencirpauli_estimation_on_own_data": tc_estimation[
                "median_seconds"
            ]
            / native_estimation["median_seconds"],
            "tensorcircuit_over_tencirpauli_estimation_on_same_data": tc_estimation[
                "median_seconds"
            ]
            / imported_estimation["median_seconds"],
        },
    }


def _run_clifford_timing(
    state: np.ndarray[Any, Any],
    dense_target_statevector: np.ndarray[Any, Any],
    nqubits: int,
    nsettings: int,
    repeats: int,
    repetitions: int,
    seed: int,
    stabilizer_nqubits: int,
    stabilizer_settings: int,
) -> Dict[str, Any]:
    dense_target = tcp.StabilizerState.zero_state(nqubits)
    shots = nsettings * repeats
    dense_sampling, dense_snapshots = _benchmark(
        lambda: tcp.Snapshots.sample(
            state,
            nsettings=nsettings,
            repeats=repeats,
            protocol="clifford",
            seed=seed ^ 0xC11F,
            max_bytes=MAX_BENCHMARK_BYTES,
        ),
        repetitions,
    )
    dense_estimation, dense_estimate = _benchmark(
        lambda: dense_snapshots.fidelity(dense_target), repetitions
    )
    dense_target_estimation, dense_target_estimate = _benchmark(
        lambda: dense_snapshots.fidelity_statevector(dense_target_statevector),
        repetitions,
    )
    dense_arrays = dense_snapshots.to_arrays()
    true_stabilizer_fidelity = float(abs(state[0]) ** 2)
    true_dense_target_fidelity = float(
        abs(np.vdot(dense_target_statevector, state)) ** 2
    )
    dimension = float(1 << nqubits)
    fidelity_variance = (
        dimension * (1.0 + 2.0 * true_stabilizer_fidelity) / (dimension + 2.0)
        - true_stabilizer_fidelity**2
    )
    fidelity_standard_error = (
        math.sqrt(fidelity_variance / nsettings) if repeats == 1 else None
    )
    dense_target_variance = (
        dimension * (1.0 + 2.0 * true_dense_target_fidelity) / (dimension + 2.0)
        - true_dense_target_fidelity**2
    )
    dense_target_standard_error = (
        math.sqrt(dense_target_variance / nsettings) if repeats == 1 else None
    )

    stabilizer = tcp.StabilizerState.zero_state(stabilizer_nqubits)
    stabilizer_sampling, stabilizer_snapshots = _benchmark(
        lambda: tcp.Snapshots.sample(
            stabilizer,
            nsettings=stabilizer_settings,
            repeats=1,
            protocol="clifford",
            seed=seed ^ 0x57AB,
            max_bytes=MAX_BENCHMARK_BYTES,
        ),
        repetitions,
    )
    stabilizer_target = tcp.StabilizerState.zero_state(stabilizer_nqubits)
    fidelity_timing, fidelity = _benchmark(
        lambda: stabilizer_snapshots.fidelity(stabilizer_target), repetitions
    )
    return {
        "dense_non_clifford": {
            "estimator": "StabilizerState.fidelity",
            "target": "zero_state",
            "true_fidelity": true_stabilizer_fidelity,
            "one_repeat_normal_standard_error": fidelity_standard_error,
            "one_repeat_normal_95_percent_half_width": (
                None
                if fidelity_standard_error is None
                else 1.96 * fidelity_standard_error
            ),
            "nqubits": nqubits,
            "nsettings": nsettings,
            "repeats": repeats,
            "sampling": _per_shot(dense_sampling, shots),
            "estimation": dense_estimation,
            "estimate": float(dense_estimate),
            "frame_shape": list(dense_arrays.frame_x_words.shape),
        },
        "dense_non_clifford_target": {
            "estimator": "Snapshots.fidelity_statevector",
            "target": "dense deterministic non-Clifford statevector",
            "true_fidelity": true_dense_target_fidelity,
            "one_repeat_normal_standard_error": dense_target_standard_error,
            "one_repeat_normal_95_percent_half_width": (
                None
                if dense_target_standard_error is None
                else 1.96 * dense_target_standard_error
            ),
            "nqubits": nqubits,
            "nsettings": nsettings,
            "repeats": repeats,
            "estimation": dense_target_estimation,
            "estimate": float(dense_target_estimate),
        },
        "stabilizer_input": {
            "estimator": "StabilizerState.fidelity",
            "nqubits": stabilizer_nqubits,
            "nsettings": stabilizer_settings,
            "sampling": _per_shot(stabilizer_sampling, stabilizer_settings),
            "fidelity_estimation": fidelity_timing,
            "fidelity_estimate": float(fidelity),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nqubits", type=int, default=20)
    parser.add_argument("--nsettings", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--stabilizer-nqubits", type=int, default=64)
    parser.add_argument("--stabilizer-settings", type=int, default=128)
    parser.add_argument(
        "--tensorcircuit-backend",
        choices=("numpy", "jax", "tensorflow", "pytorch"),
        default="jax",
    )
    parser.add_argument(
        "--tensorcircuit-batch-size",
        type=int,
        default=None,
        help="compile TensorCircuit shadow sampling in fixed-size batches",
    )
    parser.add_argument("--skip-tensorcircuit", action="store_true")
    args = parser.parse_args()
    _positive(args.nqubits, "nqubits")
    _positive(args.nsettings, "nsettings")
    _positive(args.repeats, "repeats")
    _positive(args.repetitions, "repetitions")
    _positive(args.stabilizer_nqubits, "stabilizer-nqubits")
    _positive(args.stabilizer_settings, "stabilizer-settings")
    if args.tensorcircuit_batch_size is not None:
        _positive(args.tensorcircuit_batch_size, "tensorcircuit-batch-size")

    state = _non_clifford_state(args.nqubits)
    dense_target_statevector = _dense_non_clifford_target(
        args.nqubits, args.seed ^ 0xD3E5E
    )
    result: Dict[str, Any] = {
        "workload": {
            "nqubits": args.nqubits,
            "nsettings": args.nsettings,
            "repeats": args.repeats,
            "shots": args.nsettings * args.repeats,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "state": "non-Clifford single-qubit rotation of zero state",
        },
        "tencirpauli_clifford_timing": _run_clifford_timing(
            state,
            dense_target_statevector,
            args.nqubits,
            args.nsettings,
            args.repeats,
            args.repetitions,
            args.seed,
            args.stabilizer_nqubits,
            args.stabilizer_settings,
        ),
    }
    if not args.skip_tensorcircuit:
        result["pauli_comparison"] = _run_pauli_comparison(
            state,
            args.nqubits,
            args.nsettings,
            args.repeats,
            args.repetitions,
            args.seed,
            args.tensorcircuit_backend,
            args.tensorcircuit_batch_size,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
