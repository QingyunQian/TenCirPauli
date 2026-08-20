"""End-to-end release benchmarks for native classical-shadow snapshots."""

from __future__ import annotations

import os

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


def _state(nqubits: int) -> np.ndarray[object, object]:
    values = np.zeros(1 << nqubits, dtype=np.complex128)
    values[0] = 1.0
    return values


def _dense_target(nqubits: int) -> np.ndarray[object, object]:
    rng = np.random.default_rng(20260820 + nqubits)
    values = rng.normal(size=1 << nqubits) + 1j * rng.normal(size=1 << nqubits)
    values = values.astype(np.complex128, copy=False)
    values /= np.linalg.norm(values)
    return np.ascontiguousarray(values, dtype=np.complex128)


@pytest.mark.parametrize("protocol", ("pauli", "clifford"))
@pytest.mark.parametrize("nqubits", (4, 8, 10, 16))
def test_dense_sampling(
    benchmark: BenchmarkFixture, protocol: str, nqubits: int
) -> None:
    state = _state(nqubits)
    expected = tcp.Snapshots.sample(
        state, nsettings=16, repeats=1, protocol=protocol, seed=17
    )
    result = benchmark(
        tcp.Snapshots.sample,
        state,
        nsettings=16,
        repeats=1,
        protocol=protocol,
        seed=17,
    )
    assert result.to_arrays().outcomes.shape == expected.to_arrays().outcomes.shape
    benchmark.extra_info.update(
        {
            "protocol": protocol,
            "nqubits": nqubits,
            "nsettings": 16,
            "repeats": 1,
            "state_bytes": state.nbytes,
            "thread_count": int(os.environ.get("RAYON_NUM_THREADS", "1")),
        }
    )


@pytest.mark.parametrize("nqubits", (8, 32, 64))
def test_stabilizer_sampling_and_estimation(
    benchmark: BenchmarkFixture, nqubits: int
) -> None:
    state = tcp.StabilizerState.zero_state(nqubits)
    observable = tcp.PauliWord.from_codes([3] * nqubits)
    snapshots = tcp.Snapshots.sample(
        state, nsettings=128, repeats=1, protocol="clifford", seed=23
    )
    expected = snapshots.expectation(observable)
    result = benchmark(snapshots.expectation, observable)
    assert result == expected
    benchmark.extra_info.update(
        {
            "protocol": "clifford",
            "nqubits": nqubits,
            "nsettings": snapshots.nsettings,
            "repeats": snapshots.repeats,
            "thread_count": int(os.environ.get("RAYON_NUM_THREADS", "1")),
        }
    )


@pytest.mark.parametrize(
    "nqubits, nsettings, repeats", ((64, 16, 1), (256, 8, 1), (64, 8, 8))
)
def test_stabilizer_sampling(
    benchmark: BenchmarkFixture, nqubits: int, nsettings: int, repeats: int
) -> None:
    state = tcp.StabilizerState.zero_state(nqubits)
    expected = tcp.Snapshots.sample(
        state, nsettings=nsettings, repeats=repeats, protocol="clifford", seed=23
    )
    result = benchmark(
        tcp.Snapshots.sample,
        state,
        nsettings=nsettings,
        repeats=repeats,
        protocol="clifford",
        seed=23,
    )
    expected_arrays = expected.to_arrays()
    result_arrays = result.to_arrays()
    np.testing.assert_array_equal(result_arrays.outcomes, expected_arrays.outcomes)
    np.testing.assert_array_equal(
        result_arrays.frame_x_words, expected_arrays.frame_x_words
    )
    benchmark.extra_info.update(
        {
            "protocol": "clifford",
            "estimator": "sampling",
            "nqubits": nqubits,
            "nsettings": nsettings,
            "repeats": repeats,
            "thread_count": int(os.environ.get("RAYON_NUM_THREADS", "1")),
        }
    )


@pytest.mark.parametrize("nqubits", (64, 256))
def test_pauli_stabilizer_sampling(benchmark: BenchmarkFixture, nqubits: int) -> None:
    state = tcp.StabilizerState.zero_state(nqubits)
    expected = tcp.Snapshots.sample(
        state, nsettings=4, repeats=1, protocol="pauli", seed=29
    )
    result = benchmark(
        tcp.Snapshots.sample,
        state,
        nsettings=4,
        repeats=1,
        protocol="pauli",
        seed=29,
    )
    expected_arrays = expected.to_arrays()
    result_arrays = result.to_arrays()
    np.testing.assert_array_equal(result_arrays.bases, expected_arrays.bases)
    np.testing.assert_array_equal(result_arrays.outcomes, expected_arrays.outcomes)
    benchmark.extra_info.update(
        {
            "protocol": "pauli",
            "estimator": "sampling",
            "nqubits": nqubits,
            "nsettings": result.nsettings,
            "repeats": result.repeats,
            "thread_count": int(os.environ.get("RAYON_NUM_THREADS", "1")),
        }
    )


def _random_observables(nqubits: int, count: int) -> list[tcp.PauliWord]:
    rng = np.random.default_rng(20260820)
    return [
        tcp.PauliWord.from_codes(
            rng.integers(0, 4, size=nqubits, dtype=np.uint8).tolist()
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize("term_count", (1, 16, 128, 512))
def test_clifford_estimate_many(benchmark: BenchmarkFixture, term_count: int) -> None:
    snapshots = tcp.Snapshots.sample(
        tcp.StabilizerState.zero_state(12),
        nsettings=64,
        repeats=1,
        protocol="clifford",
        seed=102,
    )
    observables = _random_observables(12, term_count)
    expected = snapshots.estimate_many(observables)
    result = benchmark(snapshots.estimate_many, observables)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "protocol": "clifford",
            "estimator": "estimate_many",
            "nqubits": 12,
            "nsettings": snapshots.nsettings,
            "repeats": snapshots.repeats,
            "term_count": term_count,
        }
    )


@pytest.mark.parametrize("subsystem_size", (4, 8, 10))
def test_clifford_rdm(benchmark: BenchmarkFixture, subsystem_size: int) -> None:
    snapshots = tcp.Snapshots.sample(
        tcp.StabilizerState.zero_state(12),
        nsettings=16,
        repeats=1,
        protocol="clifford",
        seed=113,
    )
    keep = list(range(subsystem_size))
    expected = snapshots.rdm(subsystem_to_keep=keep)
    result = benchmark(snapshots.rdm, subsystem_to_keep=keep)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "protocol": "clifford",
            "estimator": "rdm",
            "nqubits": 12,
            "subsystem_size": subsystem_size,
            "nsettings": snapshots.nsettings,
            "repeats": snapshots.repeats,
        }
    )


@pytest.mark.parametrize("method", ("shadow", "randomized_measurement"))
def test_renyi_entropy(benchmark: BenchmarkFixture, method: str) -> None:
    repeats = 2 if method == "randomized_measurement" else 1
    protocol = "pauli" if method == "randomized_measurement" else "clifford"
    snapshots = tcp.Snapshots.sample(
        _state(8),
        nsettings=16,
        repeats=repeats,
        protocol=protocol,
        seed=127,
    )
    expected = snapshots.renyi_entropy(subsystem_to_keep=list(range(4)), method=method)
    result = benchmark(
        snapshots.renyi_entropy,
        subsystem_to_keep=list(range(4)),
        method=method,
    )
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "estimator": "renyi_entropy",
            "method": method,
            "nqubits": 8,
            "subsystem_size": 4,
            "nsettings": snapshots.nsettings,
            "repeats": snapshots.repeats,
        }
    )


def test_stabilizer_fidelity(benchmark: BenchmarkFixture) -> None:
    target = tcp.StabilizerState.zero_state(64)
    snapshots = tcp.Snapshots.sample(
        target, nsettings=16, repeats=1, protocol="clifford", seed=131
    )
    expected = snapshots.fidelity(target)
    result = benchmark(snapshots.fidelity, target)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "estimator": "stabilizer_fidelity",
            "nqubits": 64,
            "nsettings": snapshots.nsettings,
            "repeats": snapshots.repeats,
        }
    )


@pytest.mark.parametrize("nqubits", (8, 10, 12))
def test_dense_target_fidelity(benchmark: BenchmarkFixture, nqubits: int) -> None:
    state = _state(nqubits)
    target = _dense_target(nqubits)
    snapshots = tcp.Snapshots.sample(
        state, nsettings=64, repeats=1, protocol="clifford", seed=29
    )
    expected = snapshots.fidelity_statevector(target)
    result = benchmark(snapshots.fidelity_statevector, target)
    np.testing.assert_allclose(result, expected)
    benchmark.extra_info.update(
        {
            "protocol": "clifford",
            "estimator": "dense_target_fidelity",
            "nqubits": nqubits,
            "nsettings": snapshots.nsettings,
            "repeats": snapshots.repeats,
            "target_bytes": target.nbytes,
            "thread_count": int(os.environ.get("RAYON_NUM_THREADS", "1")),
        }
    )
