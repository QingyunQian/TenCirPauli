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
@pytest.mark.parametrize("nqubits", (4, 8, 10))
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
