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
