"""Numerical and interchange contracts for classical-shadow snapshots."""

from __future__ import annotations

import numpy as np
import pytest

import tencirpauli as tcp


def test_pauli_import_and_readback_are_numeric_and_immutable() -> None:
    bases = np.array([[3], [3]], dtype=np.uint8)
    outcomes = np.zeros((2, 2, 1), dtype=np.uint8)
    snapshots = tcp.Snapshots.from_pauli_measurements(bases, outcomes)
    arrays = snapshots.to_arrays()
    assert isinstance(arrays, tcp.PauliSnapshotArrays)
    assert arrays.bases.dtype == np.uint8
    assert arrays.outcomes.shape == (2, 2, 1)
    assert not arrays.outcomes.flags.writeable
    np.testing.assert_allclose(
        snapshots.expectation(tcp.PauliWord.from_string("Z")), 3.0
    )
    np.testing.assert_allclose(
        snapshots.rdm(), np.diag([2.0, -1.0]).astype(np.complex128)
    )


def test_clifford_frame_import_and_numeric_estimators() -> None:
    snapshots = tcp.Snapshots.from_clifford_measurements(
        np.array([[1]], dtype=np.uint64),
        np.array([[0]], dtype=np.uint64),
        np.array([[1]], dtype=np.int8),
        np.array([[[0]]], dtype=np.uint8),
        nqubits=1,
    )
    arrays = snapshots.to_arrays()
    assert isinstance(arrays, tcp.CliffordSnapshotArrays)
    assert arrays.frame_signs.shape == (1, 1)
    np.testing.assert_allclose(
        snapshots.expectation(tcp.PauliWord.from_string("X")), 3.0
    )
    np.testing.assert_allclose(
        snapshots.rdm(),
        np.array([[0.5, 1.5], [1.5, 0.5]], dtype=np.complex128),
    )


def test_seeded_sampling_roundtrips_and_stabilizer_fidelity() -> None:
    state = tcp.StabilizerState.zero_state(1)
    snapshots = tcp.Snapshots.sample(
        state, nsettings=32, repeats=1, protocol="clifford", seed=11
    )
    arrays = snapshots.to_arrays()
    replay = tcp.Snapshots.from_clifford_measurements(
        arrays.frame_x_words,
        arrays.frame_z_words,
        arrays.frame_signs,
        arrays.outcomes,
        nqubits=1,
    )
    assert replay.nsettings == snapshots.nsettings
    np.testing.assert_array_equal(replay.to_arrays().outcomes, arrays.outcomes)
    assert replay.fidelity(state) == snapshots.fidelity(state)


def test_blocks_and_shadow_bound_contracts() -> None:
    bases = np.full((4, 1), 3, dtype=np.uint8)
    outcomes = np.zeros((4, 1, 1), dtype=np.uint8)
    snapshots = tcp.Snapshots.from_pauli_measurements(bases, outcomes)
    blocks = snapshots.expectation(
        tcp.PauliWord.from_string("Z"), blocks=2, return_blocks=True
    )
    np.testing.assert_allclose(blocks, [3.0, 3.0])
    nsettings, block_count = tcp.shadow_bound(
        np.array([[3], [1]], dtype=np.uint8), epsilon=0.5
    )
    assert block_count > 0 and nsettings % block_count == 0


def test_invalid_shadow_inputs_fail_at_the_boundary() -> None:
    with pytest.raises(TypeError):
        tcp.Snapshots.from_pauli_measurements(
            np.array([[1.0]], dtype=np.float64), np.zeros((1, 1, 1), dtype=np.uint8)
        )
    with pytest.raises(ValueError):
        tcp.Snapshots.sample(
            np.array([1.0, 0.0], dtype=np.float64),
            nsettings=1,
            protocol="clifford",
            seed=-1,
        )
    with pytest.raises(ValueError):
        tcp.Snapshots.from_clifford_measurements(
            np.array([[1]], dtype=np.uint64),
            np.array([[0]], dtype=np.uint64),
            np.array([[2]], dtype=np.int8),
            np.zeros((1, 1, 1), dtype=np.uint8),
            nqubits=1,
        )
