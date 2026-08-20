"""TensorCircuit state and measurement-only interchange for shadows."""

from __future__ import annotations

import numpy as np
import tensorcircuit as tc

import tencirpauli as tcp


def test_tensorcircuit_state_feeds_local_pauli_shadow_measurements() -> None:
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    circuit = tc.Circuit(1)
    circuit.h(0)
    state = np.asarray(circuit.state(), dtype=np.complex128)
    state /= np.linalg.norm(state)
    snapshots = tcp.Snapshots.sample(
        state, nsettings=192, repeats=1, protocol="pauli", seed=303
    )
    np.testing.assert_allclose(
        snapshots.expectation(tcp.PauliWord.from_string("X")), 0.953125
    )
