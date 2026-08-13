"""Small Jaynes--Cummings Hamiltonian in a fixed-excitation sector."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def main() -> None:
    space = tcp.OperatorSpace(bosons=1, qubits=1)
    boson = space.boson
    qubit = space.qubit

    # n_qubit = (I - Z) / 2.
    number_builder = space.builder()
    number_builder.add_product(0.5, qubits=((0, "I"),))
    number_builder.add_product(-0.5, qubits=((0, "Z"),))
    qubit_number = number_builder.finish()

    boson_number = boson.create(0) * boson.annihilate(0)
    qubit_down = 0.5 * (qubit.x(0) + 1j * qubit.y(0))
    qubit_up = 0.5 * (qubit.x(0) - 1j * qubit.y(0))
    hamiltonian = (
        0.7 * boson_number
        + 1.1 * qubit_number
        + 0.3 * (boson.create(0) * qubit_down + boson.annihilate(0) * qubit_up)
    )

    excitation = tcp.AdditiveCharge(
        space, name="total_excitation", bosons={0: 1}, qubits={0: (0, 1)}
    )
    assert hamiltonian.is_hermitian()
    assert hamiltonian.conserves(excitation)

    sector = tcp.ChargeSector(((excitation, 1),), boson_cutoffs={0: 2})
    restricted = hamiltonian.restrict_charge(sector)
    expected = np.asarray([[1.1, 0.3], [0.3, 0.7]], dtype=np.complex128)
    np.testing.assert_allclose(restricted.dense(), expected)

    state = np.asarray([1.0 + 0j, 2.0 + 0j])
    action = restricted.mvp_plan().apply(state)
    np.testing.assert_allclose(action, expected @ state)
    print("Jaynes--Cummings excitation-1 sector:")
    print("basis:", sector.basis_states().tolist())
    print("dense block:\n", restricted.dense())


if __name__ == "__main__":
    main()
