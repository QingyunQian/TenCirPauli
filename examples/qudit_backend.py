"""Small one-qutrit Weyl Hamiltonian and backend-MVP example."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def main() -> None:
    # This is a one-qutrit (d=3) Hamiltonian.  In the Weyl basis, X shifts
    # |j> -> |j+1 mod 3> and Z applies the clock phase omega**j.
    # from_terms(d, [(word, coefficient)], n_sites=...) uses
    # word = ((site, X_exponent, Z_exponent), ...).  Thus the one-factor
    # word ((0, 0, 1),) means X_0**0 Z_0**1 = Z_0; the trailing comma makes
    # it a one-element word rather than just one factor tuple.
    shift = tcp.QuditWeylOperator.from_terms(3, [(((0, 1, 0),), 1.0)], n_sites=1)
    clock = tcp.QuditWeylOperator.from_terms(3, [(((0, 0, 1),), 1.0)], n_sites=1)
    # H = -J/2 (X + X†) - h/2 (Z + Z†) is the qutrit analogue of a
    # transverse-field clock model on one site.  The adjoints make H Hermitian.
    hamiltonian = -0.5 * (shift + shift.adjoint()) - 0.3 * (clock + clock.adjoint())
    assert hamiltonian.is_hermitian()

    backend_plan = hamiltonian.compile("backend_mvp")
    state = np.arange(3, dtype=np.complex128)
    np.testing.assert_allclose(
        tcp.backend_mvp(backend_plan)(state),
        hamiltonian.compile("dense") @ state,
    )
    print("One-qutrit clock Hamiltonian: backend and dense targets agree")


if __name__ == "__main__":
    main()
