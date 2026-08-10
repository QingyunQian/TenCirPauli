# Research-driven examples

These examples exercise complete scientific workflows and are intentionally manual. They are not imported as tests and are not added to the continuous-integration execution allowlist; the normal quality checks still format and lint their Python source.

Each study has its own directory. Keep the first version small: one executable script per implementation, a README describing the question and conventions, and independent reference code only when it is needed. Large scans, figures, and machine-specific outputs stay local.

- [`fermi_hubbard/`](fermi_hubbard/): matrix-vector products for a half-filled, spin-balanced open-boundary Fermi–Hubbard model, with a TenCirPauli restricted MVP script and a QuSpin `quantum_LinearOperator` companion.
- [`bch_convergence/`](bch_convergence/): fixed-order Pauli BCH expansion through fourth order, dense small-system convergence checks, and a pure-Python dictionary algebra baseline.
- [`bch_convergence/run_structured.py`](bch_convergence/run_structured.py): the same fixed-order BCH recurrence for Fermion and Boson operators, with independent CAR/CCR dictionary references and separate native/plain/materialization timings.
- [`syk_majorana/`](syk_majorana/): a quartic SYK ground-state calculation built from Majorana terms, with a reusable mapped native MVP plan and an independent QuSpin comparison.
- [`lie_closure/`](lie_closure/): bounded Pauli-word and Pauli-sum Lie closure, dimension/rank reporting, Jacobi checks, and a pure-Python dictionary baseline.
- [`fermion_mapping/`](fermion_mapping/): cross-environment correctness and timing comparison for Jordan–Wigner, parity, and Bravyi–Kitaev mappings against OpenFermion.
- [`holstein/`](holstein/): a one-dimensional spinful Holstein electron–phonon Hamiltonian with native MVP/Lanczos execution and an optional QuSpin tensor-basis comparison.
- [`u1_vqe_60q/`](u1_vqe_60q/): a 60-qubit, two-particle, 10-layer U(1) VQE comparison between native TenCirPauli gradients and TensorCircuit/JAX `jit(value_and_grad)`.
- [`pauli_propagation_julia_vqe/`](pauli_propagation_julia_vqe/): a matched 28-qubit, six-layer Pauli-propagation VQE comparison against PauliPropagation.jl, with an independent dense/parameter-shift correctness gate.
