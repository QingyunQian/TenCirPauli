# Examples

These scripts are executable smoke examples for the public facade API. The circuit examples construct a circuit, create a Pauli Hamiltonian, evaluate an energy, and (where applicable) apply a short gradient-based update; the structured-operator examples exercise the corresponding algebra and execution paths. They are intentionally small so an Agent can run them as contract checks.

Run them from the repository root after installing TenCirPauli:

```bash
python examples/vqe_u1.py
python examples/vqe_propagation.py
python examples/vqe_spps.py
python examples/tensorcircuit_interop.py
python examples/qudit_backend.py
python examples/majorana_charge.py
python examples/jaynes_cummings_sector.py
```

`U1Circuit`, `PropagationCircuit`, and `SPPSCircuit` have the same user-level construction, `theta=`, `expectation()`, and `value_and_grad()` shape. Their result contracts remain different where the underlying executor is different: deterministic circuits return scalar/gradient results, while SPPS returns an estimator with sampling metadata.

`qudit_backend.py` is the deliberately separate minimal uniform-qudit Weyl example. It compares direct TensorCircuit backend-MVP execution with the dense target.

`majorana_charge.py` demonstrates exact Majorana conversion, a reusable Bravyi–Kitaev plan, additive particle-number sectors, and restricted matrix-free execution.

`jaynes_cummings_sector.py` builds a mixed boson–qubit Jaynes–Cummings Hamiltonian, verifies its total-excitation symmetry, and executes the two-dimensional fixed-excitation block without mapping the model to Pauli strings.

`quantum_chemistry_pyscf.py` is an optional PySCF example. It constructs an H2 molecule, imports its default RHF Hamiltonian through `from_molecule()`, maps it explicitly to Pauli form, and runs a TensorCircuit JAX VQE with gradient descent through the backend MVP plan. Install PySCF with `pip install "tencirpauli[chemistry]"` before running it.

Heavier, manually run end-to-end studies live under [`research/`](research/). They are not part of the CI example execution list. The [`research/holstein/`](research/holstein/) study builds and diagonalizes a one-dimensional electron–phonon Hamiltonian in a fixed spin sector.
