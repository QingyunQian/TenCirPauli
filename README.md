<div align="center">
  <img src="docs/assets/tencirpauli-logo.png" alt="TenCirPauli logo" width="180">
  <h1>TenCirPauli</h1>
  <p><strong>Rust-native Pauli algebra and structured quantum execution for <a href="https://github.com/tensorcircuit/tensorcircuit-ng">TensorCircuit-NG</a>.</strong></p>
  <p>
    <a href="https://github.com/tensorcircuit/TenCirPauli/actions/workflows/ci.yml"><img src="https://github.com/tensorcircuit/TenCirPauli/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
    <a href="https://tensorcircuit.github.io/TenCirPauli/"><img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation"></a>
    <a href="https://pypi.org/project/tencirpauli/"><img src="https://img.shields.io/pypi/v/tencirpauli.svg" alt="PyPI version"></a>
    <a href="https://github.com/tensorcircuit/TenCirPauli/blob/main/LICENSE"><img src="https://img.shields.io/github/license/tensorcircuit/TenCirPauli.svg" alt="Apache 2.0 license"></a>
  </p>
</div>

TenCirPauli moves Pauli-heavy quantum workflows from Python object graphs into compact native data and Rust execution. It is the [TensorCircuit-NG](https://github.com/tensorcircuit/tensorcircuit-ng)-facing layer for Hamiltonians, measurement grouping, symmetry reduction, restricted sectors, structured operators, classical-shadow snapshots, and observable propagation.

## What it does

| Capability | Main entry points |
| --- | --- |
| Pauli algebra and Hamiltonians | `PauliWord`, `PauliOperator`, products, commutators, dense/COO/CSR targets, matrix-free MVP |
| Measurement planning | `group_commuting(mode="qubit_wise")`, `QWCGroupingResult`, basis-aware bitstring reconstruction |
| Classical-shadow snapshots | `Snapshots.sample()`, Pauli/global-Clifford protocols, native estimators, RDM, Rényi-2 entropy, and fidelity |
| Static stabilizer-code analysis | `StabilizerCode`, ordered syndromes, logical representatives, error classification, and decoder-correction verification; distinct from `StabilizerState` |
| Symmetry reduction | `find_z2_symmetries()`, `taper_z2()`, `restrict_charge()` with `U1Sector` or `ChargeSector` |
| Native circuit execution | `U1Circuit`, `PropagationCircuit`, `SPPSCircuit`, value/gradient and expectation terminals |
| Structured quantum operators | `FermionOperator`, `BosonOperator`, `QuditWeylOperator`, `HybridOperator`, `MajoranaOperator` |
| Fermion and chemistry workflows | Jordan–Wigner, parity, Bravyi–Kitaev, optional PySCF ingestion, TensorCircuit-NG integration |

## Why it is fast

- Scalable symbolic work stays in Rust over packed Pauli, occupation, charge, and structured representations.
- Public results are lazy native-backed handles by default; Python materializes terms or arrays only when an API explicitly asks for them.
- Matrix-free and restricted-sector plans avoid dense matrices and full Hilbert spaces when the workload has useful structure.
- TensorCircuit-NG and JAX can remain at the backend boundary through reusable `backend_mvp_plan()` plans, while native CPU paths use coarse-grained Rust execution and default parallelism.

## Performance snapshots

These are representative runs from the linked research studies, not universal maxima. The [benchmark page](docs/performance.md) collects the workload definitions and runnable examples.

| Workload | Comparison | Result |
| --- | --- | --- |
| 60-qubit, two-particle U(1) VQE | [TensorCircuit-NG Python/JAX path](https://github.com/tensorcircuit/tensorcircuit-ng) | 688× faster on the first compiled value-and-gradient call; 2.6× faster when steady |
| 28-qubit Pauli-propagation VQE | PauliPropagation.jl | 7.8× faster on the first call; 1.43× faster when steady |
| 12-mode fermion mapping | OpenFermion | 22× faster for Jordan–Wigner; 12.6× faster for Bravyi–Kitaev |
| 64-qubit, 1,024-term QWC grouping | Qiskit | 38× faster grouping on the linked workload, with native reconstruction included in the study |
| 4×4 Fermi–Hubbard restricted MVP | QuSpin | 1.73× faster and 2.16× lower peak memory |

## Choose an entry point

| If you need to... | Start with... |
| --- | --- |
| Build or transform Pauli operators | `PauliOperator.from_terms()` and `PauliWord.from_string()` |
| Group terms for measurements | `operator.group_commuting()`; use QWC mode for measurement-ready bases |
| Estimate observables from randomized measurements | `Snapshots.sample()` followed by `expectation()`, `estimate_many()`, or `energy()` |
| Analyze a static qubit stabilizer code | `StabilizerCode.from_css()` / `from_generators()`; the code space has dimension `2**nlogical`, while `StabilizerState` is a pure-state tableau |
| Apply a large Hamiltonian without a matrix | `operator.native_mvp_plan()` or `operator.mvp()` |
| Keep TensorCircuit-NG/JAX active | `operator.backend_mvp_plan()` and `tencirpauli.backend_mvp()` |
| Reduce a symmetry sector | `operator.find_z2_symmetries()` / `operator.taper_z2()` |
| Work at fixed particle number or additive charge | `U1Sector`, `ChargeSector`, and `operator.restrict_charge()` |
| Run a native circuit objective | `U1Circuit`, `PropagationCircuit`, or `SPPSCircuit` |
| Map or compile structured operators | `FermionQubitMapping`, `MajoranaOperator`, and the structured operator classes |

`mode="general"` is available for algebraic commuting groups, but its result is intentionally `measurement_ready=False`; use QWC grouping when the output must directly describe product-basis measurements.

## TensorCircuit-NG ecosystem

TenCirPauli is designed as a [TensorCircuit-NG](https://github.com/tensorcircuit/tensorcircuit-ng) companion, not a replacement for its circuit frontend. TensorCircuit-NG circuits can be converted with `U1Circuit.from_circuit()` or `PropagationCircuit.from_circuit()`, and Pauli backend plans can be called through TensorCircuit-NG's NumPy or JAX backend. The Rust core itself has no Python or TensorCircuit-NG dependency.

## Research examples

The [research index](examples/research/README.md) links to reproducible, manual studies covering [Fermi–Hubbard](examples/research/fermi_hubbard/), [Holstein](examples/research/holstein/), [SYK Majorana](examples/research/syk_majorana/), [BCH convergence](examples/research/bch_convergence/), [Lie closure](examples/research/lie_closure/), [fermion mapping](examples/research/fermion_mapping/), [U(1) VQE](examples/research/u1_vqe_60q/), [Pauli propagation](examples/research/pauli_propagation_julia_vqe/), [measurement grouping](examples/research/measurement_grouping/), and [classical shadows](examples/research/classical_shadows/).

## Install

```bash
python -m pip install tencirpauli
```

Released wheels cover common CPython 3.9+ platforms. Source builds require Rust, Cargo, and maturin; chemistry interop is optional via `python -m pip install 'tencirpauli[chemistry]'`.

## Documentation and development

Read the [documentation](https://tensorcircuit.github.io/TenCirPauli/) for concepts and API details, [performance notes](docs/performance.md) for benchmark context, and [CONTRIBUTING.md](CONTRIBUTING.md) for local development. The local quality gate is `python scripts/check.py --benchmark smoke`.

## License

Apache License 2.0.
