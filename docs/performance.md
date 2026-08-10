# Performance highlights

TenCirPauli is built to keep structured quantum workloads compact from the Python call to the native execution kernel. The research examples below collect representative measured wins against TensorCircuit/JAX, QuSpin, PauliPropagation.jl, OpenFermion, and independent Python baselines.

## Headline results

<div class="tp-benchmark-grid" markdown>
  <div class="tp-benchmark-card tp-benchmark-card-primary">
    <p class="tp-card-index">01 / U(1) VQE</p>
    <p class="tp-benchmark-number">688×</p>
    <p class="tp-benchmark-label">faster first compiled value + gradient</p>
    <h3>60-qubit, two-particle VQE</h3>
    <p class="tp-benchmark-copy">37.3× faster end-to-end, including JAX compilation, and 2.60× faster in steady execution.</p>
    <a href="https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/u1_vqe_60q">Open the Research Example →</a>
  </div>
  <div class="tp-benchmark-card">
    <p class="tp-card-index">02 / FERMION MAPPING</p>
    <p class="tp-benchmark-number">22.3×</p>
    <p class="tp-benchmark-label">faster Jordan–Wigner mapping</p>
    <h3>12-mode Hubbard mapping</h3>
    <p class="tp-benchmark-copy">Bravyi–Kitaev mapping is 12.6× faster; end-to-end mapping is 9.5× and 6.0× faster.</p>
    <a href="https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/fermion_mapping">Open the Research Example →</a>
  </div>
  <div class="tp-benchmark-card">
    <p class="tp-card-index">03 / PAULI PROPAGATION</p>
    <p class="tp-benchmark-number">7.8×</p>
    <p class="tp-benchmark-label">faster first value + gradient</p>
    <h3>28-qubit Pauli VQE</h3>
    <p class="tp-benchmark-copy">1.43× faster steady execution against PauliPropagation.jl at <code>max_weight=5</code>.</p>
    <a href="https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/pauli_propagation_julia_vqe">Open the Research Example →</a>
  </div>
  <div class="tp-benchmark-card">
    <p class="tp-card-index">04 / RESTRICTED MVP</p>
    <p class="tp-benchmark-number">1.73×</p>
    <p class="tp-benchmark-label">faster Fermi–Hubbard MVP</p>
    <h3>4×4 half-filled sector</h3>
    <p class="tp-benchmark-copy">The same run also uses 2.16× less peak RSS than QuSpin.</p>
    <a href="https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/fermi_hubbard">Open the Research Example →</a>
  </div>
</div>

The cards show the strongest headline results. The tables below add the resource and symbolic-algebra comparisons without competing with the main message.

## Memory wins

| Workload | Comparison | Resource result | Research example |
| --- | --- | --- | --- |
| 6-site eager Holstein sparse execution | QuSpin explicit sparse path | About 6× lower peak RSS; at four threads, the native MVP is also about 1.8× faster | [Holstein example](https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/holstein) |
| 7-site lazy Holstein MVP; 20,070,400-dimensional sector | QuSpin matrix-free execution | 1.13× faster with about 2× lower peak RSS | [Holstein example](https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/holstein) |
| 4×4 half-filled Fermi–Hubbard MVP | QuSpin matrix-free execution | About 2.16× lower peak RSS with the same 2.650 GB input/output buffers | [Fermi–Hubbard example](https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/fermi_hubbard) |

## Native baseline comparisons

These studies compare the native implementation with small independent Python dictionary prototypes, making the benefit of moving repeated symbolic work into the Rust core visible without presenting a prototype as a competing production library.

| Workload | Baseline | Measured result | Research example |
| --- | --- | --- | --- |
| Fourth-order Pauli BCH; 8 qubits and 44 terms per generator | Python dictionary recurrence | About 4.7× faster native end-to-end | [BCH convergence example](https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/bch_convergence) |
| Bounded four-qubit Pauli Lie closure | Python dictionary recurrence | About 1.6× faster native closure | [Lie-closure example](https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/lie_closure) |

All numbers are representative results from the linked studies, not universal maxima. Each example contains the workload definition, correctness checks, and the command used to reproduce the comparison.
