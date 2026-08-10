# Pauli-propagation VQE: TenCirPauli vs PauliPropagation.jl

This study compares TenCirPauli's native Pauli-propagation value-and-gradient path with the registered Julia package [PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl) v0.8.0. Both sides propagate the same observable backwards through the same parameterized circuit and evaluate it in the all-zero product state; neither side simulates a dense statevector for the benchmark.

The workload is an open-boundary transverse-field Ising Hamiltonian

`H = -sum_i Z_i Z_(i+1) - 0.5 sum_i X_i`

with a six-layer hardware-efficient ansatz. Every layer applies an independent `RY` rotation to every qubit followed by an even/odd brickwork layer of `CNOT` gates. There is no parameter sharing. The benchmark uses 28 qubits, 6 layers, 168 parameters, and `max_weight=5`; both implementations use default multithreading and disable coefficient-magnitude truncation. The VQE loop is the same fixed-step gradient descent on both sides.

Correctness is checked independently before timing the larger case. An 8-qubit, 2-layer instance is run with `max_weight=8`, and both implementations are compared against a dense statevector reference with exact parameter-shift gradients. The same small case also checks the TenCirPauli/Julia cross-result. A second 8-qubit check uses `max_weight=5` and finite differences of each implementation's own truncated value. The 28-qubit comparison checks the shared truncated value and reports the gradient difference explicitly: TenCirPauli uses its frozen-support reverse gradient, while PauliPropagation.jl uses `rewindgradient`, and these gradient contracts need not agree after weight projection.

The difference is substantive in the larger case. TenCirPauli replays each truncated forward frame and differentiates only retained aggregated edges. PauliPropagation.jl v0.8.0's `rewindgradient` computes a commutator against the current truncated operator, then propagates a dual sum and intersects its support with the operator cache. That reverse formula is exact for the unprojected unitary recurrence, but after repeated weight projection it does not generally differentiate the package's own projected value map: on the default 28-qubit case, selected Julia gradient components differ from finite differences by about `0.25`, while the corresponding TenCirPauli errors are about `1e-9`. This is not merely an alternative target documented by the package: `rewindgradient` is documented as returning the gradient of `overlapfunc(propagate(...))`. The observed mismatch therefore indicates an implementation/semantic defect in the Julia truncation path unless the authors explicitly redefine it as an approximate surrogate gradient.

The recorded default run used Julia `--threads=auto` with 10 Julia threads and no single-thread override. TenCirPauli's Rust Rayon runtime likewise used its default thread policy. Setup, first-call, steady-state, and 20-step VQE timings were:

| Implementation | Setup | First value+grad | Steady value+grad | 20-step VQE |
| --- | ---: | ---: | ---: | ---: |
| TenCirPauli | 0.0003 s | 0.388 s | 0.408 s | 8.41 s |
| PauliPropagation.jl | 0.0728 s | 3.018 s | 0.583 s | 12.32 s |

This makes TenCirPauli about 7.8x faster on the first call, 1.43x faster in steady value-and-gradient calls, and 1.47x faster for the 20-step loop. The initial truncated value was identical (`-25.1313577382`) on both sides. The final VQE energies (`-28.34341134` and `-28.02852009`) are not a valid quality ranking because the two gradient paths are not the same mathematical derivative under `max_weight=5`.

Therefore the forward value and steady propagation timings are directly comparable; the gradient and VQE-loop timings are useful package-level measurements but must be labeled as different gradient contracts until the Julia truncation reverse path is corrected.

## Requirements

The Python side uses the repository environment. The Julia side requires Julia 1.10 or newer and the registered package:

```julia
using Pkg
Pkg.add(["PauliPropagation", "JSON"])
```

The scripts use the Julia environment active when `julia` is invoked. The actual Julia thread count is printed; no single-thread override is applied.

## Run

Run the complete correctness and benchmark comparison from the repository root:

```bash
conda run -p .conda python examples/research/pauli_propagation_julia_vqe/compare.py
```

Run one implementation directly when debugging:

```bash
conda run -p .conda python examples/research/pauli_propagation_julia_vqe/run_tencirpauli.py --mode correctness
julia --threads=auto --startup-file=no examples/research/pauli_propagation_julia_vqe/run_julia.jl --mode correctness
```

The benchmark output separates setup, first value-and-gradient execution, warm steady-state execution, and the fixed VQE loop. The first objective call includes native/JIT compilation after package import; package startup is not mixed into that number.
