# Concepts

TenCirPauli is a Python-first interface over a compact Rust implementation for structured quantum workloads. Its main idea is simple: describe a quantum structure, organize it for the task, compute in the right runtime, and inspect the result as a value, an estimate, or a diagnosis.

## One workflow: Describe → Organize → Compute → Inspect

The four verbs are a memory aid, not four separate products. A Hamiltonian may use all four stages; a classical-shadow workflow may begin with a supplied state and go straight to sampling and estimation; a stabilizer-code workflow may focus on organizing checks and inspecting syndromes. The package keeps the same structure-first viewpoint across these routes.

### 1. Describe the structure

Use `PauliWord` and `PauliOperator` for canonical qubit-Pauli objects and Hamiltonians. Products, phases, duplicate terms, and ordering are explicit and deterministic, so the same operator can feed matrix construction, measurement grouping, symmetry reduction, or observable execution.

Structured operators extend the model to fermions, bosons, qudits, Majorana words, and hybrid spaces. Circuit facades describe numerical gate workloads, while `StabilizerCode` describes a qubit code space and its checks. `StabilizerState` is a pure stabilizer-state representation; it is distinct from a code-space analysis.

### 2. Organize the structure for the task

This is where TenCirPauli turns a mathematical object into a useful computational or measurement problem. It can canonicalize and aggregate operators, map fermions to qubits, group terms for measurements, find and taper Z₂ symmetries, restrict Hamiltonians to U(1) or additive-charge sectors, and build packed stabilizer-check, logical-operator, and syndrome structures.

Organization also chooses the shape of the computation. Small systems can use dense matrices, sparse consumers can use COO or CSR, large CPU workloads can use native matrix-vector products, and TensorCircuit-NG or JAX workflows can keep numerical execution in a backend-owned plan.

The contracts remain explicit. A qubit-wise commuting group carries local measurement bases; a general commuting group is not silently presented as measurement-ready. A restricted sector checks for leakage instead of assuming conservation, and stabilizer-code analysis reports syndrome and logical structure instead of pretending to be a full decoder.

### 3. Compute in the right runtime

The operator layer supports dense, sparse, native matrix-free, TensorCircuit, and JAX-oriented execution. The circuit facades cover fixed-particle-number circuits, deterministic Pauli propagation, and stochastic Pauli-path estimation. They share a Python-level style while keeping their native numerical contracts separate.

Native CPU plans keep discrete, scalable work in compact Rust data and release the GIL around coarse-grained computation. Backend plans keep NumPy, TensorCircuit-NG, JAX, JIT, or backend autodiff in control of numerical execution. Circuit terminals expose concrete-angle `expectation(observable)`, `value_and_grad(observable)`, and `expectation_jax(observable)` paths; public symbolic circuit parameters and circuit-plan compilation are not part of this interface.

### 4. Inspect the result

The output is not always one deterministic matrix element. Measurement grouping reconstructs Pauli terms from basis-aware bitstring data. Classical shadows turn randomized snapshots into estimates of Pauli expectations, Hamiltonian energy, reduced density matrices, Rényi-2 quantities, variance, and fidelity.

`Snapshots.sample()` accepts either a complete normalized statevector or an immutable `StabilizerState`. `protocol="pauli"` samples independent local X/Y/Z bases, while `protocol="clifford"` samples packed global-Clifford frames. The same snapshot handle can serve many estimators, and external measurements can enter through `from_pauli_measurements()` or `from_clifford_measurements()`.

Error-correction workflows use the same inspection idea in a different form. A `StabilizerCode` can return syndromes, classify errors as visible, stabilizer, or logical, expose deterministic logical representatives, and verify correction-times-error behavior. This is static code analysis and error diagnosis; it is not a fault-tolerant decoder, a noise simulator, or an active correction runtime.

## Common routes through the workflow

| Goal | Describe | Organize | Compute or inspect |
| --- | --- | --- | --- |
| Build and apply a Hamiltonian | `PauliOperator` | Canonicalize, restrict, or choose a target | Dense, sparse, native MVP, or backend execution |
| Run a circuit observable | `PropagationCircuit`, `U1Circuit`, or `SPPSCircuit` | Store concrete gates and observable structure | Expectation, value and gradient, or JAX terminal |
| Estimate from randomized measurements | Statevector or `StabilizerState` | Choose Pauli or Clifford snapshots | Classical-shadow expectation, energy, RDM, or fidelity |
| Analyze an error-correcting code | `StabilizerCode` | Pack checks, logical representatives, and syndrome structure | Classify errors and verify corrections |
| Prepare structured chemistry work | Fermion, Majorana, or other structured operators | Map, group, and restrict sectors | Hamiltonian plans or TensorCircuit-NG integration |

The useful question is therefore not “which feature do I use?” Start with the structure you already have, identify the structure that can be exploited, choose the runtime that owns the computation, and then choose the kind of evidence you need from the result.

## Two runtime lanes after organization

Once the structure is ready, choose the runtime lane that matches the next operation. Both lanes use the same public objects and can appear in the same workflow; the difference is who owns the numerical execution.

<div class="tp-path-grid">
  <div class="tp-path tp-path-native">
    <p class="tp-card-index">NATIVE CPU</p>
    <h3>Keep the computation compact</h3>
    <p>Use Rust-native plans and propagation when the workload is discrete, CPU-oriented, and does not need to be traced by a backend.</p>
    <code>operator.native_mvp_plan()</code>
  </div>
  <div class="tp-path tp-path-backend">
    <p class="tp-card-index">BACKEND PLAN</p>
    <h3>Keep the tensor graph alive</h3>
    <p>Use a stable structural plan when TensorCircuit, JAX, JIT, or backend autodiff should own the numerical execution.</p>
    <code>operator.backend_mvp_plan()</code>
  </div>
</div>

## What it is not

TenCirPauli is not a general-purpose statevector simulator, tensor-network engine, fault-tolerant decoder, or replacement for TensorCircuit backends. Its dense shadow sampler accepts a supplied statevector for randomized measurement, but does not turn the package into a general state-preparation or circuit-simulation frontend. It prepares, executes, and inspects the structured Pauli work around those systems, and it fails explicitly when a target would require an unbounded or unsupported expansion.

## Where to go next

- [Quickstart](quickstart.md) for a small Hamiltonian and propagation example.
- [API reference](api.md) for public classes and methods generated from the Python source docstrings.
- [GitHub repository](https://github.com/tensorcircuit/TenCirPauli) for benchmarks, design notes, and contribution details.
