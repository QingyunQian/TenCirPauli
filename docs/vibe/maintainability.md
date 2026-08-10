# Performance-Preserving Maintainability Direction

## Status and purpose

This is a durable trunk document for future maintainability work. It is not a numbered phase plan and does not authorize a broad rewrite. It defines how TenCirPauli should reduce accidental implementation entropy while preserving numerical correctness, representative performance, memory behavior, and the specialized execution paths required by the scientific workload.

## Non-negotiable priority

Numerical and semantic correctness is non-negotiable. Once correctness is protected by trusted references and regression tests, performance is the first engineering priority: a maintainability change must not trade away representative runtime, throughput, peak memory, allocation behavior, scaling, or Python/Rust boundary efficiency merely to make the source look simpler.

Every material refactor must therefore have a release-mode before/after comparison on representative end-to-end workloads. The comparison must include input normalization, native plan construction where relevant, Python/Rust boundary costs, first execution, steady execution, memory, allocation behavior when material, and numerical agreement. A refactor with a repeatable material regression is not accepted without a new owner decision; the losing implementation or abstraction must be removed rather than retained as a silent fallback.

## The central distinction

TenCirPauli has real complexity that must remain: Pauli, fermion, boson, hybrid, qudit, and Majorana operators have different algebraic contracts; dense, COO, CSR, lazy MVP, and eager MVP have different output and storage contracts; U1, general charge, factorized, Product, Cartesian, and Generic execution have different applicability conditions and time/space trade-offs. Removing these paths indiscriminately would damage the reason the project exists.

The maintainability target is accidental complexity: duplicated selection logic, repeated validation and metadata plumbing, parallel ABI shapes for the same result, hidden module-import dependencies, stale compatibility paths, duplicated read-back and CSR mechanics, diagnostic code in hot loops, and Python wrappers that know details belonging to native plans. Preserve distinct algorithms and representations when their complexity or measured performance differs; unify contracts and mechanics when their semantics and data layout are genuinely the same.

## Target shape of the architecture

The execution system should behave like a tree with one authoritative selection point, not a mesh in which every layer knows every combination:

```text
Public operator or sector
        |
        v
One construction-time plan selector
        |
        +--> lazy or eager storage policy
        |
        +--> specialized kernel or generic kernel
        |
        v
One plan contract: metadata, memory policy, apply, apply_into, and read-back
```

The selector may choose different native algorithms for different capabilities. The selector must not be duplicated in Python, PyO3, and several Rust modules, and the public API should not expose the Cartesian product of internal variants unless a distinction is an intentional public contract such as storage policy or target format.

## Main maintainability directions

### 1. Keep specialized execution, centralize ownership

U1, general charge, factorized, Product, Cartesian, Generic, Structured, and Pauli kernels may remain separate. They should have one clear owner for basis selection, one construction-time capability classification, and one immutable plan publication boundary. Python should request a capability or target and receive a plan; it should not reimplement native eligibility tests or pass physical fast-path hints that the native sector already owns.

### 2. Split Python structured code by responsibility

The structured Python package should separate shared data/model primitives, native-handle conversion, validation and integral ingestion, family-specific operator facades, and finite compilation. The shared layer may provide narrow internal protocols and constructors, but it must not become a module that imports every family at the bottom merely to repair circular dependencies. Family modules should depend on explicit shared contracts, and native result materialization should remain an explicit boundary.

The first goal is dependency direction and local reasoning, not a new public abstraction hierarchy. Public Fermion, Boson, Hybrid, and Weyl types remain distinct where their algebraic contracts differ.

### 3. Use a small number of private Python↔Rust ABI forms

The private ABI should converge on these forms:

- Bulk construction: contiguous NumPy arrays or flat buffers with offsets and lengths for scalable term, gate, transition, and basis data.
- Symbolic algebra: native handle to native handle, without materializing complete operators into Python between operations.
- Execution: state array to result array, or `apply_into(input, output)` for caller-owned non-overlapping buffers.
- Materialization: fixed-arity flat NumPy arrays with explicit order, dtype, and shape; no new deeply nested sequence read-back forms.
- Metadata: small read-only properties or one explicit metadata record, with consistent memory and transition-count semantics.

The existence of multiple operator families and native plan classes does not require multiple incompatible ABI grammars. Construction payloads can remain family-specific when their layouts differ, while common mechanics such as handle ownership, GIL release, memory errors, execution buffers, and flat read-back must not drift.

### 4. Merge mechanics, not hot algorithms

The following are good candidates for convergence when release-mode A/B evidence shows no material regression: plan metadata, state/output validation, memory preflight, native handle forwarding, CSR graph storage, destination-row apply mechanics, flat CSR/COO conversion, and SciPy wrapper plumbing. Keep specialized action compilation, factorized indexing, packed U1 rank/unrank, fermion sign handling, and structured local kernels separate when they have different asymptotic behavior or data layout.

Do not introduce a universal trait object, a runtime-heavy mega-enum, or a single executor that adds branches to every hot loop merely to reduce file count. Module boundaries should improve ownership without changing the selected kernel or its data layout.

### 5. Make the ABI typed and testable

`_native.pyi` is a private contract and should describe array dtype, flat shape conventions, tuple arity, handle types, and optional metadata accurately enough for Python wrappers to be checked. Replace `object` and open-ended tuple types where the actual result is known. Add focused ABI tests for arity, dtype, contiguity, read-only guarantees, buffer ownership, and handle-to-handle paths. Update the stub in the same change as the Rust boundary and delete obsolete private entry points instead of retaining compatibility overloads.

### 6. Treat compatibility as a bounded migration

Deprecated public aliases may forward to the authoritative implementation, but they must not receive new behavior or new internal branches. `PauliOperator.restrict_u1()` is a compatibility alias for `restrict_charge(U1Sector(...))`; migrate tests, examples, and internal callers, document the removal version, and then delete the alias. Keep the U1 sector algorithm and specialized backend: removing the redundant entry point is not removing the optimized capability.

### 7. Keep diagnostics out of scalable hot loops

Review counters and profile metadata are useful for selecting kernels and proving performance, but permanent atomic counters, per-term allocations, and repeated validation must not define ordinary execution. Prefer construction-time classification and call-local diagnostic deltas published once per call. If diagnostics materially slow a representative workload, keep them behind a private diagnostic path rather than burdening the production kernel.

### 8. Make benchmark evidence part of refactor design

Every shared-mechanism refactor needs a small representative A/B matrix covering the affected capability classes. Measure end-to-end Python calls as well as Rust kernels, and separate construction, first execution, steady execution, memory, and accuracy. Keep benchmark source and workload definitions in the repository, keep machine-specific results untracked, and do not turn one machine's wall time into a universal CI gate.

## Ordered work program

### P0: freeze ownership and ABI rules

Maintain one capability-ownership table and one private ABI grammar. New features must reuse an existing ABI form or explicitly justify a new one. No new parallel Python storage, nested read-back, compatibility implementation, or duplicated plan selector should be added.

### P1: remove accidental structural coupling

Split the Python structured common layer by responsibility, replace wildcard module coupling with explicit imports, and make charge/structured plan construction phases visible in the Rust module tree. Preserve the existing native kernels and verify each move with focused numerical tests and release A/B measurements.

### P2: converge shared plan and data-plane mechanics

Unify common plan metadata, buffer contracts, memory checks, CSR storage/apply/read-back, and handle forwarding. Keep U1, charge, structured, and Pauli producers specialized where their representations differ. Delete the old implementation immediately after the new owner is accepted.

### P3: close ABI and compatibility debt

Type the native stub, remove nested read-back forms, delete obsolete private FFI, migrate `restrict_u1`, and add ABI contract tests. Treat warning cleanup as migration work rather than suppressing warnings indefinitely.

### P4: maintain the performance guardrail

Run the complete quality gate and the affected release benchmark matrix after each vertical slice. A refactor is complete only when correctness, ABI behavior, memory policy, and representative performance remain within the documented engineering envelope.

## Explicit non-goals

This direction does not seek one universal operator class, one universal execution kernel, one universal storage format, or one giant abstraction over all algebraic domains. It does not replace measured specialized paths with aesthetically uniform but slower code, and it does not require historical phase documents to remain globally self-consistent after their work is complete.

## Completion criterion

The repository is moving in the right direction when a new operator family or execution optimization adds one owned native plan and reuses the existing construction, handle, execution, metadata, memory, and read-back contracts; when Python does not duplicate native selection logic; when deprecated paths have a removal boundary; and when every maintainability improvement is accompanied by evidence that the representative performance envelope was preserved.
