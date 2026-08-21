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

Deprecated public aliases may forward to the authoritative implementation, but they must not receive new behavior or new internal branches. `PauliOperator.restrict_u1()` remains a deprecated thin forwarding API for `restrict_charge(U1Sector(...))`; its warning and equivalence contract are covered by tests. Removing it requires a separately approved breaking release and is outside the current maintenance plan. Keep the U1 sector algorithm and specialized backend: preserving the compatibility alias does not preserve a second optimized implementation.

### 7. Keep diagnostics out of scalable hot loops

Review counters and profile metadata are useful for selecting kernels and proving performance, but permanent atomic counters, per-term allocations, and repeated validation must not define ordinary execution. Prefer construction-time classification and call-local diagnostic deltas published once per call. If diagnostics materially slow a representative workload, keep them behind a private diagnostic path rather than burdening the production kernel.

### 8. Make benchmark evidence part of refactor design

Every shared-mechanism refactor needs a small representative A/B matrix covering the affected capability classes. Measure end-to-end Python calls as well as Rust kernels, and separate construction, first execution, steady execution, memory, and accuracy. Keep benchmark source and workload definitions in the repository, keep machine-specific results untracked, and do not turn one machine's wall time into a universal CI gate.

## Operational rules for new changes

The rules below turn the maintainability direction into a review contract. They apply to new features, bug fixes, performance work, ABI changes, and refactors. A change may make a deliberate exception, but the exception must be explicit in the change description and in the affected source or trunk document; an unrecorded exception is treated as accidental complexity.

### 9. Establish ownership before implementation

Every scalable behavior must have one authoritative owner. Before adding code, identify the owner for input normalization, capability selection, representation, execution, metadata, memory checks, and materialization. The public Python layer may validate friendly inputs and choose a public target; it must not independently reproduce native applicability or kernel-selection rules. PyO3 should translate the established contract, not become a second policy layer.

For each new plan or execution family, record a compact ownership row with the public entry point, native owner, retained representation, selector, supported targets, and read-back path. If a new row duplicates an existing owner, reuse or extend the existing owner instead of adding a parallel path.

### 10. Prefer explicit dependencies over namespace wiring

Implementation modules must import the shared symbols they use through explicit module imports or narrow private protocols. Do not add module-wide `globals()` copying, `module.__dict__` mutation, wildcard imports, or bottom-of-file imports whose main purpose is to repair a circular dependency. These techniques hide the dependency graph and make static analysis, deletion, and local reasoning unreliable.

The structured namespace bridge at `python/tencirpauli/_structured/base.py` has been removed. Sibling modules now import shared symbols explicitly, family composition is owned by `compile.py` and concrete facades, and `scripts/check_architecture.py` rejects namespace mutation, wildcard imports, whole-file Ruff suppression, and unregistered broad native annotations. The public facade's module, repr, identity, pickle, and narrowly documented private forwarder contracts remain covered by tests.

The public facade may retain a narrowly documented dynamic compatibility forwarder when it is required for an existing import contract. Such a forwarder must not be used to hide implementation dependencies or to provide a second implementation.

### 11. Keep one representation and one fallback policy

Do not retain Python symbolic storage, an array mirror, and a native handle for the same scalable state unless each representation has a separately documented public purpose. If a native handle is mandatory for a completed path, construct it once and fail clearly when the invariant is violated; do not preserve a no-handle execution branch merely because it existed historically.

Every fallback must answer four questions: what capability makes it applicable, why the authoritative path cannot handle that case, what correctness contract it shares with the authoritative path, and what evidence or owner decision justifies its retained cost. A fallback without a removal condition is a second implementation. Compatibility aliases may forward to the owner, but they must not gain new behavior.

### 12. Treat the ABI as a finite grammar

New native endpoints must use one of the established ABI forms unless a new form is justified in the ABI contract:

- handle to handle for scalable symbolic operations;
- flat contiguous arrays with explicit offsets, lengths, order, and dtype for bulk construction or materialization;
- state array to result array, or caller-owned `apply_into`, for execution;
- small immutable metadata records or scalar properties for control information.

Do not introduce a nested numeric read-back form, split real/imaginary arrays when a complex array is the canonical representation, or a Python materialize-and-rebuild round trip when the destination is native. Update the Rust binding, `_native.pyi`, wrapper, and ABI tests in the same change. Known dynamic tensor inputs may remain typed as `object` when the backend protocol genuinely requires it; known native outputs, handles, tuple arity, and array contracts should be precise.

### 13. Make plan contracts capability-based

Different algorithms and storage formats should remain different implementations, but their common execution contract must be testable. Every reusable MVP plan should be classified against the applicable capabilities: dimension, target, storage, estimated retained bytes, construction peak estimate, `apply`, `apply_into`, transition count, dense/COO/CSR read-back, parallel execution, and backend execution.

Optional capabilities must be represented as intentional differences, not discovered through scattered `hasattr()` checks or duplicated string selectors. A capability matrix and parameterized contract tests should verify output dtype, shape, contiguity, ownership, writeability, overwrite behavior, concurrent reuse, budget behavior, and numerical equality across all applicable plan families.

### 14. Centralize errors and boundary validation

Validate friendly Python input once at the public boundary, validate native invariants once at plan construction, and trust immutable handles during execution. Do not repeatedly rescan canonical terms, revalidate fixed layouts, or add layered defenses for implausible numerical extremes without representative evidence.

Each public invalid-input category must have a deliberate exception contract: type errors for wrong kinds, value errors for invalid values or incompatible domains, overflow errors for checked dimension arithmetic, memory errors for enforced allocation limits, and explicit runtime or dependency errors for unavailable capabilities. Tests should assert the exception class as well as the message fragment and failure timing.

### 15. Make deletion part of completion

When a representation, FFI entry point, compatibility branch, helper, diagnostic counter, or benchmark name is superseded, delete it in the same change unless it is an intentional public or historical contract. Search for sibling occurrences of the same pattern before declaring the change complete. Do not preserve dead code with `Any`, `cast`, `type: ignore`, Clippy allowances, or compatibility wrappers solely to keep an abandoned path compiling.

Historical design and audit records may preserve why a path existed, but they must not be mistaken for active implementation requirements. Durable trunk documents and current source win when historical records disagree with them.

## Change review checklist

Before merging a material change, the author and reviewer should be able to answer yes to the applicable questions below:

- Is the authoritative owner and selector identified, with no duplicate policy in Python, PyO3, or Rust?
- Was an existing primitive, handle method, ABI form, validator, plan contract, or read-back path reused before new code was written?
- Does the change avoid parallel scalable storage, nested read-back, per-term FFI, hidden imports, and silent fallbacks?
- Are public input validation, native invariant validation, error class, error timing, and `max_bytes` behavior explicit?
- Does every numeric or gradient entry point have a known-correct numerical assertion, not only shape or metadata checks?
- Are thresholded, parallel, eager/lazy, and capability-specific branches tested for both numerical parity and intended metadata?
- If the change affects a shared mechanism, is there release-mode end-to-end A/B evidence covering the dominant cost and the Python/Rust boundary?
- Were abandoned helpers, stubs, branches, compatibility paths, tests, benchmark aliases, and stale documentation removed or explicitly retained?
- Were the relevant ABI stub, public API manifest, changelog, architecture document, and release metadata updated together?

The checklist should remain short enough to use on every meaningful change. Detailed audits may expand it for a specific workload, but they should not turn every hypothetical extreme into a permanent production rule.

## Enforceable guardrails

The prose rules should be backed by cheap repository checks where possible. The preferred direction is one architecture check invoked by `scripts/check.py`, with a small reviewed allowlist for intentional exceptions. It should initially cover:

- forbidden imports across the core, PyO3, public Python, integration, and test-oracle boundaries;
- new namespace mutation, wildcard imports, or unregistered dynamic compatibility forwarding;
- new nested numeric native returns and unregistered `_native.pyi` `object`/`Any` annotations;
- duplicate target or execution-family selectors outside the owning module;
- native-handle optional branches and fallback markers outside their allowlist;
- version, public-export, ABI-stub, and changelog consistency at release boundaries.

The check should report a file, line, rule, and registered exception identifier. It should not attempt to prove exact peak RSS, infer algorithmic quality from line count, or replace numerical tests and release benchmarks. Mechanical checks protect ownership and contracts; scientific correctness and performance still require focused tests and measurements.

## Exception ledger

Intentional deviations should be recorded in a small table in the relevant trunk document or change record:

| Exception | Owner | Reason | Scope | Evidence | Removal or review condition |
| --- | --- | --- | --- | --- | --- |
| Dynamic compatibility forwarder | Public facade owner | Preserve an existing import or pickle contract | Named symbols only | Contract tests | Remove after the documented compatibility window |
| Family-specific read-back | Native family owner | Retained representation is materially different | Named plan type and target | Numerical and memory evidence | Revisit when the representation changes |
| Backend capability probe | Integration owner | External framework has version-dependent optional capability | Integration boundary only | Compatibility tests | Replace with a versioned contract when dependency support is fixed |

The ledger is not permission to accumulate exceptions. Each row must describe a bounded scope and a condition under which it is deleted, narrowed, or re-evaluated.

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
