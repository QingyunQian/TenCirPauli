# TenCirPauli Current-Tree Audit — 2026-08-11

> **Source:** consolidated from [Deep Audit Round 5](audit-report-2026-08-08-round5.md). **Baseline:** `e38533f712cb458f261b3000525f646e7eaa3663`.

## Baseline and scope

- Source baseline: commit `e38533f712cb458f261b3000525f646e7eaa3663` (`Cache static backend dtype casts`, 2026-08-11).
- The worktree is dirty. The documentation reorganization and unrelated pre-existing user changes are not treated as fixes for the findings below.
- This report lists only active or partially active findings in the current source tree and private ABI. It does not modify production code, tests, or benchmarks.
- This is a static source review, not a release-mode performance measurement. The performance items require matched benchmarks before implementation decisions.

## Executive disposition

| Disposition | Findings | Meaning |
| --- | --- | --- |
| Partially relevant | 1.1 | Integral ingestion uses one visitor per layout, but the count and materialization passes traverse the same O(n⁴) domain twice. |
| Open | 2.1–2.6, 2.8, 3.1–3.5, 4.1–4.2 | The current code exposes the cited API/ABI inconsistency or slow path. |

The highest-value current items are 4.1, the serial Structured eager CSR apply; 1.1, the repeated integral-domain traversal; 4.2, the U1 eager materialization clone; and 2.1, the missing public `PauliOperator` memory-budget argument.

## Dimension 1 — Dead code, verbose residue, and duplicated work

### 1.1 — Partially relevant, medium: integral ingestion performs two O(n⁴) visits

The current implementation has one `visit_spin_orbital_terms` visitor and one `visit_spin_block_terms` visitor in `crates/tencir-pauli-core/src/structured/integrals.rs`. `canonicalize_fermion_integrals` invokes each visitor once with a counting sink and once with a materializing sink (`structured/integrals.rs:56-160`). Each visit recomputes the Hermitian averages and exact-zero checks, so integral ingestion performs a redundant O(n⁴) traversal.

Disposition: keep open as a measured ingestion optimization.

## Dimension 2 — Python API consistency

### 2.1 — Open, medium: `PauliOperator.from_terms` has no `max_bytes`

`PauliOperator.from_terms` accepts only `nqubits` and `terms` at `python/tencirpauli/pauli.py:368`. Its construction paths use `DEFAULT_MAX_BYTES` internally, while the structured and Majorana families expose budget-bearing constructors. This is a public consistency and documented-budget gap.

Disposition: retain as a medium API item. Add focused constructor tests before changing the signature, including custom low budgets and array-backed inputs.

### 2.2 — Open, low: `PauliOperator.__init__` and `from_code_arrays` cannot receive a custom budget

`PauliOperator.__init__` at `python/tencirpauli/pauli.py:324` and `from_code_arrays` hardcode the default budget. The signatures are inconsistent with `MajoranaOperator.__init__`.

Disposition: retain, preferably together with 2.1 as one Pauli-constructor budget-contract change.

### 2.3 — Open, low: `BosonOperator.compile` strengthens the base precondition

`python/tencirpauli/_structured/boson.py:209` requires `boson_cutoffs` without a default and suppresses the override warning. The base compile contract keeps the argument optional and performs the runtime validation.

Disposition: retain as a small signature-consistency fix; no evidence of a numerical bug.

### 2.4 — Partly substantive: restricted facades lack unified `compile`

`ChargeRestrictedOperator` and `U1RestrictedOperator` provide `dense`, `coo`, `csr`, `apply`, and `mvp_plan`, but no unified `compile` method. The missing `mvp` method is mostly a naming alias because `apply` already implements the one-shot operation. Restricted operators have a narrower materialization contract than general operator facades, so any `compile` method should expose only supported targets.

Disposition: retain only the `compile` question as an API-contract item. Treat `mvp` as optional ergonomics, not a correctness gap.

### 2.5 — Open, low: U1 storage annotations use `str`

`python/tencirpauli/symmetry.py:214,226,310,494,605` annotates U1 storage values as `str`, while the general charge path uses the `ChargeStorage` literal. Runtime validation is present, so this is a static contract inconsistency.

Disposition: retain as a typing cleanup.

### 2.6 — Open, low: `NativeMVPPlan.csr` is not shared by sibling plan types

`NativeMVPPlan.csr` is defined at `python/tencirpauli/hamiltonian.py:310`, while the other MVP plan classes do not expose the same method. The `MVPPlan` protocol does not promise `csr`, so this is a public capability asymmetry rather than a protocol violation.

Disposition: retain for an owner decision: either make CSR a shared capability or keep it only on operator facades and remove the one-off plan method.

### 2.8 — Open, requires an API decision: U1 gradient checkpoint argument

`U1Circuit.value_and_grad` and `_U1CircuitPlan.value_and_grad` do not accept `checkpoint_interval` (`python/tencirpauli/u1_circuit.py:283,575`). The native U1 reverse pass has no checkpoint parameter, so accepting the argument and silently ignoring it would be wrong.

Disposition: retain as a contract-clarification item. The valid choices are to document that U1 gradients are checkpoint-free or to accept the keyword and reject non-`None` values explicitly. Do not implement a superficial forwarding branch.

## Dimension 3 — Python/Rust ABI and typing

### 3.1 — Open, low: Hybrid materialization shape is the ABI outlier

`NativeHybridOperatorHandle.materialize` is `tuple[Any, ...]` in `python/tencirpauli/_native.pyi:204`, and the Rust/Python boundary groups flat arrays in nested tuples. This is a tuple-shape and stub-accuracy inconsistency; it does not remove the Python list reconstruction performed by the public wrapper.

Disposition: retain as a private ABI cleanup, below numerical and hot-path work.

### 3.2 — Open, low: backend-plan array return stub is inaccurate

`pauli_backend_plan_handle` is typed as returning `Sequence[int]` and `Sequence[complex]` in `_native.pyi:769-771`, while Rust returns NumPy arrays. The runtime caller handles the current result correctly.

Disposition: retain as a stub-only correction; no performance claim should be attached to it.

### 3.3 — Open, low: CSR read-back arity and budget contract differ

The native ABI has dimension-prefixed, budget-bearing CSR methods alongside `StructuredEagerMvpPlan.csr()` and `NativeChargeEagerMvpPlan.csr()` returning three arrays without a `max_bytes` argument (`_native.pyi:51,353,513`; `crates/tencirpauli-native/src/charge_sector.rs:709`). Python-level preflight exists, but the native arity and method-contract inconsistency is present.

Disposition: retain as one ABI contract cleanup. Verify the actual wrapper call sites before changing the private signatures.

### 3.4 — Open, low: compact charge-plan nested inputs are stubbed as `object`

`charge_sector_plan_compact` declares the nested weight and qubit-level parameters as `object` in `_native.pyi:362-372`, despite neighboring native entry points using nested `Sequence` types.

Disposition: retain as a typing-only cleanup; the supported Python wrapper path is functional.

### 3.5 — Open, low: Hybrid canonicalization inputs are stubbed as `object`

`structured_hybrid_canonicalize` and `structured_hybrid_jordan_wigner` use `input: object` in `_native.pyi:442-466`. The private ABI is difficult to inspect directly, although the public wrapper constructs the expected tuples.

Disposition: retain as a private-stub/documentation item. A new handle-accepting construction ABI would be a separate design change, not a mechanical audit fix.

## Dimension 4 — Slow fallback and performance paths

### 4.1 — Open, highest-value: Structured eager CSR apply is serial

The current `StructuredEagerMvpPlan::apply_into` calls `CsrApplyPolicy::serial()` at `crates/tencir-pauli-core/src/structured/execution/lazy.rs:111`. The same repository has parallel CSR policies for other execution families. This is a reachable eager path for large non-charge Structured operators and is the strongest performance candidate in this report.

Disposition: retain as P0 performance investigation. First run a release-mode A/B on representative large Boson/Qudit/Hybrid eager CSR workloads, with one and four threads, numerical equality, construction time, steady apply time, and peak memory.

### 4.2 — Open, low-to-medium: U1 warm materialization rebuilds a plan to read `transition_count`

`U1RestrictedOperator._ensure_eager_for_target` calls `native.mvp_plan(_effective_max_bytes(max_bytes))` on the cached warm path at `python/tencirpauli/symmetry.py:368-377` before `.dense()`, `.coo()`, or `.csr()`. The native plan is then discarded. This is a live clone/allocation concern, although dense materialization can be dominated by its own O(D²) output allocation.

Disposition: retain as P1 targeted measurement and cleanup candidate. Measure warm CSR/COO calls separately from dense calls and confirm whether a cached transition count or native getter removes a material part of the end-to-end time.

## Recommended follow-up order

1. Measure 4.1 on large non-charge Structured eager CSR workloads and decide whether the shared CSR parallel policy is justified.
2. Measure 4.2 on warm U1 CSR/COO/dense materialization and cache `transition_count` only if the clone is material to the complete call.
3. Measure 1.1 on representative molecular integral ingestion before removing the count pass; preserve the current two-pass budget contract unless evidence supports a safe change.
4. Resolve the Pauli constructor budget contract in 2.1–2.2 with numerical and low-budget tests.
5. Batch the low-risk signature and stub cleanups in 2.3, 2.5–2.6, and 3.1–3.5 after the public/private contracts are explicitly chosen.
6. Record owner decisions for 2.4 and 2.8 instead of adding aliases or ignored parameters by analogy alone.

## Verification boundary

This audit performed source and stub inspection against `e38533f`; it did not run a full Rust/Python quality gate or release benchmark suite. No production, test, or benchmark source was changed.
