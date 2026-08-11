# TenCirPauli Current-Tree Audit — 2026-08-11

> **Source:** consolidated from [Deep Audit Round 5](audit-report-2026-08-08-round5.md). **Baseline:** `e38533f712cb458f261b3000525f646e7eaa3663`.

## Baseline and scope

- Source baseline: commit `e38533f712cb458f261b3000525f646e7eaa3663` (`Cache static backend dtype casts`, 2026-08-11).
- The worktree is dirty. The documentation reorganization and unrelated pre-existing user changes are not treated as fixes for the findings below.
- This report preserves the original findings and records the implementation or measured decision for each one, including items intentionally closed without production changes.
- This update includes focused Rust/Python tests and same-machine release A/B measurements for the two performance findings. The exact peak-RSS and stage-level chemistry profile remain outside this slice.

## Executive disposition

The following dispositions were agreed after a current-tree source review and owner discussion. Specifications and earlier audit labels are treated as design evidence rather than authority; the deciding criteria are reachable behavior, representative scientific workloads, interface ownership, maintainability, and measured performance.

| Disposition | Findings | Meaning |
| --- | --- | --- |
| Measured and implemented | 1.1, 2.1–2.2, 2.4–2.5, 3.2–3.5, 4.1–4.2 | Release A/B evidence supported the scoped runtime changes; the remaining rows are mechanical API, ABI, typing, or metadata fixes. |
| Intentional asymmetry or no production fix | 2.3, 2.8, 3.1 | Preserve the domain-specific contract or representation ownership; clarify the reason where useful instead of expanding the API by analogy. |
| Documentation clarification | 2.6 | Keep the representation-specific capability and clarify its required eager CSR-backed plan. |

The highest-value measured change is 4.1, where a Structured-specific threshold enables parallel CSR rows only for large graphs. Finding 1.1 now uses a one-pass incremental budget guard after release measurements showed a repeatable construction win. Finding 3.3 performs the native CSR budget check before read-back allocation, and 4.2 reads the authoritative U1 transition count without cloning a temporary plan.

## Dimension 1 — Dead code, verbose residue, and duplicated work

### 1.1 — Implemented after release A/B: integral ingestion performed two O(n⁴) visits

The previous implementation had one `visit_spin_orbital_terms` visitor and one `visit_spin_block_terms` visitor in `crates/tencir-pauli-core/src/structured/integrals.rs`. `canonicalize_fermion_integrals` invoked each visitor once with a counting sink and once with a materializing sink, recomputing Hermitian averages and exact-zero checks over the same O(n⁴) domain.

The validation pass is itself O(n⁴), so the former complete path performed Hermitian validation, nonzero counting, and materialization over the same domain before CAR rewriting and aggregation. Removing the count visit therefore removes one simple traversal, not one third of the complete ingestion cost.

Implementation: validation remains a separate O(n⁴) pass, while term counting and materialization are now fused. `push_integral_term` increments the exact raw-term count, checks the existing `raw_count * 768` guard before each append, and then stores the factor/coefficient. This preserves exact-zero handling, canonical output, and the native budget error while removing the duplicate visitor.

Evidence: on the same release build and machine with dense Hermitian synthetic spin-orbital inputs, baseline versus one-pass construction medians were n=8: 0.844 ms versus 0.835 ms, n=10: 2.089 versus 1.866 ms, n=12: 5.902 versus 4.079 ms, and n=16: 16.955 versus 13.974 ms. The n=12 and n=16 gains are repeatable and material; focused integral-ingestion and structured-algebra tests pass. A separate streaming accumulator was not justified because raw-term materialization was not profiled as the dominant remaining cost.

## Dimension 2 — Python API consistency

### 2.1 — Implemented, medium: `PauliOperator.from_terms` had no `max_bytes`

`PauliOperator.from_terms` accepts only `nqubits` and `terms` at `python/tencirpauli/pauli.py:368`. Its construction paths use `DEFAULT_MAX_BYTES` internally, while the structured and Majorana families expose budget-bearing constructors. This is a public consistency and documented-budget gap.

Implemented with 2.2: `max_bytes` is keyword-only, validated at the Python boundary, and forwarded to both native Pauli construction paths. Values below the public 1 GiB enforcement floor remain advisory.

### 2.2 — Implemented with 2.1, low: Pauli constructors could not receive a custom budget

`PauliOperator.__init__` at `python/tencirpauli/pauli.py:324` and `from_code_arrays` hardcode the default budget. The signatures are inconsistent with `MajoranaOperator.__init__`.

Implemented across `__init__`, `from_terms`, `from_code_arrays`, and `from_strings`, with source compatibility preserved by the keyword-only argument. Focused tests cover invalid values, `None`, advisory forwarding, array-backed input, and unchanged canonical terms. Canonicalization-result APIs remain outside this constructor change.

### 2.3 — Closed without change: `BosonOperator.compile` strengthens the base precondition

`python/tencirpauli/_structured/boson.py:209` requires `boson_cutoffs` without a default and suppresses the override warning. The base compile contract keeps the argument optional and performs the runtime validation.

Disposition: do not fix. The concrete Boson contract genuinely requires finite cutoffs, and making the argument optional would weaken type checking and defer a useful call-site error to runtime. The base class is private, and the repository has other family-specific compile overrides; eliminating every override suppression would require a broader hierarchy redesign with no demonstrated user or performance benefit.

### 2.4 — Implemented: restricted facades lacked unified `compile`

`ChargeRestrictedOperator` and `U1RestrictedOperator` provide `dense`, `coo`, `csr`, `apply`, and `mvp_plan`, but no unified `compile` method. The missing `mvp` method is mostly a naming alias because `apply` already implements the one-shot operation. Restricted operators have a narrower materialization contract than general operator facades, so any `compile` method should expose only supported targets.

Implemented as thin Python-only dispatchers on both restricted facades for `dense`, `coo`, `csr`, and `native_mvp`. `backend_mvp` and an `mvp` alias remain intentionally unsupported; dispatch reuses the existing cached paths.

### 2.5 — Implemented as typing-only cleanup: U1 storage annotations used `str`

`python/tencirpauli/symmetry.py:214,226,310,494,605` annotates U1 storage values as `str`, while the general charge path uses the `ChargeStorage` literal. Runtime validation is present, so this is a static contract inconsistency.

Implemented with a dependency-safe `U1Storage = Literal["lazy", "eager"]` alias. Runtime validation is unchanged and strict mypy passes.

### 2.6 — Resolved as intentional capability asymmetry: `NativeMVPPlan.csr` is not shared by sibling plan types

`NativeMVPPlan.csr` is defined at `python/tencirpauli/hamiltonian.py:310`, while the other MVP plan classes do not expose the same method. The `MVPPlan` protocol does not promise `csr`, so this is a public capability asymmetry rather than a protocol violation. The representations are also materially different: a non-charge Structured eager plan retains a CSR graph, a Pauli eager plan retains X-mask diagonals rather than CSR, and the Charge/U1 restricted facades own synchronized eager caches.

Calling `operator.compile("native_mvp", storage="eager")` and then `operator.compile("csr")` on an ordinary Structured operator currently rebuilds the transition graph because the operator retains only its symbolic native handle. Consequently, the Structured eager plan's existing `csr()` is a useful read-back of an already retained CSR representation and can avoid material reconstruction. By contrast, `ChargeRestrictedOperator.csr()` and `U1RestrictedOperator.csr()` already reuse the eager graph cached on their facades, so adding plan-level `csr()` there is unnecessary for reuse. Pauli eager plans cannot provide the same operation without building a different representation.

Implemented the docstring and error clarification while retaining `NativeMVPPlan.csr()` only for eager CSR-backed Structured plans. It was not added to the common protocol or sibling plan types, and no general Structured graph cache was introduced.

### 2.8 — Closed without API expansion: U1 gradient checkpoint argument

`U1Circuit.value_and_grad` and `_U1CircuitPlan.value_and_grad` do not accept `checkpoint_interval` (`python/tencirpauli/u1_circuit.py:283,575`). The native U1 reverse pass has no checkpoint parameter, so accepting the argument and silently ignoring it would be wrong.

No argument was added. The two public docstrings now state that inverse-gate replay retains a constant number of sector state vectors and has no propagation-style checkpoint interval. Unexpected keywords continue to fail explicitly.

## Dimension 3 — Python/Rust ABI and typing

### 3.1 — Closed without runtime change: Hybrid materialization shape is the ABI outlier

`NativeHybridOperatorHandle.materialize` is `tuple[Any, ...]` in `python/tencirpauli/_native.pyi:204`, and the Rust/Python boundary groups flat arrays in nested tuples. This is a tuple-shape and stub-accuracy inconsistency; it does not remove the Python list reconstruction performed by the public wrapper.

Disposition: do not change the runtime ABI in this work. The payload already consists of flat NumPy arrays grouped into fixed tuples, and Python reconstructs typed objects only at the explicitly requested `terms`/`to_dict` materialization boundary; it is not materialized and fed back into Rust. Runtime flattening would be contract churn without a demonstrated performance or maintainability gain.

### 3.2 — Implemented as stub-only cleanup: backend-plan array return stub was inaccurate

`pauli_backend_plan_handle` is typed as returning `Sequence[int]` and `Sequence[complex]` in `_native.pyi:769-771`, while Rust returns NumPy arrays. The runtime caller handles the current result correctly.

Implemented in `_native.pyi` with NumPy-array return types matching the native result. The Rust endpoint is unchanged.

### 3.3 — Implemented, medium: eager-plan CSR checked the budget after read-back allocation

Before this change, the native ABI had dimension-prefixed, budget-bearing CSR methods alongside `StructuredEagerMvpPlan.csr()` and `NativeChargeEagerMvpPlan.csr()` returning three arrays without a `max_bytes` argument (`_native.pyi:51,353,513`; `crates/tencirpauli-native/src/charge_sector.rs:709`). `_read_csr_arrays` consequently called the unbounded native read-back first and checked the resulting NumPy `nbytes` afterward, allowing an enforced-budget caller to briefly retain an over-budget CSR copy.

Implemented as a cheap best-effort preflight: both private eager-plan `csr` methods accept `max_bytes`, pass it to bounded `to_u64_arrays`, and only then allocate NumPy read-back arrays. The three-array arity and Python post-check remain unchanged; no zero-copy or allocator accounting was introduced. Native under-budget failures and `uint64`/`complex128` ABI checks are covered by focused tests.

### 3.4 — Implemented as stub-only cleanup: compact charge-plan nested inputs were stubbed as `object`

`charge_sector_plan_compact` declares the nested weight and qubit-level parameters as `object` in `_native.pyi:362-372`, despite neighboring native entry points using nested `Sequence` types.

Implemented in the private stub using nested integer and qubit-level sequence types; the construction ABI is unchanged.

### 3.5 — Implemented as stub-only cleanup: Hybrid canonicalization inputs were stubbed as `object`

`structured_hybrid_canonicalize` and `structured_hybrid_jordan_wigner` use `input: object` in `_native.pyi:442-466`. The private ABI is difficult to inspect directly, although the public wrapper constructs the expected tuples.

Implemented with precise private aliases for the existing raw and Jordan–Wigner Hybrid tuples. Tuple order and runtime conversion are unchanged; no flat-buffer compatibility path was added.

## Dimension 4 — Slow fallback and performance paths

### 4.1 — Implemented after release A/B: Structured eager CSR apply was serial

The previous `StructuredEagerMvpPlan::apply_into` called `CsrApplyPolicy::serial()` at `crates/tencir-pauli-core/src/structured/execution/lazy.rs:111`. This is a reachable eager path for large non-charge Structured operators.

Implemented `CsrApplyPolicy::parallel(1 << 20)` for Structured eager plans. The threshold is intentionally higher than the shared 1<<14 policy because A/B measurements showed overhead on small and medium graphs. With five nearest-neighbor boson hopping terms, serial versus four-thread steady `apply_into` medians were 0.0531 versus 0.0525 ms (16,807 dimensions; 49,392 transitions), 0.1689 versus 0.1695 ms (59,049; 186,624), 0.3841 versus 0.3684 ms (117,649; 432,180), and 1.9057 versus 0.5153 ms (531,441; 2,099,520). The one-thread path remains serial, and numerical eager/lazy differential tests pass. No public parallel flag was added.

### 4.2 — Implemented as a small cleanup: U1 warm materialization rebuilt a plan to read `transition_count`

Before this change, `U1RestrictedOperator._ensure_eager_for_target` called `native.mvp_plan(_effective_max_bytes(max_bytes))` on the cached warm path at `python/tencirpauli/symmetry.py:368-377` before `.dense()`, `.coo()`, or `.csr()`, then discarded the plan. Source inspection showed that `U1MvpPlan::clone_with_budget` shares both the CSR graph and choose table through `Arc`; it does not clone the transition arrays. The remaining cost was a small wrapper/metadata operation, and the repeated graph-budget check was semantically unnecessary when the graph was already retained.

Implemented the read-only native getter and used it in warm materialization preflight, first eager-cache accounting, and `estimated_bytes`. The authoritative native value replaces temporary `mvp_plan` construction; focused cache/materialization and numerical tests pass. This is accepted as a metadata/budget simplification, not as a claimed standalone speedup.

## Implementation handoff matrix

This matrix is the authoritative implementation brief for the findings above. An implementer should not broaden a row into adjacent API redesign, exact allocator accounting, compatibility overloads, or speculative fallback paths.

| Finding | Required action | Focused correctness and contract checks | Performance evidence and decision rule |
| --- | --- | --- | --- |
| 1.1 | Fuse validation-count/materialization with `push_integral_term`; keep the streaming accumulator out of scope. | Integral-ingestion and structured differential tests pass for exact zeros, Hermitian averaging, non-finite rejection, constants, and canonical coefficients. | Release medians improved from 5.902 to 4.079 ms at n=12 and 16.955 to 13.974 ms at n=16 on dense fixtures; retain the one-pass implementation. |
| 2.1–2.2 | Add keyword-only `max_bytes` to all four Pauli constructors and forward through existing native calls; canonicalization-result APIs remain unchanged. | Focused constructor tests cover invalid budgets, `None`, advisory values, array-backed input, and canonical equality. | No dedicated benchmark required; construction tests pass without a regression signal. |
| 2.3 | No code or signature change. Keep `boson_cutoffs` required on concrete Boson compilation. | Existing Boson compilation tests remain the guard. Do not add tests whose only purpose is to permit an omitted cutoff. | None. |
| 2.4 | Thin Python-only `compile` dispatchers support `dense`, `coo`, `csr`, and `native_mvp`; `backend_mvp` and `mvp` remain unsupported. | Charge and U1 dispatch tests compare direct numerical targets and invalid-target errors. | No dedicated benchmark; no extra native compilation is introduced. |
| 2.5 | Use a dependency-safe `U1Storage = Literal["lazy", "eager"]` alias without changing runtime validation. | Strict mypy passes. | None. |
| 2.6 | Retain Structured eager `NativeMVPPlan.csr()` as a representation-specific read-back and clarify its docstring/error; do not broaden the protocol or add a general graph cache. | Existing eager-plan CSR and facade cache tests remain the guard. | No benchmark required. |
| 2.8 | Do not add `checkpoint_interval`; clarify the two U1 gradient docstrings. | Existing U1 gradient numerical tests remain sufficient; unexpected keywords continue to fail. | None. |
| 3.1 | Do not change the Hybrid runtime materialization ABI in this work. | Existing lazy-materialization and structured-algebra tests remain sufficient. Do not add a compatibility read-back variant. | None. |
| 3.2 | Correct `pauli_backend_plan_handle` return typing to NumPy arrays. | Strict mypy and backend-plan tests pass; Rust endpoint unchanged. | None. |
| 3.3 | Add `max_bytes` to both eager-plan CSR methods and call bounded `to_u64_arrays` before read-back allocation; preserve three-array arity and Python post-check. | Native under-budget and dtype/shape tests pass. | No repeatable successful-call regression observed; no zero-copy work pursued. |
| 3.4 | Replace `object` with nested integer/qubit-level sequence types in the compact charge stub. | Strict mypy and charge tests pass. | None. |
| 3.5 | Define precise aliases for the existing Hybrid canonical/Jordan–Wigner input tuples. | Strict mypy and Hybrid construction tests pass; runtime ABI unchanged. | None. |
| 4.1 | Use Structured CSR row-parallel application only above the measured `1 << 20` transition threshold; no public parallel flag. | Eager/lazy numerical differential tests pass; one-thread execution remains serial. | Same-machine release A/B: 531,441 dimensions/2,099,520 transitions improved 1.9057 to 0.5153 ms at four threads, while 16,807–117,649 dimension cases stayed within noise. |
| 4.2 | Add the authoritative native U1 `transition_count` getter and use it for warm preflight, eager accounting, and `estimated_bytes`; no Python cache. | Focused U1 cache/materialization and numerical tests pass. | Accepted as metadata/budget simplification; no standalone speedup claim. |

## Delivery and verification rules

- Keep benchmark evidence beside the approved mechanical/API fixes. The 1.1 and 4.1 production policies landed only after the release A/B measurements recorded above.
- Capture the before baseline on the same machine, release build, workload, thread count, and environment used for the after measurement. Record absolute time and measurement variability, not only percentages.
- For approved runtime changes, add numerical tests that bind outputs to trusted results; shape-only or replay-only checks are insufficient.
- Run the focused tests listed above during each slice. Before handoff or commit, run `python scripts/check.py` in the project environment, including Rust formatting/Clippy/tests, Black, Ruff, strict mypy, release extension build, Python tests, and benchmark smoke.
- Run the affected release benchmark subsets for 1.1, 3.3, 4.1, and 4.2 as specified above. Do not treat smoke timings as performance evidence and do not commit `.benchmarks/` results. The temporary A/B branches used for 1.1 and 4.1 were removed after the decisions.
- Keep this report's finding status synchronized with accepted implementations and measured decisions.

## Recommended follow-up order

1. Keep the `1 << 20` Structured CSR threshold in the release benchmark matrix to detect crossover regressions.
2. Retain the focused CSR budget, Pauli constructor, restricted-dispatch, U1 metadata, typing, and docstring tests as the maintenance guard.
3. Expand chemistry fixtures and stage-level profiling only if future workloads make integral validation/materialization the dominant cost.
4. Leave 2.3, 2.8, and 3.1 unchanged; keep 2.6 as an intentional representation-specific capability.

## Verification boundary

The original audit performed source and stub inspection against `e38533f`. This implementation pass ran Rust format, Clippy, workspace tests, strict mypy, Ruff, the full Python suite (444 passed), doctests (7 passed), focused regression tests, and release extension builds. `maturin develop --release` built successfully; its final pip-install step could not write `/Users/shixin/.local/lib` in this managed environment, while the built extension was available in the source tree for test execution. Release A/B measurements were run on the same machine; benchmark result files were not committed.
