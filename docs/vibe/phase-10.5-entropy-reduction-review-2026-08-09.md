# Phase 10.5 Entropy-Reduction and Maintainability Review, 2026-08-09

Status: Phase 10.5 capability and performance acceptance is closed, and the ER-00–ER-09 entropy-reduction program is complete in the current worktree. The owner explicitly selects one uniform raw-Fermion CPU-native default and rejects a topology-dependent raw/Jordan--Wigner runtime selector. The completed refactor preserves the measured specialized kernels.

Implementation update: the direct public-index Raw-Fermion path removes the former unchecked cache; all confirmed dead paths are deleted; Rust, PyO3, and Python structured ownership is physically split; Charge owns one prepared-plan selector; Product construction has named phases; and the final audit/quality/performance gates in Section 7.13 have passed. The current final smoke gate reports 56 Rust unit tests and 436 Python tests. The one-thread bounded QuSpin rerun covers nine directly comparable rows (all numerical preflights pass, all remain within the owner ceiling) plus the two intentional no-fair-baseline rows. Its raw local artifacts remain untracked under `.benchmarks/`.

## 1. Executive verdict

The latest implementation addresses the substance of the sixth-round review. S6-1 is closed by an explicit superseding owner decision rather than by claiming that raw Fermion wins every topology: raw descriptors remain the uniform CPU-native default because one implementation is easier to maintain and because raw wins materially on nonlocal hopping and quartic transfer, while explicit `map_fermions("jordan_wigner")` remains available when a mapped Pauli representation is deliberately requested. No automatic topology classifier or second default execution policy should be added.

S6-2 is technically closed. The committed capability matrix contains twelve directly comparable QuSpin capabilities and two explicit no-fair-optimized-baseline capabilities. The complete local run at one and four threads produced 24 directly comparable rows and four internal rows; all direct rows passed numerical preflight, all were faster than QuSpin in the recorded public-call comparison, the worst public-call ratio was approximately `0.575`, and the largest mapped-action error was approximately `2.22e-16`. The primary seven-site lazy and six-site eager/explicit Holstein workloads also remain inside the owner-approved engineering envelope.

The formal coverage is not inferred from a model name: the twelve direct manifests are `spin-full`, `spin-full-throughput`, `spin-fixed-hamming`, `fermion-full`, `boson-full`, `fermion-boson-full`, `fermion-boson-full-throughput`, `fermion-fixed-boson-spectators`, `boson-fixed-fermion-spectators`, `independent-spin-factors`, `independent-spin-factors-throughput`, and `reversed-boson-spin-factors`. `generic-coupled-charge` and `qudit-weyl` are the two internal rows because no equivalent optimized QuSpin baseline exists. The acceptance count is therefore exactly `(12 direct + 2 internal) * 2 thread settings`; adding another alias of an existing manifest is not additional capability coverage.

The complete current-worktree quality gate passes: Rust formatting, Black, stage-label validation, Clippy with warnings denied, Ruff, strict mypy, `git diff --check`, 56 Rust tests, the release native build, 436 Python tests, and the bounded benchmark smoke suites.

The former Raw-Fermion cache-budget defect is closed. `StructuredMvpPlan::apply_into` now uses the descriptor-owned direct public-index mask view for eligible one-word contiguous and Fermion-prefix layouts, with no per-call `O(D)` packed-source cache. The caller-owned `apply_into(max_bytes=0)` path consequently performs no major workspace allocation; multiword and non-contiguous layouts retain only bounded caller-owned scratch. Section 6 records the selected no-cache strategy.

The closure actions are complete: the uniform-Raw owner decision is recorded, the specification/acceptance/status documents are closed, and the behavior-preserving maintainability refactor below has passed its final audit and quality gates.

## 2. What the code-size number actually means

The tracked Rust and Python files currently total approximately 68,051 physical lines. That number combines production code, tests, benchmarks, scripts, and examples, so it overstates the production implementation.

| Area | Effective code lines by `cloc` | Interpretation |
| --- | ---: | --- |
| Rust production source and bindings | 28,138 | Core algorithms and PyO3 bindings, including unit tests embedded beside the source. |
| Installed Python package | 11,552 | Public API, facades, integrations, and Python-side plan objects. |
| Rust benchmarks | 724 | Criterion benchmark implementations. |
| Python tests | 7,236 | Correctness and API regression coverage. |
| Benchmarks | 10,504 | Benchmark implementations and frozen workload protocol. |
| Examples | 3,035 | User and research examples. |
| Scripts | 190 | Quality and repository utilities. |

Production-scale code is therefore at most about 39,690 effective lines, not 68,000; the actual production-only number is slightly lower because Rust unit tests are embedded beside source. Separate tests, benchmarks, examples, and scripts account for another 21,689 effective lines and are not dead weight merely because they contribute to the repository total.

Phase 10.5 is nevertheless the main recent entropy source. From `c50038b` through the reviewed HEAD, `crates/` plus `python/` gained 9,728 lines and deleted 1,362, a net production increase of 8,366 physical lines. Validation code gained a net 5,473 lines and vibe documentation gained a net 2,475 lines. Most of the size increase is therefore explainable, but too much of the new production code is organized as parallel live implementations rather than shared mechanisms.

The right diagnosis is not “most of the project is dead code.” The confirmed dead code is modest. The larger problem is architectural entropy: several correct and benchmark-relevant kernels expose overlapping metadata, validation, CSR application, plan wrapping, diagnostics, and read-back machinery through different implementations.

## 3. Confirmed dead code and residue that should be deleted

The following Python functions have exactly one repository occurrence, their definitions, across the package, tests, benchmarks, scripts, and examples. They are abandoned reference helpers and should be deleted rather than retained for hypothetical future use.

| File and location | Dead symbol | Required action |
| --- | --- | --- |
| `python/tencirpauli/structured.py:483` | `_aggregate_terms` | Delete the function and any imports made unused by its removal. Canonical aggregation is native-authoritative. |
| `python/tencirpauli/structured.py:1766` | `_multiply_terms` | Delete the complete unused Python multiplication path. Native handle multiplication is authoritative. |
| `python/tencirpauli/structured.py:1922` | `_adjoint_qubit_codes` | Delete the identity helper. |
| `python/tencirpauli/structured.py:1962` | `_codes_on_space_axes` | Delete the unused Python-axis reconstruction helper. |
| `python/tencirpauli/u1_circuit.py:138` | `_normalize_ps` | Delete the abandoned observable-normalization path. |

There is also one confirmed dead parallel representation centered in `python/tencirpauli/hamiltonian.py`. `_generic_entries` is never populated by any factory, yet it is stored on both `NativeMVPPlan` and `BackendMVPPlan`, checked by their apply methods, and interpreted by `_apply_generic_entries`; `python/tencirpauli/structured.py:_with_plan_metadata` copies the value forward. Delete both fields, the `NativeMVPPlan.generic_entries` constructor parameter, the Structured copy-through, all three unreachable apply/apply-into branches, and `_apply_generic_entries`. This removes a silent Python sparse fallback that cannot occur in the supported factory graph and restores one authoritative native/backend representation per plan.

The following Rust fragments are not independent live features and should be removed in the same P0 cleanup:

- `crates/tencir-pauli-core/src/structured.rs:2274`: replace the infallible `checked_integer_to_f64(value, context)` wrapper and its discarded `context` with the ordinary casts selected by the project numeric policy, then delete the helper.
- `crates/tencir-pauli-core/src/structured.rs:3238`: delete the computed-and-discarded Weyl `a` value and `let _ = a`.
- `crates/tencir-pauli-core/src/charge.rs:4409`: delete `term.mapped_present && !term.mapped_codes.is_empty()` because the immediately preceding `term.mapped_present` already makes the clause redundant.

These deletions should be one small commit. Run formatting, Ruff/mypy, Clippy, the focused structured/U1/public-plan tests, and then the ordinary full quality gate. No performance rerun is necessary for pure unreachable-code deletion unless generated code or hot assembly changes unexpectedly.

## 4. Largest entropy concentrations

| Module | Physical lines | Main responsibilities currently mixed together |
| --- | ---: | --- |
| `crates/tencir-pauli-core/src/structured.rs` | 5,361 | Batch algebra, chemistry ingestion, canonicalization, mapping, lazy compilation, raw Fermion descriptors, eager CSR, dense/COO/CSR targets, diagnostics, and tests. |
| `crates/tencir-pauli-core/src/charge.rs` | 5,313 | Generic charge schedules, accumulators, CSR, Qubit/Qudit/Boson/Fermion/Product specialized plans, public indexing, diagnostics, and tests. |
| `python/tencirpauli/structured.py` | 3,551 | Public words and terms, spaces, all structured operator families, algebra facades, mapping, compilation, and native marshalling. |
| `crates/tencirpauli-native/src/structured.rs` | 2,523 | Fermion, Boson, and Hybrid handles; algebra methods; materialization; hashing/equality; and structured compilation bindings. |

`build_fast_product_mvp_plan` alone spans about 508 lines, and the workspace contains 32 `clippy::too_many_arguments` suppressions. Neither metric proves bad code by itself, but together they show that data ownership and phase boundaries are not explicit enough.

The first structural refactor should be a behavior-preserving module split, not an algorithm rewrite. Rust module boundaries compile away and should not cost runtime. The authoritative target files, symbol ownership, and migration restrictions are fixed in ER-02, Section 7.5; do not invent a parallel directory layout during implementation.

For Product construction, introduce one borrowed `ProductBuildInput` and a `ProductPlanBuilder` with named phases: discover independent components, build factor ranks and the public indexer, compile actions, group schedules, account memory, and publish the immutable plan. This removes argument bundles and makes abandoned intermediate paths harder to retain. The final plan layout and apply kernel remain unchanged during this step.

## 5. Duplicated live paths that should converge

### 5.1 CSR application

There are at least three destination-row CSR MVP implementations: the parallel shared charge path in `charge.rs:186`, the packed U1 path in `sector.rs:700`, and the serial Structured eager path in `structured.rs:3058`. They differ in index type, validation placement, parallel threshold, and arithmetic spelling, so fixes and tuning can drift.

Converge them on one generic `CsrGraph<I>` and one `csr::apply_into` source implementation with compile-time monomorphization. Row pointers use native `usize`; column type `I` remains `usize` for U1/Charge and `u64` for Structured so hot execution and large flat read-back preserve their current best representation. Never use trait objects in the hot loop. Preserve each current parallel crossover as explicit policy metadata. Pauli compact-graph execution is a different representation and should not be forced through CSR.

### 5.2 Python plan facades and metadata

`NativeMVPPlan`, `ChargeMvpPlan`, `ChargeLazyMvpPlan`, and `U1MvpPlan` repeat immutable metadata, state/buffer validation, `apply`, `apply_into`, callable behavior, SciPy wrapping, memory fields, and native-handle forwarding.

Create one private frozen `NativePlanMetadata` and one `_NativeMvpFacade` method implementation. Preserve `NativeMVPPlan`, `ChargeMvpPlan`, `ChargeLazyMvpPlan`, and `U1MvpPlan` as distinct public advanced classes; they inherit the shared methods and retain only their domain-specific fields and capabilities. They are not aliases. ER-03, Section 7.6, fixes the exact field and compatibility-property layout.

### 5.3 Packed U1 and general charge

Packed U1 and general additive-charge execution are parallel stacks, but their basis construction and compilation algorithms are genuinely different. Keep `PackedU1Basis`, `U1BasisIterator`, `U1Sector`, `U1LazyMvpPlan`, and the packed U1 term compiler in `sector.rs` separate from `ChargeSectorPlan`, compiled charge schedules, Product classification, and the general charge kernel selector. Do not place packed U1 inside the general `FastChargeMvpPlan` enum.

Unify only the layers that are semantically identical: the owned validated CSR graph, destination-row CSR apply, flat CSR/COO read-back, common Python MVP facade methods, and plan metadata. The packed U1 kernel and general charge kernels remain separate producers that publish the same graph/facade contract. This preserves arbitrary-width packed rank/unrank and avoids a universal compiler full of U1-specific branches.

### 5.4 Structured, Product, and Generic diagnostics

Structured counters are always allocated and atomically incremented in scalable apply loops. Product counters are always present, and the general Product path still contains inner-loop atomic updates; Generic charge counters are also retained and wired into ordinary execution. These counters were useful for review evidence, but review instrumentation should not permanently define production hot loops.

Accumulate diagnostic deltas in ordinary stack-local integers and publish them once per call, as the new contiguous two-factor path already does. ER-06 first tests always-collected local statistics; only if that measured candidate causes a large regression does the package select diagnostics-disabled production plus a private diagnostic constructor. Inner-loop atomics are deleted in either outcome.

### 5.5 PyO3 structured handles

Fermion, Boson, and Hybrid native handles repeat add, scale, multiply, commutator, anticommutator, adjoint, content hash, materialization, equality, and GIL-release scaffolding. Keep the three Python-visible types and payloads, and share only GIL release, budget conversion, error mapping, and same-layout checks. Domain algebra, adjoints, hashes, and materialization remain explicit where their data differs. ER-07, Section 7.10, fixes the exact boundary; do not introduce a universal handle trait or a macro that merely hides three full implementations.

### 5.6 Chemistry integral ingestion

`count_spin_orbital_terms` and `count_spin_block_terms` duplicate the same O(n^4) formulas later used by the append passes. The prepass currently protects allocation and provides exact capacity, so deleting it without measurement is not justified.

Create one authoritative visitor for each integral layout and invoke it first with a counting sink and then with a materializing sink. This removes duplicated formulas while preserving the current two-pass memory and performance behavior. A one-pass growable builder is outside this entropy-reduction program and must not be introduced while applying the report.

### 5.7 Benchmark protocol

The capability work added substantial and valuable benchmark code, but environment-specific child runners repeat artifact, timing, provenance, and record-shape logic. Move library-neutral manifest validation, canonical occupation ordering, timing-record construction, and artifact I/O into one importable protocol module; leave only operator construction and library-specific apply calls in the TenCirPauli and QuSpin children. Preserve isolated interpreters and flat JSON/NPZ exchange.

## 6. Mandatory cache-budget remediation

The raw-Fermion implementation currently builds a complete `Vec<u64>` packed-source cache inside every applicable `StructuredMvpPlan::apply_into` call. The measured source-packing share is small, so repeatedly materializing and reading the cache is not obviously the best long-term kernel, and its `8D` bytes are ignored by `_max_bytes`.

Preferred solution: complete the construction-time one-word public-index mask view already described by the sixth-round review. Derive public-orientation occupied, empty, parity, and flip masks from the authoritative `FermionMaskDescriptor`; gather each destination directly using `source = destination XOR flip`; keep CAR lowering and sign derivation solely in the descriptor; and eliminate the `D`-entry cache. Retain the width-generic descriptor path for multiword or non-contiguous layouts. This preserves the owner's uniform Raw policy without adding a Raw/JW selector.

Fallback solution if the direct-mask A/B loses materially: retain the current complete packed-source cache, but calculate its exact major workspace with checked arithmetic before allocation, reject the call when `max_bytes` is smaller, and expose the workspace bytes in plan diagnostics. For a contiguous full-space plan the estimate is `D * size_of::<u64>()`; for a Fermion-prefix plan it is `fermion_dimension * size_of::<u64>()`; the small width-generic descriptor scratch is accounted separately. A destination-local chunk cache is not an approved candidate because `source = destination XOR flip` can leave the destination chunk and would require either redundant reconstruction or a more complex cross-chunk policy. Do not retain an unchecked cache and do not move the cache into persistent plan storage merely to hide a per-call allocation.

Exit tests are numerical equality against explicit Jordan--Wigner and dense references for hopping, density, quartic, complex, interleaved, and multiword cases; no per-term allocation growth; `apply_into(max_bytes=0)` either performs no major allocation or fails according to one documented contract; and representative Raw performance stays in the same engineering class. The owner does not require topology-by-topology Raw wins and no runtime mapping selector should be introduced.

## 7. Authoritative implementation plan

This section is the executable specification for the entropy-reduction work. It supersedes any broader wording elsewhere in this report. Each work package is one review unit; do not combine a mechanical module move with an algorithm change, and do not retain both old and new production paths after a package is accepted.

### 7.1 Fixed merge boundary and abstraction budget

The program follows one rule: unify contracts and mechanics that are semantically identical; keep algorithms and representations separate when they encode different complexity, data layout, or measured performance. “One class” or “one enum” is not a goal by itself.

| Area | Fixed decision | Exact stopping point |
| --- | --- | --- |
| Python MVP plans | Merge common metadata, input/output validation, native `apply` forwarding, callable behavior, and SciPy wrapping. | Keep `NativeMVPPlan`, `ChargeMvpPlan`, `ChargeLazyMvpPlan`, and `U1MvpPlan` as distinct public advanced types with their domain metadata and materialization capabilities. |
| CSR | Merge validated generic CSR storage, row scheduling, apply, and flat read-back logic. | Keep `usize` columns for U1/Charge and `u64` columns for Structured; keep their three construction algorithms separate and keep Pauli compact graph outside CSR. |
| Charge execution | Merge one general-charge ownership object, one selector, and shared builder inputs. | Keep Fermion, Qubit, Qudit/Boson additive, Product, and Generic numerical kernels as enum variants; keep packed U1 outside the general-charge enum. |
| Raw Fermion | Merge lowering/sign authority in `FermionMaskDescriptor` and use one uniform Raw default. | Keep width-generic and eligible one-word application kernels because their machine representation differs; do not add automatic Raw/JW selection. |
| Structured handles | Merge GIL release, budget conversion, error mapping, and same-layout checks. | Keep Fermion, Boson, and Hybrid payloads, canonicalization, multiplication, adjoint, and materialization separate. |
| Diagnostics | Merge per-call local statistics and one publication mechanism. | Keep different stat fields per kernel; do not create a universal string-keyed metrics map in hot code. |
| Chemistry | Merge duplicated formulas through one visitor per input layout. | Keep spin-orbital and spin-block visitors separate because their indexing and mode-order semantics differ. |
| QuSpin benchmarking | Merge timing, artifacts, record schema, provenance, and validation. | Keep TenCirPauli and QuSpin processes and their operator builders separate; do not invent a common operator DSL. |

The abstraction budget is fixed as follows:

- A public `apply_into` may perform at most one enum dispatch before entering its selected kernel. No trait object, callback, hash lookup, or per-term variant switch is allowed in a scalable hot loop.
- A construction-time selector may choose a documented kernel variant, but it may not retain losing variants for runtime retry. Memory failure is reported; it does not trigger a second implementation.
- A shared abstraction is accepted only when it deletes an existing implementation or makes one invariant authoritative. Renaming wrappers around unchanged duplicate logic does not count.
- When a shared mechanism causes a repeatable large regression, retain specialization at the smallest layer demonstrated by profiling. For example, two row-arithmetic functions inside one CSR scheduler are acceptable; two complete CSR stacks are not.
- A specialized kernel remains only for a distinct capability or a material measured advantage. Low-teens timing differences are the same engineering class and favor the simpler shared mechanism; large repeatable differences favor the specialized kernel.
- File splitting and compile-time generic helpers must have zero runtime policy cost. Do not add factories, registries, or generic intermediate representations merely to make the source tree look symmetrical.

Only three decisions in this plan are intentionally left to A/B evidence: direct public-index Fermion masks versus one checked complete packed-source cache; one versus two CSR row-arithmetic functions; and always-collected local diagnostics versus diagnostics-disabled production. Every other merge/separation boundary above is fixed and should not be reopened during implementation.

### 7.2 Dependency order and pull-request boundaries

| Package | Depends on | Required result | Performance decision? |
| --- | --- | --- | --- |
| ER-00 | None | Remove or explicitly budget the raw-Fermion `O(D)` apply cache and close Phase 10.5 documentation. | Yes: direct public-index masks versus one checked complete cache. |
| ER-01 | None | Delete every confirmed dead Python path and Rust residue in Section 3. | No. |
| ER-02 | ER-01 | Split giant Rust, PyO3, and Python modules without changing behavior or storage. | No. |
| ER-03 | ER-02 | Introduce one Python native-MVP facade implementation while preserving the four public advanced return-type names. | No design A/B; run one latency sanity check. |
| ER-04 | ER-02 | Introduce one validated native CSR graph and one destination-row apply implementation. | Only the row-arithmetic choice requires A/B; existing parallel policies are preserved. |
| ER-05 | ER-02, ER-04 | Move general-charge plan ownership and selector dispatch into one core plan; retain all measured kernel variants. | No design A/B; selector and final hot-path regression checks are mandatory. |
| ER-06 | ER-05 | Replace inner-loop atomic diagnostics with per-call local reductions and one publication point. | Yes: local counters versus diagnostics-disabled production. |
| ER-07 | ER-02 | Consolidate PyO3 handle scaffolding without merging domain algebra or materialization payloads. | No. |
| ER-08 | ER-02 | Replace duplicated chemistry count/append formulas with two authoritative visitors. | No for the two-pass visitor; one-pass ingestion is explicitly outside this package. |
| ER-09 | ER-03 | Consolidate benchmark timing/artifact protocol without merging library-specific operator builders. | No. |

ER-01 and ER-00 may proceed independently. ER-02 starts only after ER-01 so dead functions are not moved into the new tree. ER-03, ER-04, and ER-07 may proceed independently after ER-02. ER-05 must use the ER-04 graph abstraction rather than introducing another eager graph. ER-06 follows ER-05 so diagnostics are fixed once in the final charge ownership model.

### 7.3 ER-00 — Raw-Fermion workspace and formal closure

#### Production implementation

Add a private construction-time view in `crates/tencir-pauli-core/src/fermion_mask.rs`:

```rust
#[derive(Clone, Copy)]
pub(crate) struct ContiguousFermionAction {
    required_occupied: usize,
    required_empty: usize,
    flip_mask: usize,
    source_parity_mask: usize,
    fixed_sign: i8,
}

impl FermionMaskDescriptor {
    pub(crate) fn contiguous_public_action(&self) -> Option<ContiguousFermionAction>;
}
```

`contiguous_public_action` is eligible only when there is one descriptor word, `mode_axes.len() <= usize::BITS`, and `mode_axes[mode] == mode` for every mode. It converts each logical-mode mask into public-index orientation once using the same bit reversal already used by `pack_contiguous_binary_index` and `apply_flip_to_contiguous_binary_index`. It does not accept a second mode-count argument that could disagree with the descriptor, redo CAR lowering, change `fixed_sign`, or introduce a second sign convention.

Add two inline operations on the view: `source_for_destination(destination) -> usize`, which returns `destination ^ flip_mask`, and `amplitude(source, coefficient) -> Option<Complex64>`, which performs the two requirement comparisons, one parity popcount, and fixed-sign application. Store `contiguous_public: Option<ContiguousFermionAction>` inside `FermionMaskDescriptor`, compute it once in `from_steps` when the identity-axis/one-word conditions hold, and make `contiguous_public_action()` return that stored copy. Do not add a parallel action array to `StructuredMvpPlan`. Plan construction declares a full-space or prefix plan eligible only when every Fermion descriptor it will execute has this view. `StructuredMvpPlan::contiguous_fermion_destination_value` reads the first group member's view for the shared source index and each member's view for amplitude; `apply_compiled_schedule_member` reads the descriptor view for the Fermion-prefix case instead of reading a packed-source cache. If the direct candidate wins, delete `StructuredScheduleGroup.contiguous_source_flip_mask` because the descriptor view becomes the sole public-orientation flip authority. If the cache candidate wins, delete `contiguous_public` and keep the existing group flip field.

Delete `apply_contiguous_fermion_cached_gather_into`, its `packed_sources: Vec<u64>`, and the `prefix_sources` allocation in `StructuredMvpPlan::apply_into` when the direct view wins the A/B below. The prefix case uses the same view on `fermion_source = public_index / suffix_dimension` and retains only `suffix_dimension`; it must not cache every Fermion-prefix source. The existing width-generic `pack_index` plus caller-owned `fermion_mask_scratch` path remains authoritative for multiword or non-contiguous layouts.

#### Required A/B and selection rule

Implement the direct view and compare it once against the current complete-cache kernel after adding checked workspace accounting to that kernel; do not expose a runtime strategy. Use the existing 12-mode hopping/density workload and the 18-mode local-chain, long-range-hopping, and nonlocal-pair-transfer workloads at one and four threads. A short steady-state run sufficient to distinguish noise-level/low-teens movement from a large repeatable regression is enough; do not repeatedly rerun the matrix to police an exact percentage. Record apply median, allocation count, workspace bytes, and numerical error against explicit Jordan--Wigner.

- Select the direct view if none of the four representative classes shows a large repeatable regression against the complete cache. Delete the complete-cache kernel and its workspace-accounting branch.
- Select the complete cache only if profiling attributes a large repeatable advantage to source packing reuse on at least one representative class and it is not materially worse on the others. Add one `checked_contiguous_cache_bytes(&self, max_bytes: u128) -> Result<usize, PauliError>` helper that chooses `cache_len = dimension` for the pure contiguous layout or `cache_len = dimension / suffix_dimension` for the prefix layout, performs checked byte multiplication, and returns `MemoryLimit` before allocation when the budget is smaller. Add `estimated_apply_workspace_bytes() -> u128` on the core plan and a private native `debug_apply_workspace_bytes` getter so the exact selected cache size is testable; it returns zero on plans that do not select this cache. Keep the cache per call; do not store it on the plan. Delete the direct-view apply kernel after the decision, while retaining the descriptor's existing public-index helpers only if another live caller uses them.
- If results are within the low-teens band, select the direct view because it has no scalable workspace and fewer states to maintain.

#### Tests and closure edits

Add core tests for public-bit orientation at 1, 8, and `usize::BITS` modes, fixed sign, invalid occupied/empty sources, destination inversion, and equality with the width-generic descriptor. Preserve the mode-128 multiword tests. Update `tests/test_execution_invariants.py` so caller-owned `apply_into(max_bytes=0)` proves the selected direct path allocates no major workspace; if the complete cache wins, change the success case to pass its documented exact workspace, add a failing zero-budget case, and cover both full-space and Fermion-prefix cache lengths.

After the implementation passes, append the uniform-Raw decision to `phase-10.5-owner-decision-log.md`, replace the open status in `phase-10.5-spec.md`, `phase-10.5-acceptance-evidence.md`, and `implementation-status.md`, and record that raw/JW topology results are descriptive evidence rather than a runtime selector.

### 7.4 ER-01 — Exact dead-code deletion

Delete the following in one change and do not add replacement helpers:

1. `_aggregate_terms`, `_multiply_terms`, `_adjoint_qubit_codes`, and `_codes_on_space_axes` from `python/tencirpauli/structured.py`.
2. `_normalize_ps` from `python/tencirpauli/u1_circuit.py`.
3. `_generic_entries` from `NativeMVPPlan` and `BackendMVPPlan`, the `generic_entries` constructor parameter, the `_with_plan_metadata` copy-through in `python/tencirpauli/structured.py`, the three unreachable branches (`NativeMVPPlan.apply`, `NativeMVPPlan.apply_into`, and `BackendMVPPlan.apply`), and `_apply_generic_entries` from `python/tencirpauli/hamiltonian.py`.
4. `checked_integer_to_f64` from core Structured code; replace each call with the ordinary `i128 as f64` cast already selected by the project numeric policy.
5. The unused Weyl `a` computation and `let _ = a` in `compile_mvp_operations`.
6. The redundant `term.mapped_present && !term.mapped_codes.is_empty()` clause in `build_fast_fermion_mvp_plan`.

Use Ruff/Clippy to remove imports made unused by the deletion. Required verification is `cargo test --workspace`, `pytest tests/test_structured_algebra.py tests/test_u1_circuit.py tests/test_hamiltonian.py tests/test_public_api.py`, strict mypy, and `git diff --check`. No performance run is required because no reachable operation changes.

### 7.5 ER-02 — Mechanical module split

This package moves code only. Do not rename public types, change function signatures, alter visibility beyond the minimum `pub(super)` needed by sibling modules, change field order, or modify kernels.

#### Core Structured target tree

Keep `crates/tencir-pauli-core/src/structured.rs` as a facade containing submodule declarations and the existing public re-exports. Target files and ownership are:

| Target file | Move these existing responsibilities and symbols |
| --- | --- |
| `structured/types.rs` | `FermionSpinOrdering`, `FermionSpinBlocks`, `FermionIntegralSource`, `FermionBatch`, `HybridBatch`, `HybridRawBatch`, canonical result structs, `HybridLayout`, and `StructuredOperation`. |
| `structured/hybrid.rs` | Embedding, charge analysis/conservation, hybrid multiplication/canonicalization, hybrid Jordan--Wigner, and hybrid aggregate helpers. |
| `structured/fermion.rs` | Fermion canonicalization, multiplication/binary algebra, Jordan--Wigner word expansion, CAR rewrite, and Fermion aggregate helpers. |
| `structured/boson.rs` | Boson canonicalization, multiplication/binary algebra, CCR rewrite, block products, and Boson aggregate helpers. |
| `structured/integrals.rs` | Integral validation, Hermitian averaging, index/order helpers, term enumeration, and `canonicalize_fermion_integrals`. |
| `structured/execution/common.rs` | Compiled operation/net-action/schedule types, mixed-radix utilities, input validation, grouping, packed-Pauli detection, and `StructuredExecutionFamily`. |
| `structured/execution/lazy.rs` | `StructuredMvpPlan`, `FermionMaskPhaseProfile`, schedule apply, lazy builders, and lazy diagnostics. |
| `structured/execution/eager.rs` | `StructuredEagerMvpPlan`, `StructuredSparseResult`, CSR construction, dense/COO/CSR materialization, and eager builders. |
| `structured/tests.rs` | The current `#[cfg(test)]` tests, split further only when a file exceeds roughly 1,000 lines. |

Shared exact-zero, deterministic-sum, coefficient validation, byte-check, and Pauli-code helpers go in `structured/common.rs`; there must be exactly one copy. `fermion_mask.rs` remains a separate module because charge and Structured execution both consume it.

#### Core Charge target tree

Keep `crates/tencir-pauli-core/src/charge.rs` as the facade and use:

| Target file | Move these existing responsibilities and symbols |
| --- | --- |
| `charge/types.rs` | `ChargeTransitionTerm`, `ChargeCsrResult` until ER-04 replaces its arrays with the shared graph, and public metadata types. |
| `charge/schedule.rs` | `PreparedChargeTransitionPlanLayout`, `CompiledChargeSchedule`, position/validation/compile helpers, fixed-width accumulators, and generic schedule application. |
| `charge/kernels/common.rs` | `FastProductLayout` under its existing name, mixed-radix helpers, shared Boson local actions, and common factor-action aliases. A semantic rename, if still useful, belongs in ER-05 rather than this move-only package. |
| `charge/kernels/qubit.rs` | `FastQubitMvpPlan`, combination rank/unrank helpers used only by it, and `build_fast_qubit_mvp_plan`. |
| `charge/kernels/additive.rs` | `FastAdditiveMvpPlan`, shared Qudit/Boson factor-action construction, `build_fast_qudit_mvp_plan`, and `build_fast_boson_mvp_plan`. |
| `charge/kernels/fermion.rs` | `FastFermionSectorIndex`, `FastFermionMvpPlan`, its term/layout structs, and `build_fast_fermion_mvp_plan`. |
| `charge/kernels/product.rs` | Every `Product*` action/index/group type, `FastProductMvpPlan`, Product apply helpers, and `build_fast_product_mvp_plan`. |
| `charge/kernels/mod.rs` | The selected-kernel enum and its one-call dispatch/metadata methods. |
| `charge/generic.rs` | Generic lazy apply and destination aggregation. |
| `charge/eager.rs` | Generic eager graph compilation and CSR construction profiling. |
| `charge/tests.rs` | Current charge unit tests. |

Do not move `PackedU1Basis`, `U1BasisIterator`, or the U1 term compiler out of `sector.rs` in this package.

#### PyO3 and Python target trees

Keep `crates/tencirpauli-native/src/structured.rs` as a registration/re-export facade. Move shared flat-array conversion and error/GIL scaffolding to `tencirpauli-native/src/structured/common.rs`; move the three handles to `fermion_handle.rs`, `boson_handle.rs`, and `hybrid_handle.rs`; move `StructuredMvpPlan`, `StructuredEagerMvpPlan`, and structured target functions to `plans.rs`. Do not combine handle payload structs.

Create the private package `python/tencirpauli/_structured/` with `words.py`, `space.py`, `base.py`, `fermion.py`, `boson.py`, `hybrid.py`, `mapping.py`, and `compile.py`. Keep `python/tencirpauli/structured.py` as the stable public re-export module, not as a second implementation. Move code once; imports within the package must point to `_structured`, while external imports continue to use `tencirpauli.structured` or package-root names. Moving a public class changes its default `__module__`, so the facade must explicitly restore `__module__ = "tencirpauli.structured"` for every name in the existing `structured.__all__`; add regression assertions for `__module__`, import identity, repr, and pickle round trips of the value classes. Do not create facade subclasses merely to recover the module name.

Each mechanical move must pass formatting and the focused module tests. Run the complete quality gate once after the entire ER-02 tree is assembled, not after every file move.

### 7.6 ER-03 — One Python native-MVP facade implementation

Add `python/tencirpauli/_native_mvp.py` with this exact common metadata object:

```python
@dataclass(frozen=True, slots=True)
class NativePlanMetadata:
    dimension: int
    term_count: int
    storage: Literal["lazy", "eager"]
    basis_ordering: str
    target: Literal["native_mvp"]
    execution_family: str
    native_variant: str
    retained_format: str
    source_term_count: int
    schedule_count: int
    estimated_retained_bytes: int
    estimated_construction_peak_bytes: int
    transition_count: Optional[int]
```

Add a private `_NativeMvpFacade` method mixin with `__slots__ = ()`. Each concrete class declares and stores exactly one `_execution` field and one `_metadata` field using its existing frozen-dataclass or locked-slots immutability mechanism; the mixin must not add a `__dict__` or replace observable public equality/repr behavior. It owns the shared metadata properties, strict input/output validation, `apply`, `apply_into`, `__call__`, and `to_scipy_linear_operator`. Its apply methods read `self._execution`, call exactly one native handle method, and perform no term materialization. Reuse `_validate_max_bytes`, `_effective_max_bytes`, `_check_allocation`, and `_validate_apply_into_buffers` by moving them into this module; delete their duplicate imports and implementations elsewhere.

`NativePlanMetadata.__post_init__` performs the common non-negative integer checks, validates `storage`, `target`, and non-empty family/variant/format strings, and enforces `transition_count is None or transition_count >= 0`; subclass constructors validate only their domain fields before creating this object. The mixin exposes every metadata field plus the compatibility property `estimated_bytes = estimated_retained_bytes`. `apply` validates a one-dimensional complex128 input of length `dimension`, accounts for the newly allocated `dimension * sizeof(complex128)` output against `max_bytes`, makes at most one contiguous input copy, and calls `_execution.apply`. `apply_into` accepts only the existing strict C-contiguous complex128, non-aliasing caller buffers, allocates no state-sized Python output, and calls `_execution.apply_into`. Do not make `apply` call `apply_into` through a Python-allocated buffer, because that would change native allocation accounting and add an extra dispatch.

Keep the four advanced public names and their distinct domain metadata:

- `NativeMVPPlan(_NativeMvpFacade)` retains `nqubits`, `local_dimensions`, mapping, Boson cutoffs/boundary, Qudit dimension/convention, schema version, and `plan_term_count`. Its private `_native_plan` property returns `_execution` so existing internal callers do not gain a second stored handle.
- `ChargeMvpPlan(_NativeMvpFacade)` is the eager charge facade and adds only `_csr_arrays`, `_coo_arrays`, and the four read-back properties. Its `_native_plan` property returns `_execution`.
- `ChargeLazyMvpPlan(_NativeMvpFacade)` retains the sector-plan lifetime owner separately as `_sector_plan`; its compatibility `_native_plan` property returns `_sector_plan`, and `_native_execution` returns `_execution`. All numerical apply logic comes from the base.
- `U1MvpPlan(_NativeMvpFacade)` retains only `sector`; `_native_plan` returns `_execution`.

Do not make these names aliases: `tests/test_scipy_linear_operator.py` and advanced API users observe their type names. Do not move `BackendMVPPlan` under the base because it is an array/backend executor with different apply semantics. Delete the four duplicate implementations of apply/apply_into/call/SciPy wrapping only after all subclasses use the base.

Add one `_read_csr_arrays(execution, dimension, max_bytes) -> CSRMatrix` helper and one `_read_coo_arrays(...) -> COOMatrix`; eager facades call them, while lazy facades expose no materialization method. Native read-back remains flat NumPy arrays.

Required tests are `tests/test_public_api.py`, `tests/test_scipy_linear_operator.py`, `tests/test_native_mvp_resources.py`, `tests/test_hamiltonian.py`, `tests/test_charge.py`, and `tests/test_symmetry.py`, plus strict mypy. Run one bounded public-call latency sanity check over Pauli, Charge lazy/eager, and U1 plans. There is no competing design: if Python overhead increases materially, remove avoidable property construction or copying inside the base rather than restoring four implementations.

### 7.7 ER-04 — One validated CSR graph and apply kernel

Add the following statically dispatched core type to `crates/tencir-pauli-core/src/csr.rs`:

```rust
pub trait CsrIndex: private::Sealed + Copy + Send + Sync {
    fn checked_usize(self) -> Option<usize>;
    fn trusted_usize(self) -> usize;
    fn to_u64(self) -> Result<u64, PauliError>;
    fn into_u64_vec(values: Vec<Self>) -> Result<Vec<u64>, PauliError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct CsrGraph<I: CsrIndex> {
    dimension: usize,
    indptr: Vec<usize>,
    columns: Vec<I>,
    values: Vec<Complex64>,
}

#[derive(Clone, Copy)]
pub struct CsrApplyPolicy {
    pub parallel: bool,
    pub min_parallel_entries: usize,
    row_arithmetic: CsrRowArithmetic,
}
```

Seal `CsrIndex` and implement it only for `usize` and `u64`; mark its scalar methods `#[inline]`, and never use the trait as `dyn CsrIndex`. `CsrGraph::try_new(dimension, indptr, columns, values)` takes ownership and uses `checked_usize` to validate pointer length, monotonic pointers, final entry count, column range, platform conversion, and equal column/value lengths once. Expose read-only `dimension()`, `transition_count()`, `estimated_bytes()`, `indptr()`, `columns()`, and `values()` accessors rather than letting consumers retain parallel arrays. The hot row loop then uses the invariant-backed `trusted_usize` cast and ordinary safe slice indexing; it must not return or branch on a per-entry conversion `Result`. `into_u64_vec` converts `usize` columns once and returns the input vector unchanged for `u64`, which is how consuming Structured read-back remains zero-copy without specialization or `unsafe`.

Give `CsrApplyPolicy` public `const fn serial()` and `const fn parallel(min_parallel_entries)` constructors; they select `ManualRealImag`, so consumers do not depend on the private arithmetic enum. Give the graph an exact repeated-execution entry point `apply_into(&self, state: &[Complex64], output: &mut [Complex64], policy: CsrApplyPolicy) -> Result<(), PauliError>`. It validates only state/output lengths, chooses serial or destination-row Rayon scheduling from the policy, and performs no graph rescan or allocation. Define `CsrRowArithmetic` privately with `ManualRealImag` as the initial and normally sole variant; add `NativeComplex` and one `pub(crate)` policy constructor only if the ER-04 A/B selects it. `to_u64_arrays(max_bytes)` is the sole borrowed flat read-back conversion, `into_u64_arrays(self, max_bytes)` delegates column ownership to `I::into_u64_vec`, and `to_coo_u64(max_bytes)` derives rows without rebuilding or reaggregating the operator.

Keep `CsrSink` and the public `ChargeCsrResult` column arrays in `u64` so the independently publishable core API and existing flat construction result do not break. Add `CsrGraph::<usize>::try_from_u64(dimension, indptr, columns, values)` as the one consuming conversion used by the native eager Charge plan; this is the same one-time conversion Charge already performs, now owned by the graph implementation. Structured constructs `CsrGraph<u64>` directly. U1 constructs `CsrGraph<usize>` directly.

Keep the public `apply_charge_csr_into` symbol as a one-line compatibility wrapper over the shared slice apply helper; it must contain no validation, scheduling, or arithmetic implementation of its own. Remove it from native-plan internals after `NativeChargeEagerMvpPlan` owns a validated graph.

Migrate consumers in this order:

1. `NativeChargeEagerMvpPlan` stores `CsrGraph<usize>` plus its existing construction-profile scalar fields; delete its independent `indptr`, `columns`, `values`, `apply_values`, and inline `from_csr` conversion loops. Call `CsrGraph::<usize>::try_from_u64` once during construction and preserve the current charge parallel threshold in `CsrApplyPolicy`.
2. `StructuredEagerMvpPlan` stores `CsrGraph<u64>`; delete its separate row loop while preserving its zero-copy consuming `u64` column read-back. Set `parallel=false` to preserve behavior. Enabling Structured eager parallel apply is not authorized by this report.
3. `U1MvpPlan` stores `Arc<CsrGraph<usize>>` plus `U1Sector` so cloned plans share the graph; delete its row loop and duplicate CSR conversion. Preserve `U1_PARALLEL_TRANSITION_THRESHOLD` in the policy.
4. `U1RestrictedOperator::dense/coo/csr` and charge/Structured materializers consume graph read-back methods. They must not re-evaluate terms.

Keep these producers separate: the U1 two-pass combinatorial builder, the general charge chunked `CsrSink` builder, and the Structured destination-row builder. They solve different construction problems but publish the same graph. Keep Pauli `compact_graph` separate because it is not CSR.

Use the current manual real/imag row accumulation from `apply_charge_csr_row` as the first shared arithmetic implementation. Run targeted release A/B for U1 MVP, Structured eager, and general charge eager against their prior implementations. If every representative case is within the low-teens engineering band or faster, delete all old row kernels and keep `CsrRowArithmetic` private with only `ManualRealImag`. If one family has a large repeatable regression and profiling attributes it specifically to complex-row arithmetic, add exactly one second private row function, `apply_row_native_complex`, and a second enum variant selected once before row scheduling; validation, scheduling, storage, and read-back still remain single implementations. No family-specific complete CSR apply function may remain, and the enum must not become a public tuning option.

Required correctness covers empty rows, sorted columns, exact-zero omission, cross-chunk cancellation, caller-owned overwrite, serial/parallel equality, U1 dense/COO/CSR equality, Structured dense/CSR equality, and charge eager/lazy equality. Run `cargo bench ... symmetry -- --test` and the bounded `representation_ab.py`; run the six-site eager workload only once at the end of ER-04.

### 7.8 ER-05 — General charge ownership and selector consolidation

This package consolidates general additive-charge execution only; packed U1 remains in `sector.rs`.

In core Charge code, replace `FastChargeMvpPlan` plus an outer `Option` with:

```rust
pub enum ChargeLazyKernel {
    Generic,
    Fermion(FastFermionMvpPlan),
    Qubit(FastQubitMvpPlan),
    Qudit(FastAdditiveMvpPlan),
    Boson(FastAdditiveMvpPlan),
    Product(FastProductMvpPlan),
}

pub struct PreparedChargeMvpPlan {
    sector: Arc<ChargeSectorPlan>,
    layout: PreparedChargeTransitionPlanLayout,
    terms: Box<[ChargeTransitionTerm]>,
    termwise_conserved: bool,
    kernel: ChargeLazyKernel,
}
```

`PreparedChargeMvpPlan` owns `estimated_bytes`, `execution_family`, `native_variant`, `schedule_count`, `apply`, `apply_into`, `compile_eager`, and diagnostic snapshot methods. `NativeChargeMvpPlan` becomes a PyO3 wrapper with one field, `plan: PreparedChargeMvpPlan`; delete its duplicate ownership and the free `apply_charge_mvp_from_prepared_plan*` public surface after internal callers migrate.

Introduce the following borrowed build context and pass it by shared reference; it is a construction-only argument bundle and is never stored in a kernel:

```rust
struct ChargeBuildContext<'a> {
    sector: &'a ChargeSectorPlan,
    layout: &'a PreparedChargeTransitionPlanLayout,
    terms: &'a [ChargeTransitionTerm],
    max_bytes: u128,
}
```

Change each specialized builder to accept `&ChargeBuildContext` plus only its domain-specific hint. Domain positions and local dimensions are read from the validated layout through borrowed accessors; do not copy those arrays into the context or each builder. This removes repeated local-dimension/position/term argument bundles without adding heap allocation or trait-object dispatch.

In Python, replace `_fast_qubit_particles`, `_fast_qudit_factor`, `_fast_boson_factor`, `_supports_independent_factor_product`, and `_fast_fermion_particles` as separately called helpers with this immutable private result:

```python
@dataclass(frozen=True, slots=True)
class ChargeCompileHints:
    fermion_particles: Optional[int]
    fermion_mode_order: Optional[Tuple[int, ...]]
    qubit_particles: Optional[int]
    qubit_positions: Optional[Tuple[int, ...]]
    qudit_weights: Optional[Tuple[int, ...]]
    qudit_target: Optional[int]
    qudit_positions: Optional[Tuple[int, ...]]
    boson_weights: Optional[Tuple[int, ...]]
    boson_target: Optional[int]
    boson_positions: Optional[Tuple[int, ...]]
    product_positions: Optional[Tuple[Tuple[int, ...], ...]]

    @classmethod
    def from_sector(cls, operator: _StructuredOperator, sector: ChargeSector) -> "ChargeCompileHints": ...
```

`from_sector` executes the existing classification rules once and validates that at most one of the Qubit/Qudit/Boson single-factor hint families is populated. The PyO3 method may retain its current flat argument list until a future ABI break; call sites unpack this one object and no longer run five classifiers independently. Rust remains the only authority that accepts or rejects a specialized kernel and publishes `native_variant`.

The Rust selector order is fixed and must match the current measured policy:

1. Eligible specialized Fermion, including the approved Cartesian-Boson spectator envelope.
2. Eligible Qubit, Qudit, or Boson single-factor specialization for its corresponding hint.
3. General Product when independent factor positions are available and the preceding specialization rejected the layout or budget.
4. Generic otherwise.

The selector runs once at construction. `ChargeLazyKernel::apply_into` performs one enum match per public call, never per term or state. Raw/Jordan--Wigner selection is not part of this enum. Do not add a runtime performance selector and do not delete a specialized variant based only on code similarity.

Split `build_fast_product_mvp_plan` into a `ProductPlanBuilder<'a>` with five methods called in order: `discover_components`, `build_factor_indexes`, `compile_actions`, `group_actions`, and `finish`. Builder scratch is dropped by `finish`; only immutable plan fields survive. The new methods must use the existing algorithms, ordering, and memory accounting. After migration, no individual builder method should exceed roughly 150 lines; if one does, split by a named construction phase rather than adding a generic utility layer.

Required selector tests assert the exact variant for Qubit, Qudit, Boson, pure Fermion, Fermion×Boson, two-factor Product, reversed-factor Product, interleaved Product, and coupled Generic. Run `tests/test_charge.py`, `tests/test_execution_invariants.py`, `product_variant_matrix.py`, and `product_u1_16_ab.py`. The final hot-path medians may move within the low-teens band; a material regression requires profiling before acceptance, not restoration of duplicate ownership.

### 7.9 ER-06 — Diagnostics outside inner loops

Keep the public/private diagnostic values used by current tests, but change how they are collected.

Define plain non-atomic per-call structs `StructuredApplyStats`, `ProductApplyStats`, and `ChargeApplyStats`, each containing exactly the fields currently exposed by its snapshot. Every serial loop receives `&mut Stats`; every Rayon loop returns a local `Stats` and combines it with `reduce`. The plan retains its current atomic cumulative counters only as a publication sink. At the end of one `apply_into` call, publish each field with at most one relaxed atomic addition. Construction-time validation counts are published once during construction.

Specific deletions are all `fetch_add` calls inside `apply_compiled_schedule_member`, the Structured block/member loops, Product group/member/spectator loops, and `ChargeDestinationAccumulator::insert/drain`. The contiguous two-factor `ProductApplyCounterDeltas` pattern becomes the shared Product pattern rather than a special case.

Run one A/B on `fermion_mapping_topology_profile.py`, `product_u1_16_ab.py`, and `generic_wide_profile.py`. Select local accumulation when results remain in the same low-teens engineering class. If a dominant path instead shows a clearly large repeatable slowdown and profiling attributes it to local counter maintenance, make diagnostics disabled by default in production and add one private diagnostic-plan constructor for tests/benchmarks; do not restore inner-loop atomics. This is an engineering classification, not an exact percentage gate. After selection, delete the losing instrumentation path.

### 7.10 ER-07 — PyO3 handle scaffolding, with algebra kept separate

Do not merge `NativeFermionOperatorHandle`, `NativeBosonOperatorHandle`, and `NativeHybridOperatorHandle` payloads. Their canonical batches, multiplication rules, adjoints, and materialization arrays remain domain-specific.

Add `structured/common.rs` helpers for only repeated boundary mechanics:

```rust
fn require_same_layout<K: Eq>(left: &K, right: &K, message: &'static str) -> PyResult<()>;

fn run_binary_handle<T: Send + Sync>(
    py: Python<'_>,
    left: &T,
    right: &T,
    max_bytes: usize,
    operation: impl FnOnce(&T, &T, u128) -> Result<T, PauliError> + Send,
) -> PyResult<T>;

fn run_unary_handle<T: Send>(
    py: Python<'_>,
    operation: impl FnOnce() -> Result<T, PauliError> + Send,
) -> PyResult<T>;
```

Each `#[pymethods]` add/scale/multiply/commutator/anticommutator method remains visible but becomes a short layout check plus one helper call. Domain-specific closures call the existing `merge_*`, `multiply_*`, and `binary_*` core functions. Adjoint implementations remain domain-specific throughout this program; do not force Hybrid Weyl phases through the Fermion/Boson adjoint helper.

Share content-hash framing and exact coefficient hashing, but keep each payload's field sequence explicit. Keep `materialize`, `to_hybrid`, `jordan_wigner`, `embed`, direct-Weyl extraction, and charge analysis in their family modules. The target is one GIL/error/budget pattern, not one universal operator-handle trait.

Required verification is the complete structured-algebra, value-semantics, mapping, charge-analysis, and public-API test set. Add one test per family proving that add/multiply/commutator releases the GIL through the shared helper and returns the same flat materialization as before. No performance A/B is required because scalable algebra still calls the same coarse native function once.

### 7.11 ER-08 — Authoritative chemistry visitors

In `structured/integrals.rs`, replace `count_spin_orbital_terms` plus `append_spin_orbital_integrals` with one generic traversal:

```rust
fn visit_spin_orbital_terms<E>(
    n_modes: usize,
    one_body: &[Complex64],
    two_body: &[Complex64],
    mut visit: impl FnMut(&[(usize, u8)], Complex64) -> Result<(), E>,
) -> Result<(), E>;
```

Use fixed local arrays of length two and four for factor words so the counting pass allocates nothing. The counting sink performs checked increment; the materializing sink copies the short factor slice into the final vector and pushes the coefficient. Implement `visit_spin_block_terms` analogously, using `spin_mode` in exactly one place. Constant handling remains outside the visitor and is executed once.

Keep the two-pass algorithm: validate, visit to count and enforce the memory preflight, reserve exact capacity, then visit to materialize. Delete both old count functions and both old append functions after numerical equality passes. Do not implement a one-pass growable builder in ER-08; it would weaken the pre-allocation contract and requires a separate ingestion benchmark decision.

Required tests compare complete and spin-block ingestion against frozen dense Hamiltonians, cover grouped/interleaved spin order, complex Hermitian averaging, exact zeros, non-Hermitian rejection, and the memory limit before materialization. Run `tests/test_integral_ingestion.py`, `tests/test_quantum_chemistry_example.py`, the chemistry interop benchmark smoke, and Clippy.

### 7.12 ER-09 — Benchmark protocol consolidation

Keep `run_quspin_tencirpauli.py` and `run_quspin_quspin.py` separate because they import incompatible library environments and contain different operator builders. Keep `_operator_and_sector` and `_basis_static_and_node` separate and capability-explicit; do not create a cross-library operator DSL.

Move only duplicated protocol into `benchmarks/manual/quspin_capabilities.py`:

- `measure_apply(callable, warmups, repeats) -> {first, samples, median}`;
- `write_preflight_artifact(validation_kind, states, apply, path)`;
- `build_timing_record(...)` for common manifest, dimension, dtype, sample count, warmups, RSS, thread environment, exact command, and provenance fields;
- validation of record-required keys before JSON output.

The two child runners supply basis states, construction timings, the public apply callable, optional caller-owned apply callable, native/library-specific metadata, and output norm. The parent ABBA process and action comparison remain in `quspin_capability_matrix.py`.

Add unit tests for manifest enumeration, record schema, matrix versus vector artifact canonicalization, missing-field failure, and deterministic command/provenance fields. Run the bounded direct rows once at one thread to verify protocol equality; no large performance rerun is required because timing boundaries and callables must be byte-for-byte the same expressions after extraction.

### 7.13 Final integration and deletion audit

After ER-00 through ER-09, run `rg` for every deleted type/function and confirm that only historical `docs/vibe/` references remain. Confirm these architectural counts:

- one Python implementation of native-plan apply/apply_into/SciPy wrapping;
- one retained-plan `CsrGraph` storage and scheduling implementation; `ChargeCsrResult` may remain only as a transient construction DTO until it is consumed by the graph;
- one general-charge selector and one Product builder;
- zero unchecked raw-Fermion apply caches; the selected production path contains either no `O(D)` cache or exactly one checked per-call complete cache, never both;
- zero inner-loop atomic diagnostic increments;
- one spin-orbital integral traversal and one spin-block traversal;
- three distinct domain handle payloads but one PyO3 boundary-helper implementation;
- two library-specific QuSpin child builders but one timing/artifact schema.

Then run `python scripts/check.py --benchmark smoke`, the bounded capability matrix, `representation_ab.py`, `product_variant_matrix.py`, and the primary seven-site lazy and six-site eager comparisons once. This is the only mandatory full performance integration run for the complete entropy-reduction program.

### 7.14 Implementation closure

All ER packages are implemented. The final deletion audit finds no live occurrence of the confirmed Python/Rust residue and no old textual module assembly in the required trees. The architectural counts in Section 7.13 are satisfied: one native MVP facade implementation, one retained CSR graph implementation, one Charge selector, one Product builder, no unchecked Raw-Fermion cache, no inner-loop diagnostic atomic increment, one visitor per chemistry layout, one PyO3 helper boundary, and one shared QuSpin timing/artifact schema.

Final current-worktree verification passed `python scripts/check.py --benchmark smoke`; release focused structured/charge suites; `representation_ab.py`; `product_variant_matrix.py --repeats 7`; `product_u1_16_ab.py --repeats 7`; `generic_wide_profile.py --repeats 7`; the one-thread bounded nine-row direct QuSpin matrix and the two intentional no-baseline rows; seven-site lazy Holstein (`1.511 s` median); and six-site eager/CSR Holstein (`17.62 ms` median). These are local informational measurements; raw records remain untracked.

## 8. Performance and correctness guardrails

The refactor should use a small number of meaningful gates rather than repeatedly rerunning every historical benchmark after every file move.

- Every change first passes the relevant dense/differential/property tests; the complete quality gate runs at the end of each roadmap slice.
- Preserve the owner's uniform raw-Fermion default. Do not introduce automatic topology selection to improve one benchmark row.
- Preserve construction-time native-variant selection and coarse-grained FFI. No refactor may materialize native terms into Python only to feed them back into Rust.
- Preserve deterministic public ordering and exact cancellation-before-leakage semantics.
- Use targeted release A/B only for a changed hot path. Treat roughly 10% and low-teens variation as the same engineering class; investigate repeatable large regressions rather than noise-level threshold crossings.
- Keep the current representative capability matrix and the primary seven-/six-site workloads as final integration gates, not as a command to rerun all large cases after every mechanical edit.
- Never replace a specialized winner with a shared abstraction using trait-object or per-term dispatch. Shared facades and compile-time generics are acceptable; hot-loop indirection requires measured justification.
- Track retained bytes and major per-call workspace together with runtime. A faster refactor that reintroduces `O(pD)` worker buffers or an unchecked `O(D)` cache is rejected.

## 9. Completion matrix

| Requirement | Current verdict | Evidence or remaining action |
| --- | --- | --- |
| Product inactive-factor remediation | Complete | Numerical and structural regressions pass; current specialized and Product routes remain selected as documented. |
| Raw Fermion policy | Complete by owner decision | Uniform Raw default retained; explicit mapping remains user-directed; no topology selector. Performance is topology-dependent and must be described honestly. |
| Formal QuSpin capability coverage | Complete | 12 direct capabilities at one/four threads plus two explicit no-baseline capabilities; all 24 direct rows passed. |
| Primary seven-site lazy workload | Complete | Latest HEAD measured approximately `1.554 s` versus `1.574 s` at one thread and `0.411 s` versus `1.581 s` at four threads. |
| Primary six-site eager/explicit workload | Complete as parity/win | Latest HEAD one-thread fresh-process records were approximately `22.52/20.51 ms` and `22.11/21.26 ms`; the four-thread record was approximately `12.76/21.12 ms`. This is parity at one thread and a win at four, not a one-thread speedup claim. |
| Full quality gate | Complete | 55 Rust tests, 430 Python tests, 8 doctests, three Rust benchmark smoke suites, and 353 Python benchmark cases pass with all required static checks. |
| Checked major `apply_into` workspace | Complete | Direct public-index masks eliminate the raw-Fermion `D`-entry packed-source cache; full-space and prefix `apply_into(max_bytes=0)` are covered. |
| Documentation closure | Complete | Specification, acceptance, owner-decision, implementation-status, and this review record the selected Raw policy and completed maintenance program. |

## 10. Final recommendation

TenCirPauli does not contain evidence of a repository dominated by dead code. The 10.5-era maintainability problem is resolved by one authoritative implementation for identical contracts and mechanics, while retaining distinct algorithms and specialized numerical kernels where they encode a different capability, data layout, or material measured advantage. Future changes should preserve these boundaries and reuse the final integration suite rather than recreating parallel ownership paths.
