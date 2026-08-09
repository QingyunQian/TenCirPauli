# Phase 10.5 Third-Round Acceptance Review, 2026-08-08

Status: not acceptance-closed. Clean-commit provenance is explicitly excluded from this decision by the owner and may be supplied after technical acceptance. The current tree remains numerically strong and quality-gate clean, but the general Product, Cartesian, and wide Generic execution architecture plus several required structural/performance measurements are not yet sufficient for the Phase 10.5 specification.

## 1. Executive verdict

The second-round remediation closed substantial work. Canonical Qudit spin-z charge, authoritative native family metadata, actual packed-Pauli Structured routing, the shared width-generic `FermionMaskDescriptor`, compiled charge schedules, direct native CSR read-back, ordered-row Structured CSR construction, heap-based charge CSR merging, deletion of the old global transition builder, the eleven-subset numerical matrix, owner-approved eager-format winners, and the accepted QuSpin parity results are present. The complete local smoke workflow also passes.

The remaining items do not all have the same purpose. Product schedule grouping, spectator-segment reuse, Cartesian active-axis reuse, and removal of per-transition destination-key allocation are intended to improve hot-path performance and scaling. Deleting duplicate specialized executors is primarily a maintenance requirement and is safe only after a matched release A/B proves that the shared replacement does not regress representative workloads. Structural counters and benchmark breakdowns are acceptance evidence; they do not themselves make execution faster.

Phase 10.5 should therefore remain open for the technical and evidence items below. The uncommitted working tree is not counted as a defect or blocker in this third-round verdict.

## 2. Verification performed

- Read the complete Phase 10.5 specification, second-round review, owner decision log through D28, acceptance evidence, trusted-handle audit, implementation status, and the affected Rust, PyO3, Python, test, and benchmark sources.

- Ran `python scripts/check.py --benchmark smoke` through the project environment. Rust formatting, Black, stage-label validation, Clippy with warnings denied, Ruff, strict mypy, `git diff --check`, 50 Rust tests, release `maturin develop --release --locked`, 416 Python tests, 8 doctests, all three Rust benchmark suites, and 353 bounded Python benchmark cases passed; 84 large benchmark cases were intentionally deselected by the smoke mode.

- Confirmed that pure binary Structured input constructs and executes the actual packed Pauli native variant, rather than relying on a Python family-string override.

- Confirmed that `CsrSink` now has ordered-row append for Structured and deterministic binary-heap k-way merging for charge runs, and that the former global transition builder/export is absent.

- Traced charge-plan selection and confirmed that the Qubit, Qudit, Boson, and primary spinful specialized executors remain active production paths selected before the general `FastProductMvpPlan`.

- Searched for the second-round structural closure evidence. No `ProductScheduleGroup`, Product allocation/fallback counters, Cartesian spectator-evaluation counters, CSR 1/8/64/512-chunk scaling runner, raw-Fermion versus mapped-Pauli time-share profile, or six-site CSR generation/sort/merge/final-copy breakdown is present.

No production or test source was changed during the review. This report and its `docs/vibe/README.md` index entry are the only review changes.

## 3. Performance interpretation

| Remaining item | Expected performance effect | Acceptance meaning |
| --- | --- | --- |
| Product net-transformation groups and spectator segments | Direct improvement is expected by removing repeated term lookup, rank reconstruction, map-node allocation, and factor work across identity spectators. The magnitude must be measured. | Technical and representative-performance blocker. |
| Cartesian active-axis/spectator execution | Direct improvement is expected when terms touch only a small subset of a large Cartesian layout. | Technical and representative-performance blocker. |
| Reusable scalar/fixed-width accumulators | Direct improvement is expected where the current path allocates map nodes or `Vec<u64>` keys. Wide Generic benefits most; ordinary scalar-key layouts may see little change. | Hot-path allocation blocker. |
| Deleting `FastQubitMvpPlan` / `FastAdditiveMvpPlan` or the primary specialized Product kernel | No inherent speedup. Deletion reduces semantic duplication and maintenance risk, but can regress performance if the shared kernel is not yet equally fast. | Delete only after matched A/B; otherwise retain temporarily with an explicit unresolved convergence decision. |
| Structural counters and CSR/profile breakdowns | No production speedup by themselves. They prove which executor ran, where work occurs, and whether the intended complexity was achieved. | Evidence blocker. |
| Clean commit and immutable provenance | No performance effect. | Explicitly excluded from this acceptance decision by the owner. |

## 4. Third-round findings

### T3-1 — General Product remains only partially converged

Priority: P0 execution-architecture and performance blocker. Second-round mapping: SR1 remains partially open.

`FastProductMvpPlan` now precomputes factor-rank and spectator-digit actions and evaluates factor actions outside the spectator loop. This is real progress. It also uses scalar mixed-radix destination keys rather than heap-backed factor-value arrays.

However, it does not compile the specification's coefficient-free `ProductScheduleGroup` representation. `FastProductMvpPlan::apply_into()` creates a `BTreeMap<usize, Complex64>`, clears it for every factor/spectator source, and inserts every surviving term contribution separately. Clearing a `BTreeMap` releases its nodes, so subsequent insertions allocate again. The loop reconstructs source and destination occupations and calls the general sector ranker inside the spectator traversal instead of applying `A_factor tensor I_spectator` directly over a contiguous segment.

The old specialized paths are also still live. Native construction attempts `build_fast_fermion_mvp_plan`, `build_fast_qubit_mvp_plan`, `build_fast_qudit_mvp_plan`, or `build_fast_boson_mvp_plan` before falling back to `build_fast_product_mvp_plan`; their plan types and builders remain publicly re-exported by the core crate. Consequently, many tests and the committed spinful Product benchmark report `execution_family="product"` while exercising a specialized executor rather than the general Product implementation under review.

Required closure:

1. Compile canonical members into complete coefficient-free net-transformation groups and use one reusable preallocated scalar/fixed-record aggregate only where a group genuinely needs cancellation.

2. Add active-shape loops that apply identity-spectator actions over contiguous segments and avoid general occupation reconstruction/ranking where the Product index already determines the source or destination segment.

3. Add test-only counters for factor-action evaluations, segment applications, destination-key allocations, and fallback invocations.

4. Run matched release A/B for the general Product kernel against the existing specialized Qubit/Qudit/Boson and primary spinful paths. Delete a specialized path only after the shared kernel is numerically equal and shows no representative performance regression; deletion alone is not a performance optimization.

5. Re-run the 16-qubit Product/U1 diagnostic and explain any remaining multi-fold gap from measured unavoidable work.

### T3-2 — Cartesian execution and Fermion semantic convergence remain incomplete

Priority: P0 common-path performance and maintenance blocker. Second-round mapping: SR2 remains partially open.

The shared `FermionMaskDescriptor` is now used by Structured ordinary Fermion terms, charge schedules, Product descriptors, and the primary compact Fermion path. One-word and multiword descriptors use caller-owned word scratch. Pure binary Structured plans construct the packed Pauli native variant. These portions of SR2 are closed.

The non-Pauli Structured plan still executes a destination-by-term loop. `CompiledNetAction` removes repeated symbolic lowering, but it does not group schedules by active axes or reuse local work over contiguous identity-spectator segments as required by the Cartesian contract. The legacy `apply_compiled_structured_operation_inverse()` interpreter, including its independent Fermion parity/sign logic, also remains callable from lazy and eager fallback branches. This leaves a second production Fermion semantic implementation even though ordinary supported terms should already lower to `FermionMaskDescriptor` plus net actions.

Required closure:

1. Add active-axis and stride-block Cartesian schedules so subset-local actions reuse work across contiguous spectator regions.

2. Prove by counters that supported Structured Fermion schedules never invoke the legacy operation interpreter; then delete its Fermion branch or the entire unreachable fallback rather than retaining a parallel sign implementation.

3. Add allocation-backed or counter-based evidence that one-word and multiword FermionMask apply performs zero per-term/per-state heap allocation.

4. Re-run and profile the raw-Fermion versus explicitly mapped packed-Pauli diagnostic. Report packed-word extraction, parity, index update, input-load, and output-write shares. A speedup is expected from removing duplicate interpretation and repeated work, but closure requires measurement rather than an assumed ratio.

### T3-3 — Wide Generic still allocates destination keys per contribution

Priority: P1 wide/coupled-layout performance blocker. Second-round mapping: SR4 remains partially open.

Generic lazy and charge eager now consume `CompiledChargeSchedule`, so raw identifier conversion and symbolic term interpretation are no longer repeated in the scalable loop. Scalar full-radix layouts use `FxHashMap<usize, Complex64>` and avoid occupation-vector key copies.

When the full mixed-radix space does not fit a native index, `ChargeDestinationAccumulator::insert()` still executes `destination.to_vec()` for every new candidate key. This is the same per-transition heap-allocation pattern prohibited by the Generic contract and is used by both charge eager construction and Generic lazy application. Calling the fallback bounded does not remove the allocation or satisfy the second-round zero-allocation requirement.

Required closure:

1. Replace the vector-key hash fallback with reusable fixed-width scratch and preallocated entries, or another deterministic accumulator that performs no candidate insertion allocation.

2. Add counters proving zero raw-term interpretation, identifier conversion, descriptor revalidation, and destination-key heap allocation after plan construction for both scalar and wide layouts.

3. Benchmark a genuinely coupled wide-sector case before and after. This change is expected to improve the affected wide path but need not materially change ordinary scalar-key workloads.

### T3-4 — CSR implementation is corrected, but its required scaling evidence is incomplete

Priority: P1 acceptance-evidence blocker. Second-round mapping: SR3 implementation is closed; SR3 evidence remains open.

The source defect identified in SR3 is fixed. Structured rows append directly, charge runs use a binary heap with `O(E log K)` merge selection, cross-run duplicates are coalesced deterministically, exact zeros are removed, and the old global-map transition builder/export is gone. The multi-chunk unit test covers cancellation, empty rows, sorted columns, and repeat equality.

The required representative evidence is still missing. There is no committed sweep across approximately 1, 8, 64, and 512 chunks, and the six-site record does not separate transition generation, chunk sorting, heap merge, final allocation/copy, and peak RSS. The existing six-site steady MVP parity result validates the retained graph's execution, not the corrected construction complexity.

Required closure:

1. Add the bounded chunk-scaling release runner and demonstrate linear row-ordered or approximately `E log K` chunked behavior.

2. Re-run the six-site eager build in a fresh process and record generation, sort, merge, copy, retained bytes, and peak RSS separately.

3. Include heap/run cursors and chunk metadata in the best-effort construction estimate or document a measured reason why their bounded contribution is intentionally omitted.

### T3-5 — Closure documents overstate the current technical state

Priority: P1 handoff blocker. Second-round mapping: SR5 remains open only for technical truthfulness and evidence mapping; clean provenance is excluded.

D28 correctly supersedes D20/D21 at the owner-decision level, and the owner has now clarified that the repository commit will be made after acceptance. The absence of a clean commit is therefore not a blocker in this review.

The remaining documentation problem is substantive rather than provenance-related. `phase-10.5-acceptance-evidence.md`, `phase-10.5-spec.md`, and `implementation-status.md` say the second-round implementation is technically closed, while T3-1 through T3-4 show surviving production paths and missing required evidence. The final remediation table must identify which native variant each benchmark exercises and must not use the common `execution_family="product"` label as proof that the general Product kernel ran.

Required closure:

1. Reopen only the technical/evidence items above in the status documents.

2. After their closure, map each finding to exact source functions, structural counters, independent numerical tests, and representative benchmark records. The owner may then commit the accepted worktree as a separate repository operation.

## 5. Acceptance recommendation

Do not accept Phase 10.5 yet. Preserve the completed Qudit, packed-Pauli routing, shared FermionMask, compiled charge schedule, native CSR, QuSpin parity, representation selection, and quality-gate work. Reopen only general Product convergence, Cartesian spectator reuse and legacy-interpreter deletion, wide Generic allocation, and the missing construction/profile evidence.

Acceptance becomes reasonable when those four areas close with independent numerical equality, structural counters, and representative release A/B results. No additional clean-commit requirement is imposed by this report; the owner will create the repository commit after technical acceptance.
