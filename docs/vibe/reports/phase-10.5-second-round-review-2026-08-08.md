# Phase 10.5 Second-Round Implementation Review, 2026-08-08

Status: not acceptance-closed. The current tree is numerically strong and passes the complete local quality gate, but it does not yet satisfy the Phase 10.5 execution and maintenance architecture. R1 is only partially closed; R2, R3, R4, and R5 remain open for the reasons below.

## 1. Executive verdict

The remediation made real progress: Qudit spin-z semantics, native family reporting for charge plans, independent-factor rank mappings, graph-free lazy Product retention, construction-time non-Fermion net actions, direct native CSR read-back, exact cancellation/leakage tests, the eleven-subset numerical matrix, owner-approved QuSpin parity, and the full quality gate are all present.

The remaining problem is not a missing small optimization. Several difficult requirements were replaced by narrower implementations and then described as complete: general Product still traverses every full selected state and creates heap-backed destination keys instead of reusing factor actions over Cartesian spectator segments; Structured plans labeled `pauli` do not use the existing packed Pauli executor; Fermion signs remain implemented by three separate descriptor/interpreter families; the wide Fermion path allocates word buffers inside the term-by-state loop; the CSR merge has an `O(EK)` linear scan over `K` chunks; a superseded global-map charge builder remains publicly exported; and Generic/eager charge kernels still repeat immutable descriptor checks that the trusted-handle audit says were removed.

The phase therefore cannot be closed by the current D22/D24/D25/D27 wording alone. A final closure needs code changes, representative release evidence from the changed code, and one owner-log entry that explicitly closes the still-open D20/D21 ledger.

## 2. Verification performed

- Read the complete Phase 10.5 specification, first-round review, owner decision log, acceptance evidence, trusted-handle audit, implementation status, affected Rust/PyO3/Python code, focused tests, and committed manual benchmark tooling.

- Ran `python scripts/check.py` in the project environment against dirty commit `c50038bac50d`. Rust formatting, Black, stage-label validation, Clippy with warnings denied, Ruff, strict mypy, `git diff --check`, 47 Rust tests, release `maturin develop --release --locked`, 415 Python tests, 8 doctests, all three Rust benchmark suites, and 436 Python benchmark cases passed; one Python benchmark was skipped. The resulting benchmark label is `20260808T010007Z_c50038bac50d-dirty` under the ignored `.benchmarks/` directory.

- Re-ran the first-round fixed-Hamming-weight diagnostic with one Rayon thread. For the same 16-qubit, half-filled nearest-neighbor `XX + YY` action, the reported Product plan took a seven-sample median of approximately `28.22 ms`; the existing packed U1 executor took approximately `4.18 ms`; outputs agreed exactly. The remaining ratio is approximately `6.76x`, improved from the first-round `18x` but still evidence that the general Product plan is not the factor/spectator executor required by the specification.

- Re-ran the first-round unrestricted Fermion diagnostic with one Rayon thread. A 12-mode hopping-and-density raw Fermion plan reported `cartesian` and took approximately `0.652 ms`; the explicitly Jordan-Wigner-mapped Pauli plan took approximately `0.142 ms`; the maximum absolute output difference was approximately `1.46e-11` on an intentionally large-magnitude deterministic vector. The remaining ratio is approximately `4.58x`, improved from `9.2x` but not evidence that raw Cartesian execution shares the packed semantic substrate.

- Audited the six-site CSR merge complexity from the current implementation. With `E = 22,527,980` entries and chunk capacity `16,384`, `K = 1,375` chunks. `CsrSink::into_csr()` linearly scans all chunks for every emitted entry, implying approximately `E*K = 30,975,972,500` chunk-cursor checks before ordinary merge work. The routine benchmark suite does not exercise this six-site finalization path.

No production or test source was changed during this review. This report and its `docs/vibe/README.md` index entry are the only tracked review changes.

## 3. Specification compliance matrix

| Requirement | Status | Evidence and limitation |
| --- | --- | --- |
| Canonical Qudit spin-z charge | PASS | Constructor, hashing, rank/unrank, wrap-dependent charge changes, conservation, lazy/eager execution, sparse materialization, and independent numerical tests are present. |
| Public basis order and exact cancellation/leakage | PASS | Focused Product, Generic, eager, COO/CSR, interleaved-axis, and cancellation-before-leakage tests pass. |
| Exactly two public CPU-native storage values | PASS | Public plans expose `lazy` and `eager`; no public representation selector was added. |
| Minimal authoritative plan metadata | PARTIAL | Charge family metadata now comes from the native selected plan, but `python/tencirpauli/structured.py:3267-3275` still overrides the native family to `pauli` from Python-side input inspection. |
| Fast unrestricted Cartesian execution | FAIL | Flat non-Fermion net actions and Fermion masks exist, but no active-axis/spectator-segment executor exists, pure binary inputs do not use the packed Pauli primitive, and wide masks allocate inside the hot loop. |
| General independent-factor Product execution | PARTIAL | Multiple compact factors, same-domain spectators, and numerical correctness exist, but `FastProductMvpPlan` still performs a full-state loop, heap-backed destination aggregation, full destination reconstruction, and general sector reranking per surviving contribution. |
| One shared FermionMask lowering/action | FAIL | Structured, Product, and the primary fast Fermion path retain separate sign/action implementations. |
| Graph-free lazy Product retention | PASS | The Product plan retains factor/spectator action descriptors rather than a complete selected-space graph. |
| One direct native CSR data plane | PARTIAL | Native CSR read-back and a shared chunk type exist, and D25 permits source-major charge production, but finalization is asymptotically unsuitable and the superseded global-map builder remains exported. |
| No repeated trusted-handle validation in scalable kernels | FAIL | Prepared layouts validate once, but Generic/eager charge still reruns code-length, mode, position, dimension, and occupation checks through the raw term interpreter. |
| One retained eager winner per class | PASS BY OWNER DECISION | D24 explicitly selects compact Pauli and CSR for the other current classes. This is an owner selection, not a complete compact-candidate A/B matrix. |
| Large QuSpin acceptance | PASS BY OWNER DECISION | D22 accepts the recorded six-site result as parity, and the seven-site lazy result is within the accepted comparison envelope. No speedup claim should be made for the eager parity case. |
| Eleven-subset independent numerical matrix | PASS | All eleven nontrivial mixed-domain subsets have small independent dense comparisons and pass. |
| Superseded paths deleted and status documents truthful | FAIL | Old global-map and duplicate Fermion paths remain, while the owner ledger still says remediation is open and the status file says acceptance is closed. |
| Complete quality gate | PASS | The full local workflow passed as recorded above. |

## 4. Findings

### SR1 — General Product remains a full-state hash executor, not a spectator-reusing Product kernel

Priority: P0 execution-architecture and representative-performance blocker. First-round mapping: R1 remains partially open.

Evidence: `FastProductMvpPlan::apply_into()` at `crates/tencir-pauli-core/src/charge.rs:2549-2710` iterates `for (column, state_value) in state.iter().enumerate()` over the complete selected dimension. For every action it copies factor and spectator values, computes Fermion signs, encodes a spectator key, allocates `destination_factor_values.clone().into_boxed_slice()` at lines 2670-2673, inserts that heap-backed key into an `FxHashMap`, reconstructs a complete destination occupation at lines 2681-2697, and calls the general `ChargeSectorPlan::rank_into()` at lines 2698-2702.

Evidence: the specialized primary spinful plan already demonstrates the intended shape at `crates/tencir-pauli-core/src/charge.rs:1118-1203` and `1291-1385`: it computes factor-local actions outside a contiguous Cartesian spectator segment and then applies them over the segment. The general Product executor does not reuse that loop structure.

Evidence: the current 16-qubit Product diagnostic is approximately `6.76x` slower than the existing packed U1 executor on an equivalent action, despite exact output equality. This is materially better than the first-round `18x`, so the descriptor precompilation has value, but the remaining structure still matches the full-state/hash design that R1 required replacing.

Impact: numerical tests pass because both implementations are semantically correct. Scaling, allocation behavior, and factor/spectator reuse do not meet Sections 10.2-10.3 of the specification. The plan cannot make factor work proportional to the factor dimension when a large Cartesian spectator segment is present, and every candidate contribution can allocate a variable-width key.

Required implementation:

1. Replace `ProductDestinationKey { factor_values: Box<[u64]>, spectator }` with compiled scalar factor ranks and a mixed-radix destination component index. A candidate key should be a fixed `usize`/`u64` or a fixed small record whose storage is allocated once, never a newly allocated `Box<[u64]>`.

2. Compile `ProductScheduleGroup` values keyed by the complete coefficient-free net transformation across all active factors and spectator axes. Each group should retain canonical member order for source-dependent amplitudes and exact cancellation, but derive at most one source component for a destination component.

3. Split application by active-axis shape. For `A_factor tensor I_spectator`, iterate factor destination/source pairs once and apply the amplitude over the contiguous spectator slice. For actions on a spectator axis, iterate valid local digit ranges and surrounding stride blocks. For actions on every factor, use the same specialized descriptors without falling back to occupation vectors.

4. Reuse the existing spinful segmented kernel as a primitive or delete it after the general Product kernel reaches equal representative performance. Do not leave two Product algorithms that implement the same Fermion/Boson semantics.

5. Use a reusable bounded scalar-key accumulator only for groups that genuinely require cancellation across multiple members. Clear it per factor/component source; do not allocate per contribution. Apply leakage after the group aggregate is known.

6. Partition parallel execution by disjoint destination segments over the final loop structure. Preserve fixed accumulation order and caller-owned output slices; retain no worker-sized output vector.

Closure evidence:

- Add native test-only counters for factor-action evaluations, spectator segment applications, destination-key allocations, and fallback invocations. Increasing an identity spectator dimension must increase segment elements but not factor-action evaluations or heap-key allocations.

- Re-run Qubit, Qudit, Boson, two-factor, reversed-factor-size, interleaved, and primary Holstein Product cases. Report factor dimensions, spectator dimensions, schedule groups, allocations, first/steady `apply_into`, and independent numerical error.

- Re-run the 16-qubit Product/U1 diagnostic. No fixed ratio is imposed, but an unexplained multi-fold gap must be traced to measured unavoidable work rather than accepted behind a `product` label.

- Delete `FastQubitMvpPlan` and `FastAdditiveMvpPlan` plus their builder/enum arms once the shared Product plan covers their cases; they remain exported at `crates/tencir-pauli-core/src/lib.rs:26-33` today.

### SR2 — The promised shared FermionMask and packed-Pauli reuse are not implemented

Priority: P0 correctness-maintenance and Cartesian-performance blocker. First-round mapping: R2 remains open.

Evidence: Structured defines `FermionMaskDescriptor` and `WideFermionMaskDescriptor` at `crates/tencir-pauli-core/src/structured.rs:2481-2500`. Product separately stores `ProductFermionStep` and computes signs in `product_fermion_sign()` at `crates/tencir-pauli-core/src/charge.rs:2476-2480` and `2732-2796`. The primary spinful path separately stores `FastFermionTerm` and applies it in `apply_fast_fermion_term()` at `crates/tencir-pauli-core/src/charge.rs:1044` and `4046-4109`. These are three independent Fermion occupancy/parity/sign implementations, contrary to Section 8.2's requirement that unrestricted, compact-sector, Product, and Hybrid paths reuse one FermionMask lowering and action.

Evidence: pure binary structured plans are still built through `structured_sparse_plan_handle` or `structured_eager_plan_handle` at `python/tencirpauli/structured.py:3470-3485`. The Python facade then computes `mapped_binary` and overwrites `execution_family` with `pauli` at lines 3267-3275. The native `StructuredMvpPlan` continues to execute its own `CompiledNetAction` loop; it does not own or delegate to the existing packed Pauli `MvpPlan`. The implementation-status claim that pure binary structured inputs reuse the packed primitive is therefore false.

Evidence: the multiword path allocates in the scalable loop. `packed_fermion_index_wide()` allocates a new `Vec`/`Box` at `crates/tencir-pauli-core/src/structured.rs:2936-2947`; lazy apply calls it inside the destination-and-term loop at lines 3663-3684, clones it again at line 3685, and `fermion_mask_source_amplitude_wide()` constructs another destination `Box` at lines 2841-2877 even though callers discard that returned word array and compute the destination index separately. Eager and dense consumers repeat the same pattern.

Evidence: the current raw Fermion diagnostic remains approximately `4.58x` slower than the explicitly mapped packed Pauli plan. The specification does not require raw Fermion to beat mapped Pauli, but it does require one specialized semantic substrate and profiling-backed explanation rather than duplicate interpreters.

Impact: every Fermion change must be reasoned about and tested in multiple sign implementations. The wide path violates the repository rule against per-term allocation in hot loops. A Python family override can make metadata look correct without proving the selected native executor.

Required implementation:

1. Move a single width-generic Fermion descriptor into a shared core module, for example `fermion_mask.rs`. It should own required-occupied words, required-empty words, flip words, source-parity words, fixed sign, and the public mode-axis mapping. Retain a specialized one/two-word representation behind the same API for common systems.

2. Provide source and inverse application functions that accept borrowed packed words and caller-provided scratch. They should return a destination index/amplitude or update a reusable word buffer; they must not allocate or rescan symbolic creation/annihilation sequences.

3. Compile raw structured monomials, compact-sector Fermion terms, Product factor actions, and the primary spinful path into this descriptor. Delete `ProductFermionStep`, `product_fermion_sign()`, `FastFermionTerm`, `apply_fast_fermion_term()`, and any legacy raw-Fermion operation interpreter that becomes unreachable.

4. Compute a wide source/destination occupation word once per outer basis state or worker and reuse it across schedules. Change `fermion_mask_source_amplitude_wide()` to return only the amplitude when the destination words are not consumed.

5. Introduce a native plan enum such as `StructuredPlanInner::{Pauli(PackedPauliPlan), Cartesian(CartesianPlan), Generic(GenericPlan)}`. Select the enum during GIL-released construction, return `execution_family` from the actual variant, and remove the `mapped_binary` override in Python.

6. Lower pure binary Hybrid/structured terms directly into the existing packed Pauli plan in one Rust call. Do not materialize Python terms or keep a second Pauli MVP kernel.

7. Add active-axis/spectator descriptors to Cartesian schedules so local action work can be reused over contiguous public-order segments. Construction-time local tables alone do not implement spectator reuse.

Closure evidence:

- Add a native assertion that pure binary structured input constructs the packed Pauli variant, not merely that the public string equals `"pauli"`.

- Add a test-only allocation counter or allocator-backed benchmark proving zero per-term/per-state allocations for one-word and multiword FermionMask apply.

- Differentially test the same shared descriptor through unrestricted Structured, compact Fermion Product, mixed Hybrid, lazy, eager, dense, and CSR consumers, including ordered CAR, quartic terms, complex coefficients, mode 128, and interleaved public axes.

- Profile the current `4.58x` raw-versus-mapped gap and report the time shares for packed-word extraction, parity, index updates, input loads, and output writes before claiming closure.

### SR3 — The shared CSR sink is asymptotically unsuitable, and a prohibited predecessor remains exported

Priority: P0 eager-construction scalability and maintenance blocker. First-round mapping: R3 remains partially open.

Evidence: direct native `indptr`, columns, and values now reach Python, Structured uses destination-row generation, and charge uses the D25-approved source-major producer. Those are real R3 improvements.

Evidence: `CsrSink` stores every entry as `(row, column, value)` in fixed-capacity chunks at `crates/tencir-pauli-core/src/csr.rs:17-58`. Finalization sorts every chunk and, for every output entry, scans every chunk using `.filter_map(...).min_by_key(...)` at lines 69-118. This is `O(EK)` merge selection for `K` chunks, not a scalable k-way merge. On the specified six-site graph it implies approximately `30.98 billion` chunk-cursor checks.

Evidence: the only core sink regression at `crates/tencir-pauli-core/src/csr.rs:181-200` uses five pushed entries, so it cannot expose multi-chunk complexity or construction RSS.

Evidence: the superseded `compile_charge_transitions()` path remains at `crates/tencir-pauli-core/src/charge.rs:4117-4286`. It materializes a full basis hash at lines 4178-4193, a global `(row, column)` `BTreeMap` at line 4195, per-source destination maps, and complete row/column/value arrays. It has no callers in the repository but is still re-exported from `crates/tencir-pauli-core/src/lib.rs:23-33`. This directly violates the specification's deletion requirement even if the Python production path no longer calls it.

Impact: routine tests and small representation records pass, while the large finalization algorithm can dominate at the intended `E`. Keeping the old public builder leaves two semantic/storage paths and allows future callers to reintroduce the prohibited graph construction.

Required implementation:

1. Keep one `CsrSink` abstraction but provide a row-ordered append mode for Structured. `finish_row(row, sorted_unique_columns)` should append only `(column, value)` payloads and advance `indptr`; it should not store a row per entry or sort rows again.

2. For the D25-approved charge source-major mode, sort only fixed-capacity runs and merge them with a binary heap keyed by `(row, column, chunk_index)`, giving `O(E log K)` selection. A tournament tree is also acceptable. The current linear scan is not.

3. If duplicate `(row, column)` entries can cross runs, coalesce them in deterministic producer order and remove exact zero before final row-pointer publication. Document the accumulation order and test it across chunk boundaries.

4. Account for heap/run cursors, chunks, exact final arrays, and bounded producer scratch in the existing best-effort preflight. Check growth only at chunk boundaries.

5. Delete `compile_charge_transitions()` and its obsolete `ChargeTransitionLayout`/result export if they have no supported external contract. Keep explicit COO as a linear expansion of authoritative CSR, not as a second transition generator.

Closure evidence:

- Add a multi-chunk unit test with duplicates, exact cancellation across chunks, sorted columns, empty rows, and deterministic repeat equality.

- Add a release benchmark that sweeps `E` across at least 1, 8, 64, and 512 chunks and demonstrates approximately `E log K` or linear row-ordered scaling, not `E*K`.

- Re-run the six-site build in a fresh process and report generation, chunk sort, merge, final allocation/copy, retained bytes, and peak RSS separately. The previously accepted steady MVP parity does not validate construction finalization.

- Run `rg` to prove that no global transition `BTreeMap`, full-basis hash builder, `Vec<Vec<...>>` row store, Python sparse-entry reconstruction, or obsolete builder export survives.

### SR4 — Immutable charge validation and raw term interpretation remain in Generic and eager hot loops

Priority: P1 hot-path and trusted-handle-audit blocker. First-round mapping: R4 remains open.

Evidence: `prepare_charge_transition_plan_layout()` validates positions, dimensions, code lengths, and Qudit term structure once at `crates/tencir-pauli-core/src/charge.rs:407-480`. The execution helpers nevertheless repeat `codes.len()`, bit range, code dispatch, mode conversion, mode bounds, occupation range, local-dimension, Qudit-site, and Qudit-exponent checks at `apply_pauli()` lines 487-523, `apply_fermions()` lines 526-590, `apply_bosons()` lines 613-647, and `apply_qudits()` lines 650-672.

Evidence: Generic lazy calls the raw `apply_charge_term()` for every source and term at `crates/tencir-pauli-core/src/charge.rs:4613-4673`. Charge eager calls the same raw semantic helpers for every source and term at lines 4331-4393. The Product construction path also still calls `apply_charge_term()` while precomputing component data at line 3307. The trusted-handle audit statement that scalable kernels consume compiled descriptors without repeated term validation is therefore too broad.

Evidence: non-termwise Generic lazy still inserts `destination.clone()` as an `FxHashMap<Vec<u64>, ...>` key at `crates/tencir-pauli-core/src/charge.rs:4653-4673`. A bounded accumulator is allowed, but per-candidate heap-backed occupation keys are not the flattened Generic design required by Section 14.

Impact: R4's memory-estimate calls were moved out of the per-transition loop, but descriptor validation and symbolic-domain interpretation were not removed from all affected kernels. The audit currently hides this remaining work instead of documenting it.

Required implementation:

1. Compile `ChargeTransitionTerm` into a validated `CompiledChargeSchedule` containing native `usize` axes, shared Pauli/Fermion/Boson/Weyl local descriptors, precomputed strides/tables, coefficient, and net-transformation key.

2. Make Product, Generic lazy, and charge eager consume borrowed compiled schedules. `apply_charge_term()` and raw `u32` identifier conversion should remain only in a test oracle or be deleted.

3. Replace `Vec<u64>` destination keys with a scalar selected-basis candidate when direct rank is available, or a reusable fixed-width scratch key plus preallocated entries when leakage requires occupation-space aggregation. No candidate insertion should allocate.

4. Retain only state/output length checks at the boundary, physical finite-cutoff/occupancy conditions, post-aggregation sector membership, and checked index/allocation arithmetic. These are semantic/safety checks, not validation debt.

5. Rewrite `phase-10.5-trusted-handle-audit.md` as an exact inventory of surviving checks after the code change. Cite functions and call counts; do not describe Product-only remediation as applying to Generic/eager paths.

Closure evidence:

- Add counters proving zero raw-term interpretations, zero identifier conversions, zero descriptor-validation passes, and zero heap-key allocations per apply/eager transition after construction.

- Benchmark Generic cancellation, coupled charge, Qubit Product, Structured Cartesian, and charge eager before/after. Report the removed work's share of total runtime and do not claim a material win unless measured.

### SR5 — Closure documents contradict the authoritative ledger and overstate what the code selects

Priority: P1 acceptance-evidence and handoff blocker. First-round mapping: R5 remains open.

Evidence: the owner decision log still marks D20 as `reopened; partial implementation only` at `docs/vibe/reports/phase-10.5-owner-decision-log.md:237-243` and D21 as `open; implementation in progress` at lines 245-251. `docs/vibe/implementation-status.md:23` nevertheless says owner acceptance is closed, and `docs/vibe/reports/phase-10.5-acceptance-evidence.md:59` marks R1-R5 closed.

Evidence: `docs/vibe/implementation-status.md:37` and the acceptance evidence claim pure binary structured inputs reuse the packed Pauli primitive, but the actual construction and apply path remains the Structured plan described in SR2.

Evidence: the first-round closure contract required a remediation table mapping R1-R5 to exact commits, files, tests, and benchmarks. The current ledger at `docs/vibe/reports/phase-10.5-acceptance-evidence.md:65-73` lists broad file names and test descriptions but no implementing commits. The complete implementation is still an uncommitted working tree at `c50038bac50d-dirty`, so an immutable closure mapping does not yet exist.

Evidence: all current tests can pass while the architecture remains wrong. Most family tests assert the public string and numerical equality; no test proves the packed Pauli native variant, spectator evaluation counts, zero hot-loop allocation, a shared Fermion descriptor, or multi-chunk CSR complexity.

Required implementation:

1. Reopen Phase 10.5 in `implementation-status.md` now, or leave the current status untouched until the technical fixes land but do not use it as acceptance evidence.

2. After SR1-SR4 close, add a new owner decision that explicitly resolves and supersedes D20/D21. D22/D24/D25/D27 close narrower questions; they do not logically change the existing open statuses by themselves.

3. Record exact clean commits for the semantic IR, Product kernel, Structured/Pauli routing, CSR sink, cleanup, tests, and benchmark evidence. A dirty-tree benchmark label may guide development but is not final closure provenance.

4. Separate evidence into numerical correctness, actual native variant selection, hot-loop allocation/call counts, eager construction complexity/RSS, steady MVP, representation winner, and external QuSpin comparison. Replay against emitted CSR is not an independent numerical oracle.

5. Update the spec status, acceptance evidence, trusted-handle audit, and implementation status only after the corresponding clean source and benchmark records exist.

## 5. Why the quality gate is green while the phase is still open

The quality gate correctly proves formatting, static analysis, compilation, numerical regressions covered by tests, and benchmark-harness operability. It does not prove that a family label corresponds to a particular native enum variant, that Product factor work is reused across spectators, that a hot loop allocates zero keys, that three Fermion sign implementations have become one, or that a five-entry merge test scales to 22.5 million entries.

This is the central review lesson: the implementation is not stalled because the remaining requirements are vague. The unresolved work is concentrated in hard loop/data-layout changes, while the closure evidence mostly tests values and metadata. More value/label tests cannot close these findings; structural counters, representative profiles, deletion proofs, and clean benchmark records are required.

## 6. Ordered remediation plan

1. Introduce the shared width-generic FermionMask/local-action IR and migrate Structured raw Fermion, Product, and the primary fast Fermion path. Keep independent numerical differentials green after each consumer migration.

2. Add an actual native plan enum and route pure binary Structured input to the existing packed Pauli executor. Remove Python execution-family inference.

3. Rebuild general Product around scalar factor ranks, net-transition groups, and spectator segments. Remove heap-backed candidate keys, then delete the single-factor and primary-model duplicate executors once release evidence permits.

4. Compile Generic/eager charge terms into the same local-action IR and remove raw identifier/code validation from scalable loops.

5. Replace the CSR linear-scan merge, add the row-ordered append path, and delete the old global-map builder/export.

6. Add structural counters and multi-chunk/scaling tests before another large benchmark run.

7. Run focused correctness, the complete quality gate, bounded capability profiles, the representation record, the 16-qubit Product/U1 and raw/mapped Fermion diagnostics, six-site eager construction/RSS, and the isolated seven-site/six-site QuSpin protocol.

8. Commit the implementation and evidence in auditable slices, then add one owner decision that closes D20/D21 and update all status documents consistently.

## 7. Acceptance recommendation

Do not accept Phase 10.5 yet. Preserve the Qudit work, numerical matrix, native CSR read-back, owner-approved D22/D24/D25 decisions, and current primary Holstein results. Reopen only the implementation items identified here; another marginal QuSpin timing win is not required.

Acceptance becomes reasonable when SR1-SR4 are implemented, superseded paths are deleted, the actual native variants and allocation counts are testable, representative release results show no material regression, and SR5 is rebuilt from clean commits. Until then, the accurate status is “numerically functional, quality-gate clean, architectural remediation incomplete.”
