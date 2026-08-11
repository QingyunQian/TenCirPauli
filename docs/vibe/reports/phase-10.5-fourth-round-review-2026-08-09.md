# Phase 10.5 Fourth-Round Acceptance Review, 2026-08-09

Status: not acceptance-closed. The D29 closure statement is not supported by the current production route or by the current benchmark provenance. Numerical correctness and the complete repository quality gate remain strong, and the CSR construction remediation is substantially real, but general Product execution still fails the required matched-variant performance and architecture gates; the Cartesian/Fermion and wide Generic zero-allocation claims still lack valid instrumentation; and the live closure documents overstate the evidence.

Reviewed baseline: commit `c50038bac50d` plus the uncommitted Phase 10.5/D29 worktree present on 2026-08-09. As directed by the owner before the third round, clean-commit provenance is excluded from the technical verdict. This review did not modify production source, tests, benchmark implementations, or recorded benchmark data. Only this report and its `docs/vibe/README.md` index entry were added.

## 1. Executive verdict

The third-round implementation added real capabilities: coefficient-free Product schedule keys, reusable scalar/fixed-width accumulators, Structured active-axis/suffix groups, the shared `FermionMaskDescriptor`, direct packed-Pauli native selection, heap-based CSR merge, CSR phase statistics, the chunk-scaling runner, and a reproducible six-site construction profile. The complete local smoke workflow passes.

Phase 10.5 is nevertheless still open. The highest-priority defect is not a marginal benchmark gap. The public selector disables the specialized Qubit/Qudit/Boson paths whenever a general Product descriptor is available and disables the specialized spinful Fermion path for mixed Fermion-Boson Product layouts. On the current five-site mixed workload, the selected general Product kernel is approximately `10.4x` slower than the retained specialized Fermion kernel with one thread and approximately `34.4x` slower with four threads, while the outputs agree to approximately `4.2e-17`. On the exact seven-site primary lazy workload, the current selected Product median is approximately `22.017 s`, not the `1.461 s` stated in the acceptance document; it is approximately `14.7x` slower than the cited `1.502 s` QuSpin median. The selected Product kernel shows no useful thread response because `FastProductMvpPlan::apply_into()` is serial; the Rayon destination-gather implementation remains in `FastFermionMvpPlan`, not in the general Product plan. This directly contradicts the no-regression condition required before a shared replacement may displace a specialized executor and fails the formal primary lazy performance gate.

Several diagnostic claims are also not trustworthy yet. The Product fallback counter can report zero while the fallback loop executes, the four Generic counters are never incremented anywhere, the Structured legacy counter is likewise never incremented, and the CSR chunk runner does not perform the cross-capacity CSR equality check attributed to it by the acceptance document. A zero from an unwired counter is not structural evidence.

The acceptance disposition is therefore:

| Third-round item | Fourth-round disposition | Reason |
| --- | --- | --- |
| T3-1 general Product convergence | **FAIL** | The default route regresses representative specialized kernels by an order of magnitude, remains serial/source-scatter, falls back for multi-factor layouts, and exposes an incorrect fallback counter. |
| T3-2 Cartesian/Fermion convergence | **PARTIAL** | Shared FermionMask lowering, packed-Pauli selection, active-axis groups, and removal of the production raw-Fermion interpreter are real; allocation evidence and the required raw-versus-mapped phase profile are absent, and packed-source extraction is still repeated per schedule member. |
| T3-3 wide Generic allocation | **PARTIAL** | The vector-key insertion was replaced by fixed-width preallocated storage, but the counters used as closure evidence are unwired, scalar/wide end-to-end instrumentation is incomplete, and no coupled-wide before/after record exists. |
| T3-4 CSR scaling evidence | **PASS WITH EVIDENCE CORRECTIONS** | The heap merge, memory estimate, chunk sweep, and six-site phase/RSS profile reproduce. The runner must still compare CSR arrays across capacities, use fresh processes for per-case RSS if RSS scaling is claimed, and rename the allocation-only `final_copy_nanos` metric. |
| T3-5 truthful handoff | **FAIL** | D29, acceptance evidence, and implementation status claim technical closure despite the Product regression and missing/wired evidence; the cited Product thread results do not describe the current selected kernel. |

## 2. Verification performed

- Read the complete Phase 10.5 third-round report, formal specification, owner decision log through D29, acceptance evidence, trusted-handle audit, implementation status, affected Rust/PyO3/Python sources, focused tests, and manual benchmark runners.

- Rebuilt the current worktree with `maturin develop --release --locked` and independently ran `python scripts/check.py --benchmark smoke`. Formatting, stage-label checks, Clippy with warnings denied, Ruff, strict mypy, `git diff --check`, 52 Rust tests, release native build, 416 Python tests, 8 doctests, three Rust benchmark smoke suites, and 353 bounded Python benchmark cases passed.

- Ran the focused charge, structured, and native-resource files after the release rebuild: 162 tests passed.

- Re-ran `benchmarks/manual/product_u1_16_ab.py` for the same 16-mode, four-particle action. The Generic, general Product, and specialized Qubit medians were approximately `0.520 ms`, `0.0427 ms`, and `0.00400 ms`; all outputs agreed. General Product was approximately `12.2x` faster than Generic but approximately `10.7x` slower than the specialized Qubit executor. Retained estimates were `366,736` bytes for Product and `34,984` bytes for specialized Qubit.

- Re-ran `benchmarks/manual/product_thread_ab.py` on the current five-site, cutoff-three, two-particles-per-spin workload (`D=102,400`). The actual native variant was `product`. One-thread and four-thread medians were approximately `55.36 ms` and `55.47 ms`, respectively, rather than the `23.87 ms` and `6.79 ms` recorded in the acceptance document. This is consistent with the absence of Rayon from `FastProductMvpPlan`.

- Ran a matched same-process diagnostic on that same five-site operator and sector, changing only the private selector hook so both retained native variants could be measured. With one thread, general Product was approximately `55.52 ms` and `208,320` retained bytes versus specialized Fermion at approximately `5.34 ms` and `65,508` retained bytes. With four threads, general Product was approximately `55.76 ms` versus specialized Fermion at approximately `1.62 ms`. The maximum output difference was approximately `4.16e-17`. The hook was restored after each construction and no production state was changed.

- Re-ran the exact seven-site, cutoff-three, three-particles-per-spin lazy command used by the primary acceptance workload with one thread, two warmups, and seven steady samples. The actual variant was `product`, the seven steady samples were `21.934-22.057 s`, and the median was approximately `22.017 s`; first apply was approximately `21.950 s`, retained estimate was `2,571,064` bytes, peak RSS was approximately `1.36 GB`, and complete setup plus ten applications took approximately `228.4 s`. The current median is approximately `15.1x` slower than the `1.461 s` TenCirPauli value and approximately `14.7x` slower than the `1.502 s` QuSpin value cited by the acceptance document. This is a direct failure of the primary lazy gate on the current selected route.

- Ran a two-factor Qubit-number × Boson-number Product differential. The result was numerically exact, but the debug counters after one apply were `(18, 0, 0, 6)`: factor work was repeated over the combined component, no spectator segment was used, and the general fallback was invoked once per factor component.

- Ran a three-site mixed Fermion-Boson Product differential containing both Fermion-local and Boson-active schedules. Its counters were `(207, 27, 0, 0)`. The zero fallback count is false as an execution statement: `needs_fallback` is true for the Boson-active groups and the fallback loop runs, but the counter increments only when `direct_layout.is_none()`.

- Re-ran `benchmarks/manual/csr_chunk_scaling.py`. For the same 1,120-transition workload, target counts `1/8/64/512` produced `1/8/63/374` chunks, `1,120` sorted and merged entries, and `2,240` heap operations in every case. This reproduces the recorded structural scaling.

- Re-ran the exact six-site eager construction command in a fresh process. It reproduced `D=1,638,400`, `E=22,527,980`, `1,375` chunks, generation approximately `6.576 s`, sorting approximately `0.355 s`, heap merge approximately `0.816 s`, final-array allocation call approximately `11.2 us`, retained CSR `553,778,728` bytes, estimated construction peak `1,635,202,672` bytes, and peak RSS approximately `1.45 GB`.

## 3. Findings and required remediation

### F4-1 — CRITICAL: the public Product selector displaces much faster retained variants before convergence

Third-round mapping: T3-1 items 4 and 5 remain open; T3-5 is reopened by invalid performance mapping.

Evidence: `python/tencirpauli/charge.py:1250-1269` clears the specialized Qubit, Qudit, and Boson candidates whenever `_supports_independent_factor_product()` succeeds. For Hybrid inputs, `python/tencirpauli/charge.py:1325-1347` passes the specialized spinful candidate only when no Product factor descriptor exists, so an eligible mixed Fermion-Boson layout is forced onto general Product. `crates/tencir-pauli-core/src/charge.rs:2616-2940` implements `FastProductMvpPlan::apply_into()` as a serial source-oriented loop. The Product implementation contains no Rayon iterator. The destination-owned Rayon loop at `charge.rs:1340-1445` belongs to `FastFermionMvpPlan`.

Impact: the current selected five-site general Product route is approximately `10.4x` slower than specialized Fermion with one thread and approximately `34.4x` slower with four threads. The exact seven-site primary route is approximately `14.7x` slower than the cited matched QuSpin median. The 16-mode Qubit case is approximately `10.7x` slower and approximately `10.5x` larger than the specialized Qubit executor. Existing specialized builders remain in the core and private ABI, but the public selector bypasses them for the very cases on which their A/B win is material. This is the opposite of the third-round rule: a specialized path may be deleted or displaced only after the shared kernel is numerically equal and shows no representative regression.

Required repair, in order:

1. Apply an immediate containment change at the selector boundary. Define fixed capability-class winners from measured evidence and keep the specialized executor for every class where it wins materially. At minimum, the balanced compact spinful Fermion factor with Cartesian Boson spectators must continue to select `native_variant="fermion"` until general Product passes the same-workload gate. Do not add an operator-name or Holstein branch; eligibility must remain algebraic and layout-based.

2. Apply the same rule to Qubit, Qudit, and Boson compact factors. The current 16-mode result already rejects automatic displacement of the Qubit executor. Run the matched matrix before deciding whether each specialized variant is retained or deleted. A dormant duplicate is not closure; either make it the fixed measured winner for its capability class or delete it after Product wins.

3. Change `product_thread_ab.py` and every acceptance runner to require an expected `native_variant` and fail immediately on a mismatch. Recording the field without asserting it is insufficient because the stale thread table was accepted under the coarse `execution_family="product"` label.

4. Re-run the five-site and seven-site public routes after containment. Record actual native variant, construction, first/steady apply, retained bytes, peak RSS, numerical error, and explicit thread settings in fresh processes. Do not reuse the current Product thread table.

5. Do not switch a fixed capability class back to general Product until a reverse-order, same-machine release A/B demonstrates numerical equality and no material representative regression. Product Rayon may remain only if the owner-approved `1.5x` speedup gate is met without `O(pD)` workspace.

### F4-2 — MAJOR: general Product still lacks the complete active-shape/destination-gather execution contract

Third-round mapping: T3-1 items 1-3 remain only partially closed.

Evidence: `product_direct_layout()` at `charge.rs:2963-3018` returns a direct layout only when there is exactly one compact factor and that factor occupies one contiguous public-axis range. Any two-factor Product, reversed factor layout, or other non-direct factorization enters the fallback source/spectator loop at `charge.rs:2785-2937`, reconstructs occupation vectors, and may call the general sector ranker. The reproduced two-factor counter tuple `(18, 0, 0, 6)` confirms that identity compact factors are not reused as spectator shapes.

Evidence: even when `direct_layout` exists, groups with an active Cartesian spectator enter the loop at `charge.rs:2785-2937`. `fallback_invocations` increments only under `needs_fallback && self.direct_layout.is_none()` at `charge.rs:2773-2782`, so the reproduced mixed plan reports zero fallback while executing the fallback. The regression in `tests/test_charge.py:728-754` currently asserts this false zero.

Evidence: `ProductScheduleGroup` groups coefficient-free destination fingerprints and preserves member order, which is useful, but the execution remains source-scatter. It does not own destination ranges, derive inverse active-factor sources, or provide the destination-disjoint baseline required by Sections 10.2, 10.3, and 11 of the formal specification.

Required repair:

1. Extend each Product group with an explicit active shape: active compact factors, inactive compact factors, active Cartesian axes, suffix/prefix segment metadata, and inverse local descriptors. The group must describe `A_active tensor I_inactive` directly rather than treating every compact factor rank as active.

2. Replace the single-factor-only `ProductDirectLayout` gate with a general Product index that maps factor-component ranks and Cartesian digits to public sector rank without reconstructing a complete occupation vector in the scalable loop. Reuse the existing factor rank/prefix-completion metadata; if a bounded factor-to-public permutation is required, construct it once and account for it as retained factor metadata rather than rebuilding/ranking per contribution.

3. Implement the optimized serial baseline as destination gather. For each destination active tuple, derive at most one source tuple from inverse local descriptors, evaluate grouped amplitudes in canonical member order, remove exact zero, and apply the result across the inactive-factor/Cartesian segment. A term acting on every factor may have no reuse, but it must still use Product indices and local descriptors instead of Generic occupation reconstruction.

4. Use a reusable aggregate only for groups with more than one member or genuine destination collision. Single-member injective groups should write directly. Capacity must be fixed during construction; growth must fail in diagnostics rather than allocate silently.

5. Define counter semantics before using them as acceptance evidence. `factor_action_evaluations` must count actual active-factor evaluations, `spectator_segment_applications` must count completed segments, `destination_key_allocations` must count real capacity-growth/owned-key events, and `fallback_invocations` must increment every time the fallback body executes. Add `ranker_invocations` or an equivalent counter so tests can prove that a factorable plan does not call the general sector ranker in its hot loop.

6. Add structural scaling tests. Increasing an identity spectator dimension or an inactive compact-factor dimension must increase segment elements but not active-factor evaluations, destination-key allocations, or ranker calls. Cover single factor, two factors, reversed factor sizes, interleaved public axes, Fermion-Boson, Qubit-Boson, Qudit spectators, and exact cancellation before leakage.

### F4-3 — MAJOR: the specialized-convergence A/B matrix required by T3-1 was not completed

Third-round mapping: T3-1 item 4 and the 16-mode explanation in item 5 remain open.

Evidence: `benchmarks/manual/product_u1_16_ab.py` provides a real Generic/Product/specialized-Qubit comparison and correctly reports actual variants. It still shows general Product approximately `10.7x` slower than specialized Qubit. The acceptance document lists a pure-Fermion Product-versus-specialized number, but no committed runner reproduces that exact comparison. `product_lazy_precompile_ab.py` compares Product with Generic, not Product with specialized Fermion. No matched general-Product-versus-specialized Qudit or Boson record is present.

Required repair:

1. Add one committed variant A/B harness covering Qubit, Qudit, Boson, pure balanced Fermion, mixed balanced Fermion with Boson spectators, two independent factors, reversed factor sizes, and interleaved axes. Each child process must construct the same public operator/sector and change only the explicitly recorded internal variant selector.

2. Assert `native_variant` for every record, compare outputs against an independent dense/CSR oracle on bounded cases, and report construction, first apply, seven steady samples, retained bytes, peak RSS, factor/spectator dimensions, group count, and normalized counters per apply.

3. For the 16-mode U1 case, profile the measured gap instead of merely listing it. Separate combinatorial rank/unrank, factor table lookup, group traversal, segment copy/accumulate, and Python/PyO3 overhead. Retain specialized Qubit if the remaining Product overhead is avoidable or materially larger.

4. Record an explicit owner decision per capability class after the matrix. There must be one production winner, not a slower default plus faster dormant code.

### F4-4 — MAJOR: Cartesian/Fermion implementation advanced, but the required allocation and phase evidence is still absent

Third-round mapping: T3-2 items 1 and 2 are substantially implemented; items 3 and 4 remain open.

Evidence accepted: `crates/tencir-pauli-core/src/fermion_mask.rs` now owns the shared width-generic descriptor. Structured, charge schedules, Product, and the specialized Fermion plan consume it. The old raw-Fermion operation interpreter is `#[cfg(test)]` only at `structured.rs:3331-3373`; the production non-Fermion fallback at `structured.rs:3541-3609` cannot execute Fermion sign logic. Pure binary Structured plans select the packed native variant and focused numerical differentials pass.

Evidence missing: no allocator-backed test or meaningful allocation counter covers one-word and multiword end-to-end apply. The mode-128 test proves numerical behavior but not allocation behavior. `StructuredApplyCounters::legacy_interpreter_invocations` is read at `structured.rs:2501-2516` but is never incremented, so its zero value does not prove that an instrumented branch was avoided. No current benchmark reports the required raw-Fermion versus mapped-Pauli shares for packed-word extraction, parity, index update, input load, and output write.

Remaining hot-work concern: `apply_compiled_schedule_member()` calls `FermionMaskDescriptor::pack_index()` for each Fermion schedule member at `structured.rs:3561-3588`. The caller owns one scratch buffer, so this is allocation-free inside the member, but packed-source extraction may still be repeated across groups and members for the same source block. The missing profile is necessary to decide whether source-word reuse, a one/two-word specialization, or loop reordering is justified.

Required repair:

1. Replace the unwired legacy counter with construction-time structural metadata that is actually populated, such as total Fermion schedules, schedules lowered to FermionMask, and descriptor-ineligible schedules. Reject or label a supported plan if these counts do not match. Keep the repository-wide residual search proving that the production raw-Fermion interpreter is absent.

2. Add an allocator-backed Rust regression in a serialized test binary. Warm construction, preallocate caller scratch, reset a thread-local allocation counter, and exercise one-word and three-word/mode-128 `pack_index`, `amplitude_from_words`, `apply_flip_to_index`, and `apply_to_words` loops. Assert zero allocation in the inner application methods. For end-to-end plan apply, assert that allocation count is constant per call and does not scale with basis states or term count; the contract is zero per-term/per-state allocation, not necessarily zero setup allocation per public call.

3. Add a reproducible 12-mode raw-versus-mapped runner using the frozen hopping-and-density fixture from the earlier reviews. Report actual native variants, first/steady apply, numerical error, and release single-thread phase measurements for packed extraction, parity, index update, input load, and output accumulation. Use batched phase microbenchmarks or coarse instrumented kernels so timer calls do not dominate individual contributions.

4. Profile before changing loop structure. If packed extraction dominates, compute source words once per source block and reuse them across compatible Fermion schedule groups. If parity dominates, specialize the common one/two-word representation behind the same descriptor API. If input/output bandwidth dominates, record that fact and do not add speculative tables.

### F4-5 — MAJOR: wide Generic storage is fixed, but its closure counters and benchmark are not valid evidence

Third-round mapping: T3-3 item 1 is implemented; items 2 and 3 remain open.

Evidence accepted: `WideDestinationAccumulator` at `charge.rs:380-442` stores destination digits in one preallocated flat buffer, `insert()` uses `extend_from_slice()` within construction-time capacity, and the old per-candidate `destination.to_vec()` key allocation is absent. The unit test confirms capacity reuse, and the 65-mode coupled Generic numerical regression passes.

Evidence missing: `ChargeApplyCounters` declares raw-term, identifier-conversion, descriptor-revalidation, and destination-key-allocation atomics at `charge.rs:259-275`, but there is no `fetch_add` for any of them in the repository. The wide test at `tests/test_charge.py:101-137` therefore proves only that four default atomics remain zero; it does not prove that the intended wide path executed without allocation. There is no corresponding scalar Generic counter regression and no genuinely coupled wide-sector release before/after record.

Required repair:

1. Instrument events that can actually occur. At minimum record scalar inserts, wide inserts, accumulator capacity-growth events, owned destination-key creations, raw schedule fallback invocations, and descriptor validation invocations after plan publication. Pass the diagnostic handle into both lazy apply and eager CSR construction so both consumers are covered.

2. Make the test prove path execution: a scalar coupled case must report scalar inserts greater than zero and zero growth/owned-key/raw-interpretation events; the 65-mode case must report wide inserts greater than zero and the same forbidden-event zeros. Run repeated apply and eager construction, not only one allocating `apply()`.

3. Add an allocator-backed check around the accumulator itself, verifying unchanged capacities and zero allocations across repeated clear/insert/drain cycles at full planned width/capacity. The test callback must not call `to_vec()` while allocation counting is active.

4. Add a committed coupled-wide manual runner and record the pre-change vector-key baseline versus fixed-width current code from separate worktrees/commits. Report construction, first/steady lazy apply, eager construction, peak RSS, width, selected dimension, schedule count, candidate inserts, and numerical equality. Do not retain the vector-key implementation as a second production path solely for benchmarking.

### F4-6 — MINOR: CSR implementation closes the principal blocker, but two evidence claims should be corrected

Third-round mapping: T3-4 implementation and required representative measurements are closed.

Evidence: the source-major sink uses a binary heap, coalesces cross-run duplicates, removes exact zeros, and includes chunk headers, cursors, and heap metadata in the best-effort estimate. The target `1/8/64/512` sweep and exact six-site fresh-process profile reproduce the acceptance values.

Remaining evidence issues: `benchmarks/manual/csr_chunk_scaling.py:41-80` constructs and deletes each eager plan but never calls `csr()` or compares the returned arrays across capacities, while `phase-10.5-acceptance-evidence.md` attributes deterministic output equality to the sweep. The small Rust sink regression does prove repeated equality for a synthetic case, but it is not the same release workload. The runner also executes all target capacities sequentially in one process, so per-record high-water RSS is not an independent per-capacity measurement. Finally, `final_copy_nanos` at `csr.rs:257-260` times only `Vec::with_capacity()` calls; direct final emission occurs inside `merge_nanos`, and there is no final copy.

Required repair:

1. Materialize `(indptr, columns, values)` for each capacity in the release runner and compare every record byte-for-byte or array-for-array with the one-chunk reference. Also compare a deterministic MVP result so ABI conversion is covered.

2. If RSS scaling is reported per target, make the parent runner launch one fresh child process per capacity. Otherwise label RSS as the cumulative process high-water mark and do not compare records as independent peaks.

3. Rename `final_copy_nanos` to `final_array_allocation_nanos`, or add separate final-emission accounting. Do not describe the allocation-call time as copy cost. Preserve the direct-emission implementation; adding a real second copy solely to make the phase name literal would be a regression.

### F4-7 — MAJOR: live status and acceptance evidence must be reopened and tied to actual variant records

Third-round mapping: T3-5 remains open.

Evidence: `phase-10.5-acceptance-evidence.md:3,69`, `implementation-status.md:7,23`, and D29 state technical closure. The current five-site public Product record contradicts the accepted thread table, the current matched A/B rejects the claimed no-regression convergence, and the exact seven-site current Product median is approximately `22.017 s` rather than the cited `1.461 s`. The remediation table maps broad R1-R5 labels to files rather than mapping T3-1 through T3-5 to exact functions, executed counters, numerical tests, and identifiable benchmark records. The complete smoke workflow does not execute the critical manual Product, raw/mapped Fermion, wide Generic, chunk-scaling, or isolated QuSpin acceptance cases; its success cannot close those gates.

Required repair:

1. Reopen Phase 10.5 in `implementation-status.md` and the acceptance evidence before further optimization. Preserve D29 as a historical decision entry, but append a superseding owner entry after the fourth-round measurements; do not silently rewrite the historical record.

2. Mark the existing five-site thread table and seven-site Product/QuSpin record as requiring rerun unless their raw records prove the same current `native_variant` and worktree. The current five-site rerun already shows the table is stale for the selected general Product path.

3. Give every manual acceptance record a stable label containing the repository commit, dirty marker or worktree fingerprint, native variant, thread environment, workload manifest version, and exact command. Machine-specific raw outputs may remain ignored under `.benchmarks/`, but the acceptance document must cite the label and transcribe enough fields to audit the conclusion.

4. Replace the broad remediation table with a T3-1 through T3-5 closure matrix. Each row must name the exact source function, structural assertion/counter whose nonzero activity proves the intended path ran, independent numerical test, manual record label, and owner decision that accepts or retains a specialized variant.

5. Keep manual performance local and informational; do not add wall-time CI gates. Add deterministic semantic guards to ordinary tests: expected native variant, factorable-versus-Generic selection, counter path activity, zero capacity growth, and exact CSR equality. These guards prevent the same false closure without introducing flaky timing thresholds.

## 4. Ordered remediation plan

The following order minimizes risk and prevents new measurements from being taken on a route that should not be the default.

### Step 0 — Reopen and contain the regression

- Mark Phase 10.5 technically open and invalidate the stale Product thread/QuSpin mapping.

- Restore fixed specialized winners for capability classes where the current A/B shows a material win, starting with balanced compact Fermion plus Cartesian Boson spectators and the 16-mode Qubit sector.

- Add expected-native-variant assertions to the manual runners before collecting more data.

Exit gate: the public five-site workload selects the documented winner, matches the numerical oracle, and no longer exhibits the current order-of-magnitude avoidable regression.

### Step 1 — Make Product instrumentation truthful

- Correct fallback counting for both `direct_layout=None` and partially segmentable direct layouts.

- Add ranker-call and actual accumulator-growth counters.

- Add one-factor-active-spectator and two-factor structural tests whose expected counts are derived analytically.

Exit gate: every counter has a reachable increment site and at least one test proving the increment; zero-valued acceptance counters are paired with positive activity counters showing that the intended executor ran.

### Step 2 — Implement the complete Product active-shape/destination-gather baseline

- Generalize Product indexing beyond one contiguous factor.

- Compile inverse group descriptors and active/inactive factor shapes.

- Remove full occupation reconstruction and general rank calls from factorable hot loops.

- Apply identity factors and Cartesian spectators through contiguous or explicitly strided segments while preserving public order.

- Add Rayon only after the optimized serial baseline is measured; retain it only after the `1.5x` gate and memory/determinism checks pass.

Exit gate: single-factor, two-factor, reversed, interleaved, and primary mixed workloads have zero Generic/ranker fallback where the Product index is sufficient, numerical equality, deterministic counters, and no material regression versus the retained specialized winner.

### Step 3 — Close Fermion and Generic allocation evidence

- Add serialized allocator-backed tests for FermionMask and scalar/wide accumulators.

- Replace unwired counters with activity plus forbidden-event instrumentation.

- Add and run the raw/mapped Fermion phase profile and coupled-wide Generic before/after runner.

Exit gate: one-word/multiword Fermion and scalar/wide Generic paths show positive intended-path activity, zero per-term/per-state allocation growth, numerical equality, and profile-backed explanations for remaining overhead.

### Step 4 — Correct CSR evidence and rerun acceptance records

- Add cross-capacity CSR equality to the chunk runner, fresh-process RSS where claimed, and accurate phase names.

- Re-run the six-site profile, full Product variant matrix, five-site thread matrix, exact seven-site lazy comparison, representation matrix, raw/mapped profile, coupled-wide Generic case, and isolated QuSpin protocol on the same reviewed worktree.

Exit gate: every cited record identifies the actual native variant and worktree, every claimed equality is computed by the cited runner, and no stale record is reused across a selector or kernel change.

### Step 5 — Final quality and handoff

- Run `python scripts/check.py --benchmark smoke` after the focused structural and manual acceptance cases pass.

- Update the T3 closure matrix and append the superseding owner decision.

- Keep clean-commit creation as the separate owner operation already excluded from technical acceptance.

## 5. Required final evidence matrix

| Area | Required numerical evidence | Required structural evidence | Required performance/resource evidence |
| --- | --- | --- | --- |
| General Product | Independent dense/CSR equality for Qubit, Qudit, Boson, pure/mixed Fermion, two-factor, reversed, and interleaved layouts; exact cancellation before leakage | Actual `native_variant`, active-factor counts, segment counts, zero unexpected ranker/fallback/allocation events | Matched Product/specialized/Generic construction and first/steady A/B, retained bytes, peak RSS, 1/2/4/physical threads, current five-site and seven-site workloads |
| Cartesian Fermion | Ordered CAR, quartic, complex, interleaved, mode-128, lazy/eager/dense/CSR equality | Fermion schedule count equals FermionMask-lowered count; allocator-backed one/multiword inner-loop zero allocation | Frozen 12-mode raw/mapped end-to-end ratio and phase shares |
| Generic | Coupled scalar and 65+-axis wide lazy/eager equality | Positive scalar/wide insert activity; zero capacity growth, owned-key allocation, raw interpretation, identifier conversion, and revalidation after publication | Coupled-wide before/after time, RSS, width, schedule/candidate counts |
| CSR | Exact arrays and MVP equality across chunk capacities; cancellation and empty-row regressions | Ordered-row or heap mode, chunk count, heap operations, exact-zero removal, no retained chunks | 1/8/64/512 target sweep; fresh six-site generation/sort/merge/allocation/emission, retained bytes, peak RSS |
| Handoff | Full 52-Rust/416-Python/8-doctest gate or current successor counts | Exact function/test/record mapping for T3-1 through T3-5 | Record labels with commit/worktree, actual variant, thread environment, and workload manifest |

## 6. Acceptance recommendation

Do not accept Phase 10.5 at the current worktree. Preserve the shared FermionMask, packed-Pauli native selection, compiled charge schedules, fixed-width accumulator implementation, ordered-row/heap CSR builder, and reproduced CSR evidence. Immediately reopen the Product route and live documents, restore measured specialized winners as fixed capability-class implementations, then complete the shared Product architecture and variant A/B before reconsidering convergence.

No clean-commit requirement is added by this review. The blocker is technical and evidentiary: the currently selected Product kernel is materially slower than an already present numerically equivalent executor, its counters do not faithfully describe fallback execution, and the acceptance records do not identify a current equivalent route.
