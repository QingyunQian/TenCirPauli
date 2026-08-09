# Phase 10.5 First-Round Implementation Review, 2026-08-07

Status: open remediation report. The primary interleaved Fermion-factor × Cartesian-Boson Holstein slice, canonical Qudit spin-z charge, and the complete local quality gate are working, but Phase 10.5 is not acceptance-closed against `phase-10.5-spec.md`. The remaining blockers are architectural and hot-path implementation gaps rather than a demand for another small QuSpin timing win.

## 1. Review scope and verdict

This review covers the current uncommitted Phase 10.5 working tree against `docs/vibe/phase-10.5-spec.md`, with emphasis on numerical correctness, actual execution-family selection, common scientific workloads, retained representations, scalable Python/Rust ownership, and avoidable work in native hot loops. It deliberately does not request exact allocator accounting, recovery paths for implausible numeric extremes, new public strategy controls, optimized Block execution, or speculative architecture unrelated to a supported workload.

The implemented primary workload should be preserved. The seven-site interleaved Holstein lazy plan reaches approximately `1.53 s` at one thread versus approximately `1.57 s` for the matched QuSpin matrix-free path, and the six-site eager CSR applies in approximately `22.04 ms` versus approximately `21.27 ms` for QuSpin at one thread while winning materially at four threads and in peak RSS. The owner direction for this review is to accept the small single-thread eager difference as performance parity. Remediation must not spend time forcing another few-percent win unless later profiling exposes the same structural problem as one of the findings below.

The phase nevertheless remains open because several plans report `product` or `cartesian` without using the execution substrate those labels promise. Qubit, Qudit, and fixed-number Boson Product plans still execute through full occupation vectors, generic term application, destination-vector cloning, and per-source hash aggregation. Multiple independent compact factors are not implemented. Unrestricted Cartesian execution remains a source-scatter term interpreter rather than grouped domain-specialized gather. Charge eager construction creates a global COO intermediate before converting it to CSR, structured CSR construction uses a separate full-dimension row-buffer implementation, and public structured `compile("csr")` performs an `O(E)` row-pointer reconstruction loop in Python. Repeated memory checks and repeated validation remain inside these scalable loops.

No production or test source was modified during the review. This report and its `docs/vibe/README.md` index entry are the only review changes.

## 2. Verification performed

- Read the complete Phase 10.5 specification, owner decision log, acceptance evidence, implementation status, changed Rust/PyO3/Python paths, focused tests, routine benchmarks, representation runner, Product thread runner, and isolated QuSpin runner.

- Ran the focused structured/charge/resource suite: 154 tests passed.

- Ran `python scripts/check.py --benchmark smoke` in the project environment. Rust formatting, Black, stage-label validation, Clippy with warnings denied, Ruff, strict mypy, `git diff --check`, the release PyO3 build, 43 Rust tests, 408 Python tests, 8 doctests, three Rust benchmark smoke suites, and 353 selected non-large Python benchmark cases all passed.

- Confirmed at runtime that a sector with independent Fermion-number and Boson-number constraints reports `execution_family="product"`, although the native selector cannot construct more than one `FastChargeMvpPlan` factor and therefore uses the Generic executor.

- Confirmed at runtime that overlapping same-domain constraints also report `execution_family="product"`, because the Python classifier checks only the set of active domains rather than disjoint-factor eligibility or the selected native plan variant.

- Ran a release, one-thread diagnostic for a 16-qubit fixed-Hamming-weight operator with a dimension-one Boson spectator. The reported Product path took approximately `29.0 ms`, while the same numerical action through the existing packed U1 executor took approximately `1.62 ms`; outputs agreed. This approximately `18x` gap is not a new public performance gate, but it demonstrates that the reported Product family is still the generic vector/hash path for this required case.

- Ran a release, one-thread diagnostic for a 12-mode unrestricted Fermion hopping-and-density operator. The raw plan reported `cartesian` and took approximately `1.41 ms`; the explicitly mapped Pauli plan took approximately `0.153 ms`; outputs agreed. This approximately `9.2x` gap likewise demonstrates missing FermionMask-style Cartesian lowering rather than a numerical defect.

## 3. Findings requiring remediation

### R1 — Public Product metadata is not authoritative, and required Product families still use the Generic hot loop

Priority: P0 semantic, architecture, and performance blocker.

Evidence: `python/tencirpauli/charge.py:766-787` labels a restricted plan `product` whenever every individual constraint touches at most one domain. It does not verify that constraints split into disjoint factors, that factor axes are representable by the Product index, or that a native Product plan was selected. `crates/tencirpauli-native/src/charge_sector.rs:131-200` constructs at most one `FastChargeMvpPlan` variant in a priority chain. It cannot compose independent Fermion, Boson, Qubit, and Qudit factors in one plan. The Python label is then copied onto lazy and eager public plans without consulting the actual native variant.

Evidence: `crates/tencir-pauli-core/src/charge.rs:1454-1539` implements `FastQubitMvpPlan::apply_into()` by unranking a complete factor state, decoding the full mixed-radix source vector, applying every original `ChargeTransitionTerm`, cloning every valid destination vector into an `FxHashMap<Vec<u64>, Complex64>`, reranking the factor, and scattering to output. `crates/tencir-pauli-core/src/charge.rs:1624-1707` uses the same generic vector/hash shape for Qudit and fixed-number Boson factors. These types add factor-specific names and rankers, but they do not implement the grouped analytic Product executor required by Sections 10.2-10.4 of the specification.

Evidence: `_fast_qubit_particles()`, `_fast_qudit_factor()`, and `_fast_boson_factor()` at `python/tencirpauli/charge.py:790-839` accept only one constraint and require all axes of the relevant domain to participate with the expected nonzero weight. A zero-charge Qubit, Qudit, or Boson spectator in the same domain disables the fast builder. Multiple separate compact factors also disable every builder. The standard Fermion × Boson Holstein slice is specialized separately and does not close these general Product requirements.

Evidence: non-termwise-conserved but exactly conserved Qubit operators such as `XX + YY` may select `FastQubitMvpPlan`, whose cancellation mechanism is still the per-source destination-vector hash at `crates/tencir-pauli-core/src/charge.rs:1478-1513` rather than compiled net-transition grouping. In layouts that do not match one of the single fast-factor helpers, the same semantic case falls through to the Generic hash path at `crates/tencir-pauli-core/src/charge.rs:3045-3084`. Tests and benchmarks nevertheless assert only `execution_family == "product"`, so they currently validate the Python label rather than a Product grouping implementation.

Impact: users cannot trust the required diagnostic metadata, and several required scientific shapes have Generic scaling while appearing accepted. The problem is not that every plan must beat a separate library by a fixed ratio; it is that a compact factor plan still performs full-state decoding, generic term interpretation, `O(R)` heap-backed destination-key creation, and repeated rank/unrank work that the Product contract explicitly moves outside spectator repetitions.

Required remediation route:

1. Make execution-family selection a Rust-owned result of plan construction. The native plan enum must expose its actual family, and Python must copy that value rather than infer the family independently. If a specialized Product plan cannot be built, report `generic` until the missing capability is implemented.

2. Replace the single-variant `Option<FastChargeMvpPlan>` selector with one Product plan that owns a list of disjoint compact factor descriptors plus Cartesian spectators. The plan must support more than one independent compact factor and preserve public axis order through bounded factor-level mappings rather than a full `D` permutation.

3. Lower Qubit, Qudit, Boson, and Fermion actions into flat factor-local descriptors once. Do not retain original `ChargeTransitionTerm` values merely to call `apply_charge_term()` in every source/term iteration.

4. Group schedules by complete coefficient-free net transformation across all active factors. Evaluate inverse/local actions once per factor destination, aggregate schedule members in canonical order, remove only exact final zero, and apply leakage checks only after the grouped contribution is known.

5. Support Cartesian spectators inside the same domain, including zero-charge Qubit, Qudit, or Boson axes, without routing the plan to Generic. Support separate constraints over multiple domains when their active axis sets are disjoint.

6. Delete `FastQubitMvpPlan` and `FastAdditiveMvpPlan` if they become redundant wrappers around the new shared Product executor. Do not retain them as compatibility or fallback variants.

Closure evidence:

- Add native tests that assert the actual selected enum variant, not only a Python-computed label.

- Add independent dense numerical differentials for Qubit, Qudit, Boson, Fermion, same-domain spectators, two independent compact factors, reversed factor sizes, exact cancellation before leakage, and interleaved public axes.

- Add a regression showing that a deliberately unsupported coupled charge reports `generic`, while every supported independent-factor case reports and executes `product`.

- Retain bounded routine release benchmarks for Qubit, Qudit, Boson, two-factor Product, and the primary Holstein path. Record operation counts, factor dimensions, spectator dimensions, retained bytes, first/steady `apply_into`, and numeric error. No new fixed speed ratio is required beyond eliminating the demonstrated Generic-shaped execution and preventing order-of-magnitude regressions against an existing equivalent native executor.

### R2 — Cartesian execution is a renamed source-scatter interpreter rather than the required domain-specialized executor

Priority: P0 architecture and performance blocker.

Evidence: `crates/tencir-pauli-core/src/structured.rs:2521-2544` classifies every finite structured plan that is not a pure all-binary Pauli plan as `cartesian`. The classifier does not distinguish an optimized Cartesian executor from the legacy structured term interpreter.

Evidence: `StructuredMvpPlan::apply_into()` at `crates/tencir-pauli-core/src/structured.rs:3038-3060` iterates every source column, every symbolic term, and every local operation, then scatters one contribution. It does not group equal net transformations, derive sources from destination configurations, or reuse active-factor work over contiguous spectators. `schedule_count()` is therefore just the original term count.

Evidence: raw Fermion operations use per-operation digit extraction and parity scans in `apply_compiled_structured_operation()` at `crates/tencir-pauli-core/src/structured.rs:2926-2951`. There is no packed required-occupied/required-empty/flip/parity descriptor equivalent to the specification's FermionMask, even though the public plan reports `cartesian` and default raw Fermion execution intentionally avoids mandatory Jordan-Wigner expansion.

Impact: numerical results are correct on the reviewed tests, but the path does not meet the phase's central performance architecture. Mixed Cartesian layouts, unrestricted Fermions, and active-factor-with-spectator cases continue to pay work proportional to symbolic operation traversal for every source. The approximately `9.2x` raw-versus-mapped diagnostic is consistent with this structure.

Required remediation route:

1. Define one validated semantic-lowering IR shared by Cartesian lazy execution, eager graph construction, and explicit sparse materialization. It must carry authoritative Pauli phases, Fermion signs, Boson ladder amplitudes, Weyl phases, finite cutoffs, exact-zero behavior, and public strides.

2. Lower the IR into domain-specific flat descriptors: packed Pauli masks, packed Fermion occupancy/emptiness/flip/parity masks, Boson inverse ladder descriptors with bounded local amplitude tables, and Qudit modular shift/phase descriptors.

3. Group terms by coefficient-free net basis transformation and preserve canonical member accumulation order. `source_term_count` and `schedule_count` must become meaningful distinct diagnostics when grouping occurs.

4. Implement destination-gather or equivalent disjoint destination-range execution using integer indices, masks, strides, and local tables. Move rank, parity, square-root, and phase work outside spectator repetitions whenever the active-factor layout permits it.

5. Route pure Pauli structured inputs to the existing packed Pauli primitive rather than labeling the generic structured loop `pauli`. Reuse existing core primitives before adding another implementation.

6. Delete the old source-scatter structured MVP path after all required domains pass independent numerical and performance gates. Do not keep it as a silent fallback under the `cartesian` label; unsupported layouts must use the explicit `generic` family.

Closure evidence:

- Extend the eleven mixed-domain subset matrix so that it includes sign-sensitive multi-mode Fermion hopping/quartic terms, off-diagonal Boson ladders with cutoff boundaries, nontrivial Pauli `X/Y` phases, Qudit wrap shifts, and cross-domain terms, rather than relying mainly on diagonal one-axis factors.

- Compare lazy Cartesian, eager CSR, dense, COO, and direct independent matrix-vector references for the same small inputs.

- Add bounded release benchmarks for pure Fermion, pure Boson, pure Qudit, Fermion-Boson Cartesian, reversed active/spectator sizes, construction, first apply, steady caller-owned apply, and retained/workspace bytes.

- Preserve the existing raw Fermion semantic contract. The acceptance requirement is an actually specialized native path and absence of an unexplained order-of-magnitude gap, not a demand that raw Fermion execution beat every mapped Pauli case.

### R3 — Eager construction retains parallel builders and a prohibited global COO/Python reconstruction path

Priority: P0 storage, scalability, and maintenance blocker.

Evidence: `compile_charge_transitions_from_prepared_plan()` at `crates/tencir-pauli-core/src/charge.rs:2740-2859` allocates one `Vec` header per destination row, appends all transitions into `row_buffers`, then allocates complete global `rows`, `columns`, and `coefficients` arrays. `NativeChargeEagerMvpPlan::from_coo()` at `crates/tencirpauli-native/src/charge_sector.rs:505-585` scans those arrays again, allocates CSR pointers and converted columns, and retains the final CSR. The intermediate global row vector is precisely the global COO representation prohibited by Sections 4 and 13 of the specification.

Evidence: structured eager construction has a separate memory estimator and builder at `crates/tencir-pauli-core/src/structured.rs:3214-3400`. It traverses source columns, stores every result in `Vec<Vec<(u64, Complex64)>>` indexed by destination row, sorts each row, and then flattens into CSR. This is not the destination-row, bounded row-local scratch builder described by the specification, and it duplicates the charge-sector builder rather than sharing one production CSR sink.

Evidence: public finite structured `compile("csr")` calls the native COO path and then loops over every returned row in Python to reconstruct `indptr` at `python/tencirpauli/structured.py:3320-3340`. This moves `O(E)` matrix-element work back across the FFI boundary and retains a complete row array that direct CSR output does not need.

Impact: the current six-site workload fits and its retained CSR applies well, but the construction data plane contradicts the frozen architecture, performs avoidable full-graph copies and conversions, and leaves three production sparse routes to maintain: charge COO-to-CSR, structured row-buffer-to-CSR, and Python COO-to-CSR. Correctness changes must currently be kept synchronized across them.

Required remediation route:

1. Introduce one core CSR construction sink shared by charge-restricted, structured Cartesian/Product, and Generic families. Family-specific code should generate one deterministic finalized destination row at a time; the shared sink should append columns/values to bounded chunks or growable flat buffers, record the next row pointer, and perform at most one final linear copy when exact-size arrays are required.

2. Use destination-row generation from the same family descriptors used by lazy gather. Equal source columns and exact cancellation must be resolved in bounded row-local scratch. Do not build `Vec<Vec<...>>` for all rows, a global `(row, column)` hash, a global row vector, or a global sort.

3. Return an authoritative native CSR result directly from the core. The PyO3 layer should move or expose flat `indptr`, `columns`, and `values` arrays without reconstructing or revalidating a second graph.

4. Derive COO and dense materializations from the authoritative eager graph in native code. Remove the Python `for row in rows` reconstruction and remove `NativeChargeEagerMvpPlan::from_coo()` once no caller needs it.

5. Keep one fixed retained representation per capability class. Representation A/B remains a development decision; do not introduce a public builder selector or runtime compact-versus-CSR policy.

Closure evidence:

- Add deterministic exact CSR comparisons against independent dense references for Cartesian, Product, Generic, cancellation, and leakage cases.

- Add a construction-residency test or internal counters proving there is no global COO row array, no `D`-element vector-of-vectors row store, no second transition-generation pass, and no retained temporary chunks after return.

- Test failure before publication under a small `max_bytes`, retry where the facade permits it, sorted columns, exact-zero removal, and equality of eager direct MVP with CSR MVP.

- Re-run the six-site eager construction/RSS measurement after the data-plane change. Performance parity with QuSpin remains accepted; the purpose is removal of redundant storage/work and preservation of current steady MVP behavior.

### R4 — Repeated validation and memory accounting remain inside scalable hot loops

Priority: P1 hot-path performance and maintainability blocker.

Evidence: the non-termwise Generic lazy path calls `check_bytes()` after every destination insertion at `crates/tencir-pauli-core/src/charge.rs:3062-3065`, although the destination aggregate cardinality is already bounded by the immutable term/schedule count and can be preflighted once per plan or source.

Evidence: charge eager construction calls `charge_csr_memory_estimate()` after every emitted transition at `crates/tencir-pauli-core/src/charge.rs:2827-2839`; structured eager construction performs the corresponding full estimate after every transition at `crates/tencir-pauli-core/src/structured.rs:3345-3357`. The primary six-site graph has approximately `22.5 million` final transitions, so this design performs at least that many repeated multi-step checked estimates even though allocated capacity changes only at buffer-growth boundaries.

Evidence: structured eager and dense paths call `apply_structured_operation()` for every operation of every term of every source at `crates/tencir-pauli-core/src/structured.rs:3310-3325` and `3529-3551`. That function rechecks axes, local dimensions, Pauli codes, Fermion mode mappings, and operation kinds at `crates/tencir-pauli-core/src/structured.rs:3627-3754`, despite receiving operations derived from a trusted native handle. Lazy structured MVP separately validates and compiles the same semantics into `CompiledStructuredOperation`, leaving two implementations that can diverge.

Evidence: generic charge term application repeatedly converts and bounds-checks Fermion/Boson/Qudit mode identifiers and code lengths in `apply_pauli()`, `apply_fermions()`, `apply_bosons()`, and `apply_qudits()` at `crates/tencir-pauli-core/src/charge.rs:402-587`, although `prepare_charge_transition_plan_layout()` already validates immutable layout and term metadata.

Impact: these are not boundary validation checks protecting arbitrary user input. They are repeated checks over immutable handles in loops scaling with `D`, `R`, or `E`. They add branches and checked arithmetic to the exact paths Phase 10.5 is intended to accelerate, while the duplicate semantic implementations increase correctness risk.

Required remediation route:

1. Validate public arrays, dimensions, codes, mode indices, cutoffs, and canonical structure once during plan construction. Convert all hot descriptors to native `usize`, masks, strides, direct table offsets, and prevalidated enum variants before the plan is published.

2. Preflight bounded per-source/per-row scratch from the compiled schedule count. Remove `check_bytes()` from the per-term destination insertion loop.

3. Check CSR temporary growth only when a chunk or backing buffer capacity grows, or at deterministic row/chunk checkpoints. Do not recompute the complete peak estimate for every transition.

4. Reuse the compiled semantic operation implementation in lazy execution, eager construction, dense, COO, and CSR. Delete `apply_structured_operation()` or reduce it to a construction-time compiler helper; do not maintain raw and compiled numerical kernels in parallel.

5. Preserve checked arithmetic for dimensions, index products, row pointers, buffer lengths, and major allocation estimates. Preserve physical boundary tests, exact cancellation, and leakage checks. This finding does not authorize `unsafe`, unchecked public input, removal of allocation guards, or weakened numerical semantics.

6. Profile before adding any new micro-optimization. The immediate deletions above are justified by exact call counts and the frozen architecture; further SIMD, hashing, or threshold work still requires representative release evidence.

Closure evidence:

- Add a repository audit listing every surviving execution-time validation or memory check in the affected kernels and why it cannot be moved to construction or capacity growth.

- Add counters or focused tests demonstrating zero term-validation passes per apply, zero full memory-estimate calls per transition, and bounded checks per row/chunk instead.

- Run before/after release measurements for Generic cancellation, Qubit Product, structured Cartesian, and six-site eager construction. Record the share of total time changed; do not claim a material win if the removed checks are not a dominant term.

### R5 — Acceptance evidence and status documents overstate the implemented capability envelope

Priority: P1 closure-evidence and handoff blocker.

Evidence: `docs/vibe/implementation-status.md:7`, `23`, and `33` describe a unified structured eager CSR builder and completed native compact Qubit/Qudit/Boson Product factors with Cartesian spectators. The reviewed implementation instead has distinct charge and structured builders, a global charge COO conversion, and generic vector/hash execution behind several Product labels.

Evidence: `benchmarks/manual/representation_ab.py:81-99` records only a small Pauli sample, one two-mode Cartesian Boson sample, and one fixed-number Boson sector without a Cartesian spectator. It does not record every implemented eager capability class, Generic irregular CSR, compact-sector mask graphs, Product spectator graphs, or a candidate-versus-CSR median for each class.

Evidence: `examples/research/holstein/run_tencirpauli.py:199-206` and `run_quspin.py:150-157` perform one measured first apply and then the timed repetitions. They do not perform the specification's two untimed warmups. The current parity-level conclusion remains acceptable under the owner direction, but the runner should match the documented protocol before it is reused for a future small percentage claim.

Evidence: several benchmark error fields compare `plan.apply_into()` with another call to the same plan, for example `benchmarks/python/test_native_mvp_resources_benchmark.py:298-378` and `benchmarks/manual/representation_ab.py:72`. Correctness tests provide independent references for some small cases, but replay equality must not be described as independent numerical benchmark evidence.

Impact: another implementer could treat the remaining work as only a 3.6% QuSpin timing issue and leave the actual execution/storage gaps untouched. The current evidence also cannot distinguish a genuinely specialized Product plan from a Generic plan carrying a Product label.

Required remediation route:

1. Update implementation status immediately after R1 makes metadata authoritative. Until then, describe only the optimized Fermion-factor × Boson-spectator slice as performance-complete and describe Qubit/Qudit/Boson Product support as numerically correct but not yet on the required executor.

2. Expand the representation record to every retained capability class that remains after remediation. Delete records for losing production variants rather than retaining runtime selectors.

3. Add two untimed warmups to both isolated comparison runners and retain at least seven timed samples. Record machine, environment versions, thread variables, dimensions, exact workload, and numerical preflight.

4. Keep the accepted interpretation of the existing QuSpin results: seven-site lazy is at parity/slightly lower, six-site eager single-thread is at parity/slightly higher, four-thread and memory are materially favorable, and no further few-percent optimization is required for closure.

5. Replace replay-only accuracy metadata with an independently computed error where a benchmark claims numerical evidence. It is acceptable for a pure timing benchmark to omit an error field when a separately identified correctness test covers the exact workload semantics.

6. Correct `phase-10.5-acceptance-evidence.md` and `implementation-status.md` only after the corresponding source changes and measurements exist. Do not change wording alone to close this review.

Closure evidence:

- A remediation table must map R1-R5 to exact commits/files/tests/benchmarks and mark each item open or closed.

- The final acceptance document must distinguish numerical correctness, actual executor selection, representation choice, construction memory, steady MVP, and external comparison evidence.

- The full quality gate must pass after deletion of old paths, and a repository search must show no production Python loop scaling with sparse entry count and no obsolete execution-family inference.

## 4. Ordered remediation roadmap

### Work package A — Restore truthful plan selection and diagnostics

1. Add an authoritative native execution-family getter to every relevant plan variant.

2. Remove `_restricted_execution_family()` as a production classifier; retain only any small constraint-normalization helpers that the native compiler genuinely needs.

3. Add negative tests proving unsupported or coupled layouts report `generic` rather than `product` or `cartesian`.

4. Keep this change small and land it before performance refactoring so subsequent benchmarks cannot confuse Generic and specialized results.

### Work package B — Build the shared semantic IR and complete Cartesian/Product executors

1. Reuse existing Pauli, charge-sector, rank/unrank, Fermion sign, Boson ladder, and Weyl primitives before writing new logic.

2. Introduce one construction-time semantic IR and compile it to Cartesian or Product descriptors.

3. Implement the missing Qubit/Qudit/Boson and multiple-factor Product cases, then the unrestricted Cartesian cases, with net-transition grouping and spectator reuse.

4. Differentially test each vertical slice against an independent small dense reference before benchmarking it.

5. Delete `FastQubitMvpPlan`, `FastAdditiveMvpPlan`, the old structured source-scatter executor, and any helper made unreachable by the new descriptors in the same changes that replace them.

### Work package C — Replace all eager routes with one direct native CSR data plane

1. Implement the shared destination-row CSR sink over the new family descriptors.

2. Move charge eager, structured eager, explicit CSR, COO expansion, and dense expansion onto the authoritative graph.

3. Remove global COO output from eager construction, `from_coo()`, full-dimension row buffers, and Python CSR reconstruction.

4. Verify construction peak accounting at chunk/capacity boundaries and preserve the existing public best-effort `max_bytes` contract.

### Work package D — Remove hot-path defensive checks and duplicate kernels

1. Move immutable validation and identifier conversion into lowering.

2. Replace per-term/per-transition memory checks with plan preflight and bounded row/chunk growth checks.

3. Delete raw structured numerical execution after lazy/eager/dense/sparse consumers share the compiled kernel.

4. Profile the affected paths and retain only checks whose safety or public semantic purpose is documented.

### Work package E — Rebuild acceptance evidence without reopening accepted performance parity

1. Run the focused independent correctness matrix and full quality gate.

2. Run bounded capability benchmarks for actual Cartesian, Product, Generic, eager, and CSR paths.

3. Run the representation matrix for every retained capability class.

4. Re-run isolated QuSpin tooling with two warmups and seven timed samples. Treat results comparable to the current report as passed; investigate only material regressions introduced by remediation.

5. Update implementation status and acceptance evidence to describe actual variants and remaining limits.

## 5. Non-goals and guardrails for the remediation

- Do not implement optimized Block execution, arbitrary Qudit level-charge tables, model-name dispatch, public strategy controls, exact peak-RSS accounting, or a second CSR builder.

- Do not retain old Generic-shaped paths under specialized labels for compatibility. Public numerical behavior is the compatibility contract; private executor structure may be replaced.

- Do not add automatic compact-versus-CSR selection, table-versus-analytic public options, or fallback recovery after a fixed representation exceeds `max_bytes`.

- Do not weaken checked dimension/index/allocation arithmetic, exact cancellation, deterministic accumulation order, finite Boson boundaries, Fermion signs, Qudit phases, or strict leakage behavior.

- Do not require TenCirPauli to beat QuSpin by a fixed percentage where current matched results are already comparable. Correctness, honest execution-family selection, removal of obvious scalable overhead, and absence of material regression are the closure priorities.

## 6. Acceptance recommendation

Do not close Phase 10.5 by changing only the benchmark wording or accepting the current public family labels. The primary Holstein path and Qudit charge work are valuable and should remain, and the current QuSpin comparison is sufficient under the owner direction. Phase closure requires R1-R4 to be implemented and R5 to be rebuilt from the resulting code: authoritative native family selection, actual specialized Cartesian and general independent-factor Product execution, one direct native CSR data plane, removal of repeated trusted-handle validation and per-transition memory estimates, deletion of superseded parallel kernels, and evidence that exercises the real selected variants.

After those changes, the current passing quality gate and accepted QuSpin parity can be reused as the baseline. A material regression in the primary Holstein or eager CSR workload must be investigated, but another marginal single-thread speed win is not a prerequisite.
