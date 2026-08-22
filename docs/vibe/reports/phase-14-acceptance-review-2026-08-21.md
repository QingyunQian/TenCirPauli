# Phase 14 Stabilizer-Code Analysis Acceptance Review, 2026-08-21

Status: functionally implemented, acceptance remediation open. The public Rust/PyO3/Python feature slice is present, but the current tree does not satisfy the complete correctness, packed-execution, resource, performance, maintainability, and retained-evidence contract in `docs/vibe/specs/phase-14-stabilizer-code-analysis-spec.md`.

Reviewed tree: `c957b31` with the Phase 14 worktree changes present. This review added only this report and its report-index entry; production source, tests, examples, specifications, and benchmark sources were not modified.

## 1. Scope and evidence boundary

This review compares the authoritative Phase 14 specification with the current Rust core, PyO3 binding, public Python facade, focused QEC tests, examples, benchmark sources, saved local release records, and trunk status documents. It focuses on ordinary stabilizer-code, CSS, repetition-code, finite-rate, QLDPC-style, batched syndrome/classification, error-set, and external-decoder verification workflows. Exact peak-RSS accounting, implausible allocation extremes, arbitrary-precision algebra, decoder implementation, distance search, circuit semantics, subsystem codes, and speculative abstractions are out of scope.

Per owner direction, the complete test suite and benchmark matrix were not rerun. Existing passing test and release records are accepted as evidence. Two small read-only probes were used only to reproduce specific review findings: the lexicographic conflict-witness defect and the missing logical-storage `max_bytes` guard.

## 2. Executive verdict

The main capability is available. `StabilizerCode` has signed general and CSS constructors, original-order syndromes, deterministic relations and logical representatives, singular and batched classification, supplied-error-set analysis, supplied-correction analysis, flat public read-back, root exports, and Rust/Python benchmark entry points. The separation from `StabilizerState`, the original-check syndrome contract, the shared GF(2) migration, and the basic Python/native ownership direction are sound.

Phase 14 is not acceptance-complete. One public result is demonstrably wrong: `analyze_error_set()` does not always return the specified lexicographically smallest conflicting pair. Existing release evidence also shows that construction dominates the representative workload, while the current construction repeatedly reduces already canonical rows and uses a high-cost logical-pairing path that is not exercised by the finite-rate workloads required by the Spec. Packed classification and correction analysis still materialize complete unpacked syndromes, packed conversion performs scalable work while holding the GIL, correction analysis is not fused, and public batch outputs are copied twice. The construction memory guard omits an exactly estimable retained logical basis. Required independent correctness oracles and representative performance rows remain absent.

The correct repository status is therefore “functionally implemented; acceptance remediation open.” It must not be changed to “acceptance closed” until every MAJOR item below is implemented and retained evidence satisfies the closure checklist, or the owner explicitly amends the corresponding Spec requirement.

## 3. Compliance checklist

| Area | Status | Evidence |
| --- | --- | --- |
| Public module, root exports, and code/state separation | PASS | `python/tencirpauli/qec.py`, `python/tencirpauli/__init__.py`, and `docs/vibe/architecture.md` expose a separate immutable code facade and keep QEC code-space semantics out of `StabilizerState`. |
| Signed commuting-check construction | PASS with evidence gap | The Rust construction tracks signs and provenance and rejects a reproduced `-I` duplicate dependency, but retained tests do not cover nontrivial Pauli multiplication phases or a dense signed projector oracle. |
| Original-order syndrome semantics | PASS | Original packed checks are retained separately and singular/batched syndrome kernels evaluate them in caller order. |
| Deterministic logical representatives | PASS with evidence gap | Canonical full elimination and deterministic null-space selection are implemented; retained tests cover reordering but not the complete row-operation/signed invariance matrix. |
| Singular and batched classification values | PASS with performance remediation open | Existing fixed vectors agree, but syndrome-visible rows still perform unnecessary bit conversion and the zero-syndrome logical path uses scalar per-qubit inner products. |
| Error-set correctability | FAIL public witness contract | Correctability detection is structurally sound, but the early return does not guarantee the lexicographically smallest conflicting pair. |
| Decoder-correction analysis | FAIL fused-path contract | Numerical XOR semantics are present, but full residual arrays are materialized before classification and packed output still goes through an unpacked result. |
| Packed syndrome output | PASS for `syndrome_many()` | The dedicated packed syndrome endpoint returns the expected public representation, although it still allocates one unpacked row per input internally. |
| Packed classification/correction output | FAIL | The core ignores `packed_syndromes`, allocates complete unpacked syndromes, and PyO3 repacks while holding the GIL. |
| Coarse FFI and GIL release | PARTIAL | Public batch operations use one native method, but packed classification conversion is scalable GIL-held work and mixed Python normalization/output copying remains material. |
| `max_bytes` major-output/workspace policy | FAIL | Eager logical storage and other construction state are absent from the estimate; packed classification is estimated as unpacked instead of by the requested representation. |
| Shared GF(2) reuse and Z2 preservation | PASS for implemented migration | The compatible Z2 null-space/rank helpers use `gf2.rs`, and saved records document the regression and its forward-echelon correction. |
| Complete correctness gate | FAIL | The Section 8 projector, random differential, lexicographic oracle, signed-phase, correction, row-operation, and concurrency matrix is not retained. |
| Representative performance gate | FAIL | Saved QEC rows cover only repetition codes at 64/256 qubits and omit 1,024+, finite-rate/general/redundant codes, error-set, corrections, output materialization, and the eager/lazy decision. |
| ABI and maintainability contract | FAIL | Misleading packed fields/parameters, duplicate canonical storage/reduction, double Python copies, and inaccurate singular `_native.pyi` return types remain. |
| Documentation and example closure | FAIL | The Spec and index still say implementation in progress, trunk status overstates gate closure, and the required non-CSS five-qubit example is absent. |

## 4. MAJOR findings and required remediation

### M1 — `analyze_error_set()` returns the first conflict by second index, not the lexicographically smallest pair

Priority: P0 correctness blocker.

Locations: `crates/tencir-pauli-core/src/stabilizer_code.rs:660-724`; Spec lines 128, 386, 469, and 513; `tests/test_qec.py:179-182`.

Current behavior: the implementation scans inputs by increasing `index` and immediately returns when the current syndrome group contains a different logical coset. This minimizes the second component `b` at the moment of discovery, but the public contract compares pairs lexicographically by `a` and then `b` across the complete batch.

The defect was reproduced on the three-qubit bit-flip repetition code with errors `[III, IXI, XIX, XXX]`. Their syndromes are `[[0,0], [1,1], [1,1], [0,0]]`. Both `(1,2)` and `(0,3)` are conflicts with logical difference `XXX`, but the implementation returns `(1,2)` even though `(0,3)` is lexicographically smaller.

Required implementation changes:

1. Do not return when the first conflict is encountered. Scan the complete batch and retain the minimum candidate tuple under ordinary `(a,b)` ordering.

2. Simplify each syndrome group to one earliest reference error. For a fixed syndrome, the lexicographically smallest possible first index is the group's first input index. Compare each later row with that reference until the first nonzero logical difference is found; that pair is the group's unique lexicographic candidate. Any pair starting from a later row is lexicographically worse.

3. Store only `reference_index`, packed reference X/Z words, and an optional completed candidate per syndrome. Delete `ErrorSetEntry`, the per-group logical-coset `BTreeMap`, and retained copies for every distinct coset. They are unnecessary for deciding correctability or the required witness.

4. Track the global best candidate and its already computed logical coordinates while scanning. At the end, return the global minimum or the correctable result. Determinism must come from explicit tuple comparison, not map iteration order.

5. Prefer a packed syndrome key such as `Vec<u64>` rather than `Vec<u8>` so grouping storage and comparison scale with `ceil(nchecks/64)`. Reuse the same packed row kernel required by M3 rather than implementing a second syndrome convention.

Required tests:

1. Retain the exact `[III, IXI, XIX, XXX]` regression and assert `(0,3)` plus the known logical coordinates.

2. Add a small independent pairwise oracle that enumerates every `a < b`, applies the Knill–Laflamme condition through independent `PauliWord` operations, sorts all conflicts, and compares the first pair and logical coordinates with `analyze_error_set()`.

3. Run the oracle on deterministic random small codes/error batches containing multiple syndrome groups, duplicates, stabilizer-equivalent errors, and at least three logical cosets.

Closure condition: the fixed regression and independent pairwise matrix return the exact lexicographically smallest pair and coordinates, and the production implementation does not retain one entry per logical coset.

### M2 — Construction is the measured dominant cost and contains repeated canonicalization and an unnecessarily expensive logical-pairing loop

Priority: P0 representative-performance blocker.

Locations: `crates/tencir-pauli-core/src/stabilizer_code.rs:217-315`, `902-987`, and `989-1120`; `crates/tencir-pauli-core/src/gf2.rs:83-175`; `crates/tencir-pauli-core/benches/stabilizer_code.rs:21-54`; saved baselines `phase14-current-rust-final-20260821` and `phase14-current-python-2-20260821`.

Existing release evidence: at 256 qubits the saved Rust median is approximately 18.878 ms for construction and 0.164 ms for classifying 256 errors, so construction is about 115 times the classification call. From 64 to 256 qubits, construction increases by about 58 times, close to cubic scaling, while classification increases by about four times. The Python construction-plus-classification median is approximately 19.687 ms, making construction roughly 96% of the recorded end-to-end value by scale comparison.

The code has four concrete repeated-work sources. The main elimination reduces above and below every pivot and computes every row-product phase by scanning individual qubits. The resulting canonical rows are then passed through `RrefBasis::from_rows()` for the retained span, again for quotient selection, and again for logical pair tracking. During logical symplectic pairing, each iteration clones the growing `pair_span` and reinserts every projected remaining vector even though the input quotient rows are already an independent basis and symplectic Gram–Schmidt projection preserves independence modulo the extracted pair.

Required profiling changes before the structural edit:

1. Add temporary call-local component timing or a narrow benchmark harness for input packing/statistics, commutation validation, signed elimination, span construction, centralizer null space, quotient selection, and logical pairing. Do not add permanent global counters or atomics.

2. Extend the release source matrix before optimization to CSS and general checks at 64, 256, and 1,024 qubits; include `k=1`, finite-rate `k≈n/2`, high-rate/empty-check, and redundant-check cases. Preserve a pre-change release record on the same machine and thread settings.

Required implementation changes:

1. Preserve canonical pivot columns from the first full elimination and add a narrow trusted `RrefBasis::from_reduced_rows(rows, pivots)` or equivalent internal constructor. It must validate only cheap debug/test invariants, not rerun elimination. Use it wherever the rows are already the canonical RREF produced by the same construction.

2. Before writing a new phase formula, search `word.rs` and existing packed Pauli multiplication for an authoritative phase primitive. Reuse or extract that primitive so signed elimination does not maintain a separate Pauli multiplication convention. The optimized implementation should operate on packed words/popcounts rather than call `code_at_bits()` for every qubit of every row combination.

3. Remove `pair_span` and `projected_span` rebuilding from logical symplectic Gram–Schmidt after proving the invariant: the quotient starts as an independent basis, removing a paired `x,z` and projecting the remaining vectors into their symplectic orthogonal complement preserves independence modulo the stabilizer and extracted pairs. Retain only the projected vectors in their deterministic order.

4. Remove CSS double validation. `from_css_binary()` currently performs explicit X/Z orthogonality and the shared packed constructor then repeats general pairwise commutation. Factor one private construction pipeline that accepts already validated CSS commutation while keeping the public general packed constructor fully validating untrusted input. Do not expose a public skip-validation flag.

5. Run the Spec-required eager-versus-lazy logical-basis A/B using construction-only, syndrome-only, first classification, steady classification, correction analysis, retained bytes, and concurrent first access. If lazy wins the representative syndrome-first and finite-rate workloads without materially hurting classification, use one synchronized `OnceLock`-style cache and delete the eager alternative. If eager remains selected, record the matched evidence and delete the lazy experiment. Do not retain a runtime mode.

6. Keep deterministic pivot and logical output byte-for-byte identical unless the owner explicitly approves a new canonicalization version. Every construction optimization must pass the independent matrix in M6 before its performance result is accepted.

Closure condition: the component profile identifies the dominant term, the selected implementation removes repeated RREF/pair-span work, the complete construction/classification workload improves materially or its remaining cost is explicitly justified, 1,024-qubit and finite-rate rows are recorded, and deterministic outputs remain unchanged.

### M3 — Packed classification and correction analysis materialize unpacked data, repack under the GIL, and do not fuse residual execution

Priority: P0 scalable batch-path blocker.

Locations: `crates/tencir-pauli-core/src/stabilizer_code.rs:520-657` and `726-762`; `crates/tencirpauli-native/src/qec.rs:284-364`, `405-549`, and `606-650`; `python/tencirpauli/qec.py:335-444` and `555-681`; Spec lines 314, 363-365, 402, 465, and 475-479.

Current behavior: `classify_many()` accepts `_packed_syndromes` but ignores it, allocates `batch * nchecks` bytes, and sets `syndromes_packed` to false. The packed PyO3 endpoints then loop over every unpacked bit in `packed_classification_output()` while the GIL is held. `analyze_corrections()` allocates complete residual X and Z batches and calls classification afterward. Python then copies every returned array in `_call_*` and copies it again in `_readonly_copy()` to make the public result owned and read-only.

Required core changes:

1. Replace `syndrome_for_words() -> Vec<u8>` in batch loops with `syndrome_row_unpacked_into()` and `syndrome_row_packed_into()` helpers that write into caller-provided row slices and return whether the syndrome is nonzero. Do not allocate one temporary `Vec` per error.

2. Give packed and unpacked classification distinct truthful internal result variants or endpoints. The chosen type must make it impossible to return unpacked bytes while claiming `syndromes_packed=true`. Remove the ignored boolean and the always-false field pattern.

3. Write packed bits directly in the GIL-released Rust kernel. The PyO3 layer may only validate shapes, borrow normalized arrays, release the GIL, and wrap the resulting flat vector into the requested NumPy shape. It must not perform a `batch * nchecks` conversion loop after reacquiring the GIL.

4. Fuse correction XOR with classification. Reuse one call-local residual row scratch buffer, or compute XOR words directly in the syndrome/logical kernels, and immediately classify each row into final outputs. Do not allocate residual arrays of shape `(batch, word_count)` merely to pass them to another Rust method.

5. Size `max_bytes` from the selected packed or unpacked output, kinds, zero indices, logical-coordinate output, and call-local scratch. Packed mode must not be rejected merely because the corresponding unpacked matrix would exceed the limit.

Required Python/PyO3 changes:

1. Remove the first `np.array(..., copy=True)` layer in `_call_syndrome_many()`, `_call_classify_many()`, and `_call_corrections()`. Either return the native arrays to one common public ownership/read-only finalizer or create directly owned two-dimensional arrays in PyO3. There must be exactly one required ownership conversion, not two.

2. Preserve public `OWNDATA`, C-contiguous, dtype, shape, and `writeable=False` contracts with direct ABI/public tests. Do not weaken the public ownership contract merely to avoid a copy.

3. Keep one coarse Python-to-native method per public batch operation. Do not move packing, residual multiplication, or per-row result construction into Python.

Required benchmarks:

1. Add packed and unpacked `syndrome_many()`, `classify_many()`, and `analyze_corrections()` rows with the same inputs and record kernel time, Python end-to-end time, output bytes, and errors per second.

2. Include a decoder-like batch where most residuals have zero syndrome and a raw-noise batch where most errors are syndrome-visible.

3. Record at least one wide-check case where unpacked output is materially larger than packed output. The packed path must show lower retained/output bytes and must not contain a GIL-held bit loop.

Closure condition: packed results are produced directly in released Rust work, correction analysis has no full residual batch, public arrays receive at most one ownership copy, and packed/correction release rows cover the intended high-frequency workloads.

### M4 — Construction `max_bytes` omits the eager logical basis and other cheaply estimable major storage

Priority: P0 resource-contract blocker.

Locations: `crates/tencir-pauli-core/src/stabilizer_code.rs:109-160` and `277-315`; `build_logical_basis()` at `902-987`; Spec lines 461, 479-481, 496, and 576.

Current behavior: the construction estimate includes copies of original checks, signs, one elimination matrix, and provenance. It does not include retained canonical clones, the duplicate span basis, relations, centralizer/null-space work, quotient work, or eager logical X/Z rows. For an empty-check code the estimate is zero even though construction eagerly retains a complete logical basis.

The defect was reproduced with a supported `[[1024,1024]]` empty-check code. Construction succeeds with `max_bytes=0`; the four split packed logical read-back arrays alone contain 524,288 bytes. This is a deterministic major retained allocation, not exact-RSS or allocator-overhead accounting.

Required implementation changes:

1. Introduce one checked `ConstructionEstimate` calculation owned by the Rust constructor. It must cover original retained checks/signs/statistics, canonical/relation/provenance state actually retained after M2/N1 cleanup, the selected eager or lazy logical cache payload, and the largest clearly dominant elimination/null-space/quotient workspace.

2. Keep the estimate best-effort. Do not model allocator headers, PyO3 wrapper objects, transient reshape views, or exact RSS. Do include exact flat-vector payloads and obvious simultaneous matrices because their dimensions are already known.

3. If lazy logical construction is selected, preflight the eventual retained logical payload during construction and store the approved limit or estimate with the handle as needed. A later first classification must not bypass the constructor's public resource contract.

4. Apply the selected output representation to batch estimates as required by M3. Add the major packed syndrome/coset key storage used by M1 to the error-set estimate.

Required tests:

1. Add a small deterministic limit just below and at the required logical retained bytes for empty-check and finite-rate codes.

2. Assert that `max_bytes=0` succeeds only for genuinely zero-major-allocation cases and fails for an eager/later logical payload.

3. Add packed-versus-unpacked batch limits where packed succeeds and unpacked fails for the expected output-size reason.

Closure condition: every exactly sized major retained vector and dominant workspace has checked arithmetic and a cheap guard, while the implementation remains explicitly best-effort rather than claiming exact peak RSS.

### M5 — The logical-classification hot path converts and compares one bit at a time and performs redundant stabilizer membership work

Priority: P1 high-frequency performance blocker; measure after M2-M4 establish truthful workloads.

Locations: `crates/tencir-pauli-core/src/stabilizer_code.rs:635-648`, `814-852`, and `1045-1093`; `crates/tencir-pauli-core/benches/stabilizer_code.rs:23-51`.

Current behavior: every classification row is converted by `bits_from_words()` before `classify_from_syndrome_and_bits()` checks whether the syndrome is visible, so the common syndrome-visible case pays an allocation and a full qubit scan that is discarded. For zero-syndrome rows, every logical coordinate invokes `symplectic_bits()`, which scans all qubits through scalar bit access instead of using the existing split packed X/Z words and word-level popcount parity. The implementation then separately calls `stabilizer_span.contains()` after computing complete logical coordinates.

The existing benchmark uses only a `k=1` repetition code and single-X syndrome-visible inputs. It therefore does not measure the finite-rate zero-syndrome and decoder-residual path for which logical-coordinate evaluation is central.

Required implementation changes:

1. Compute the syndrome and nonzero flag first. Return `SYNDROME_VISIBLE` without constructing symplectic `bits` or logical coordinates.

2. Store logical representatives in the same split packed X/Z layout used by input words, or maintain one authoritative split representation after logical construction. Evaluate each coordinate through the existing packed symplectic popcount primitive. Do not retain both split and interleaved logical bases after construction merely for convenience.

3. Prove and use the quotient invariant: for a zero-syndrome Pauli, all logical X/Z coordinates are zero if and only if the Pauli belongs to the stabilizer span. Add a differential test against the current `RrefBasis::contains()` result, then classify from coordinate zero/nonzero and delete the retained span solver if it has no remaining consumer.

4. If a span solver remains necessary for another documented operation, do not run it in ordinary classification after the complete coordinates already determine the kind.

Required benchmarks:

1. Separate syndrome-visible, zero-syndrome stabilizer, and zero-syndrome logical batches.

2. Cover `k=1`, finite-rate `k≈n/2`, and high-rate cases at 64/256/1,024 qubits.

3. Report the classification share independently from construction and Python normalization so a construction improvement cannot hide a logical-kernel regression.

Closure condition: syndrome-visible rows perform no interleaved-bit conversion, logical coordinates use packed popcount kernels, redundant membership work is deleted or justified by a separate consumer, and finite-rate release rows demonstrate the resulting path.

### M6 — The retained correctness package does not satisfy the independent Section 8 matrix

Priority: P0 correctness-evidence blocker before M2/M5 algorithm changes are accepted.

Locations: Spec lines 498-519 and 574-582; `tests/test_qec.py:1-230`; `crates/tencir-pauli-core/src/stabilizer_code.rs:1189-1232`.

Current evidence: the public file has six focused test functions and the Rust module has three unit tests. They cover valuable basic repetition, phase-flip, five-qubit, Steane, packed, decoder, boundary, and invalid-input vectors, but there is no dense signed projector oracle, no independent random pairwise correctability oracle, no random singular/batch/correction differential, no nontrivial signed Pauli-phase dependency, no stabilizer row-operation matrix, no logical-coordinate invariance under stabilizer multiplication, and no concurrency/worker matrix. The existing error-set test checks only `(0,1)` and cannot detect M1.

Required retained tests:

1. Add the M1 pairwise error-set oracle and regression first.

2. Add a nontrivial signed dependency using commuting `XX`, `ZZ`, and `YY`: verify that signs `[+1,+1,-1]` produce a consistent relation and `[+1,+1,+1]` produce `-I`. Bind the exact relation bits and canonical behavior.

3. Add a test-only dense projector for small `n`: construct `Π_i (I + s_i P_i)/2`, assert rank/trace `2**k`, and verify representative supplied stabilizers, reported logical pairs, and signed inconsistent cases without calling production GF(2) routines.

4. Generate deterministic small commuting codes and errors, then compare every singular syndrome/classification row with the batched packed and unpacked results. For zero-syndrome errors, multiply by each stabilizer and assert unchanged logical coordinates.

5. Compare `analyze_corrections()` with explicit test-side `correction.multiply(error)` followed by singular classification for word and code-array inputs, including zero-syndrome successes, logical residuals, and residual syndromes.

6. Construct reordered and independently row-operated signed generator sets for the same group. Assert identical canonical logical arrays while asserting the separately expected public syndrome coordinate order.

7. Retain five-qubit, Steane, degenerate/nonidentity-stabilizer, empty/full-rank, and widths crossing 64-bit boundaries with exact numeric assertions rather than shape-only checks.

8. Add concurrent first access and worker-count identity tests only if M2 selects lazy logical construction or a batch path gains parallel execution. Do not invent a worker option solely to satisfy a test bullet when the measured implementation remains serial.

Closure condition: every numerical native entry point has an independent known-correct assertion, the signed/dense conventions are bound outside production GF(2), M1 is covered, and the matrix protects the construction/classification refactors required by this report.

## 5. MINOR and maintainability findings

### N1 — The native state retains duplicate or unconsumed canonical data, and construction performs dead bookkeeping

Locations: `crates/tencir-pauli-core/src/stabilizer_code.rs:82-98`, `230-315`, and `417-475`.

`canonical_bits` is cloned into `stabilizer_span`; `canonical_signs` and `canonical_provenance` are retained after relations and signed validation are complete but are consumed only by public Rust getters with no Phase 14 public use; the local `pivots` vector is populated but not read. `ErrorSetEntry` and the per-coset map become unnecessary under M1. `BatchClassification.syndromes_packed` and `syndrome_width` are misleading when the packed argument is ignored.

Required remediation: after M1-M5 settle the selected eager/lazy and membership designs, list every retained field and its runtime consumer. Delete fields, getters, helper structs, parameters, and result metadata with no documented consumer. If canonical signed provenance is intentionally retained for a future public Rust contract, that contract and read-back use must be added explicitly; otherwise amend Spec Section 5.4 and delete the data rather than retaining it speculatively. Use the existing pivot vector for the trusted reduced-basis constructor from M2 or delete it. Do not silence dead paths with allowances or compatibility wrappers.

Closure condition: one retained representation owns each required concept, no construction-only provenance survives without a consumer, and ignored output-mode fields/parameters are absent.

### N2 — Singular native ABI types are inaccurate and batch ownership is implemented through redundant copies

Locations: `crates/tencirpauli-native/src/qec.rs:167-192` and `568-755`; `python/tencirpauli/_native.pyi:1001-1083`; `python/tencirpauli/qec.py:327-444` and `555-691`.

The stub declares `syndrome_one()` and the syndrome component of `classify_one()` as NumPy `UInt8Array`, while the current PyO3 `Vec<u8>` conversion produces Python `bytes`. Direct native batch arrays are reshape views whose base owns the Rust vector; the public wrapper then performs an intermediate copy in `_call_*` and a second copy in `_readonly_copy()`.

Required remediation: select one singular ABI deliberately. Prefer flat NumPy arrays for scalable syndrome/logical vectors to preserve the repository's single read-back grammar, then convert to ergonomic tuples only in the public Python facade. Update Rust return types and `_native.pyi` in the same change. Add direct private ABI tests for exact Python type, tuple arity, dtype, ndim, shape, C-contiguity, base/ownership behavior, and numeric values. Apply the one-copy output finalization from M3 to batch paths.

Closure condition: `_native.pyi` matches runtime types exactly, direct ABI tests bind the contract, and every public batch array has one justified ownership conversion.

### N3 — Result records and documentation do not match the frozen public/documentation contract

Locations: `python/tencirpauli/qec.py:22-86`; `examples/stabilizer_code.py`; `docs/vibe/specs/phase-14-stabilizer-code-analysis-spec.md:3` and `533-541`; `docs/vibe/specs/README.md:27`; `docs/vibe/implementation-status.md:20` and `35-41`.

The Spec declares frozen slotted result records, but the current records omit `slots=True`. The single example combines repetition-code and external-decoder material but does not include the required non-CSS five-qubit example. The Spec and index correctly say implementation is in progress, while trunk status says the local quality/performance gate passed even though the same status entry acknowledges that the expanded Section 8/9 matrix remains follow-up evidence.

Required remediation: add `slots=True` to the frozen public result records with a focused construction/repr/pickle compatibility check if those behaviors are part of the package contract. Extend the existing example or add one compact non-CSS five-qubit example that verifies the returned logical pair. After M1-M6 and the performance matrix close, update the Spec status, `docs/vibe/specs/README.md`, `docs/vibe/implementation-status.md`, and report index together. Until then use “functionally implemented; acceptance remediation open.”

Closure condition: public record shape matches the Spec, required examples exist, and all active status documents distinguish availability from acceptance closure consistently.

## 6. Required performance evidence package

The existing records are valid but too narrow for acceptance. Do not discard or rerun them merely because of this review; retain them as the pre-remediation reference and add matched rows needed to decide and verify the changes above.

| Workload dimension | Required rows |
| --- | --- |
| Code family | General non-CSS, CSS with both halves, repetition, finite-rate/QLDPC-style, redundant checks, and empty/high-rate boundary. |
| Width | 64, 256, and 1,024+ qubits where the complete estimated peak remains within the local 16 GiB policy. |
| Rank/logical ratio | `k=1`, `k≈n/2`, and high-rate `k≈n`; do not infer finite-rate performance from repetition codes. |
| Construction | Input normalization/packing, validation/statistics, signed elimination, logical basis, total construction, retained bytes, and first logical access for the selected eager/lazy design. |
| Syndrome | Packed and unpacked, small-call overhead and steady batch throughput, original and redundant check counts. |
| Classification | Syndrome-visible, stabilizer, and logical batches; packed/unpacked output; construction excluded from steady rows. |
| Error-set | Many unique syndromes, repeated syndromes in one coset, and multiple conflicting cosets; record grouping bytes and errors per second. |
| Corrections | Successful corrections, logical residuals, and residual syndromes; packed/unpacked; confirm no full residual batch. |
| Python boundary | Code-array normalization, word-array path, native kernel, NumPy output finalization, and complete public call. |
| Baseline/accuracy | Independent vectorized NumPy/GF(2) reference for manageable sizes, exact output equality, source-tree package/native paths, thread settings, and computed maximum error where applicable. |

Acceptance does not require a universal speed ratio or wall-time CI gate. It requires that the dominant cost be identified, the selected fix materially improves the complete representative path or moves a scaling/resource boundary, and every claim includes the Python/Rust boundary and output materialization.

## 7. Ordered remediation plan

1. Add the M1 regression and independent pairwise oracle, then fix lexicographic error-set selection and delete the per-coset retained map.

2. Add the compact M6 signed-phase, dense projector, batch/singular, correction, logical-invariance, and row-operation matrix against the current canonical outputs.

3. Record the missing pre-change construction components and representative 64/256/1,024, finite-rate, packed, error-set, and correction rows. Do not begin a broad construction rewrite without identifying the dominant share.

4. Implement M3's direct packed row sinks, fused correction classification, truthful result types, released-GIL packing, and one-copy Python finalization. Close its numerical, ABI, ownership, and memory tests before construction work.

5. Implement M4's checked construction/output estimates using the post-M3 representations.

6. Apply M2's trusted reduced-basis initialization, shared packed phase primitive, removal of repeated logical pair-span rebuilding, CSS validation deduplication, and eager/lazy decision. Run the M6 matrix after every canonicalization-affecting step.

7. Optimize M5 only after profiles separate visible and zero-syndrome shares. Move logical coordinates to split packed popcount, delete redundant membership work after differential proof, and retain only the authoritative representation.

8. Complete N1-N3 deletion, ABI typing, slotted records, examples, and active status-document reconciliation.

9. Rebuild the source-tree extension in the exact project environment, run the complete repository quality gate, record the post-change release matrix, and compare it with the saved Phase 14 pre-remediation records.

## 8. Acceptance checklist

| Area | Closure evidence |
| --- | --- |
| Error-set correctness | Exact lexicographic regression plus independent pairwise oracle across multiple syndrome groups/cosets. |
| Signed/code-space correctness | Nontrivial signed dependency, dense projector rank/action, five-qubit/Steane/degenerate vectors, row-operation invariance, and logical-coordinate stabilizer invariance. |
| Batch numerical parity | Singular, packed/unpacked batch, error-set, and correction results agree with independent operations on deterministic random small systems. |
| Packed execution | Packed syndrome/classification/correction outputs are written directly in released Rust work with no complete unpacked temporary. |
| Correction execution | No full residual batch; one fused classification path owns XOR, syndrome, kind, and coordinates. |
| Construction performance | Dominant component identified; no repeated RREF of trusted canonical rows; finite-rate and 1,024+ release rows recorded; eager/lazy decision documented. |
| Logical performance | Syndrome-visible rows skip logical conversion; zero-syndrome coordinates use packed popcount; no redundant span solve. |
| Resource policy | Checked estimates cover major retained logical/canonical/relation state and dominant workspaces; packed budgets use packed sizes. |
| FFI/ABI | One coarse call, scalable work outside the GIL, exact `_native.pyi`, direct ABI tests, and one public ownership copy. |
| Maintainability | Abandoned coset/state/result-mode helpers and unconsumed provenance/storage are deleted or have an explicit current contract. |
| Benchmarks | General/CSS/redundant/finite-rate cases at 64/256/1,024+, separated construction/first/steady/output boundaries, accuracy, memory, paths, and thread settings. |
| Documentation | Non-CSS example present; Spec, index, trunk status, and report index consistently state acceptance closure only after the evidence passes. |

## 9. Final acceptance decision

Phase 14's central public capability is usable, and the reviewed syndrome, logical-pair, and basic correction semantics are structurally aligned with the intended design. The phase is not complete against its owner-approved Spec. The lexicographic witness defect alone prevents correctness acceptance; the measured construction dominance, fake packed classification path, unfused correction path, missing major-allocation guard, scalar finite-rate logical path, and incomplete Section 8/9 evidence independently prevent performance and engineering acceptance.

No production source, test, example, specification, or benchmark source was changed by this review. Closure requires implementation and retained evidence for M1-M6, cleanup of N1-N3, a rebuilt source-tree extension, the complete quality gate, and matched release-mode results on the representative matrix above.
