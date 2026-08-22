# Phase 14 Stabilizer-Code Analysis Second-Round Review, 2026-08-22

Status: the central Phase 14 capability and most first-round implementation remediation are complete, but acceptance closure remains open for a narrowed resource-guard correction, retained correctness evidence, benchmark-evidence completeness, and small ABI/maintainability cleanup. The previously requested eager-versus-lazy logical-basis A/B is waived by owner decision; eager logical construction is the accepted design for the intended tens-to-thousands-of-qubits workload.

## 1. Scope and evidence boundary

This second-round review checks the current working tree against `docs/vibe/specs/phase-14-stabilizer-code-analysis-spec.md` and the M1-M6/N1-N3 findings in `docs/vibe/reports/phase-14-acceptance-review-2026-08-21.md`. It reviews the Rust core, PyO3 boundary, public Python facade, focused tests, benchmark sources, retained local benchmark records, example, and active status documents.

Per owner direction, correctness and performance suites were not rerun. Existing successful records are accepted as evidence of their reported executions; this review evaluates implementation structure, test coverage, benchmark coverage, and claim-to-evidence traceability. No production source, test, benchmark, example, or specification file was changed by this review.

The owner clarified the `max_bytes` policy during this review: it is an advisory best-effort guard, not an OOM guarantee. Underestimation is acceptable and ordinary execution may exhaust system memory. A cheap exact or conservative-under estimate should be included when available, but the implementation must not add costly accounting or reject an otherwise runnable request because it conservatively overestimated peak memory.

## 2. Executive verdict

The main remediation is real. Error-set analysis now selects the global lexicographically smallest witness; packed classification and correction syndromes are written directly by distinct Rust paths; correction analysis fuses residual XOR with classification; public batch outputs use one final ownership copy; construction reuses trusted reduced rows, the shared packed phase primitive, and a pairing path without repeated pair-span rebuilding; classification skips logical work for syndrome-visible rows and uses packed symplectic coordinates; unused retained canonical/provenance/span state is gone; result records are slotted; and the example includes a non-CSS five-qubit code.

The eager design is accepted. The saved successful Python record covers 64, 256, and 1,024 qubits across repetition, redundant repetition, finite-rate CSS, balanced CSS, finite-rate general, and empty/high-rate families. Its 1,024-qubit public construction medians are approximately 18-42 ms across those families. Together with the removal of the measured repeated-RREF and pair-span bottlenecks, this is sufficient for the owner-selected target scale. A lazy cache would add synchronization, first-access state, and a second ownership path without a demonstrated product need. No lazy implementation or A/B is required for closure, and active documents should record this item as waived rather than pending.

Acceptance is nevertheless not yet closed. The construction guard currently combines upper estimates for data that do not coexist as independently allocated payloads, which conflicts with the clarified no-overestimation policy. The retained test matrix still does not independently bind signed projector semantics and nontrivial logical action as required by the first review. The expanded benchmark records demonstrate broad execution and favorable absolute times, but the sources and records do not retain several required workload distinctions, accuracy metadata, or source-tree module paths. Minor singular-ABI and duplicate-helper cleanup also remains.

## 3. First-round closure matrix

| First-round item | Second-round status | Disposition |
| --- | --- | --- |
| M1 lexicographically smallest error-set witness | CLOSED | `analyze_error_set()` keeps one earliest reference per packed-syndrome group, scans the complete batch, and explicitly compares global `(a,b)` candidates. The exact `[III, IXI, XIX, XXX] -> (0,3)` regression and an exhaustive three-qubit pairwise oracle are retained. |
| M2 dominant construction work | IMPLEMENTATION CLOSED; EVIDENCE PARTIAL | Trusted RREF initialization, packed phase reuse, CSS validation deduplication, and removal of pair-span rebuilding are present. Broad 64/256/1,024 release rows are retained. The eager/lazy A/B is owner-waived, not open work. Component and provenance evidence still needs the narrow cleanup in SR3. |
| M3 direct packed and fused correction execution | CLOSED | Packed/unpacked result types are distinct, packed rows are emitted directly inside released Rust work, correction analysis retains only row scratch, and public arrays receive one ownership copy. |
| M4 construction and batch resource guard | SUPERSEDED; SR1 OPEN | The first review's broad peak-accounting requirement is replaced by the owner policy above. Exact accounting is not required, but the current estimator may overestimate and reject runnable work, so it must be simplified. |
| M5 logical-classification hot path | CLOSED | Syndrome-visible rows return before logical conversion, coordinates use split packed popcount parity, and the retained stabilizer-span solve is gone. |
| M6 independent correctness matrix | PARTIAL; SR2 OPEN | Lexicographic, signed-dependency, batch/singular, packed/unpacked, correction, row-operation, boundary, and logical-invariance coverage improved materially, but the dense independent oracle and input-form matrix remain incomplete. |
| N1 retained state and dead bookkeeping | MOSTLY CLOSED; SR4 OPEN | Duplicate retained canonical/provenance/span representations are gone. One duplicate singular syndrome helper remains. |
| N2 native ABI and ownership | MOSTLY CLOSED; SR4 OPEN | Singular syndrome typing and public one-copy batch ownership are corrected. Optional native logical vectors still cross as Python sequences, and direct native batch ABI coverage is absent. |
| N3 records, examples, and status | IMPLEMENTATION CLOSED; STATUS UPDATE PENDING | Public result records are slotted and the example includes a non-CSS logical pair. Active status documents must be reconciled after SR1-SR4 and must record the lazy decision as waived. |

## 4. Remaining required changes

### SR1 — Simplify `ConstructionEstimate` so the advisory guard cannot reject work through conservative overestimation

Severity: MAJOR contract issue under the clarified owner policy.

Locations: `crates/tencir-pauli-core/src/stabilizer_code.rs:91-210` and `275-282`; resource regression at `tests/test_qec.py:415-442`.

The estimator currently adds complete logical representatives, the complete normalizer basis, logical quotient span, current quotient rows, and projected quotient rows as separate payloads. Several of these rows are moved between containers or exist at different construction stages rather than as independent simultaneous payloads. For an empty code, for example, the estimate counts the normalizer, logical rows, current quotient, and projected quotient at full size. This is intentionally conservative peak accounting, but it can reject a request whose actual construction would fit the supplied advisory limit. The existing `max_bytes=0` regression proves that a nonzero code is guarded but cannot detect over-rejection.

Required minimal change:

1. Replace the summed upper-bound model with cheap quantities that are exact or known not to exceed the corresponding allocation. At minimum, count the two retained original check arrays separately, signs/statistics, and other obvious flat payloads.
2. After elimination has produced the actual `rank`, perform a second cheap arithmetic check for the exact retained relation and eager logical-row payloads. Do not infer those sizes from `maximum_rank`.
3. Omit difficult transient allocator, nested-vector, and exact peak-RSS accounting. Do not add allocation instrumentation, allocator hooks, or a second analysis pass.
4. Add one regression with a limit between the former conservative sum and the cheap retained-payload estimate, demonstrating that a runnable empty/high-rate code is not rejected. Add one heavily dependent-check case so the post-rank relation/logical arithmetic is exercised.

Closure condition: the guard catches cheaply known major retained payloads, may underestimate difficult transient memory, and has no deliberate sum of incompatible worst-case/lifetime maxima that can reject a request solely because of conservative accounting.

### SR2 — Complete the independent correctness evidence rather than relying on production classification for the remaining conventions

Severity: MAJOR acceptance-evidence gap; no reproduced numerical defect.

Locations: `tests/test_qec.py:254-272`, `275-382`, and `445-477`; first-round M6; Spec Section 8.

The current dense projector test uses only the unsigned repetition-code projector. Its logical assertion checks `(P L P)^2 == P`, which also holds for identity or a stabilizer and therefore does not independently prove that the returned representative acts as a nontrivial logical Pauli. The `[XX, ZZ, YY]` signed dependency is tested through the production constructor, but the independent dense oracle is not used to show that `[+1,+1,-1]` has the expected nonzero code space and `[+1,+1,+1]` has the zero projector. Correction differentials exercise code-array batches only; the native/public PauliWord-batch correction path is not retained in the matrix. Five-qubit and Steane checks establish algebraic properties but do not freeze exact expected canonical logical arrays.

Required minimal change:

1. Extend the existing dense helper to evaluate the two signed `[XX, ZZ, YY]` cases independently: assert the consistent projector's exact rank/trace and signed stabilizer action, and assert that the inconsistent sign choice produces the zero projector before separately checking that construction rejects it.
2. For the repetition-code logical pair, use dense projected operators to assert nonzero/non-stabilizer action and the required X/Z anticommutation on the code space, not only involution.
3. Add one PauliWord-batch `analyze_corrections()` differential, including stabilizer, logical, and syndrome-visible residuals. The expected residual remains test-side `correction.multiply(error)` followed by the already required singular comparison.
4. Freeze exact trusted canonical logical packed arrays for at least the five-qubit and Steane fixtures, then retain the existing reordering/row-operation invariance checks. The expected vectors must be reviewed algebraically rather than copied blindly from one production run.
5. Add one small deterministic generated commuting-code matrix only if the fixed fixtures above do not cover all constructor/classification branches; do not create a broad random-test framework solely for count.

Closure condition: the signed phase and nontrivial logical-action conventions are independently bound, both scalable input forms reach correction analysis, and deterministic canonical output has at least one nontrivial exact reference outside the production algorithm.

### SR3 — Make the retained benchmark evidence match the claims actually used for acceptance

Severity: MAJOR acceptance-evidence gap; existing performance direction and reported measurements remain valid.

Locations: `benchmarks/python/test_stabilizer_code_benchmark.py:120-153`, `163-230`, and `233-275`; `crates/tencir-pauli-core/benches/stabilizer_code.rs:86-176`; `.benchmarks/runs/stabilizer-code-remediation-20260821-r2.json`; `docs/vibe/reports/phase-14-acceptance-remediation-2026-08-21.md:9-19`.

The successful record contains 136 Python rows and the Rust Criterion baseline contains 144 rows across the required widths and broad code families. This supports the eager implementation and demonstrates that the rewritten construction path is no longer the prior 0.70-second bottleneck. The retained benchmark interface nevertheless records only output bytes and a few dimensional fields. It does not record the actual `tencirpauli.__file__` and `tencirpauli._native.__file__` values, exact-output accuracy metadata, or an auditable construction-component boundary. Correction rows always use a visible error with identity correction, so the decoder-like zero-syndrome success/logical paths are absent. Error-set rows cover only one repetition workload at 64/256 qubits and do not distinguish unique syndromes, repeated same-coset rows, and multiple conflicting cosets. The deleted temporary component profile is summarized in prose but is not reproducible evidence.

Required minimal change:

1. Record the package and native-extension paths from the benchmark process and assert that they resolve to the current source tree. Keep machine-specific absolute paths only in ignored benchmark results.
2. Add exact equality or `max_abs_error=0` metadata outside the timed region for representative packed/unpacked classification and correction rows. Shape assertions alone are not benchmark accuracy evidence.
3. Parameterize correction rows over residual `SYNDROME_VISIBLE`, `STABILIZER`, and `LOGICAL` cases; retain packed and unpacked forms. This replaces the current single raw-noise-only correction workload.
4. Add compact error-set rows for many unique syndromes, repeated rows in one coset, and multiple conflicting cosets. One manageable 1,024-qubit row is sufficient; do not construct a broad Cartesian product.
5. Retain one reproducible construction breakdown or narrow profiling driver for packing/statistics, signed elimination, normalizer/quotient work, and logical pairing. It may be an explicit manual profiling script and need not add counters to production code.
6. Amend Spec Section 9 to remove exact retained-byte/peak-workspace expectations that conflict with the clarified advisory `max_bytes` policy. Output bytes and cheaply known payloads remain useful; exact peak memory is not a closure condition.

No repeat of the already completed full benchmark matrix is required for this review. After the benchmark source changes, one targeted QEC record is sufficient to retain the new rows and provenance; unrelated suites need not be rerun unless production code changes affect them.

Closure condition: the retained record proves which source-tree extension ran, binds representative outputs to exact values, covers the three correction outcomes and the three error-set grouping regimes, and preserves a reproducible basis for the construction improvement claim without exact memory accounting.

### SR4 — Finish the single-read-back ABI and delete the duplicate singular syndrome kernel

Severity: MINOR maintainability and private-ABI gap.

Locations: `crates/tencirpauli-native/src/qec.rs:39-44`, `187-201`, and `552-565`; `python/tencirpauli/_native.pyi:1022-1083`; `crates/tencir-pauli-core/src/stabilizer_code.rs:622-625`, `1089-1107`, and `1144-1153`; `tests/test_qec.py:445-463`.

The singular syndrome is now a flat NumPy array, but `classify_one()` and error-set logical coordinates still cross PyO3 as optional `Vec<u8>` values and therefore become Python sequences. This leaves the private ABI with two read-back grammars. The existing direct native ABI test checks only singular syndrome and the visible-classification `None` case; it does not bind nonempty logical arrays or representative packed/unpacked batch tuples. Separately, `syndrome_for_words()` duplicates the loop already owned by `syndrome_row_unpacked_into()`.

Required minimal change:

1. Return nonempty singular/error-set logical coordinates as flat one-dimensional NumPy arrays, while preserving `None` for syndrome-visible/no-witness cases. Update `_native.pyi` in the same change; the public facade may continue materializing ergonomic tuples.
2. Extend the private ABI test with one zero-syndrome logical classification, one conflicting error-set result, and one representative packed and unpacked batch tuple. Bind tuple arity, dtype, ndim, shape, contiguity, base ownership, and numeric values without duplicating every endpoint/form combination.
3. Implement `syndrome_one()` by allocating its output once and calling `syndrome_row_unpacked_into()`, then delete `syndrome_for_words()`.

Closure condition: all native vector read-back is flat NumPy, the stub matches runtime values, representative direct ABI tests protect the one-copy design, and only one singular/unpacked syndrome kernel remains.

## 5. Documentation reconciliation after remediation

After SR1-SR4 close, update the active documents together:

1. Replace “acceptance closure deferred pending the eager-versus-lazy A/B” with an owner-decision statement that eager construction is selected and the A/B is waived in `docs/vibe/specs/phase-14-stabilizer-code-analysis-spec.md`, `docs/vibe/specs/README.md`, `docs/vibe/implementation-status.md`, and `docs/vibe/reports/README.md`.
2. Reconcile Spec Section 5.3, which still says classification uses a retained stabilizer span solver, with the implemented quotient-coordinate invariant and the state contract in Section 5.4.
3. Amend the resource and benchmark language so `max_bytes` is explicitly a non-overestimating advisory guard: cheap known payloads are guarded, difficult transient memory may be underestimated, and OOM remains a normal possible outcome for sufficiently large requests.
4. Change the Phase 14 status to acceptance closed only after the focused SR1-SR4 changes and their targeted evidence are retained. Historical first-round and remediation reports remain unchanged as dated records.

## 6. Recommended order

1. Simplify the resource estimate and add the two focused guard regressions.
2. Complete the signed projector, logical-action, canonical-vector, and PauliWord correction tests.
3. Normalize the remaining native vector ABI and remove the duplicate syndrome helper.
4. Extend the QEC benchmark source and retain one targeted record with paths and exact-output metadata.
5. Reconcile the active specification/status documents and mark the lazy A/B as owner-waived.

## 7. Second-round acceptance decision

Phase 14 is functionally implemented and its main algorithms are substantially remediated. The decision not to implement lazy logical construction is accepted and is not a blocker. Acceptance closure should remain open only for SR1-SR4 above; none requires a new architecture, exact memory accounting, a lazy runtime path, or a repeat of unrelated correctness/performance suites.

This review changed no production source, test, benchmark source, example, or specification. It added only this report and its report-index entry.
