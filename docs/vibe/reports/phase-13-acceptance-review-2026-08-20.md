# Phase 13 Classical Shadow Acceptance Review, 2026-08-20

Status: functionally implemented, acceptance remediation open. The main local-Pauli and global-Clifford capabilities are present and no representative-path numerical defect was reproduced, but the current tree does not satisfy the complete acceptance, resource, validation, and performance-evidence contract in `docs/vibe/specs/phase-13-classical-shadow-snapshots-spec.md`.

Reviewed tree: `c2a7d85` (`Optimize classical shadow hot paths`). Production source, tests, and benchmark code were not modified by this review.

## 1. Scope and evidence boundary

This review compares the authoritative Phase 13 specification, the current Rust/PyO3/Python implementation, the committed correctness and benchmark sources, `docs/vibe/implementation-status.md`, and `docs/vibe/reports/phase-13-performance-review-2026-08-20.md`. It focuses on common scientific workloads, numerical semantics, usable resource scaling, coarse native ownership, and performance claims. Extreme numerical rescue paths, exact RSS accounting, speculative representations, and low-payoff micro-optimizations are out of scope.

Per owner direction, the complete test suite and benchmark matrix were not rerun, and the A/B numbers recorded in the Performance Review are accepted as correct evidence. Before that direction was clarified, focused release-build probes confirmed a two-qubit non-stabilizer global-Clifford joint distribution against explicit commuting projectors, fixed-value checks for both Rényi-2 methods, energy variance and Clifford RDMs, and TensorCircuit measurement-only interchange. These probes found no numerical mismatch, but they are review evidence rather than retained regression coverage.

## 2. Executive verdict

The public/core feature slice is substantially implemented. Both protocols, packed imports and read-back, dense and stabilizer sampling, setting-level estimators, RDM, two Rényi-2 methods, energy variance, stabilizer fidelity, the dense-target extension, and `shadow_bound()` are present. The fused statevector Pauli action, cached Clifford span, and direct stabilizer-RDM accumulation match the intended architecture.

The Performance Review was partially addressed correctly. D1 was measured and reasonably left unchanged; D3/A3, D4/A4, and the fused part of A5 have material matched A/B evidence and are implemented; D5 gained useful dense, stabilizer, many-observable, RDM, and dense-fidelity cases. The review itself still records D2 as unresolved and A1, A2, A6, A7, and A8 as open. Therefore its final statement that Phase 13 is “functionally implemented” is supportable, but it is not evidence that the authoritative Spec is acceptance-closed.

No CRITICAL numerical issue was found. Five MAJOR acceptance issues remain: unbounded shadow Rényi-2 memory growth, an acknowledged frame-generation bottleneck in the scalable stabilizer path, incomplete performance-contract closure around allocation/reuse/parallel execution, missing required regression and benchmark evidence, and a scalable operator clone performed while holding the GIL. Documentation also overstates closure and conflicts with the Spec status.

## 3. Compliance checklist

| Area | Status | Evidence |
| --- | --- | --- |
| Public API and native ownership | PASS | `python/tencirpauli/shadows.py` exposes the required facade and immutable read-back records; estimator work is native and normal pipelines remain handle-to-handle. |
| Local-Pauli sampling and estimators | PASS with evidence gap | The implementation follows setting-major repeats and correlated bitstring sampling; retained tests do not cover the full Spec matrix. |
| Global-Clifford dense sampling | PASS with evidence gap | The frame sampler and sequential conditional projections are implemented at `crates/tencir-pauli-core/src/shadows.rs:731-752` and `1724-1824`; focused review probes found no mismatch. Exact one-/two-qubit exhaustive regression coverage required by the Spec is absent. |
| Stabilizer sampling and fidelity | PASS with performance remediation open | Tableau sampling and overlap kernels are present, but the hot path retains per-measurement allocation and frame construction is the acknowledged dominant large-width cost. |
| RDM and direct accumulation | PASS | The A4 change removes the dominant per-element dense temporary and has accepted 98.4% matched A/B evidence. |
| Both Rényi-2 methods | FAIL resource contract | Arithmetic appears correct, but the shadow method retains every setting matrix and has no memory guard. |
| Energy variance | PASS with evidence gap | The cross-setting square and native `H**2` path match the Spec structurally; required independent retained tests are absent. |
| GIL and coarse FFI | FAIL | `estimate_many()` clones every complete native operator before releasing the GIL. |
| Performance Review remediation | PARTIAL | D1, D3/A3, D4/A4, fused A5, and part of D5 are addressed; D2 and A1/A2/A6/A8 remain materially relevant. |
| Complete correctness gate | FAIL | The required frame, dense differential, Rényi-2, variance, tableau/tape, concurrency, memory, docstring, and TensorCircuit test matrix is not present. |
| Representative benchmark gate | FAIL | Recorded A/B results are useful, but the source matrix does not meet the Spec's complete workload and component-separation contract. |
| Status/document consistency | FAIL | The Spec says “not implemented,” while trunk status says implemented with focused checks complete. |

## 4. MAJOR findings

### M1 — Shadow Rényi-2 retains `O(nsettings * 4**k)` complex data without a guard

Locations: `crates/tencir-pauli-core/src/shadows.rs:1351-1401`, `python/tencirpauli/shadows.py:540-567`, Spec lines 432 and 445-446, Performance Review A8 at lines 115-119.

For `method="shadow"`, the implementation constructs one dense `2**k × 2**k` complex128 matrix for every setting and retains all matrices in `setting_matrices` before reducing any block. The public method has no `max_bytes` argument and the core path performs no memory preflight. At 1,024 settings this retained matrix payload is approximately 1 GiB for `k=8` and 16 GiB for `k=10`, before temporary snapshot RDMs, block sums, allocator overhead, and Python process memory. These are plausible subsystem and setting counts for the documented workflow, not unrealistic coefficient extremes.

This is more than an optional A8 speed hypothesis: it crosses an explicit memory/scaling boundary and conflicts with the Spec's bounded-accumulator and best-effort workspace policy. The U-statistic needs only the current block sum, the accumulated diagonal term, one current setting matrix, and scalar block purities. Balanced blocks are contiguous, so settings can be streamed block by block without changing arithmetic, retained order, or median semantics.

Required remediation: stream each balanced block and keep `O(4**k)` matrix storage rather than `O(nsettings * 4**k)`. Add a cheap peak estimate covering the block sum plus current setting/repeat workspace, using the repository's existing best-effort policy. This is a direct deletion of unnecessary retained state, not a new representation or defensive fallback.

### M2 — The dominant large-width stabilizer cost identified by the Performance Review remains open

Locations: `crates/tencir-pauli-core/src/shadows.rs:710-752`, Performance Review D2 at lines 37-43 and A1 at lines 71-77.

The current symplectic-frame construction repeatedly materializes byte/boolean nullspace rows and recombines each row against the complete current basis. The Performance Review describes this as allocation-heavy, roughly cubic in qubit count with an additional packed-word factor, and states that stabilizer Clifford sampling is dominated by frame generation at tested 64-1024-qubit sizes. Its accepted D1 measurement reports about 2.27 seconds for only two 1,024-qubit settings; extrapolated linearly, the documented 256-1,024-setting fidelity pilot is minutes rather than seconds before estimation.

The Review correctly avoids merging an unmeasured structural rewrite, but leaving A1 as a hypothesis means the scalable/high-performance stabilizer claim is not closed. D2 should not be fixed first merely because its allocations are visible: the Review already says frame generation dominates, so the next performance action should quantify frame time and compare one packed construction variant at 64, 256, and 1,024 qubits. Retain the variant only if complete sampling improves materially or an important scaling boundary moves.

Required remediation: perform the A1 component profile and matched packed-construction A/B requested by the Performance Review. If the gain is small, retain the current algorithm and narrow the performance claim; if the frame remains the dominant cost with a material packed-path win, implement that single change before considering a second tableau representation for D2/A2.

### M3 — The Spec's hot-loop allocation, scratch-reuse, and parallel-worker contract is not implemented

Locations: Spec lines 434 and 466-472; `crates/tencir-pauli-core/src/shadows.rs:1198-1284`, `1671-1721`, `1724-1824`, and `1826-1905`; Performance Review D2, A5, and A6.

Dense Pauli sampling allocates a transformed state and CDF for every setting. Dense Clifford sampling allocates `current` and `scratch` for every repeat. Stabilizer sampling allocates measurement words and intermediate Pauli products in the inner loop. All setting/repeat loops are serial; the Performance Review explicitly notes that the shadows core has no Rayon loop. This directly disagrees with the Spec's requirement that scratch be allocated once per active worker, reused across assigned work, worker count be memory-capped, and hot-loop allocation counters show zero allocation per generator and amplitude.

The accepted A5 A/B proves that fusing Pauli application and expectation is worthwhile at 20 qubits, but it does not implement the “reuse scratch buffers” half of the candidate. D2 and A6 are explicitly unresolved. This does not imply that Rayon or a second tableau must be added blindly: memory bandwidth can erase parallel gains, and the Review's anti-overengineering rule is sound. It does mean the current Spec and the current implementation cannot both be called complete.

Required remediation: first reuse dense `current`/`scratch`/CDF allocations in the existing serial loops, which is local and representation-neutral. Then run the bounded-worker A6 comparison before enabling parallelism. For stabilizer D2/A2, use complete-workload A/B after the frame bottleneck is resolved; if the owner rejects the added representation because the gain is small, amend the zero-allocation acceptance clause explicitly rather than leaving a permanent false requirement.

### M4 — Required numerical and integration regression evidence is largely absent

Locations: Spec lines 409-412 and 436-452; `tests/test_shadows.py:1-150`; absent `tests/test_shadow_tensorcircuit.py`; `crates/tencir-pauli-core/src/shadows.rs` contains no module test suite.

The committed public test file contains six focused tests. They bind basic Pauli/Clifford imports, a few fixed expectations and RDMs, seeded round-trip/fidelity equality, several dense-target fidelity values, one block example, one `shadow_bound` property, and a few invalid inputs. The Spec-required TensorCircuit test file does not exist. There is no retained exact one-/two-qubit frame-distribution proof, dense joint-probability differential, Bell/GHZ/non-stabilizer/toric sampling matrix, numerical test for either Rényi-2 method, energy-variance oracle, stabilizer tape/tableau round trip, concurrency/GIL test, memory-guard test, or docstring contract test.

This finding does not claim that the implementation is numerically wrong, and the owner's statement that the existing tests pass is accepted. It means the status text at `docs/vibe/implementation-status.md:9` and `:17` overstates what the committed regression package proves. A future refactor of the 2,484-line core module could break a phase/sign, q0/MSB, U-statistic, or overlap rule without the present suite detecting it.

Required remediation: add the smallest high-value closure set rather than reproducing every possible fixture: exhaustive 1/2-qubit frame validity/distribution, one explicit dense joint-probability differential with nonlocal signed generators, fixed known-correct values for both Rényi-2 methods and energy variance, one stabilizer tape/tableau round trip, one TensorCircuit measurement-only interchange test, and one concurrent GIL/replay test. These directly cover the most failure-prone conventions and every numerical native entry point.

### M5 — `estimate_many()` clones scalable operator data while holding the GIL

Locations: `crates/tencirpauli-native/src/shadows.rs:137-152`, Spec line 414.

The native binding borrows each Python operator handle and calls `PauliOperator::clone()` before entering `py.allow_threads`. This clone copies every term and packed word, so preparation scales with total observable term data while the GIL is held. The estimator kernel itself releases the GIL, but the repository's non-negotiable rule and the Phase 13 Spec both require complete scalable preparation to occur in the released section.

For the current one-word benchmark rows this cost is small, which is why it does not invalidate the accepted A3 speedup. It becomes an availability and latency issue for many large Hamiltonians or concurrent Python callers, exactly where `estimate_many()` is intended to amortize FFI.

Required remediation: make the native operator payload cheaply shareable for this call, for example by cloning an `Arc` or another constant-size owner under the GIL and performing term-scale work after release. Do not materialize terms into Python or add a second operator representation.

## 5. MINOR findings

### N1 — Status documents conflict and public docstrings do not fully meet the frozen contract

The Spec header at line 3 still says “not implemented,” while `docs/vibe/implementation-status.md:9`, `:17`, and `:44` say implemented and focused checks complete. The accurate current description is “functionally implemented; acceptance remediation open.” The linear estimator docstrings at `python/tencirpauli/shadows.py:432-511` explain blocks and shapes but do not state the local-Pauli versus global-Clifford variance tradeoff required by Spec line 420.

Required remediation: after M1-M5 are resolved or explicitly owner-waived, update the Spec status, trunk status, and report index together. Until then, preserve the distinction between feature availability and acceptance closure. Add one concise variance sentence to the linear estimator documentation rather than duplicating the full statistical discussion.

### N2 — D5 benchmark coverage improved but remains narrower than both the Spec and the Review handoff

The Python benchmark now covers dense sampling through 16 qubits, stabilizer sampling at 64/256 qubits, Clifford `estimate_many()`, Clifford RDM, and dense-target fidelity. The Rust Criterion suite still samples only 4/8/10-qubit dense states, constructs stabilizer snapshots outside the timed estimator closure, and has no isolated frame/action/projection/tableau kernels. Neither layer covers both Rényi-2 methods, energy variance, or stabilizer fidelity, and the Python rows do not provide the required 10/14/18-qubit and 64/256/1,024-setting sweeps or component shares.

Required remediation: treat D5 as partially addressed. Add only the rows needed to decide M1-M3 and validate the advertised common workloads: frame-generation share at 64/256/1,024 qubits, dense scratch/allocation and bounded-worker comparisons at 14/18 qubits, shadow Rényi-2 peak memory/runtime, and stabilizer fidelity. The accepted A3/A4/A5 numbers do not need to be rerun merely for this review.

## 6. Performance Review disposition

| Review item | Current disposition | Acceptance judgment |
| --- | --- | --- |
| D1 duplicate validation/copy | Measured at about 1.6% of the reported 1,024-qubit workload; unchanged | Addressed correctly; no action now. |
| D2 stabilizer hot-loop allocation | Acknowledged; unchanged | Open, but lower priority than dominant frame generation. |
| D3/A3 Clifford span reuse | Implemented; 82.6% reduction at 512 terms | Closed. |
| D4/A4 stabilizer RDM temporary matrices | Implemented; 98.4% reduction at `k=10` | Closed. |
| D5 benchmark gaps | Dense/stabilizer/estimate-many/RDM coverage added | Partially closed; important estimator and component rows remain absent. |
| A1 symplectic-frame construction | No matched alternative/profile closure | Open and high priority because the Review says frame generation dominates large-width stabilizer sampling. |
| A2 packed tableau measurement | No A/B; deliberately deferred | Open; do not implement before A1 establishes the remaining share. |
| A5 fused action and scratch reuse | Fusion implemented with 27.3% 20-qubit end-to-end gain; scratch reuse not implemented | Partially closed. |
| A6 bounded parallel settings/repeats | No Rayon loop or A/B | Open; Spec and implementation disagree. |
| A7 Pauli estimator metadata | No A/B | Acceptably deferred absent evidence of dominance. |
| A8 Rényi-2 streaming/outcome packing | No A/B or implementation | Streaming is now a required resource fix; packed outcomes remain optional. |

## 7. Recommended remediation order

1. Stream shadow Rényi-2 by block and add the cheap workspace guard. This removes the clearest common-workload failure boundary without changing public semantics.
2. Add the compact correctness closure set from M4 before changing frame or tableau internals.
3. Profile frame-generation share and run the A1 packed-construction A/B at 64/256/1,024 qubits. Decide from complete sampling, not an isolated kernel percentage.
4. Reuse dense sampling scratch in the current serial implementation, then run the bounded-worker A6 A/B. Enable parallelism only where throughput improves materially within the memory cap.
5. Reassess D2/A2 after A1. Implement direct packed tableau measurement only if it materially improves complete Pauli and Clifford stabilizer workloads.
6. Move `estimate_many()` term-scale preparation out of the GIL-held section.
7. Reconcile Spec, trunk status, docstrings, and benchmark claims. Close Phase 13 only when each remaining Spec requirement is implemented, measured, or explicitly amended by an owner decision.

## 8. Final acceptance decision

Phase 13 is available and its central algorithms appear numerically sound on the reviewed representative paths. It is not yet complete against its owner-approved Spec. The correct repository status is “functionally implemented; acceptance remediation open,” not “acceptance closed.” Performance Review remediation is real and valuable, especially A3, A4, and the fused A5 kernel, but it did not address the shadow Rényi-2 resource boundary, the dominant large-width frame path, the complete allocation/worker contract, the required validation matrix, or the scalable GIL-held clone.

No source, test, or benchmark file was changed by this review. The only repository changes are this archived report and its report-index entry.
