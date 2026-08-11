# Phase 10.5 Sixth-Round Acceptance Review, 2026-08-09

Status: acceptance remains open for raw-Fermion performance and the complete directly comparable QuSpin capability matrix. The fifth-round Product inactive-factor remediation is accepted, and the current six-site eager result is accepted as ordinary performance parity rather than treated as a blocker.

Reviewed baseline: commit `ae99869246200d2b0e2007d6c1318acab3704f9e` with only the pre-existing untracked `.claude/` content outside the commit before this report was written. Clean-commit provenance remains outside the technical verdict.

## 1. Executive verdict

The fifth-round Product blocker is closed. General Product now evaluates active compact factors independently of inactive compact-factor volume, preserves public order without a selected-space permutation, reports zero Generic fallback/global-ranker/destination-key allocation activity, and remains numerically exact. On the current release build, the representative two-factor Product median is approximately `2.83 us` versus `3.92 us` for Generic, while the reversed-factor Product median is approximately `2.37 us` versus `5.42 us` for Generic.

The raw-Fermion blocker is not closed. Hoisting packed-source extraction reduced the previous approximately `20x` regression, but the current one-thread frozen profile still measures approximately `0.325 ms` for the direct raw `FermionMask` path versus `0.231 ms` for explicit Jordan--Wigner packed-Pauli execution. Direct raw execution is therefore approximately `41%` slower. The owner direction is stronger than the previous ambiguous "no material regression" wording: the direct raw-Fermion path should be faster than explicit Jordan--Wigner execution. If it cannot become faster after profile-led remediation, retaining it as the default has no performance justification and any reversion to mapped execution would require an explicit superseding decision for D09.

The external comparison policy is also clarified. Every directly equivalent optimized QuSpin capability in the formal development matrix must receive a matched comparison. TenCirPauli should be faster. A repeatable slowdown of at most `10%` may be accepted as parity when correctness, memory, and scaling remain sound, but it must not be described as a speedup. A repeatable slowdown above `10%` is a blocker and is treated as evidence of an implementation problem that requires profiling and improvement. Under this policy, the current six-site eager measurements, approximately `5.14%` and `6.04%` slower than QuSpin in two independent seven-sample runs, are accepted and should not consume further optimization effort by themselves.

Phase 10.5 should remain open until the raw-Fermion path beats the explicit Jordan--Wigner packed-Pauli reference and the complete fair QuSpin capability matrix is executed under the clarified `10%` parity ceiling.

## 2. Verification performed

- Ran `python scripts/check.py --benchmark smoke` in the project environment. Rust formatting, Black, stage-label validation, Clippy with warnings denied, Ruff, strict mypy, `git diff --check`, the release native build, 55 Rust tests, 426 Python tests, 8 doctests, three Rust benchmark smoke suites, and 353 bounded Python benchmark cases passed.
- Re-ran the fifth-round Product structural regression through the complete Python suite. Expanding an inactive compact factor preserves factor-action, group, and aggregate counts while increasing only identity-segment applications; numerical dense equality and zero forbidden counters pass.
- Re-ran `product_variant_matrix.py` for the two-factor and reversed-factor cases with one thread, two warmups, and seven steady samples. Product is numerically equal to the dense reference and faster than Generic in both cases.
- Re-ran `fermion_mask_phase_profile.py` on the current HEAD. Raw and mapped actions agree to approximately `2.20e-14`; packed extraction occurs 4,096 times for 139,264 schedule-member evaluations. The one-thread medians are approximately `0.325 ms` raw and `0.231 ms` mapped. After source packing was hoisted, the diagnostic shares are approximately `3.1%` packed extraction, `36.1%` parity, `31.4%` index update, and `29.4%` input/output.
- Re-ran the seven-site lazy isolated comparison twice with the pinned QuSpin environment. TenCirPauli/QuSpin medians were approximately `1.503/1.521 s` and `1.506/1.522 s`; the primary lazy path remains narrowly faster and selects `execution_family="product"`, `native_variant="fermion"`.
- Re-ran the six-site eager/explicit isolated comparison twice. TenCirPauli/QuSpin medians were approximately `21.333/20.289 ms` and `21.685/20.451 ms`, corresponding to approximately `5.14%` and `6.04%` slowdowns. Dimensions and transition counts match the expected `D=1,638,400` and `E=22,527,980`; TenCirPauli peak RSS remains approximately `1.45 GB` versus approximately `8.8-9.0 GB` for QuSpin.

## 3. S6-1 — BLOCKER: direct raw Fermion execution must beat explicit Jordan--Wigner execution

### Evidence and diagnosis

The fifth-round implementation solved the measured repeated-packing defect: the contiguous binary path now packs each source once rather than once per schedule member. The new profile shows that packing is no longer dominant. Continuing to optimize source extraction alone would target only about `3%` of the descriptor profile and cannot close a `41%` end-to-end gap.

The current hot loop still evaluates validity and parity separately for every group member, follows descriptor indirection, and reads a transient `D`-entry packed-source cache. For a contiguous one-word binary layout, the public-index bit orientation and every required/empty/parity/flip mask are known at construction time. This permits a faster one-word view derived from the authoritative `FermionMaskDescriptor` without introducing a second CAR/sign implementation.

### Required remediation

1. Profile the current one-thread raw and mapped paths at the group/member loop level. Record group count, member count, direct single-member count, validity tests, parity popcounts, destination XORs, input reads, output writes, allocation bytes, and representative hardware counters where available. Quantify the share of the end-to-end gap before changing the kernel.
2. Add a construction-time contiguous one-word view owned or produced by `FermionMaskDescriptor`. Convert required-occupied, required-empty, source-parity, and flip masks into public-index bit orientation once. The descriptor remains the sole authority for lowering and sign semantics.
3. Evaluate `source = destination XOR flip_mask` directly in public-index orientation. Combine occupied/empty validity into the minimum mask comparisons, absorb the fixed sign into the prepared coefficient, and perform one parity popcount only for surviving members.
4. Flatten prepared group members to remove repeated `Option` checks and pointer chasing. A single-member injective group writes directly; multi-member groups aggregate in canonical order and remove exact zero before writing.
5. Delete the `D`-entry packed-source cache if the direct public-index mask A/B wins. This also removes its `8D` transient workspace and the current problem that this major apply workspace is not checked against `max_bytes`. If the cache remains, its bytes must be checked and reported explicitly.
6. Keep the width-generic multiword descriptor path for wide or non-contiguous layouts. A one-word specialization must be reached through the same descriptor API and must not duplicate CAR canonicalization, fixed-sign derivation, or parity-mask construction.

### Exit gate

- The frozen 12-mode hopping-and-density raw and explicitly mapped plans assert their actual native variants and agree numerically within the existing tolerance.
- Two warmups and seven release-mode steady `apply_into` samples show the raw direct median strictly below the explicit Jordan--Wigner packed-Pauli median at one thread and at the declared fixed multi-thread setting.
- Separate representative hopping, density-density, general quartic, complex-coefficient, ordered-CAR, interleaved-layout, and multiword correctness cases remain passing. At least hopping/density and quartic performance rows must demonstrate that the win is not peculiar to one term mixture.
- Per-state or per-term allocation growth remains zero. Any retained per-call workspace is checked against `max_bytes`, reported by the benchmark, and justified by an A/B win.
- If the raw path cannot beat mapped execution after the dominant cost is addressed, Phase 10.5 remains open until the owner explicitly chooses either further optimization or a D09-superseding mapped-default policy.

## 4. S6-2 — BLOCKER: complete the fair QuSpin capability matrix

### Acceptance policy

For every directly equivalent optimized QuSpin representation, use the same machine, complex dtype, basis dimension, sector, coefficients, boundary conditions, starting-vector policy, thread settings, warmup count, sample count, and measured boundary. Numerical equivalence is a precondition for timing. TenCirPauli should be faster; a repeatable ratio `TenCirPauli / QuSpin <= 1.10` is acceptable parity, while a ratio above `1.10` requires investigation and improvement. A parity result is not a speedup claim.

Cases that would require a QuSpin Python callback, callback-heavy `user_basis`, a different sector, or nonequivalent Weyl semantics are not fair optimized baselines. They must be labeled explicitly as nonequivalent and retain internal before/after, throughput, memory, and scaling evidence instead of a fabricated cross-library speed claim.

### Required coverage

The matrix must cover, where QuSpin has a directly equivalent optimized representation:

- pure Cartesian Fermion execution;
- pure finite-cutoff Boson execution;
- Qubit/spin execution in full and fixed-Hamming-weight spaces;
- Fermion--Boson Cartesian execution;
- fixed Fermion number with Boson spectators;
- fixed Boson number with Fermion spectators;
- independent factorized constraints over two or more factors;
- the reversed-factor-size Product case;
- the primary seven-site lazy Holstein workload;
- the primary six-site eager/explicit Holstein workload;
- any intentionally Generic coupled-charge case for which QuSpin provides an actually equivalent optimized basis, otherwise an explicit no-fair-baseline classification.

Finite Qudit/Weyl and other cases without a directly equivalent optimized QuSpin representation must still remain in the internal capability matrix, but they do not receive a QuSpin pass/fail ratio.

### Benchmark implementation requirements

1. Extend the separate-process harness beyond the Holstein-only manifest. Use capability-named frozen manifests and environment-specific child runners; TenCirPauli and QuSpin must not be imported into the same process.
2. Record operator/sector/plan construction, first apply, seven steady samples after at least two warmups, retained bytes, construction peak, measured RSS, thread variables, numerical error, actual native variant, execution family, and exact command for every row.
3. Record caller-owned TenCirPauli `apply_into` as required by the formal specification. If QuSpin lacks an equivalent caller-owned output API, also record a separate allocation-inclusive matched public-call comparison; do not substitute that result for the required `apply_into` record.
4. Alternate subprocess order or use a deterministic ABBA schedule for near-threshold cases. A ratio near `1.10` must be confirmed in independent fresh-process records before acceptance or rejection.
5. Validate each mapping before timing through action equality after the documented basis permutation, an independent dense/CSR reference for bounded cases, or matched energy/residual checks where full action comparison is impractical.

### Exit gate

- Every required capability has either a directly comparable QuSpin record or a justified no-fair-baseline classification.
- No directly comparable row is repeatably more than `10%` slower than QuSpin.
- Any row between parity and a clear win is labeled parity and makes no speedup claim.
- Rows above `10%` are profiled against their dominant cost, remediated, and rerun; changing workload semantics or using a slow QuSpin callback is not an acceptable way to pass.

## 5. Accepted items and documentation corrections

The fifth-round B5-1 Product inactive-factor work is accepted and does not need further remediation unless the wider capability matrix reveals a new material regression. The six-site eager result is also accepted under the clarified `10%` parity ceiling; the historical exact `5%` D22 wording and Section 17.4/19 wording must be reconciled with this owner direction before final closure.

The complete quality gate passes on the reviewed commit. The seven-site primary lazy path remains faster in two current matched runs, although the margin is approximately `1-1.2%` rather than the older recorded approximately `11%`; the live acceptance evidence should use the current records and describe the result conservatively.

`implementation-status.md` and the Phase 10.5 acceptance evidence currently overstate closure while the raw-Fermion gate and full QuSpin matrix remain open. Do not append a new technical-closure decision or mark the phase accepted until S6-1 and S6-2 pass. At closure, reconcile D09, D22, the formal specification, the acceptance evidence, and the implementation status in one documentation change rather than leaving contradictory thresholds.

## 6. Ordered remediation

1. Profile and optimize the raw contiguous Fermion kernel until it beats explicit Jordan--Wigner packed-Pauli execution, preserving the shared descriptor and all numerical/allocation regressions.
2. Generalize the isolated QuSpin harness into the required capability matrix and classify every unsupported comparison explicitly.
3. Run the complete quality gate, the full capability matrix at one thread and the declared fixed multi-thread setting, the representation matrix, and the primary seven-/six-site acceptance workloads on the same reviewed worktree.
4. Update the formal specification and D22 wording to the owner-approved `10%` parity ceiling, refresh measured records, and mark technical closure only if every directly comparable row and the raw-Fermion gate pass.

## 7. Acceptance recommendation

Do not accept Phase 10.5 yet. Preserve the completed Product active/inactive schedule work, current specialized selector winners, Qudit charge semantics, deterministic CSR construction, Generic allocation remediation, and passing quality gate. Accept the present six-site eager difference as parity, but keep the phase open for the direct raw-Fermion performance requirement and the complete fair QuSpin capability matrix.

This review added only this report and its `docs/vibe/README.md` index entry. It did not modify production source, tests, benchmark implementations, or benchmark result files.
