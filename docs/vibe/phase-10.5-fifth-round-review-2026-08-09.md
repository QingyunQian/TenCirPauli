# Phase 10.5 Fifth-Round Remaining-Blocker Report, 2026-08-09

Status: acceptance remains open. This report intentionally contains only the substantive implementation and performance gaps that still block the Phase 10.5 specification. It does not repeat closed fourth-round findings or minor evidence-format issues.

Reviewed baseline: commit `c50038bac50d` plus the current uncommitted Phase 10.5 worktree on 2026-08-09. Clean-commit provenance remains outside the technical verdict.

## 1. Executive verdict

Two material issues remain:

1. General Product execution has a bounded public-order index and a destination-gather loop, but it still repeats active-factor work across inactive compact factors instead of applying one active action over the complete identity-factor segment.
2. Raw Cartesian Fermion execution is numerically correct and allocation-bounded, but the required profile shows an approximately `20x` slowdown against the existing explicit Jordan--Wigner packed-Pauli path. Packed-source extraction dominates and is still repeated for each schedule member.

Phase 10.5 should remain open until both issues meet the exit gates below. Optimization should remain serial and profile-led; no Product Rayon branch or duplicate Fermion sign implementation is needed to close these gaps.

## 2. B5-1 — General Product does not yet reuse active work across inactive compact factors

### Evidence

`FastProductMvpPlan::apply_destination_gather_into()` iterates the full combined factor-component dimension before visiting a schedule group. For a group that acts on only one factor, the same inverse transition, factor amplitude, and grouped aggregate are therefore reevaluated for every state of every inactive compact factor. The current `spectator_segment_applications` counter is incremented once per surviving output element, so it does not represent application of a reused segment.

A direct structural differential used one active two-qubit fixed-weight factor and enlarged a second, completely inactive fixed-weight factor. The results remained numerically exact and selected `native_variant="product"`, but `(dimension, factor_action_evaluations)` changed as follows:

| Inactive factor | Selected dimension | Factor-action evaluations |
| --- | ---: | ---: |
| 2 qubits, weight 1 | 4 | 4 |
| 4 qubits, weight 2 | 12 | 12 |
| 6 qubits, weight 3 | 40 | 40 |

The active action is identical in all three cases, so its evaluation count should remain constant. Only the number of identity-segment elements that receive the result should grow. The current behavior violates the factor-reuse contract in Section 10.3 of the specification and D10.

### Required implementation

1. Compile an explicit active shape for every Product schedule group: active compact factors, inactive compact factors, active Cartesian axes, inactive Cartesian axes, and the public-order contiguous or strided segment description.
2. Iterate destination tuples only over the active factors and active Cartesian axes. Derive inverse source tuples and evaluate the canonical grouped amplitude once for each active tuple.
3. Apply the resulting scalar transition across the inactive compact-factor and Cartesian identity space through a public-order segment iterator. Use contiguous suffix slices when available and bounded strided iteration for interleaved layouts; do not reconstruct full occupations or call the global sector ranker.
4. For a single-member injective group, write directly without an aggregate. For a multi-member or colliding group, aggregate once per active tuple in canonical order and remove exact zero before applying the inactive segment.
5. Keep `ProductPublicIndexer` factor-local. Add only bounded prefix/base-rank metadata needed to advance through public-order segments; do not add a `D`-entry permutation.

### Exit gate

- Enlarging an inactive compact factor leaves active-factor evaluations and aggregate evaluations unchanged while increasing only applied segment elements.
- Single-factor, two-factor, reversed-factor-size, and interleaved-axis Product differentials remain numerically equal to independent dense or CSR references.
- Product apply reports zero Generic fallback and zero global-ranker calls.
- Release A/B on representative two-factor and reversed-factor workloads shows the optimized Product kernel has no material regression against the current Generic reference, with retained storage remaining independent of the complete transition count.

## 3. B5-2 — Raw Fermion execution still has a material default-path performance regression

### Evidence

The current frozen 12-mode hopping-and-density profile reports:

| Variant | Steady median |
| --- | ---: |
| Raw `FermionMask` Cartesian plan | approximately `4.87 ms` |
| Explicit Jordan--Wigner packed-Pauli plan | approximately `0.244 ms` |

The raw path is approximately `20x` slower. Its batched phase profile attributes approximately `69.9%` of descriptor time to packed-source extraction, `21.3%` to index update, `6.8%` to parity, and `2.0%` to input/output work.

`apply_compiled_schedule_member()` still calls `FermionMaskDescriptor::pack_index()` for each Fermion schedule member. Multiple members and groups processing the same source block therefore reconstruct the same packed occupation words repeatedly. This is the dominant measured cost and leaves the D09 no-regression requirement unsatisfied.

### Required implementation

1. Reorder the compiled Cartesian Fermion loop so packed source words are computed once per source or compatible source block and reused across all Fermion schedule members sharing the same mode-axis layout.
2. Keep the shared width-generic `FermionMaskDescriptor` authoritative. Pass the reused source words into validity/parity and destination-flip evaluation; do not introduce another CAR/sign kernel.
3. Preserve active-axis and suffix-segment grouping so a reused Fermion result can still be applied across inactive Cartesian segments without repacking the source for each element.
4. Re-profile after source-word reuse. Only if parity or index update then becomes dominant should a one-/two-word specialization be added behind the same descriptor API.
5. Retain the allocator-backed one-word and multiword regressions and the constant-per-call plan-allocation contract.

### Exit gate

- The frozen 12-mode raw and explicitly mapped plans remain numerically equal and assert their actual native variants.
- Two warmups and seven release-mode steady `apply_into` samples demonstrate no material performance regression of the default raw path against the explicit Jordan--Wigner packed-Pauli path, as required by D09.
- The phase record demonstrates that packed extraction is performed once per intended source unit rather than once per schedule member.
- Ordered CAR, quartic, complex-coefficient, interleaved, and multiword correctness tests remain passing with zero per-state/per-term allocation growth.

## 4. Ordered remediation

1. Implement Product active/inactive shapes and segment application, then add the inactive-compact-factor scaling regression before collecting new Product timings.
2. Hoist Fermion packed-source extraction to the source/block level, rerun the phase profile, and optimize further only if the new dominant phase justifies it.
3. Run the focused numerical suites and the complete project quality gate.
4. Rerun the Product variant matrix, the frozen raw/mapped Fermion profile, and the primary acceptance workloads on the same worktree.
5. Update the live acceptance and implementation-status documents only after both exit gates pass.

## 5. Acceptance recommendation

Do not mark Phase 10.5 technically closed yet. Close it after B5-1 and B5-2 pass their numerical, structural, and release-performance gates. No other fourth-round item needs to be reopened unless either remediation changes its already established behavior.

This review added only this report and its index entry; it did not modify production source, tests, or benchmark implementations.
