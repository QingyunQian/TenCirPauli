# Phase 12 Common Noise Channel Gradients Implementation Review, 2026-08-13

Status: open focused remediation report. Commit `c42a754` implements the intended public channel families and main native/JAX differentiation architecture, but Phase 12 is not acceptance-closed against `docs/vibe/specs/phase-12-common-noise-channel-gradients-spec.md`.

## 1. Scope and verdict

This review compares `c42a754` with its parent and the Phase 12 specification, focusing on valid-input numerical behavior, representative propagation performance, JAX composition, frozen-support semantics, and acceptance evidence. Per owner direction, the test and benchmark suites were not rerun; their reported passing results are accepted as existing evidence. Extreme numerical inputs, exact RSS accounting, broad defensive validation, arbitrary channels, parameterized PTMs, and unrelated refactors are excluded.

The implemented API and mathematics are substantially aligned: the three TensorCircuit-compatible channel families, advanced scalar slots, separate high-level angle/channel gradients, QIR conversion, one-callback JAX VJP, immutable batch execution, and SPPS rejection are present. No interior-point formula error was found by static inspection.

Four focused issues remain. The forward channel path does not implement the low-allocation specialized kernels required by the specification; the benchmark source cannot measure that requirement; the JAX channel VJP has no numerical oracle; and several frozen-support/high-level gradient contracts are covered only by shape or vacuous endpoint checks.

## 2. Why `gamma=1` must be tested on X or Y

Phase damping and the X/Y lanes of amplitude damping use

```text
s(gamma) = sqrt(1 - gamma),
ds/dgamma = -1 / (2 sqrt(1 - gamma)).
```

At `gamma=1`, the forward edge is exactly zero while the fixed-basis derivative is singular. Phase 12 defines an executed-trace/frozen-support derivative: the zero edge is absent, reverse propagation must not evaluate the singular expression, and that edge contributes zero to the returned gradient.

The current endpoint test uses Z through phase damping. Z always has multiplier `1` and no gamma derivative, so the test never reaches the square-root branch and would still pass if the X/Y endpoint returned `NaN`, `inf`, or incorrectly reintroduced the deleted edge. The correct regression uses X or Y with a nonzero matching Bloch component and asserts finite value/gradient with a contract-defined zero contribution. This test verifies that the singular derivative is skipped; it does not request regularization or approximation of that derivative.

## 3. Required remediation

### R1 — Replace generic channel aggregation with direct kernels

Priority: P1 representative performance blocker.

Current behavior: `crates/tencir-pauli-core/src/propagation.rs:1453-1478` constructs `T` or `2T` contributions for every channel and calls `aggregate()`, which creates an `FxHashMap`, materializes another vector, sorts it, and filters it. Depolarizing and phase damping therefore pay approximately `T` key clones, `T` hash operations, and an `O(T log T)` sort per occurrence although their output words are identical to their input words.

Required code changes:

1. Split the `GateKind::Channel` forward arm into explicit helpers, for example `apply_depolarizing`, `apply_phase_damping`, and `apply_amplitude_damping`, rather than routing all three through `channel_edges()` plus `aggregate()`.

2. For depolarizing, resolve `(px, py, pz)` once, precompute the four local multipliers, consume the existing `Vec<DynamicTerm>`, multiply each coefficient according to `code_at(wire)`, and retain only nonzero/projected terms. Do not clone or rewrite keys. Because input keys are already unique and sorted and no key changes, the output remains unique and sorted without aggregation.

3. For phase damping, resolve `gamma` once, compute `sqrt(1-gamma)` once, and use the same order-preserving scale/filter pass. I/Z coefficients remain unchanged; X/Y coefficients are scaled. Do not build a transition list or hash map.

4. For amplitude damping, process I/X/Y lanes directly. Only Z produces the additional I word and can collide with an existing I word having the same spectator codes. Restrict aggregation to those Z-derived I contributions and their collision targets. Preserve exact cancellation, deterministic canonical order, and projection-after-aggregation. Do not send unchanged I/X/Y lanes through a full-map aggregation merely for implementation uniformity.

5. Replace the per-operation heap `Vec<f64>` returned by `resolve_channel()` with fixed-size resolved values or a small typed enum/struct if profiling shows the allocation remains visible after removing the dominant aggregation. This is secondary to steps 2-4 and should not delay them.

6. Keep `visit_retained_edges()` and the existing reverse contract unless the direct forward representation exposes a real mismatch. The optimization must not introduce a second reverse representation or change slot ordering.

Required correctness checks for the kernel change:

- Compare every direct channel against the equivalent static PTM for all local I/X/Y/Z inputs and a multi-term operator.

- Include exact-zero multipliers, amplitude I/Z collisions, exact cancellation, and a mixed sequence in which a later propagated rotation or PTM projects out a channel-dependent branch under `max_weight`.

- Assert deterministic repeated term order and coefficients.

Closure condition: depolarizing and phase damping execute without contribution vectors, aggregation maps, or output sorting; amplitude damping limits aggregation to the actual split/collision work; all existing numerical semantics remain unchanged.

### R2 — Make the benchmark package answer the specialized-kernel question

Priority: P1 performance-evidence blocker.

Current behavior: `benchmarks/python/test_propagation_channels_benchmark.py:43-59` builds equivalent static PTMs, but the PTM is used only as an untimed value oracle. The 12-qubit case contains no static PTM, no batch benchmark exists, and metadata is incomplete. A clean no-regression report therefore does not establish a built-in/static-PTM performance comparison.

Required benchmark changes:

1. Preserve a release-mode baseline for the current generic channel implementation before changing R1, using the same machine and thread settings intended for the post-change record.

2. Parameterize matched forward rows by `{built_in, static_ptm}` for each channel family. Use identical observable, product state, `max_weight`, warmup, rounds, and materialization boundary. Record setup, first expectation, and steady expectation separately.

3. Keep a built-in-only value-plus-gradient row for every channel family because static PTM entries are intentionally not differentiable. For a matched reverse-through-map control, place the same runtime rotation before a built-in channel or equivalent static PTM and compare the rotation gradient timing and value under identical scalar-buffer semantics.

4. Modify the 12-qubit workload so it actually contains rotations, at least two supported channel families, and a fixed one- or two-qubit PTM. Record the numerical difference against an equivalent control, observable/input term counts, final and peak term counts, scalar-buffer bytes, estimated peak bytes, and thread settings.

5. Add one `PropagationBatch` workload with enough observables to exercise shared-tape batch scheduling. Record value and value-plus-gradient paths and verify a computed maximum error against independent per-observable execution.

6. Retain the synchronized JAX workload, but record a computed value/gradient error rather than only finiteness. Synchronize both the scalar value and every gradient leaf inside the timed callable.

7. Run `python benchmarks/run.py record` after remediation and retain the untracked manifest/label as local acceptance evidence. Compare the direct kernels with both the pre-change built-in baseline and the equivalent PTM rows. A speedup need not become a CI gate, but the record must show whether the removed hash/sort work is material on the representative mixed and batch cases.

Closure condition: the saved release record contains matched built-in/static-PTM forward rows, complete first/steady/value-gradient boundaries, a true mixed 12-qubit tape, a batch workload, synchronized JAX timing, and computed accuracy/resource metadata.

### R3 — Add numerical JAX channel-gradient and shared-upstream tests

Priority: P1 correctness-evidence blocker; no valid-input failure has been reproduced.

Current behavior: the ordinary tests contain no JAX channel case. The JAX benchmark runs depolarizing and amplitude damping but only checks that outputs are finite. Phase damping and accumulation from one upstream scalar into both rotation and channel occurrence vectors are not tested.

Required tests, preferably in `tests/test_common_noise_channels.py` or a focused JAX channel module:

1. Add one `jax.jit(jax.value_and_grad(objective))` test per channel family. Choose interior parameters and a state/observable for which the relevant derivative is nonzero. Compare both value and gradient with a central finite difference of the complete scalar objective using the specification's JAX tolerance.

2. For depolarizing, exercise asymmetric `px`, `py`, and `pz`; ensure the tested scalar affects a Pauli lane with a nonzero state expectation. Do not use a symmetric fixture where two derivatives become indistinguishable.

3. For phase damping, test an X or Y observable at an interior gamma so the square-root derivative is nonzero.

4. For generalized amplitude damping, use an observable/state combination containing both an X/Y contribution and a Z contribution, or use two small cases, so gamma's square-root and split derivatives and p's split derivative are all numerically bound.

5. Add a shared-upstream objective such as a scalar `theta` used in a rotation and in `gamma = sigmoid(theta)`, with both paths intentionally nonzero. Build a concrete counterpart at the same point, obtain its separate native angle and channel occurrence gradients, and assert the explicit chain rule

```text
dE/dtheta = angle_gradient[rotation_slot]
          + channel_gradient[gamma_slot] * sigmoid(theta) * (1 - sigmoid(theta)).
```

6. Extend the shared case so one upstream scalar is also reused by two channel occurrences, or add a second compact fixture. Assert the sum of all occurrence contributions rather than only equality between eager and JIT execution.

7. Insert a concrete static PTM between a channel and a parameterized rotation in one JAX test. Compare the total gradient with finite differences to prove that the fixed PTM transpose transports cotangents without adding PTM-entry gradients.

8. Retain one float64 numerical case at analytic tolerance and one default-float32 callback case at the existing float32 tolerance if the test environment supports both configurations. Do not add `vmap` coverage; it is explicitly outside this phase.

Closure condition: every channel family has a nontrivial numerical JAX oracle, shared rotation/channel uses are proven to sum by an explicit chain rule, and a static PTM is proven to transport the cotangent.

### R4 — Bind frozen-support and public gradient-order semantics numerically

Priority: P1 correctness-evidence blocker.

Current behavior: the existing endpoint test uses Z and bypasses the singular X/Y edge. The suite also lacks amplitude split/cancellation/projection endpoint cases, shared advanced slots, checkpoint replay, empty batch shapes, and numerical assertions for the separated high-level gradient arrays.

Required frozen-support tests:

1. Phase damping at `gamma=1`: propagate X and Y separately from Bloch states with nonzero matching components; assert value `0`, gamma gradient `0`, and finite outputs.

2. Amplitude damping at `gamma=1`: repeat the X/Y zero-edge assertions. Separately propagate Z with `p != 0.5` and assert the retained I-edge value and finite frozen-support derivatives directly. Do not use a finite-difference oracle at the support-change point.

3. Test `p=0`, `p=1`, and mixed p for the Z split. Include `p=0.5`, where the Z-to-I multiplier is exactly zero, and assert that the removed edge contributes no p or gamma derivative under frozen support while the retained Z edge behaves normally.

4. Construct one two-term case whose channel-derived contributions cancel exactly into the same output word. Assert that the canceled output is absent and contributes no reverse edge.

5. Do not try to make a one-qubit channel increase Pauli weight: depolarizing and phase damping preserve weight, while amplitude damping preserves or lowers it. Instead, arrange a mixed tape so the channel acts first in Heisenberg propagation and a subsequent two-qubit rotation or PTM turns one channel-dependent word into an overweight branch. For example, place the entangling operation before the channel in Schrödinger append order so reverse propagation applies the channel and then the entangling map. Assert that `max_weight` removes the overweight continuation and that its channel-gradient contribution is absent while a retained continuation still contributes.

Required layout/replay tests:

6. Build an advanced tape in which one runtime slot is shared by a rotation and a channel scalar, and another slot is shared by two channel scalars. Compare each returned slot gradient with finite differences of that shared-slot execution.

7. Compare `value_and_grad()` with at least two `checkpoint_interval` values and the default interval on a mixed rotation/channel/PTM tape. Assert bitwise equality where the existing deterministic contract permits it, otherwise use the analytic tolerance.

8. For `PropagationCircuit.value_and_grad()`, use at least two rotations and two channel families in interleaved append order. Compute independent finite differences for each stored concrete occurrence and assert the exact documented ordering of `gradient`, `channel_gradient`, and `channel_parameter_layout`; shape-only assertions are insufficient.

9. Assert read-only, C-contiguous `(0,)` angle and channel arrays for circuits with only channels or only rotations, plus `(0,)` values and `(0, nparameters)` gradients for an empty `PropagationBatch`.

10. Retain the current concurrent batch test, but compare its values and gradients with independent single-engine results rather than only with the first batch call.

Closure condition: endpoint tests execute the actual disappearing edges, cancellation/projection behavior is asserted directly, shared slots and checkpoint replay have numerical oracles, and public gradient arrays are bound to occurrence order rather than only shape.

## 4. Recommended implementation order

1. Add the missing correctness tests from R3 and R4 against the current implementation. This freezes the existing intended semantics before performance refactoring.

2. Record the pre-change matched benchmark baseline from R2.

3. Implement the direct kernels from R1, starting with depolarizing and phase damping, then isolate amplitude-damping split aggregation.

4. Run the focused correctness tests after each kernel change, then run the complete repository quality hook once after the implementation is stable.

5. Record the post-change release benchmark package and compare it with both the pre-change built-in rows and static-PTM controls.

6. Update the Phase 12 specification/status only after the focused closure matrix is satisfied. The feature may remain described as implemented while acceptance remediation is open.

## 5. Acceptance checklist

| Area | Closure evidence |
| --- | --- |
| Forward kernels | Depolarizing and phase damping have no contribution vector/hash/sort; amplitude damping aggregates only split collisions. |
| Frozen support | X/Y `gamma=1`, retained amplitude Z branches, zero split, exact cancellation, and projected outputs have direct contract assertions. |
| Native/public gradients | Interior finite differences cover all scalars, shared slots, checkpoint replay, and exact high-level occurrence order. |
| JAX | Each channel has a numerical `jit`/`grad` oracle; shared rotation/channel upstream values sum correctly; static PTM transport is covered. |
| Batch/concurrency | Empty/nonempty shapes and numbers are correct; concurrent results agree with independent engines. |
| Performance | Matched built-in/static-PTM rows, true mixed 12-qubit and batch workloads, synchronized JAX timing, and computed accuracy/resource metadata are recorded. |

Phase 12 may be acceptance-closed when R1-R4 and this checklist are complete. No public API redesign, parameterized PTM work, arbitrary Kraus execution, SPPS noise support, or defensive extreme-input machinery is required.

## 6. Remediation outcome, 2026-08-13

The focused remediation has been implemented in the current worktree. Depolarizing and phase damping now use direct in-place scale/filter kernels with no contribution vector, aggregation map, or output sort; amplitude damping retains unchanged I/X/Y lanes directly and sorts/aggregates only the Z-to-I split before merging it with the retained base. Runtime channel values now use a fixed-size stack buffer. `visit_retained_edges()` and the reverse slot/derivative contract were left unchanged.

The forward-kernel A/B measurement used the same release build, 12-qubit observable with 2048 terms, 12 channel occurrences, and repeated value/value-plus-gradient calls. Depolarizing forward improved from approximately 1380 to 77 microseconds and value-plus-gradient from 3737 to 1197 microseconds; phase damping improved from approximately 1375 to 78 microseconds and value-plus-gradient from 3691 to 1167 microseconds; amplitude damping improved from approximately 1469 to 432 microseconds and value-plus-gradient from 3834 to 2041 microseconds. These measurements make the specialized kernels material on the representative workload.

The Rust propagation tests now compare every direct channel with an equivalent static PTM and cover deterministic ordering, amplitude split collisions, exact cancellation, and projection. Python coverage binds the X/Y `gamma=1` frozen-support edges, amplitude Z split endpoints including `p=0`, `p=0.5`, `p=1`, shared advanced slots, checkpoint replay, projected overweight continuations, concurrent batch results against independent engines, empty read-only arrays, and high-level angle/channel occurrence order with finite differences.

The benchmark source now records matched built-in/static-PTM setup, first and steady forward paths, built-in value-plus-gradient paths, reverse-through-map rotation controls, a mixed 12-qubit rotation/channel/PTM workload, shared-tape batch value and value-plus-gradient workloads, and synchronized JAX value/gradient errors. The project-environment smoke and release record `20260813T052717Z_c42a754cca87-dirty` passed with 532 benchmark cases and one pre-existing optional skip. JAX numerical oracle tests passed for all three channel families, shared rotation/channel upstreams, and static-PTM cotangent transport.
