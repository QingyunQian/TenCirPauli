# Phase 11 Factorized Additive-Charge MVP Specification

Status: owner-directed implementation specification. This phase is not implemented by this document.

## 1. Decision and purpose

TenCirPauli already exposes the correct public way to define a symmetry-restricted problem: users provide one or more `AdditiveCharge` values and a `ChargeSector` target. That public interface remains the only interface. Users must never identify a spin block, select a kernel, provide an occupation layout, or choose an execution strategy.

The present restricted lazy-MVP selector is too dependent on the spelling of the input constraints. In particular, it has a balanced-spin Fermion fast path that accepts only `N_up = N_down`, routes a single fixed-total-number Fermion sector through a heavy general Product builder, and can route physically identical `(N, S_z)` and `(N_up, N_down)` sectors to different execution paths. The Product builder is a valid executor for genuinely independent factors, but its all-state maps, keys, prefix metadata, and transition preparation are an inappropriate representation for one compact factor.

This phase replaces physical-name-specific classification with one exact compiler for additive-charge sectors. It must identify the maximal factorization proved by the user-supplied charges, use compact factor indexers where the factor is simple, retain one general representation for truly coupled constraints, and compile all selected actions to native code once. A factor is an internal basis representation, not a public concept and not a new kind of symmetry.

The principal objective is simple:

> Every factorable restricted sector uses the fastest applicable native representation regardless of whether the user spelled it as total number, spin number, spin balance, Qudit excitation count, or an equivalent combination of `AdditiveCharge` values. A sector that cannot be proved factorable remains correct on the general path.

## 2. Public API contract

The following public interfaces and meanings are frozen:

```python
charge = tcp.AdditiveCharge(space, ...)
sector = tcp.ChargeSector(((charge_a, target_a), (charge_b, target_b)))
restricted = operator.restrict_charge(sector, storage="lazy")
plan = restricted.mvp_plan()
```

- `AdditiveCharge`, `ChargeSector`, `restrict_charge`, `mvp_plan`, `apply`, `apply_into`, and the eager/lazy storage arguments retain their public signatures and results.
- A user does not supply `up_modes`, `down_modes`, factor positions, a preferred variant, an expected block count, or a strategy hint.
- `ChargeSector` basis ordering remains `MIXED_RADIX_BASIS_ORDERING`. Equivalent user constraint systems must produce the same basis order and numerical result.
- Existing native-variant metadata remains diagnostic only. It may be renamed to describe a representation accurately, but it is not a user control surface.
- An optional read-only diagnostic summary may later report the selected representation and factor count. It must not be required to construct or execute a sector.

No API of the form `restrict_charge(..., strategy="product")`, `factorize=...`, `spin_sector=...`, or `u1_kind=...` is permitted.

## 3. Mathematical model used internally

Every `AdditiveCharge` is reduced at the Python boundary to local integer contribution tables. For axis `a`, local state `x_a`, and user charge `j`, the compiler sees an exact contribution `q[j, a, x_a]`. A sector is the finite set of local configurations satisfying:

```text
sum_a q[j, a, x_a] + offset[j] = target[j]    for every j
```

Fermions are the binary special case `x_a in {empty, occupied}`. A Qubit excitation count is another binary special case. A Qudit has a multi-value local state and may contribute a value such as `0, 1, 2, ...`, a spin value, or a vector of values to several charges. A bounded Boson uses the same representation with its selected cutoff.

This model deliberately does not use names such as particle number, spin, `S_z`, Hamming weight, or U(1) as classification keys. Those are common interpretations of a local contribution table, not execution categories.

## 4. Exact canonicalization and factor discovery

### 4.1 Required result

Native compilation receives a `CanonicalChargeSystem` rather than Python-side physical hints. It contains the normalized local contribution tables, adjusted targets, finite local dimensions, and a checked exact representation of the constraint row space.

The compiler must then produce exactly one of the following:

1. `FactorizedSectorPlan`: the allowed set is proved to be a Cartesian product of two or more independent components, possibly with unconstrained Cartesian spectators.
2. `SingleFactorSectorPlan`: there is one compact constrained component.
3. `GeneralCoupledSectorPlan`: the constraints cannot be proved to separate.

The words “single factor” and “factorized” describe the basis layout, not the physics. A single total-number Fermion sector is a single factor. Fixed `N_up` and fixed `N_down` are two factors. A Qudit sector with three independent local excitation-count regions is three factors.

### 4.2 Exact procedure

The factor analysis is compile-time work only. It uses bounded checked integer/rational elimination over the normalized charge rows and transforms targets with the same operations. It must not use floats, a machine-learned classifier, or a timing heuristic.

The analysis must:

1. Remove zero rows, duplicate constraints, and exact redundant rows without changing the selected set.
2. Derive an exact canonical row basis and transformed targets. Rational intermediate coefficients are allowed only with checked bounded arithmetic; rows are rescaled to primitive integer form before publication.
3. Find the maximal partition of axes for which the canonical constraint space is a direct sum of subspaces supported on disjoint axis groups.
4. Verify target compatibility for every derived component. For example, `N=10, S_z=2` yields `N_up=6, N_down=4`; a nonintegral derived target proves the sector empty rather than silently selecting a different basis.
5. Publish a factorization only when the equality system proves that the allowed set is the product of the component allowed sets.

If checked arithmetic cannot prove a factorization, the compiler preserves the original general system. This is a conservative optimization choice, not a semantic fallback: it can leave performance on the table but cannot change a result.

The important examples are:

| User-facing charge system | Canonical internal result |
| --- | --- |
| total Fermion number `N=10` on 20 modes | one binary fixed-population component with 10 occupied modes |
| `N=10`, `N_up=6` | fixed-population up component with 6 and down component with 4 |
| `N=10`, `S_z=2` | the same two components with occupations `[6, 4]` |
| fixed `N_up=3` but unconstrained down modes | one fixed-population component times a Cartesian binary spectator |
| first five two-level Qudits have three `up`, last five have two `up` | two finite fixed-population components with occupations `[3, 2]` |
| one total number only across two mode groups | one component; it must not be falsely split because the distribution between groups is not fixed |
| independent charges coupled across the same axes after all exact row operations | one general coupled component |

### 4.3 No semantic heuristic

There is no “probably factorable” path. Factor discovery is exact and conservative. Heuristics are permitted only after a representation is known to be correct, for example whether a bounded block-local rank cache has a measured benefit under the caller’s memory limit. A cache decision may not change the selected sector, the action semantics, or the error behavior.

## 5. Internal basis representations

`FactorizedSectorPlan` owns an ordered list of components and a public-order indexer. Components use one of these bounded native representations:

| Component form | Eligibility | Indexing representation |
| --- | --- | --- |
| `Cartesian` | no constraint touches its axes | ordinary mixed-radix digit |
| `FixedPopulation` | a binary local alphabet with an exact fixed number of the selected state | direct combination rank/unrank with checked binomial tables |
| `FiniteAdditive` | finite local alphabet with one or more additive local contribution tables | compact suffix-count dynamic program over local values and remaining targets |
| `GeneralCoupled` | constraints remain coupled after canonicalization | the existing general charge rank/unrank plan |

`FixedPopulation` is shared by Fermion, Qubit, and any two-level Qudit count. It receives an axis list, a distinguished local value, a target count, and the physical-axis mapping. It has no `spin`, `particle`, or `Hamming` field.

`FiniteAdditive` is the general Qudit/Boson representation. It receives the local contribution table directly, so a simple up-count is only a small special case of the same data model.

Each component exposes the same private operations:

```text
dimension()
rank(local_values)
unrank(rank)
prefix_completion_count(partial_local_values)
apply_local_action(compiled_local_action)
```

The public-order indexer walks physical axes and combines component prefix-completion counts to preserve `MIXED_RADIX_BASIS_ORDERING`. It must calculate this directly from compact component data. The normal factorized path must not build:

- a `BTreeMap<Vec<u64>, usize>` for every allowed public state;
- copied local-value keys for every allowed state;
- prefix/completion arrays indexed by every allowed state;
- a full `factor_to_public` / `public_to_factor` permutation merely to bridge an internal order to the public order.

Any proposed cache that scales with the full sector dimension must be separately budgeted, block-local, and retained only after it demonstrates a material representative apply win. It cannot be a hidden prerequisite for ordinary factorization.

## 6. Native MVP execution

### 6.1 One action authority

All charge-restricted terms continue to lower once into shared compiled schedules. Fermion actions retain `FermionMaskDescriptor` as the sole authority for occupation requirements, flips, CAR signs, and public-axis orientation. Qudit, Boson, and Qubit local actions retain their corresponding compact action descriptions.

The new factorized executor consumes these compiled schedules and a `FactorizedSectorPlan`. For every schedule it records the components it can change. At apply time it updates only those component ranks; untouched component digits are copied. Charge rows, charge names, and factor discovery never appear in the scalable inner loop.

### 6.2 Fast fixed-population Fermion layout

The existing `FastFermionSectorIndex` is the starting primitive, not a discarded performance experiment. It already has direct combinatorial rank/unrank plus bounded optional combination/rank caches. It must be generalized from:

```text
C(L, k) × C(L, k)
```

to:

```text
C(L_up, N_up) × C(L_down, N_down) × ...
```

with every component carrying its own mode list, population, dimension, and optional bounded cache. The old balanced path is exactly the parameter choice `[N_up, N_down] = [k, k]`; it is not retained as a second semantic kernel.

For the usual two-Fermion-component layout, the compiled hot arithmetic is simply:

```text
global_factor_rank = up_rank * down_dimension + down_rank
```

Changing `[5, 5]` to `[6, 4]` changes only plan constants. It must not add a charge-name check, a map lookup, a dynamic callback, or a per-term representation switch.

The implementation may retain compile-time-specialized one-component and two-component flat loops when profiling demonstrates that a generic component loop has a repeatable material cost. These are data-layout specializations, not `N`, `S_z`, or spin-specific implementations. All variants use the same component interface and compiled action descriptors.

### 6.3 General coupled execution

When canonicalization cannot prove factorization, `GeneralCoupledSectorPlan` retains the current exact rank/unrank and destination aggregation semantics. It is the only fallback. It must not be selected merely because the user wrote equivalent separable constraints in a different basis.

### 6.4 Eager CSR is separate

Explicit eager construction uses the same canonical sector plan and compiled schedules to generate the shared CSR graph. It must not construct an unused lazy factorized executor first. CSR graph construction and `CsrGraph.apply_into` remain the existing common representation; this phase must not create another CSR producer or another CSR apply loop.

## 7. Selector and ownership changes

The current Python functions that infer physical special cases, including the balanced-only Fermion detector and independent-factor Product hints, are removed after their behavior has migrated. Python normalizes public inputs and passes raw normalized charge data once. Rust owns canonicalization, factor discovery, indexer choice, and plan construction.

The resulting lazy selector is:

```text
canonicalize exact additive charge system
    ├─ provably factorized or one compact component
    │     └─ build FactorizedChargeMvpPlan using component indexers
    └─ otherwise
          └─ build GeneralCoupledChargeMvpPlan
```

There is no fallback chain of `balanced Fermion → Qubit → Qudit → Boson → Product → Generic`, and no losing native plan may be built merely to decide the winner.

Domain-specific local action lowering remains separate because Fermionic anticommutation and Qudit/Boson local algebra are genuinely different. Only the sector representation, public indexing, ownership, and selector are unified.

After acceptance, delete the superseded balanced-only Fermion classifier, the single-factor Product classification, the corresponding Product-only public-index metadata, and any now-unreachable variant or compatibility branch. Do not retain a legacy path behind a suppressive type cast or a runtime retry.

## 8. Correctness requirements

The implementation is accepted only with all of the following:

1. Small-system dense differential tests for every component indexer and action domain.
2. Exact rank/unrank round trips in public mixed-radix ordering for one component, multiple components, non-contiguous physical component axes, and Cartesian spectators.
3. Equivalence tests proving that `(N, S_z)` and `(N_up, N_down)` construct the same dimension, basis ordering, dense matrix, lazy result, and eager CSR result whenever they define the same sector.
4. Fermion tests for balanced and unbalanced occupations, total-only sectors with spin-changing terms, and rejection when an operator does not conserve an explicitly selected `N_up`/`N_down` sector.
5. Qudit tests for a simple up-count, independent regional counts, and a genuinely weighted multi-level charge that must use `FiniteAdditive`.
6. Multi-charge coupled tests that prove the plan remains General rather than falsely factorized.
7. Exact cancellation/leakage tests preserved from the current charge compiler.
8. Eager CSR equality with lazy/dense references, including the direct-eager construction route.

## 9. Performance and memory acceptance gates

All measurements use the existing release-mode benchmark protocol and compare saved local records. Full-suite reruns are not required to diagnose a narrow path.

| Case | Required result |
| --- | --- |
| Existing balanced Fermion fixed-spin workload | New factorized plan has no repeatable steady `apply_into` regression beyond 5% against the current native fast plan. |
| Balanced `[k, k]` construction | No repeatable construction regression beyond 5% against the current fast plan. |
| Unbalanced `[k_up, k_down]` | Per-state/per-schedule throughput remains within 5% of the equivalent balanced layout after accounting for its different sector dimension. |
| One compact total-number Fermion sector, including 20 modes / N=10 | No single-factor Product map/key/prefix construction. Restricted setup must return to the historical low-setup engineering class; steady apply must be within 10% of the current Product steady result. |
| Simple separated Qudit counts | Within 10% of the present fixed-count/Hamming-style fast case. |
| Existing general coupled-charge workloads | No repeatable lazy or eager regression beyond 5%. |
| Explicit eager construction | Does not build a lazy factorized executor that it will not execute. |

The implementation is rejected if it improves setup by routing a factorable sector through General but causes a material steady-state loss, or if it restores throughput by retaining the current Product all-state address book for one compact factor.

Every retained cache reports checked retained bytes and is covered by `max_bytes`. No cache may rely on an unaccounted `O(D)` allocation.

## 10. Delivery sequence

### 10.1 Canonical constraints and proof tests

Introduce `CanonicalChargeSystem` and exact target-preserving normalization behind the existing public `ChargeSector`. Add only algebraic and equivalence tests. No execution selector change occurs in this unit.

### 10.2 Component indexers

Extract the existing direct combinatorial Fermion indexing into `FixedPopulation`. Add `Cartesian` and `FiniteAdditive` component indexers with independent rank/unrank tests. Preserve the current balanced implementation as the benchmark oracle during this unit; do not retain it once migration is complete.

### 10.3 Factorized public indexer and action executor

Implement direct public-order ranking from compact component prefix counts and `FactorizedChargeMvpPlan`. Migrate the existing balanced Fermion workload first, with `[k, k]` as an ordinary input. Require the balanced performance gate before generalizing further.

### 10.4 Equivalent U1 forms and one-factor sectors

Enable exact `(N, S_z)` ↔ `(N_up, N_down)` canonicalization, unbalanced occupations, total-only fixed-population Fermion sectors, and independent Qudit count factors. Delete the balanced-only selector and the single-factor Product route in this unit.

### 10.5 General components and eager lifecycle

Attach `FiniteAdditive` and General coupled components to the unified factor plan. Route explicit eager requests directly to CSR construction without first building an unused lazy executor. Preserve the shared CSR graph implementation.

### 10.6 Deletion and final acceptance

Delete superseded Product metadata/builders, physical-name classification hints, obsolete benchmark names, and all losing production paths. Run the focused numerical/performance matrix once, record the result, and then run examples as the requested final compatibility check.

## 11. Contemporaneous benchmark dispositions

These are not reasons to add another sector executor, but they are tracked alongside this phase because they appeared in the same complete record.

1. `test_generic_charge_csr_materialization_after_cache` is a real Python-side read-back regression. The shared `_read_csr_arrays()` helper currently constructs an intermediate `CSRMatrix`, validates allocation, and then `ChargeMvpPlan._csr_arrays()` extracts the three arrays before `ChargeRestrictedOperator.csr()` constructs the final `CSRMatrix` again. The pre-refactor path returned the three flat arrays directly and constructed only the final container. Repair it by making the shared helper return the three flat NumPy arrays and performing one final public-container construction at the caller. This preserves the flat read-back ABI and removes the duplicate wrapper work.
2. `test_handle_native_terminal_compilation[dense]` is not a confirmed regression. The dense source path is unchanged, the full-suite current record has high variance, and three isolated records are consistently far faster than the old full-suite median. Do not change the dense compiler for this signal. Retain the old/current records as context and revisit only if a same-protocol narrow comparison later reproduces a stable loss.

## 12. Non-goals

- No public symmetry DSL beyond `AdditiveCharge` and `ChargeSector`.
- No one-kernel-per-charge-name or one-kernel-per-combination-of-charges design.
- No runtime strategy switch, promotion after an arbitrary number of applies, or hidden plan retry.
- No new CSR apply implementation.
- No arbitrary-precision algebra added solely to discover exotic factorization; checked bounded native arithmetic and conservative General classification are sufficient.
- No change to unrestricted Majorana, Pauli, Fermion, Boson, Hybrid, or Qudit MVP routes in this phase.
