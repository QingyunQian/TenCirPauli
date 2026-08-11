# Phase 11 Factorized Additive-Charge MVP Specification

Status: owner-directed implementation specification. This phase is not implemented by this document.

## 1. Review disposition and binding decisions

The architectural direction is accepted: factorization must be derived from additive-charge equations, compact factor indexers must replace selected-space address books, and the user must not choose an execution representation. The original draft was not yet executable because it did not fully specify the small row reduction, empty and unconstrained sectors, the distinction between a compact layout and General execution, direct public-order indexing, cancellation before leakage, budget ownership, private FFI deletion, or a whole-suite regression protocol. This revision closes those gaps.

The following decisions are binding:

1. The public API remains `AdditiveCharge` plus `ChargeSector`; physical charge names and Python-supplied execution hints are not compiler inputs.
2. Exact factor discovery operates on a baseline-adjusted integer contribution matrix and transformed target, not on charge names or raw row supports. It uses structural fast exits first and never constructs general rational-number objects.
3. The analyzer may conservatively decline factorization when the small checked representation is ineligible, but it may not change or approximate the selected sector. The existing exact General suffix-count plan remains authoritative.
4. Empty sectors and compatible all-zero constraint systems are explicit internal results. They are not left to accidental behavior of a later indexer.
5. A `FixedPopulation` component is a structural binary-count representation. A one-row finite weighted component uses `FiniteAdditive`. Multiple irreducible rows over the same axes use the coupled representation.
6. Public mixed-radix ordering is computed from compact completion counts. A full-sector permutation, full-sector occupation map, or full-sector prefix table is prohibited on the ordinary factorized path.
7. Exact aggregation and leakage rules remain shared across specialized and General execution. A faster indexer is not permission to validate terms independently when cancellation can occur only after aggregation.
8. Explicit eager construction lowers shared schedules and builds CSR directly. It does not first build a lazy factorized action cache that will never execute.
9. The final performance decision requires both focused charge benchmarks and the complete repository benchmark suite. A related path may be up to 10% slower than the pre-change same-machine baseline once its absolute regression is at least 1 ms. Below 1 ms, the comparison uses a relaxed but enforced 20% threshold; a multi-fold microbenchmark regression is never ignored merely because its absolute duration is small. A repeatable regression above its applicable threshold blocks acceptance until profiling and matched A/B evidence identify and resolve the cause or the owner explicitly changes the architecture.

## 2. Decision and purpose

TenCirPauli already exposes the correct public way to define a symmetry-restricted problem: users provide one or more `AdditiveCharge` values and a `ChargeSector` target. That public interface remains the only interface. Users must never identify a spin block, select a kernel, provide an occupation layout, or choose an execution strategy.

The present restricted lazy-MVP selector depends on the spelling of the input constraints. It has a balanced-spin Fermion fast path that accepts only `N_up = N_down`, routes a single fixed-total-number Fermion sector through the heavy general Product builder, and can route physically identical `(N, S_z)` and `(N_up, N_down)` sectors to different execution paths. The Product builder is a valid executor for genuinely independent factors, but its selected-state maps, copied keys, per-state prefix metadata, and transition preparation are the wrong representation for one compact factor.

This phase replaces physical-name-specific classification with one exact compiler for additive-charge sectors. It identifies the finest factorization proved by the supplied charge equations, uses compact factor indexers where their eligibility is exact, retains one General representation for a genuinely coupled component, and lowers every selected action to native descriptors once. A factor is an internal basis representation, not a public concept and not a new kind of symmetry.

The objective is:

> Every factorable restricted sector uses the fastest applicable bounded native representation regardless of whether the user writes total number, spin number, spin balance, a Qubit count, Qudit spin-z, Boson occupation, or an equivalent integer linear combination of `AdditiveCharge` values. A sector that cannot be proved factorable remains exact on the General path.

## 3. Public API and semantic boundary

The following public interfaces and meanings are frozen:

```python
charge = tcp.AdditiveCharge(space, ...)
sector = tcp.ChargeSector(((charge_a, target_a), (charge_b, target_b)))
restricted = operator.restrict_charge(sector, storage="lazy")
plan = restricted.mvp_plan()
```

- `AdditiveCharge`, `ChargeSector`, `restrict_charge`, `mvp_plan`, `apply`, `apply_into`, and eager/lazy storage arguments retain their public signatures and numerical results.
- A user does not supply `up_modes`, `down_modes`, factor positions, a preferred variant, an expected block count, a cache choice, or a strategy hint.
- `ChargeSector` basis ordering remains `MIXED_RADIX_BASIS_ORDERING`, with physical axis 0 most significant. Equivalent constraint systems produce the same dimension, rank/unrank results, dense matrix, sparse arrays, and MVP result.
- Existing `execution_family` and `native_variant` values remain diagnostic. They may be renamed once to describe the retained representations accurately, but they are never a control surface.
- A read-only diagnostic summary may report factor-analysis status, component kinds, component axis lists, dimensions, retained bytes, and cache modes. It must be derived from the native plan and must not be required for execution.
- No API of the form `strategy="product"`, `factorize=True`, `spin_sector=...`, `u1_kind=...`, or `cache="dense"` is permitted.

The current public `AdditiveCharge` expressiveness is also frozen. Fermion and Boson axes contribute integer weight times occupation, Qubit axes expose an explicit integer `(level_0, level_1)` pair, and a Qudit weight multiplies the existing canonical integer-normalized spin-z spectrum. This phase does not add an arbitrary Qudit per-level charge table. Rust receives local integer contribution tables internally so the compiler is representation-general, but public tests and documentation must not claim a charge DSL that the Python API does not expose.

The abstraction has three separate authorities: charge constraints define the selected set, `OperatorSpace` physical-axis order defines the public ordered basis by filtering the full mixed-radix basis without reordering, and `FactorizedLayout` defines private execution coordinates only. An internal component rank such as `up_rank * down_dimension + down_rank` is never a public basis rank unless it happens to equal the filtered physical order. Factor analysis may change neither public state-vector order nor rank/unrank results.

## 4. Exact mathematical input

For public axis `a`, local state `x`, and user constraint `j`, let `q[j,a,x]` be the exact native `i128` local contribution and let `t[j]` be `requested_target[j] - offset[j]`. The selected basis is

```text
sum_a q[j,a,x_a] = t[j]    for every constraint j.
```

The analyzer removes irrelevant per-axis constants before row operations. For each axis it chooses local state zero as the reference and creates one contrast column for every nonzero local state:

```text
A[j, column(a, x)] = q[j,a,x] - q[j,a,0]    for x = 1, ..., d_a - 1
b[j] = t[j] - sum_a q[j,a,0].
```

The finite-domain variables represented by these columns are not treated as independent continuous variables during indexing; the matrix is used only to transform equivalent equality systems and prove disjoint support. Row operations on `[A | b]` preserve the equality system for every legal local configuration.

The current domains lower as follows:

| Public domain | Local table supplied to Rust |
| --- | --- |
| Fermion | `(0, weight)` |
| Boson with inclusive cutoff `c` | `(0, weight, 2*weight, ..., c*weight)` |
| Qubit | the exact public `(level_0, level_1)` pair |
| Qudit of dimension `d` | `weight * s_d(x)` for the frozen canonical spin-z spectrum `s_d` |

Names, domain labels, and interpretations such as particle number or Hamming weight do not enter factor analysis. Domain information remains available only to lower local operator actions correctly, especially Fermionic CAR signs, Boson ladder amplitudes, and Qudit Weyl phases.

## 5. Factor analysis and small row reduction

### 5.1 Native ownership

`ChargeSectorPlan` remains the sole authority for the selected set and its public filtered mixed-radix ordering. Factor discovery adds one optional private execution sidecar; it does not replace public rank/unrank and does not create a second sector abstraction. Python sends charge values once when constructing the sector. Later operator compilation receives the existing native sector handle and an operator handle; it does not resend physical fast-path hints.

The retained addition is intentionally small:

```rust
struct ChargeSectorPlan {
    // Existing exact public-order fields remain authoritative.
    factorized_layout: Option<FactorizedLayout>,
    // ...
}

struct FactorizedLayout {
    components: Box<[ComponentIndex]>,
    axis_components: Box<[usize]>,
}
```

`None` means that factorization was not proved or provides no useful decomposition, so the existing General execution remains authoritative. An empty sector remains an ordinary exact sector with dimension zero. A one-component layout remains a `FactorizedLayout` of length one. Do not add `CanonicalChargeSystem`, `EmptySectorPlan`, `SingleFactorSectorPlan`, or `GeneralCoupledSectorPlan` solely to name construction outcomes.

### 5.2 Cheap structural fast exits

Factor discovery is a small construction-time analysis, not a general symbolic-algebra subsystem. The representative contract is one to four charge rows with ordinary small integer weights, most commonly `-1`, `0`, and `1`, and occasionally `2`, `3`, or `4`. It does not optimize for dozens of charges or very large coefficients; those inputs remain correct on General.

The analyzer performs these cases before any elimination:

1. Validate dimensions and perform checked baseline subtraction while building each augmented integer row.
2. Resolve zero rows and the no-active-row Cartesian case.
3. If one nonzero row remains, publish its one support component directly. A single total-number constraint therefore performs no elimination.
4. If all remaining row supports are pairwise disjoint, publish those support components directly.
5. Only when two to four rows remain and at least two supports overlap does the compiler run the small reduction in Section 5.3. More than four rows return `None` and retain General.

These fast exits cover single fixed population, ordinary one-row Qubit/Qudit/Boson charges, and already separated regional constraints without a matrix-reduction pass.

### 5.3 Small integer-preserving row reduction

The overlapping-row case must recognize exact row combinations such as

```text
N + S_z = 2 N_up
N - S_z = 2 N_down.
```

Without row combination, both `N` and `S_z` touch all spin modes and a raw-support graph falsely reports one component. This is the only reason elimination is needed.

The helper first attempts to downcast the local augmented rows to `i64`. It stores at most four rows in one contiguous buffer and uses `i128` only for the intermediate multiply/subtract. For pivot coefficients `p` and `c`, eliminating one row computes `p * row - c * pivot_row`, divides the complete augmented result by its positive gcd, normalizes the first nonzero coefficient to be positive, and converts the primitive row back to `i64`. The target entry participates in exactly the same operation.

The helper uses the leftmost nonzero pivot column, selects the largest absolute pivot in that column with original row order as the tie-breaker, and eliminates the pivot from every other row. This deterministic pivot rule limits coefficient growth without introducing a tolerance or timing heuristic.

After reduction it removes compatible zero rows and exact redundant rows, rejects an all-zero coefficient row with nonzero target as empty, and sorts retained primitive rows by pivot column and then lexicographically. The result is deterministic for the fixed contrast-column order.

With `r <= 4`, the work is at most `O(16m)` small checked operations and `O(4m)` temporary storage, where `m = sum_a (d_a - 1)` is the contrast-column count. It runs once while constructing `ChargeSectorPlan` and is cached across every operator using that sector. No elimination, gcd, reduced row, or factor-analysis branch appears in public rank/unrank or any apply loop.

If a row or target does not fit `i64`, an `i128` intermediate overflows, a primitive result does not fit `i64`, or the small analysis workspace cannot be represented safely, return `None` and retain the existing exact General plan. Do not retry with floats, rational objects, big integers, or a second algorithm. Public `max_bytes` values below 1 GiB are advisory and do not suppress this compact analysis.

The implementation is one private helper in the charge-sector module, not a public linear-algebra abstraction. Coefficients and targets are reduced together; no second row-reduction implementation is permitted.

### 5.4 Finest support partition

For the retained reduced rows, connect two rows when their physical-axis supports overlap. Because there are at most four rows, use a local nested scan and small visited array rather than introducing a reusable graph or disjoint-set abstraction. Each connected row group owns the union of its physical axes; axes absent from every retained row are Cartesian spectators. Order components by their smallest physical axis.

The reduced row support therefore proves the disjoint components needed by execution. The compiler need not discover a split that exists only because of an unusual finite-domain coincidence. A retained component may contain one or more reduced rows and is lowered to `FixedPopulation` or `FiniteAdditive` from those exact integer rows.

Small-system tests compare the selected set and ordered public basis against brute-force filtering. Production does not rescan trusted handles or retain the construction matrix.

### 5.5 Empty, unconstrained, and infeasible components

An inconsistent zero row is immediately empty. Other finite-domain infeasibility is detected while building component completion counts. For example, reduced equations may imply `2*N_up = 11`; no binary fixed-population state exists, so the exact sector dimension is zero.

An input whose reduced rows all vanish with compatible targets selects the full finite Cartesian space. It is represented by Cartesian components and is not routed through General merely because the user supplied a redundant charge.

The exact existing `ChargeSectorPlan` remains valid for empty, Cartesian, one-component, factorized, and coupled cases. The optional sidecar is retained only for a proved compact component layout. No construction-result class is added.

### 5.6 Required factor-analysis examples

| User-facing charge system | Required internal result |
| --- | --- |
| total Fermion number `N=10` on 20 modes | one binary `FixedPopulation(20, 10)` component |
| `N=10`, `N_up=6` | fixed-population up component with 6 and down component with 4 |
| `N=10`, `S_z=2` with the current integer spin convention | the same two components and targets `[6, 4]` |
| fixed `N_up=3` with unconstrained down modes | one fixed-population component plus Cartesian spectators |
| independent Qubit counts on two disjoint regions | two fixed-population components |
| independent Qudit spin-z constraints on two disjoint regions | two `FiniteAdditive` components |
| one total number across two mode groups | one component; it must not be falsely split because the distribution is not fixed |
| multiple rows that remain connected over the same axes | one coupled component |
| `N=2` together with the distinct proportional charge `2*N=5` | empty |
| zero charge with target equal to its offset | full Cartesian space |
| zero charge with a different target | empty |

## 6. Component indexers

### 6.1 Retained representation

Factorized plans store a compact enum, not a trait object dispatched once per contribution:

```rust
enum ComponentIndex {
    Cartesian(CartesianIndex),
    FixedPopulation(FixedPopulationIndex),
    FiniteAdditive(FiniteAdditiveIndex),
}
```

General execution may reuse the same suffix-count implementation as `FiniteAdditive`, but it retains the existing General execution semantics. A factorized layout may contain a multi-row `FiniteAdditiveIndex` for one component when another independent component justifies product execution.

Each component provides native methods equivalent to:

```text
dimension()
completion_count(prefix_state)
rank(local_values)
unrank(local_rank, output_values)
advance_public_prefix(value)
rewind_public_prefix(value)
```

Methods use caller-owned or stack-bounded scratch. They do not allocate a vector or hash key per state or per action.

### 6.2 Cartesian

Cartesian axes use checked mixed-radix strides. The layout records every spectator axis separately; the execution representation may coalesce adjacent spectator axes into one mixed-radix component to reduce loop overhead. Nonadjacent spectators are not reordered.

### 6.3 Fixed population

`FixedPopulation` is eligible only when all of the following hold after the optional reduction:

- the component has one retained reduced row;
- every component axis has local dimension two;
- every axis has the same nonzero contrast coefficient after primitive normalization;
- the target is exactly divisible by that coefficient; and
- the resulting selected-state count lies in `[0, axis_count]`.

The selected local state is state one in the baseline-adjusted representation. Counting state zero is represented as the equivalent count of state one. Fermion, Qubit, and any future public binary finite axis share this implementation; no field is named `spin`, `particle`, or `Hamming`.

Rank and unrank use checked binomial tables and direct combination masks. The implementation starts by extracting the existing `FastFermionSectorIndex` combination primitive, then generalizes it to an arbitrary axis list and population. The compact Pascal table is bounded by a practical internal cap and is not degraded because a caller supplied an advisory `max_bytes` value below 1 GiB. Larger components use direct combinatorial rank/unrank; this never changes the sector representation.

### 6.4 Finite additive

`FiniteAdditive` stores reachable suffix remainders for only its own axes and retained reduced rows. For each retained row, local state zero contributes zero and local state `x > 0` contributes the reduced coefficient at `column(a,x)`; the row's transformed target is the component target. Its table is sparse over exact reachable vectors; it does not allocate the numerical interval between minimum and maximum charge. Rank, unrank, and prefix completion follow increasing local values and therefore preserve component-local mixed-radix order.

A one-row weighted Boson, Qubit, or Qudit spin-z component uses scalar remainder keys where that reduces key and hash cost. A multi-row component uses fixed-width inline remainder keys for the common small row count and the existing bounded wider representation otherwise. Selection depends only on row count and checked width, not charge names.

### 6.5 Prohibited retained data

The normal factorized plan must not retain:

- a `BTreeMap<Vec<u64>, usize>` for every allowed component or public state;
- copied occupation keys for every allowed state;
- prefix/completion arrays indexed by every allowed state;
- a full `factor_to_public` or `public_to_factor` permutation;
- a full-sector packed-source cache; or
- a second Python representation of component state.

An action table may scale as `O(U_f * D_f)` for the number `U_f` of unique local actions on one component of dimension `D_f`, but it is optional, explicitly budgeted, and retained only after a release A/B demonstrates a material end-to-end apply benefit. It must never scale as the global product dimension solely to recover public order.

## 7. Direct public mixed-radix indexer

### 7.1 Rank formula

`FactorizedPublicIndexer` owns the physical-axis-to-component mapping and mutable prefix state supplied by the caller. For a valid state, it walks physical axes from most significant to least significant. At physical axis `a` belonging to component `f`, for every candidate local value `v` smaller than the selected value it adds

```text
completion_f(prefix_f followed by v)
    * product_{g != f} completion_g(current prefix_g)
```

to the public rank, then advances component `f` with the selected value. This is exactly the number of valid global states sharing the previous public prefix and using a smaller value at axis `a`.

For a valid prefix, the product of current component completion counts is maintained incrementally. The other-component product is obtained by exact division by the current nonzero completion count of `f`, so the ranker does not loop over all factors for every candidate. Checked multiplication guards the dimension and rank. An invalid selected prefix returns `InvalidSector`.

### 7.2 Unrank and public cursor

Unrank uses the same counts. At each physical axis it tries local values in increasing order, subtracts the corresponding global completion count until the target index falls inside one candidate block, then advances that component prefix.

The interleaved cursor starts from public unrank zero. To advance, it backtracks from the least-significant physical axis, rewinds that axis from its component prefix, tries larger local values in order, and accepts the first candidate with a nonzero global completion count. It then fills every following physical axis greedily with the smallest value that leaves a nonzero completion count. Only components touched by the backtracked suffix have their local rank recomputed. Reusable prefix stacks, remainder vectors, local values, and component ranks are allocated once per apply call; cursor advancement performs no heap allocation.

Lazy apply must not call general unrank from scratch for every public source. It uses one of two construction-selected loops:

1. A flat component-rank loop for one component or physically contiguous component blocks, with precomputed checked component strides.
2. A `PublicStateCursor` for interleaved axes. The cursor advances the next valid public state with reusable prefix/remainder scratch and maintains component-local ranks and compact local values incrementally.

Destination public ranks are computed from destination component values with the same direct rank formula. Fixed-population combination masks provide those values. No result depends on internal factor order.

### 7.3 Mandatory ordering cases

Rank/unrank and cursor tests cover one factor, two contiguous factors, grouped-spin axes, site-interleaved spin axes, three factors, noncontiguous factor axes, leading/middle/trailing Cartesian spectators, mixed local dimensions, and empty/full spaces. Every case compares the full ordered basis against an independent brute-force filter for small dimensions.

## 8. Native action compilation and execution

### 8.1 One schedule authority

All charge-restricted terms lower once into the shared compiled schedule types. `FermionMaskDescriptor` remains the sole authority for occupation requirements, flips, CAR signs, and public-axis orientation. Boson, Pauli, and Qudit actions retain their existing compact descriptors and exact amplitude rules.

Each compiled schedule records:

- its coefficient and coefficient-free action identity;
- the components and spectator axes it reads or changes;
- a compact local action descriptor for every changed component;
- its Fermion mask descriptor when applicable; and
- its aggregation group for non-termwise-conserved execution.

Charge rows, charge names, factor discovery, Python callbacks, and enum selection do not appear in the scalable apply loop.

### 8.2 Component-local action forms

Fixed-population actions store required/forbidden bits and flips in component-local physical-axis order. Destination masks are ranked directly through `FixedPopulationIndex`. For the ordinary two-component Fermion layout, the internal component digit remains

```text
factor_rank = up_rank * down_dimension + down_rank.
```

Changing `[5, 5]` to `[6, 4]` changes only component constants. It does not add a name check, map lookup, callback, or per-term representation switch.

Finite-additive actions apply the already compiled local Fermion/Boson/Pauli/Qudit action to compact local values and rank the destination with the component index. Repeated coefficient-free local actions are interned. A dense component-local transition table is allowed only under the `O(U_f * D_f)` budget and A/B rule in Section 6.5.

### 8.3 Exact cancellation and leakage

When every compiled term individually preserves the selected charge rows, execution may apply schedules directly. Otherwise schedules are grouped by their exact net destination transform before the destination is validated. The grouping key contains the Fermion flip, Boson net occupation delta, Pauli XOR, and Qudit modular shift needed to identify the destination, while source-dependent signs, ladder factors, Weyl phases, and coefficients remain values evaluated and summed within the group. For each source, amplitudes reaching the same destination are summed in deterministic schedule order; exact zeros are removed; only the surviving aggregate is checked for sector leakage.

This preserves cases such as `XX + YY`, in which individual Pauli terms leave a sector but their aggregate action does not. Factorized execution must share the same grouping semantics as the current Product/General compiler and must not introduce a term-by-term rejection shortcut.

No coefficient-magnitude cutoff is permitted. Floating coefficients use the existing exact-zero aggregation convention; charge arithmetic itself remains integer exact.

### 8.4 Specialized flat loops

The compiler may retain monomorphic one-component and two-component fixed-population loops if profiling shows that a generic component enum loop or public-rank call consumes a material share of representative runtime. Such loops are data-layout specializations over the same component and action descriptors. They may not reintroduce balanced-only semantics, physical charge names, or a second sign implementation.

The existing balanced `[k,k]` Fermion kernel is the migration and benchmark oracle. Once the generalized one/two-component implementation passes numerical and performance gates, the old kernel, classifier, permutation fields, and duplicate tests are deleted in the same change.

### 8.5 Apply memory and concurrency

`apply_into` writes directly to caller-owned output. Per-call scratch is `O(axis_count + constraint_count + unique_schedule_count)` unless an explicitly budgeted component-local table is retained. `apply` additionally budgets the returned complex vector. Public `max_bytes` is only a cheap major-allocation guard and values below 1 GiB are advisory; no exact peak-RSS accounting is required. All complete input conversion, schedule compilation, factor construction, source traversal, and apply work runs inside the GIL-released native section.

Plans are immutable and safe for concurrent calls. Reusable mutable scratch may not be hidden in a plan without synchronization. If per-call scratch is required, its checked estimate is compared with the call's `max_bytes` before allocation.

## 9. General and eager execution

### 9.1 General coupled execution

The existing General execution retains the current exact suffix-count rank/unrank, compiled schedules, deterministic aggregation, and leakage semantics. It is used when `factorized_layout` is `None` or when one irreducible multi-row component provides no factor-execution benefit. It is not selected merely because a compact cache exceeded budget or because component axes are interleaved.

General no-regression is a binding acceptance gate. Factor-analysis metadata must not add work to each General apply after construction.

### 9.2 Direct eager lifecycle

The schedule compiler is shared, but lazy and eager retained owners are separate:

```text
sector handle + operator handle
    -> exact public sector + optional factorized layout
    -> shared compiled schedules
        -> lazy: factorized/general action executor
        -> eager: existing deterministic CSR producer -> CsrGraph
```

`restrict_charge(..., storage="eager")` invokes the eager branch directly and does not construct factor-local transition tables, public cursors, or lazy-only schedule groups. A later `.csr()`, `.coo()`, or `.dense()` request on an existing lazy facade may compile and cache one eager `CsrGraph` from its already lowered schedules, as allowed by the existing facade lifecycle.

CSR generation uses the existing exact public sector indexer and common destination-major graph. This phase adds neither a second CSR apply loop nor a global COO sort path. `_read_csr_arrays()` returns the three flat NumPy arrays, and the public caller constructs `CSRMatrix` exactly once.

## 10. Private API and file-level migration

The implementation is complete only when ownership is visible in the code structure.

### 10.1 Rust core

- Extend `crates/tencir-pauli-core/src/charge_sector.rs` or add narrowly scoped sibling modules for the small row reducer, component indexes, and public-order indexer. Do not create a general linear-algebra layer or leave a second row-reduction implementation in a kernel module.
- Move the reusable binomial rank/unrank primitive out of `charge/kernels/fermion/index.rs` into the shared fixed-population component module.
- Replace `ChargeKernelHints` and the ordered specialized selector in `charge/generic.rs` with selection from `ChargeSectorPlan.factorized_layout`.
- Refactor `charge/kernels/product.rs` to consume component indexes and direct public ordering. Delete `ProductFactorPublicIndex`, selected-state `BTreeMap<Vec<u64>, usize>`, copied factor keys, and per-selected-state prefix/completion metadata once no caller remains.
- Keep `FermionMaskDescriptor` and shared schedule lowering authoritative. Do not clone Fermion sign logic into the component module.
- Route eager compilation through the existing `charge/eager.rs` and `CsrGraph` owner without constructing a lazy kernel.

### 10.2 PyO3 boundary

`NativeChargeSectorPlan` owns the exact public sector and optional factorized layout. The private `compile_mvp_pauli_handle` and `compile_mvp_hybrid_handle` calls remove all `fast_*` and `product_positions` parameters. Layout information already owned by the sector handle is not resent from Python. The target private shape is conceptually:

```text
compile_mvp_pauli_handle(operator_handle, termwise_conserved, max_bytes)
compile_mvp_hybrid_handle(operator_handle, termwise_conserved, max_bytes)
compile_eager_*_handle(operator_handle, termwise_conserved, max_bytes)
```

Exact signatures may retain one compact domain-layout descriptor only if the sector handle demonstrably does not own it yet; duplicated axis-position vectors and physical classification hints are not permitted.

All long native construction and execution stays inside `py.allow_threads`. Update `_native.pyi` in the same unit and delete obsolete arguments rather than retaining compatibility overloads for private APIs.

### 10.3 Python

Delete `ChargeCompileHints`, `_fast_fermion_particles`, `_fast_qubit_particles`, `_fast_qudit_factor`, `_fast_boson_factor`, and `_supports_independent_factor_product` after the native selector is wired. `python/tencirpauli/charge.py` continues friendly validation, conservation checks, public facade construction, and final result materialization. It does not inspect charge names or supports to select an executor.

For `storage="eager"`, Python calls the direct eager native constructor rather than unconditionally constructing `ChargeLazyMvpPlan` first. Existing lazy-to-eager cache synchronization remains for a lazy facade that later materializes.

## 11. Memory-budget contract

Construction maintains a checked budget ledger for:

1. temporary contrast/reduced rows and retained factor layout;
2. retained component suffix/binomial tables;
3. public index metadata;
4. compiled schedule descriptors and grouping metadata;
5. optional component-local action caches;
6. eager CSR arrays when requested; and
7. the largest simultaneous construction workspace that is cheaply estimable.

`estimated_bytes` reports retained native bytes and includes every optional cache. `estimated_construction_peak_bytes` reports the retained result plus the largest major temporary phase, without pretending to be an exact allocator/RSS measurement. Checked dimension, index, buffer-length, and allocation arithmetic is mandatory.

The following budget outcomes are distinct:

- optional cache exceeds its internal practical cap: build the same representation without that cache;
- the small factor-analysis workspace does not fit or its checked arithmetic cannot stay in the bounded `i64`/`i128` representation: retain the exact original General plan;
- selected required index or output does not fit: raise `MemoryError` before the major allocation;
- sector is empty: retain a bounded empty plan, not a failed allocation path.

No fallback is allowed to silently materialize an `O(D)` address book after a compact cache is declined.

## 12. Correctness acceptance

Correctness is a gate before performance work. At minimum, add the following tests.

### 12.1 Factor-analysis algebra

- Baseline subtraction with nonzero Qubit level-zero values and charge offsets.
- Duplicate internal rows plus public scaled, sign-flipped, permuted, redundant, and integer-linear-combination constraints. The existing Python boundary continues to reject the exact same `AdditiveCharge` object/value definition twice.
- Inconsistent duplicate rows, zero compatible rows, zero incompatible rows, and finite-domain infeasibility.
- Deterministic reduced rows and factor components across input constraint order.
- Ineligible-row-count and checked-overflow fallback cases that match the original General basis and numerical result.
- Property tests confirming the original and reduced augmented systems select the same small-system states.

### 12.2 Indexers and ordering

- Numeric dimension checks for Cartesian, fixed-population, one-row finite-additive, multi-row finite-additive, empty, and full-space plans.
- Exact component and public rank/unrank round trips.
- Full ordered-basis equality against brute-force filtering for all mandatory layouts in Section 7.3.
- Equivalent `(N,S_z)` and `(N_up,N_down)` systems with the same public rank for every small-system state.
- Nonintegral derived populations and out-of-range populations produce dimension zero.
- No retained metadata count scales with global sector dimension in the compact factor cases; test-only diagnostics assert cache/table sizes.

### 12.3 Numerical execution

- Small-system dense differential tests for Fermion, Boson, Qubit/Pauli, Qudit/Weyl, and mixed Hybrid actions.
- Known-number assertions, not only shape or replay equality, for every new native action entry point.
- Balanced and unbalanced Fermion occupations; total-only Fermion sectors with spin-changing terms; grouped and site-interleaved axes.
- Rejection when an operator genuinely leaks outside explicitly fixed `N_up` and `N_down`, plus exact cancellation cases that must not be rejected term by term.
- Qubit fixed counts, Qudit spin-z components, bounded Boson occupation components, Cartesian spectators, and a genuinely coupled multi-row case.
- Lazy `apply`, caller-owned `apply_into`, direct eager construction, CSR, COO, and dense equality.
- Flat CSR read-back creates one public container and returns arrays equal to the native graph.
- Concurrent calls on one immutable plan produce identical results.

### 12.4 Selector and deletion assertions

- Equivalent row systems report the same native representation summary.
- Every proven simple component avoids General.
- Every intentionally coupled single component reports General.
- Cache budget changes may change cache diagnostics but not component kind or numerical results.
- A repository grep confirms no production Python physical classifier, no `ChargeKernelHints`, no balanced-only selector, no selected-state Product address book, and no full-sector public permutation remains.

## 13. Benchmark implementation requirements

### 13.1 Stable focused matrix

Add a capability-named charge-factorization benchmark matrix to the existing benchmark infrastructure. It records one coarse public sample each for sector construction, restricted-plan construction, `apply_into`, allocating `apply`, direct eager construction, CSR materialization/read-back, retained bytes, native representation, component dimensions, schedule count, and a numerical checksum. The matrix is a broad regression signal, not a precision microbenchmark; detailed pytest-benchmark sampling remains available when diagnosis is needed.

The stable matrix contains at least:

| Capability case | Representative shape | Required measurements |
| --- | --- | --- |
| single total Fermion number | 20 modes, `N=10` | sector setup, lazy setup, first/steady apply, eager CSR |
| balanced two-component Fermion | 10 up + 10 down, `[5,5]` | construction and first/steady apply against the current fast baseline |
| unbalanced two-component Fermion | the same axes, `[6,4]` | construction and first/steady apply |
| equivalent row basis | `(N,S_z)` and `(N_up,N_down)` | representation, ordered-basis checksum, setup, apply |
| fixed factor plus spectators | fixed count on one region with an unconstrained interleaved region | public-rank cost and steady apply |
| Qubit fixed population | central and low-population cases | construction and steady apply |
| Qudit spin-z | one region and two independent regions | construction and steady apply |
| bounded Boson occupation | one weighted region and independent regions | construction and steady apply |
| factorized Hybrid | constrained Fermion component plus Cartesian Bosons | construction, first/steady apply, RSS |
| coupled General | at least one scalar-key and one multi-row/wide-key case | construction, steady apply, eager CSR |

Workload source, seeds, terms, cutoffs, thread counts, and input vectors are committed and stable. Raw result files remain under `.benchmarks/` and are not committed.

### 13.2 Required benchmark commands

Before changing production selection or execution, first land or stage the benchmark-only workload matrix and observability needed by Section 13.1, verify that this preparation does not change numerical or execution behavior, and then record a same-machine release baseline from that worktree. If an existing pre-change record is used, its commit, dirty fingerprint, suite, thread environment, and workload schema must match; a record made after production implementation begins must not be relabeled as the pre-change baseline:

```bash
python benchmarks/run.py record --suite all --label charge-factorization-before
```

During development, use the focused suite and manual phase profiler; exact suite naming may be added to `benchmarks/run.py`, but it must include the complete matrix above:

```bash
python benchmarks/run.py record --suite majorana_charge --label charge-factorization-focused
python benchmarks/manual/factorized_charge_matrix.py --output .benchmarks/factorized-charge-matrix.json
```

Before acceptance, run the complete repository benchmark suite against the saved baseline:

```bash
python benchmarks/run.py compare charge-factorization-before --suite all
```

This complete comparison is mandatory even if every focused case passes. It is not replaced by smoke benchmarks, a single manual script, or a narrow pytest selection. After the comparison is accepted, save the post-change full record:

```bash
python benchmarks/run.py record --suite all --label charge-factorization-after
```

### 13.3 Regression rule

For every directly affected charge-sector path, compare the same workload and metric under the same release build, thread environment, power mode, and machine:

- When the absolute regression is at least 1 ms, `current / baseline <= 1.10` is the acceptable maximum generalization cost, provided correctness, memory, and scaling gates pass. Any 5–10% loss is called out explicitly rather than described as parity.
- Below 1 ms, `current / baseline <= 1.20` is the relaxed acceptance threshold. A larger relative loss is still an acceptance blocker when repeatable; absolute time never makes a multi-fold regression automatically acceptable.
- pytest-benchmark supplies the internal sampling. Run one current suite sample per comparison; do not add an outer repetition matrix. When an isolated diagnosis is genuinely needed, record one extra same-workload run and retain both reports.
- improvement in setup does not compensate for a greater-than-10% steady apply regression, and an apply improvement does not compensate for a greater-than-10% construction or eager-materialization regression on a separately required workflow.

The threshold applies separately to sector construction, lazy-plan construction, first apply, steady caller-owned apply, allocating apply, direct eager construction, and CSR read-back. On representative nontrivial cases, retained bytes and peak RSS also use a 10% investigation threshold. Small constant-size metadata may exceed that ratio only when its absolute bytes are reported and bounded; an unplanned `O(D)` allocation is a blocker regardless of wall time.

The complete-suite report is triaged as follows:

1. Every benchmark slower by more than 10% is listed.
2. If the benchmark uses charge-sector construction, rank/unrank, restricted execution, shared schedule lowering, Fermion masks, CSR construction, or flat sparse read-back, it is presumed related until profiling proves otherwise.
3. A demonstrably unrelated case is documented from its source path and recorded result; it does not justify ignoring a related loss.
4. No benchmark is removed, renamed, resized, or given a looser fixture merely to make the comparison pass.

### 13.4 Required diagnosis above 10%

For a repeatable related regression above 10%, the implementation report must partition runtime with native phase counters or a profiler into factor analysis, component-index construction, schedule lowering, public source traversal, public destination ranking, Fermion sign evaluation, local action evaluation, aggregation, and output writes.

Then run matched A/B experiments that isolate the dominant term, including as applicable:

- direct combinatorial rank/unrank versus optional component cache;
- generic component loop versus one/two-component monomorphic loop;
- flat contiguous public indexing versus interleaved direct completion-count indexing;
- overlapping-row factor analysis enabled versus an already separated input case;
- component-local transition table enabled versus analytic action; and
- factorized executor versus the pre-change retained kernel on identical schedules and input vectors.

The report states the dominant phase's absolute time, fraction of end-to-end time, and the maximum possible win of the proposed fix. A refactor aimed at a negligible phase is not accepted as remediation.

If the loss is implementation overhead, optimize it and rerun both focused and complete suites. If evidence shows an inherent architectural cost greater than 10%, retain a representation-neutral specialized flat loop that shares the same index and action authorities, revise the architecture, or request an explicit owner decision. Do not silently accept the loss, restore a physical-name classifier, or add a runtime retry.

## 14. Performance and memory gates

In addition to the universal 10% rule, the following capability outcomes are required:

| Case | Required result |
| --- | --- |
| Existing balanced Fermion fixed-spin workload | Same component compiler as unbalanced; no full-sector permutation; every measured phase within the 10% rule. |
| Unbalanced Fermion layout | Same hot-loop shape and descriptor types as balanced, differing only in component constants and dimensions. |
| One total-number Fermion sector, including 20 modes / `N=10` | No Product map/key/prefix construction; setup returns to the compact engineering class; no greater-than-10% apply loss against the best applicable pre-change path. |
| Equivalent `(N,S_z)` and `(N_up,N_down)` | Same representation summary, component dimensions, public basis checksum, and numerical result. |
| Separated Qubit/Qudit/Boson constraints | Compact component DPs only; no selected-state address book; within 10% of the best structurally equivalent pre-change case. |
| General coupled workloads | Factor analysis is construction-only; lazy apply, eager construction, and sparse read-back each stay within 10%. |
| Explicit eager construction | Does not build lazy-only component transition tables or an unused lazy executor. |
| Memory | No hidden global `O(D)` map/permutation/cache; every retained cache is reported and budgeted. |

Cross-shape comparisons such as balanced versus unbalanced report normalized throughput for understanding scaling, but acceptance is based on same-workload pre/post comparisons. Different sector dimensions are not used to disguise a regression.

## 15. Delivery sequence

### 15.1 Baseline and observability

Add the stable focused workloads plus phase timing and retained-byte diagnostics needed to attribute factor analysis, index construction, schedule compilation, public ranking, and apply without changing the production selector. Then record the full pre-change benchmark suite and focused manual matrix. Diagnostics are test/benchmark metadata, not public strategy controls.

Exit criteria: reproducible baseline labels, worktree fingerprint, fixed thread settings, numerical checksums, and a report covering all matrix cases.

### 15.2 Small factor analysis

Implement the contrast rows, zero/single/disjoint fast exits, at-most-four-row checked reduction, support-component scan, and optional `FactorizedLayout` behind `ChargeSectorPlan`. Keep execution selection unchanged in this unit.

Exit criteria: factor-analysis property tests, equivalent-row tests, empty/full cases, more-than-four-row and overflow fallback equality, measured construction cost, and no numerical behavior change.

### 15.3 Shared component indexers

Extract the existing combination primitive into `FixedPopulation`, add Cartesian and finite-additive indexes, and implement component-local rank/unrank/completion counts. Preserve the old balanced implementation only as a benchmark oracle during this unit.

Exit criteria: brute-force ordered-basis equality, numeric dimension tests, cache-on/off equality, checked budget tests, and component microbenchmarks.

### 15.4 Direct public ordering

Implement `FactorizedPublicIndexer`, the flat contiguous loop, and the interleaved public cursor. Do not compile operator actions yet.

Exit criteria: full ordered-basis equality for every Section 7.3 layout, no `O(D)` public permutation, and measured public-rank/unrank costs recorded.

### 15.5 Factorized action executor

Attach shared compiled schedules, component-local actions, exact aggregation/leakage, and fixed-population Fermion execution. Migrate balanced `[k,k]` first as ordinary component data, then enable unbalanced and total-only sectors.

Exit criteria: dense numerical differentials, exact cancellation tests, balanced focused gate within 10%, unbalanced hot-loop structural equality, and no duplicate Fermion sign path.

### 15.6 All public charge domains and private-boundary cleanup

Enable Qubit fixed population, Qudit spin-z, bounded Boson occupation, mixed factors, Cartesian spectators, and factorized components containing a local multi-row DP. Remove Python hints and shrink the private PyO3 compile signatures.

Exit criteria: all domain numerical tests, one batched native construction boundary, strict typing updates, and repository grep deletion assertions.

### 15.7 Direct eager path and read-back

Route explicit eager requests from shared schedules directly into the existing CSR producer. Preserve lazy-to-eager cache installation for later materialization. Ensure flat CSR arrays cross the ABI once and only one public `CSRMatrix` is constructed.

Exit criteria: direct-eager lifecycle test, lazy/eager/CSR equality, no unused lazy executor in construction diagnostics, and CSR read-back benchmark within 10%.

### 15.8 Deletion and final acceptance

Delete the old balanced classifier/kernel, `ChargeKernelHints`, Python factor detectors, selected-state Product maps/keys/prefix tables, full-sector permutations, obsolete private ABI parameters, losing action-cache experiments, and compatibility branches made unreachable by the new owner.

Run formatting, lint, typing, Rust/Python correctness, focused release benchmarks, the required full benchmark comparison, and examples. Acceptance requires a written table of every related pre/post metric and the disposition of every full-suite result slower by more than 10%.

## 16. Required validation commands

Use the project environment and local TensorCircuit source described by `AGENTS.local.md`. At implementation completion run:

```bash
python scripts/check.py --fix --benchmark skip
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
black --check python tests benchmarks scripts examples
ruff check python tests benchmarks scripts examples
mypy
maturin develop --release --skip-install
pytest
python benchmarks/run.py compare charge-factorization-before --suite all
```

The first command is run before staging when formatting changes are expected; the Git hook itself remains check-only. Benchmark commands use release builds and fixed thread variables. No `.benchmarks/` result is committed.

## 17. Contemporaneous benchmark dispositions

These findings are not reasons to add another sector executor, but they remain in scope because they affect the same acceptance record.

1. `test_generic_charge_csr_materialization_after_cache` exposed a real Python-side read-back regression. The repaired contract is one flat `_read_csr_arrays()` return and one final public `CSRMatrix` construction. The final full-suite comparison must verify this path remains within the 10% gate.
2. `test_handle_native_terminal_compilation[dense]` was not a confirmed regression in the prior record. The dense source path was unchanged and isolated records were faster than the noisy full-suite median. Do not change the dense compiler for that old signal, but the mandatory full-suite comparison still reports and triages any new repeatable loss.
3. The repaired factorized executor is covered by the three formerly regressing same-workload rows: exact-cancellation Qubit aggregation, many-schedule caller-owned Qubit application, and mixed Fermion/Boson caller-owned application all returned to at least their saved-baseline throughput. A later complete Python/FFI run is retained as an informational broad signal under the stricter dual threshold. Its long mixed JAX/TensorCircuit sequence raised unrelated thermal-sensitive rows and temporarily raised the two large Hubbard rows; an immediate same-machine focused rerun of those two caller-owned paths was within the 10% gate. The unrestricted 16-qubit COO materialization remains unrelated to charge sectors. The new factorization matrix has no pre-change rows and is retained as the durable baseline workload set.

## 18. Non-goals

- No public symmetry DSL beyond the existing `AdditiveCharge` and `ChargeSector` semantics.
- No arbitrary public Qudit per-level charge table in this phase.
- No kernel per charge name or per combination of physical charge names.
- No runtime strategy switch, promotion after an arbitrary number of applies, hidden plan retry, or user-visible cache choice.
- No new CSR apply implementation or global COO producer.
- No arbitrary-precision algebra solely to discover exotic factorization; checked bounded native arithmetic plus conservative exact General fallback is sufficient.
- No change to unrestricted Majorana, Pauli, Fermion, Boson, Hybrid, or Qudit MVP routes except shared helpers whose full-suite performance remains within the acceptance rule.
- No wall-time CI gate or external benchmark service. The required performance gate is a recorded same-machine local release comparison.
