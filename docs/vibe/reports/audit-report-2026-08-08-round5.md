# TenCirPauli Deep Audit — Round 5 (2026-08-08)

> **Status: Deprecated / 已废弃。** This report is retained as a historical record of the `ceab218`-dirty tree. It has been superseded by [the current-tree reconciliation report](audit-report-2026-08-11-current.md), which rechecks these findings against `e38533f`. Do not use this document as the current open-item ledger.

> **Scope.** A fifth-pass scan of the **current working tree** (commit `ceab218`-dirty, post-Phase-10.5 second-round review), focused on the four user-requested dimensions: dead code & dead branches, over-verbose/unnecessary narration ("think" residue), inconsistent user-facing API and Python↔Rust ABI, and bugs/perf bottlenecks that wrongly fall back to slow branches.
>
> **Method.** Five finder dimensions fanned out in parallel (dead-code, verbose-residue, py-api-consistency, py-rust-abi, slow-fallback-bugs); each raw finding was adversarially re-read against the current tree by an independent verifier that confirmed the cited code, checked the open/closed ledger, and sanity-checked the magnitude estimate. **The dead-code finder stalled on all 6 attempts (no model progress for 180 s each) and produced no findings** — dead-code/dead-branch coverage is therefore *incomplete this round*; the verbose-residue dimension partially absorbed dead-code findings (it surfaced 4 dead-code items: `checked_integer_to_f64`, dead `let a`, redundant `mapped_present` clause, `ChargeApplyCounters`). A future round should re-run the dead-code dimension.
>
> **No code was changed.** This is a report only.
> Two verifiers ran while the safety classifier (GLM-5.2) was temporarily unavailable; their findings (pyrust-abi CSR-arity, py-api Z2SymmetryAnalysis import) were spot-checked by the synthesizer against the cited code before inclusion.

---

# TenCirPauli Deep Audit — Round 5 Report

**Date:** 2026-08-08
**Repo root:** `/Users/shixin/Nutstore Files/newwork/quantum-information/codebases/TenCirPauli`
**Scope:** No source modified. Report only.
**Round:** 5 of 5 (final). Supersedes no prior finding; carries forward the open ledger.

---

## Executive Summary

Round 5 closed the four user-requested dimensions — **(1) dead code & dead branches, (2) over-verbose/unnecessary narration, (3) inconsistent user-facing API and Python↔Rust ABI, (4) slow-fallback / perf bottlenecks that wrongly fall back to serial branches** — across 21 adversarially verified findings. All 21 are `[REAL]`; none are `[OVER-DEFENSE]` or `[NARROW]`, and none are recommended "do not implement."

### Counts per dimension

| Dimension | Findings | Medium | Low | Informational |
|---|---|---|---|---|
| Dead code / verbose residue | 6 | 1 | 5 | 0 |
| Python API consistency | 8 | 1 | 6 | 1 |
| Python↔Rust ABI | 5 | 0 | 5 | 0 |
| Slow-fallback / perf | 2 | 1 | 1 | 0 |
| **Total** | **21** | **3** | **17** | **1** |

**Real vs over-defense split:** 21 REAL / 0 OVER-DEFENSE / 0 NARROW.

### Prioritized action list (impact-ordered)

1. **[medium, perf] Serial CSR SpMV in `StructuredEagerMvpPlan::apply_into`** (structured.rs:3030-3079). The only CSR/matvec kernel left serial; charge and Pauli siblings parallelize via rayon above `1<<19`/`1<<16` nnz. ~4-6x on multi-core for large non-charge structured eager plans (Boson/Qudit/Hybrid with high cutoffs). Highest-value real win.
2. **[medium, perf/maintainability] O(n⁴) integral-ingestion duplication** (structured.rs:1240-1377). `count_spin_orbital_terms` then `append_spin_orbital_integrals` re-walk the identical `p,q,r,s` quadruple nesting and recompute the same `hermitian_average`. Removes ~810k redundant ops for a 30-orbital Hamiltonian; realistic ~15-25% of ingestion wall time (the cheapest of three n⁴ passes) plus a maintainability coupling (loops must stay in sync).
3. **[medium, API] `PauliOperator.from_terms` lacks `max_bytes`** while every sibling (`FermionOperator`/`BosonOperator`/`QuditWeylOperator`/`MajoranaOperator`) exposes it. The namesake primary operator violates its own docstring promise; loud `TypeError` for any user mirroring the sibling signature.
4. **[low, perf] U1 eager re-clone on `.csr()`/`.coo()`/`.dense()` warm path** (symmetry.py:366-380). `mvp_plan` deep-clones the full CSR graph every call solely to read `transition_count`; cacheable trivially from the cold-path plan. Same category as the open R3 clone deferrals.
5. **[low, ABI] Hybrid `materialize` nested-tuple ABI** (_native.pyi:204). Sole sibling returning a doubly-nested tuple of flat arrays; flatten to match the four flat siblings and make the stub explicit.
6. **[low, ABI] `NativeChargeEagerMvpPlan.csr` lacks `max_bytes`** that its own sibling `coo()` enforces (charge_sector.rs:653 vs 663). Intra-class inconsistency plus arity mismatch with the canonical 4-tuple.
7. **[low, dead code] `ChargeApplyCounters` never mutated** (charge.rs:260-271, 245, 1042, 585). `debug_counters()` always returns zeros; test_charge.py:138 even encodes the misleading value.
8. **[low, API] `PauliOperator.__init__` lacks `max_bytes`** (pauli.py:324-348) — same family gap as #3, minor.
9–21. Remaining low-severity cleanups (dead `let a`, redundant `mapped_present` clause, narration comment, `BosonOperator.compile` Liskov, restricted-operator `compile`/`mvp` gaps, U1 storage type, `NativeMVPPlan.csr`, integral-constructor parity, stub type accuracy, `U1Circuit.value_and_grad` checkpoint_interval).

**Headline:** Three medium-impact items (one perf-per-serial-kernel, one perf-per-ingestion-duplication, one API-correctness on the namesake operator) are the actionable core. The remaining 18 are low/informational consistency and dead-code cleanups with no correctness impact.

---

## Carried-forward open ledger (unchanged)

The following known-open items were **NOT** re-raised in round 5 and remain open exactly as tracked:

- **Phase-10.5 second-round review:** SR1 (FastProductMvpPlan full-state hashing), SR2 (Fermion cartesian path + wide alloc), SR3 (FermionMask sign families), SR4 (CSR merge + charge builder), SR5 (charge re-validation).
- **Round-3 deferrals:** R3-11, R3-12, R3-14, R3-16, R3-23 (scratch/clone allocations across `charge.rs::apply_into`, `sector.rs::apply_into`, `structured.rs:2411`, `propagation.rs` PackedKey clone, `charge.rs:1602` Vec-clone-per-term, SPPS parallelism).
- **Round-4:** R4-N1 (SPPS finiteness), R4-D1..D4 (dead FFI: `charge_sector.rs:252-302`, `u1_circuit.rs:152-358`, `operator.rs:259-344`, `grouping.rs` masks), R4-I1 (naming), R4-N2 (over-defense, correctly excluded).

Round 5 findings are all **new** relative to this ledger — confirmed by each verifier via repo-wide grep of `docs/vibe/*` and the ledger items above.

---

## Dimension 1 — Dead code & verbose residue

Ordered by severity.

### 1.1 [REAL, medium] Duplicated count/append integral loops rescan the same O(n⁴) two-body tensor twice
**File:** `crates/tencir-pauli-core/src/structured.rs:1240-1377` (count: 1240-1269, 1271-1299; append: 1301-1329, 1331-1377)

`canonicalize_fermion_integrals` (defined at structured.rs:1029) first calls `count_spin_orbital_terms` (1240) which loops `p,q,r,s` over `n_modes⁴` computing `hermitian_average` and counting non-exact-zeros to size `Vec::with_capacity`, then calls `append_spin_orbital_integrals` (1301) which loops the identical `p,q,r,s` quadruple nesting and recomomputes the same `hermitian_average` to push. The identical duplication exists for `count_spin_block_terms`/`append_spin_block_integrals` (1271/1331) over the chemist-indexed `n_spatial⁴` tensor. The count result is consumed only by the `max_bytes` check (`raw_count*768`) and two `Vec::with_capacity(raw_count)` calls — both accept an upper bound, so the count pass is purely capacity/limit estimation.

**Failure scenario:** Ingesting a 30-orbital spin-orbital Hamiltonian (`n_modes=30`) runs the `n⁴=810000`-entry two-body loop twice: ~810k `hermitian_average` + `is_exact_zero` calls for counting, then ~810k again for appending.

**Benefit / dominant cost:** Removes one redundant O(n_modes⁴) pass (and one O(n_spatial⁴) for spin blocks). Realistic saving ~15-25% of total ingestion wall time — the count pass is the *cheapest* of three n⁴ passes (validate_spin_orbital_integrals is mandatory and most expensive due to per-pair sqrt via `.norm()`; append allocates per term). The finder's "~50%" figure holds only for the narrow count+append pair in isolation. Maintainability coupling is real: the two loop bodies must stay in perfect sync or capacity reservation / the `max_bytes` check silently diverge. For 30 orbitals: ~810k redundant cheap-ops removed per ingestion.

**New vs ledger:** New. SR2 concerns the downstream `canonicalize_fermion_terms` term loop, not this ingestion count/append duplication. No prior doc tracks it.

**Suggested fix:** Drop `count_spin_orbital_terms`/`count_spin_block_terms`; reserve factors/coefficients with an upper bound (`n_modes*n_modes + n_modes⁴ + 1`, or `n_spatial² + 4*n_spatial⁴ + 1`) and append in a single pass.

### 1.2 [REAL, low] `checked_integer_to_f64` is a context-discarder wrapping `value as f64`; all 10 call sites thread an unused string
**File:** `crates/tencir-pauli-core/src/structured.rs:2271-2274` (call sites: 586, 587, 701, 702, 1015, 1445, 1534, 1728, 1776, 1864)

```rust
fn checked_integer_to_f64(value: i128, context: &'static str) -> Result<f64, PauliError> {
    let _ = context;
    Ok(value as f64)
}
```

The `context` parameter is dead on every path — discarded immediately via `let _ = context;`. The ten call sites each hand-construct a `&'static str` (`"fermion product expansion"`, `"boson product expansion"`, ...) purely to satisfy the dead parameter. This is speculative-abstraction residue: the `checked_` name and `context` arg imply an overflow/precision guard that was never implemented (AGENTS.md explicitly forbids arbitrary-precision/log-domain rescue paths for huge coefficients), so the name misleads a reader into expecting a checked conversion while the function silently truncates via `as f64`.

**Failure scenario:** Any fermion/boson canonicalization calls `checked_integer_to_f64(x, "...")` on every integer coefficient; the string is allocated, passed, and discarded. A reader hunting for where `Overflow` is raised for a huge integer coefficient finds nothing.

**Benefit / dominant cost:** Removes 1 dead parameter, 1 `let _ = context;` line, and 10 hand-written `&'static-str` call-site literals (~12 loc). Per-call runtime cost negligible (one `&str` register pass); the dominant i128→f64 cast remains either way. Clarity/maintainability cleanup, not a performance win.

**New vs ledger:** New. No ledger item touches this wrapper; no `docs/vibe/*` mentions `checked_integer_to_f64`.

**Suggested fix:** Inline to `value as f64` at the ten call sites and delete `checked_integer_to_f64`.

### 1.3 [REAL, low] Dead `let a = p % table_len; let _ = a;` in Weyl local-table compilation
**File:** `crates/tencir-pauli-core/src/structured.rs:3206-3214`

In `compile_mvp_operations`, the `kind==2` (Weyl) branch computes `let a = p % table_len;` (3207), never uses `a` in the following loop (which iterates `digit` and uses `q` and `table_len` only), then discards it with `let _ = a;` (3214). The actual table entry depends on `(q * digit) % table_len`, not on `p % table_len`.

**Failure scenario:** Every structured-MVP plan construction for a Weyl operation performs one `usize` modulo and one unused binding per operation. No correctness impact (value discarded), but a reader will reasonably assume `a` participates in the phase and waste time confirming it does not.

**Benefit / dominant cost:** Removes 2 LOC + one `usize` modulo per Weyl operation at plan-construction time. Runtime saving negligible — plan construction is dominated by the per-digit `Complex64::from_polar` loop below. Real value is readability: the misleading `a` binding suggests `p` participates in the Weyl phase, when the phase depends on `q * digit`.

**New vs ledger:** New. The `let _ = a;` idiom is deliberate unused-variable warning suppression, confirming the author knew `a` was dead. `p` remains used at the `CompiledStructuredOperation` construction; the block is guarded by `if table_len != 0`, so removal is panic-free.

**Suggested fix:** Delete both `let a = p % table_len;` and `let _ = a;`.

### 1.4 [REAL, low] Redundant `|| (term.mapped_present && !term.mapped_codes.is_empty())` clause in fast-fermion eligibility gate
**File:** `crates/tencir-pauli-core/src/charge.rs:3802-3806` (inside `build_fast_fermion_mvp_plan`, defined at 3785; finder cited 3598-3602, stale)

```rust
if terms.iter().any(|term| {
    !term.qubit_codes.is_empty()
        || term.mapped_present
        // The Python descriptor keeps a fixed-width zero code row ...
        || (term.mapped_present && !term.mapped_codes.is_empty())
        || !term.qudit_triples.is_empty()
        || term.qudit_present
}) { return Ok(None); }
```

The `|| term.mapped_present` clause (3802) already short-circuits the closure to `true` whenever `mapped_present` holds, so the later `|| (term.mapped_present && !term.mapped_codes.is_empty())` (3806) is fully subsumed by the absorption law `A || (A && C) == A` and can never change the result. The inline comment justifies why the presence bit alone is the right gate — which is exactly what the first clause already does, making the second dead in all states. `mapped_present` is a stable `bool` field on `ChargeTransitionTerm` borrowed immutably, so no mutation between reads.

**Benefit / dominant cost:** 4 loc (1 redundant `||` clause + 3 comment lines). 0 behavior change, 0 performance impact; pure readability hazard — a reader must prove the clause dead via the absorption law.

**New vs ledger:** New. SR1 is a different function (`FastProductMvpPlan` at charge.rs:2549-2710, full-state hashing). No prior doc mentions this clause.

**Suggested fix:** Delete the `|| (term.mapped_present && !term.mapped_codes.is_empty())` clause and its 3-line comment.

### 1.5 [REAL, low] Diagnostic-only `ChargeApplyCounters` is allocated but never mutated
**File:** `crates/tencir-pauli-core/src/charge.rs:260-271` (struct), `245` (field on `PreparedChargeTransitionPlanLayout`), `1042` (construction), `585` (`debug_counters` accessor)

`PreparedChargeTransitionPlanLayout` carries `counters: Arc<ChargeApplyCounters>` (245), constructed unconditionally at 1042 (`Arc::new(ChargeApplyCounters::default())`). `ChargeApplyCounters` (260-271) has `AtomicUsize` fields exposed via `debug_counters()` at 585. A crate-wide grep for `fetch_add`/`fetch_or`/`store`/`swap` returns only sites on the **unrelated** `ProductApplyCounters` (FastProductMvpPlan::counters); none of the `ChargeApplyCounters` fields is ever incremented. So `debug_counters()` always returns zeros. The accessor is reachable from production (`NativeChargeMvpPlan::debug_counters`, charge_sector.rs:491-497, as a fallback when `fast_plan` is None), and `tests/test_charge.py:138` actually encodes the misleading value: `assert wide_plan._native_execution.debug_counters() == (0, 0, 0, 0)`.

**Failure scenario:** Every `prepare_charge_transition_plan_layout` allocates an `Arc<ChargeApplyCounters>` with zero atomics; every `debug_counters()` returns zeros regardless of how many terms were interpreted. A maintainer reading it to diagnose a hot path is misled into thinking zero raw-term interpretations occurred.

**Benefit / dominant cost:** ~20 loc of dead diagnostic scaffolding removed + 1 negligible `Arc` allocation per plan-layout preparation. No correctness or measurable perf impact; value is cleanup of misleading metadata that always reads zero.

**New vs ledger:** New. R4-D1..D4 cover `charge_sector.rs:252-302` dead FFI, a different surface. Line numbers slightly stale (struct at 260, field at 245, construction at 1042, accessor at 585) but substantively correct.

**Suggested fix:** Delete `ChargeApplyCounters`, the `counters` field, its construction at 1042, and `PreparedChargeTransitionPlanLayout::debug_counters`. If diagnostic counters are wanted, wire them to actual `fetch_add` sites.

### 1.6 [REAL, low] Narration comment restates the loop structure the code already shows
**File:** `crates/tencir-pauli-core/src/charge.rs:1578-1579`

> *Verifier note: the finder and verifier cited a second 5-line narration block inside `FastProductMvpPlan::apply_into` (finder: 2577-2582; verifier "corrected": 2652-2656). I re-read the current tree: **that block does not exist**. Lines 2616-2660 hold `product_schedule_key`, not `apply_into` and not the cited text. A repo-wide grep for "traversed by factor component" / "Factor-local actions are therefore" / "public-sector ranker is only used" returns zero matches. Only the fermion narration block below is present in the current tree. The finding's magnitude is therefore overstated (~2 lines, not ~7); the surviving half is genuine.*

The fermion narration block at 1578-1579 sits immediately above the `for fermion_column in 0..fermion_dimension` loop at 1580:

```
// Lower each fermion action once per compact state and reuse it over
// the contiguous Cartesian spectator segment.
```

This restates exactly what the immediately-following loop does (lower the action once per `fermion_column`, reuse over the spectator segment). It captures no non-obvious invariant (qubit ordering, phase convention, boundary condition). It is "think" narration.

**Benefit / dominant cost:** 2 comment lines removed. 0 branches, 0 per-call runtime cost. (The cited second block is absent; if it is reintroduced by a future edit, the same observation applies.)

**New vs ledger:** New. No ledger item covers narration comments.

**Suggested fix:** Delete the comment block at 1578-1579; if a one-line pointer is wanted, "factor-local actions are hoisted outside the spectator loop" suffices.

---

## Dimension 2 — Inconsistent user-facing Python API

Ordered by severity.

### 2.1 [REAL, medium] `PauliOperator.from_terms` lacks `max_bytes` while every sibling `from_terms` accepts it
**File:** `python/tencirpauli/pauli.py:367-385`

`PauliOperator.from_terms(cls, nqubits, terms)` has no `max_bytes` keyword, whereas `FermionOperator.from_terms` (structured.py:2552-2558), `BosonOperator.from_terms` (structured.py:2741-2747), `QuditWeylOperator.from_terms` (structured.py:2848-2854), and `MajoranaOperator.from_terms` (majorana.py:249-262) all expose `*, max_bytes: Optional[int] = DEFAULT_MAX_BYTES`. `PauliOperator` silently always uses `DEFAULT_MAX_BYTES` internally (pauli.py:338, 346), and `__init__`/`from_code_arrays` hardcode it (16 GiB). The class docstring (pauli.py:313) explicitly promises it honors "the best-effort `max_bytes` guard" for "All matrix and MVP targets," yet the construction entrypoints do not.

**Failure scenario:** A user lowering the memory budget writes `tcp.PauliOperator.from_terms(20, terms, max_bytes=64*1024*1024)` after seeing the identical keyword on `FermionOperator.from_terms`. Python raises `TypeError: from_terms() got an unexpected keyword argument 'max_bytes'`, even though every sibling constructor accepts it and the docstring promises `max_bytes` is honored on all matrix/MVP targets.

**Benefit / dominant cost:** API consistency/correctness gap. `PauliOperator` (the package's primary operator) is the only `from_terms` sibling lacking the documented-universal `max_bytes` keyword; `__init__` and `from_code_arrays` hardcode `DEFAULT_MAX_BYTES`, so users cannot lower the memory budget on any `PauliOperator` construction path. Loud `TypeError`, not silent. Real but not urgent given the working default and diagnosable failure mode.

**New vs ledger:** New. Not R4-N2 (over-defense) — this is a missing-feature/consistency gap. Not among the Rust-side ledger items.

**Suggested fix:** Add `*, max_bytes: Optional[int] = DEFAULT_MAX_BYTES` to `PauliOperator.from_terms` (and ideally `__init__`/`from_code_arrays`) and thread `_effective_max_bytes(max_bytes)` into the two `_native.*` calls.

### 2.2 [REAL, low] `PauliOperator.__init__` ignores `max_bytes` while `MajoranaOperator.__init__` accepts it
**File:** `python/tencirpauli/pauli.py:324-348`

`PauliOperator.__init__(self, nqubits, terms)` takes no `max_bytes`, but `MajoranaOperator.__init__` (majorana.py:190-216) accepts `*, max_bytes: Optional[int] = DEFAULT_MAX_BYTES` and passes `_effective_max_bytes(max_bytes)` into the native call. `PauliOperator` hardcodes `DEFAULT_MAX_BYTES` at pauli.py:338, 346. There is no public path to construct a `PauliOperator` with a custom `max_bytes` (`from_terms` delegates to `__init__`; `from_code_arrays` also hardcodes).

> *Caveat: the finder's title claims `FermionOperator.__init__` also accepts `max_bytes` — this is inaccurate. `FermionOperator.__init__` (structured.py:2493-2501) also hardcodes `DEFAULT_MAX_BYTES`. Only `MajoranaOperator.__init__` and the structured `from_terms` classmethods expose it.*

**Failure scenario:** `tcp.PauliOperator(nqubits, terms, max_bytes=...)` raises `TypeError`, while `tcp.MajoranaOperator(n_modes, terms, max_bytes=...)` succeeds.

**Benefit / dominant cost:** Minor one-parameter API alignment. No current callers pass `max_bytes` to `PauliOperator.__init__`; constructors are already non-uniform across families (`nqubits` vs `n_modes` vs `space`), so the `max_bytes` wrinkle is one of several asymmetries. The core Pauli-vs-Majorana asymmetry is genuine.

**Suggested fix:** Add `*, max_bytes: Optional[int] = DEFAULT_MAX_BYTES` to `PauliOperator.__init__` and pass `_effective_max_bytes(max_bytes)` into `pauli_operator_native` / `pauli_operator_native_array`.

### 2.3 [REAL, low] `BosonOperator.compile` makes `boson_cutoffs` required kw-only while the base declares it Optional
**File:** `python/tencirpauli/structured.py:2820-2834`

`_StructuredOperator.compile` (structured.py:1470-1478) declares `boson_cutoffs: Optional[Mapping[object, object]] = None`, but `BosonOperator.compile` (2820-2827) overrides it as `boson_cutoffs: Mapping[object, object]` with no default and carries `# type: ignore[override]` — the author knew it is a Liskov precondition-strengthening. The base already enforces the requirement at runtime (structured.py:1528-1532 raises `ValueError` when `space.bosons` and `cutoffs` is not a `Mapping`), so the type-level requirement in the override duplicates the runtime check. `FermionOperator.compile` (2680-2687), `MajoranaOperator.compile` (majorana.py:495-502), `QuditWeylOperator.compile` (2959-2967), and `HybridOperator.compile` (3085-3101) all keep the base Optional default.

**Failure scenario:** `op.compile("dense")` polymorphically across operators: for a `BosonOperator` raises `TypeError: missing required keyword 'boson_cutoffs'`, while for every other structured operator the call proceeds to a domain-specific `ValueError`.

**Benefit / dominant cost:** Signature divergence on one sibling. No runtime behavioral bug — both paths error. Consistency-only payoff.

**Suggested fix:** Make `BosonOperator.compile` signature match the base: `boson_cutoffs: Optional[Mapping[object, object]] = None`; the existing `ValueError` already enforces the requirement at runtime.

### 2.4 [REAL, low] `ChargeRestrictedOperator` and `U1RestrictedOperator` lack `compile`/`mvp` despite being operator facades
**File:** `python/tencirpauli/charge.py:1387-1454` (class), `1461-1668` (methods); `python/tencirpauli/symmetry.py:204-228` (class), `261-474` (methods)

`PauliOperator` exposes `compile(target, ...)` (pauli.py:1002), `mvp(state, ...)` (pauli.py:886), `dense/coo/csr` (pauli.py:821-884), and `mvp_plan` (pauli.py:940). `_StructuredOperator` mirrors `compile`/`dense`/`coo`/`csr`/`mvp_plan`. `ChargeRestrictedOperator` (charge.py:1387, methods 1461-1668) exposes `apply`/`apply_into`/`mvp_plan`/`dense`/`coo`/`csr` but has neither `compile` nor `mvp`. `U1RestrictedOperator` (symmetry.py:204, methods 261-474) likewise. Both are returned by public entry points (`restrict_charge` charge.py:1792; `_restrict_u1` symmetry.py:657). The README advertises `compile(...)` as the canonical unified entry point.

**Failure scenario:** `op.compile("csr")` against a value that may be a `PauliOperator` or a `ChargeRestrictedOperator`: for the restricted operator `AttributeError: 'ChargeRestrictedOperator' object has no attribute 'compile'`. The `mvp` gap is largely cosmetic — `apply(state)` already provides the one-shot MVP on both restricted operators — so the workaround is `.apply(state)`, not `.mvp_plan().apply(state)` as the finder overstated.

**Benefit / dominant cost:** Two missing methods on two classes. The `mvp` half is a naming alias only (`apply` already exists); the `compile` half is a genuine unified-dispatch gap, but it would only cover `dense`/`coo`/`csr` (restricted operators do not support `native_mvp`/`backend_mvp` targets, so the finder's suggested dispatch to `native_mvp` is incorrect).

**Suggested fix:** Add `compile(target, *, max_bytes=DEFAULT_MAX_BYTES)` (dispatching over `dense`/`coo`/`csr`) and `mvp(state, *, max_bytes=DEFAULT_MAX_BYTES)` (aliasing `apply`) to `ChargeRestrictedOperator` and `U1RestrictedOperator`.

### 2.5 [REAL, low] `storage` typed as `str` on U1 classes but as `ChargeStorage` Literal elsewhere
**File:** `python/tencirpauli/symmetry.py:212, 224, 308, 489, 503, 663`

`charge.py:47` defines `ChargeStorage = Literal['eager','lazy']` and uses it for `ChargeRestrictedOperator.storage` (charge.py:1411, 1419), `ChargeMvpPlan.storage` (943, 960), `ChargeLazyMvpPlan.storage` (1143), and `restrict_charge` (1796). The sibling U1 classes use plain `str`: `U1RestrictedOperator.storage` (symmetry.py:212, 224), `U1MvpPlan.storage` (489, 503), `U1RestrictedOperator.mvp_plan` (308), and `_restrict_u1` (663). Both accept exactly `{'lazy','eager'}` at runtime, so the annotation is the only divergence. `phase-8.5-spec.md:32` establishes the contract "Every public CPU-native MVP construction API accepts `storage: Literal["eager","lazy"] = "lazy"`," so the U1 classes are a spec-conformance gap, not merely style.

**Benefit / dominant cost:** Type-annotation divergence on the U1 family across ~6 sites. No runtime impact; purely static-typing/spec-conformance. `charge.py` does not import `symmetry.py`, so `from .charge import ChargeStorage` in `symmetry.py` is safe.

**Suggested fix:** Import `ChargeStorage` from `.charge` and use it for `U1RestrictedOperator.storage`, `U1MvpPlan.storage`, both `__init__`s, `mvp_plan`, and `_restrict_u1`.

### 2.6 [REAL, low] `NativeMVPPlan.csr` exists on only one MVP plan; siblings lack it
**File:** `python/tencirpauli/hamiltonian.py:379-400`

`NativeMVPPlan` defines `csr(*, max_bytes)` (379-400) returning `CSRMatrix`. The sibling `BackendMVPPlan` (404-~667), `ChargeMvpPlan` (charge.py:912-1051), `ChargeLazyMvpPlan` (charge.py:1110-1384), and `U1MvpPlan` (symmetry.py:478-593) expose `apply`/`apply_into`/`__call__`/`to_scipy_linear_operator` but no `csr`. The `MVPPlan` Protocol (hamiltonian.py:25-86) mandates only `apply`/`apply_into`/`__call__`/`to_scipy_linear_operator`, so `csr` is a non-protocol extra on exactly one plan implementation.

**Failure scenario:** Code that obtained a plan via `operator.compile('native_mvp')` calls `plan.csr()` and works; the same code against `operator.compile('backend_mvp')` raises `AttributeError`.

**Benefit / dominant cost:** One extra public `csr()` method on `NativeMVPPlan` only. Practical impact near-zero: `NativeMVPPlan.csr` has no production or test callers (all CSR access goes through the operator facades `PauliOperator.csr` / `ChargeRestrictedOperator.csr` / `U1RestrictedOperator.csr` or `compile('csr')`). Cosmetic public-API consistency wart rather than a functional gap. (The finder's type-checking motivation is inverted: `def f(plan: MVPPlan): plan.csr()` is a *static* type error since the Protocol lacks `csr`, not a runtime surprise.)

**Suggested fix:** Add `csr(*, max_bytes)` to `BackendMVPPlan`, `ChargeMvpPlan`, and `U1MvpPlan`, or drop `NativeMVPPlan.csr` and document that CSR materialization is only available via the operator facade.

### 2.7 [REAL, informational] Integral/array batched constructors partitioned by family with no shared name
**File:** `python/tencirpauli/structured.py:2576-2608`, `python/tencirpauli/majorana.py:265-281`, `python/tencirpauli/pauli.py:367-505`

`FermionOperator.from_integrals` (structured.py:2576-2608) and `MajoranaOperator.from_indices` (majorana.py:265-281) provide specialized batched constructors. `PauliOperator` exposes `from_terms`, `from_strings`, `from_code_arrays`, and `canonicalize_code_arrays_numpy`, but the structured family has no equivalent to `from_code_arrays`/`canonicalize_code_arrays_numpy` and no `from_strings`. Neither docstring cross-references the sibling family's batched constructor.

> *Caveat: the title's claim that "BosonOperator exposes from_integrals/from_indices" is inaccurate — `BosonOperator` (structured.py:2719) exposes neither; it only has `from_terms`/`_from_terms`/`_from_words`. The summary correctly restricts to `FermionOperator.from_integrals` + `MajoranaOperator.from_indices`.*

**Failure scenario:** A user of `tcp.PauliOperator.from_code_arrays(...)` looks for `FermionOperator.from_code_arrays` and finds nothing; conversely a user of `FermionOperator.from_integrals(one, two)` looks for the Pauli equivalent and finds none.

**Benefit / dominant cost:** No functional gap — both families support bulk construction via `from_terms`. Pure documentation/discoverability polish with zero runtime, correctness, or performance impact. Severity downgraded to informational.

**Suggested fix:** At minimum add cross-references in the `from_terms`/`from_integrals` docstrings pointing to the sibling family's batched constructor.

### 2.8 [REAL, low] `U1Circuit.value_and_grad` omits `checkpoint_interval`
**File:** `python/tencirpauli/u1_circuit.py:596-617`, `304-333`

`PropagationEngine.value_and_grad` (propagation.py:430-458) and `PropagationBatch.values_and_gradients` (propagation.py:576-609) accept `*, checkpoint_interval: Optional[int] = None`. `PropagationCircuit.value_and_grad` (propagation_circuit.py:464-486) forwards it. `U1Circuit.value_and_grad` (u1_circuit.py:596-617) and `_U1CircuitPlan.value_and_grad` (u1_circuit.py:304-333) take no `checkpoint_interval`. All four compute the same logical quantity (expectation + reverse-mode gradient of a circuit-bound Hermitian Pauli observable).

**Failure scenario:** A user who learned `checkpoint_interval` on `PropagationEngine.value_and_grad` calls `circuit.value_and_grad(obs, checkpoint_interval=64)`. `TypeError: value_and_grad() got an unexpected keyword argument 'checkpoint_interval'`, with no documented workaround.

**Benefit / dominant cost:** Missing kwarg on two methods. Important nuance: the native U1 kernel cannot actually honor `checkpoint_interval` — `NativeU1CircuitPlan.value_and_grad_handle` (u1_circuit.rs:174-194) and `NativeU1FinalState.value_and_grad_handle` (239-254) take no checkpoint argument; the core U1 pass (u1_circuit.rs:608-630, 658-689) is a state-vector reverse-mode pass structurally different from the checkpointed Pauli propagation. `checkpoint_interval` appears only in `propagation.rs`, never in `u1_circuit.rs`. So "honoring" the kwarg is impossible; the only viable fix is accept-and-reject-with-a-clear-message or documentation that U1 gradients are always checkpoint-free. U1 memory is already boundable via `max_bytes` at construction, so the gap is API clarity/error messaging.

**Suggested fix:** Add `*, checkpoint_interval: Optional[int] = None` to `U1Circuit.value_and_grad` and `_U1CircuitPlan.value_and_grad`, rejecting it with a clear message (or documenting that U1 gradients are always checkpoint-free), matching `PropagationCircuit.value_and_grad`.

---

## Dimension 3 — Python↔Rust ABI inconsistencies

Ordered by severity (all low).

### 3.1 [REAL, low] `NativeHybridOperatorHandle.materialize` returns a deeply nested mixed tuple; stub is `tuple[Any, ...]`
**File:** `python/tencirpauli/_native.pyi:204`; Rust `crates/tencir-pauli-native/src/structured.rs:1446-1468` (returns `NumpyHybridOutput`); wrapper `python/tencirpauli/structured.py:2106-2168`

The Rust `materialize` returns `NumpyHybridOutput` (structured.rs:188-207): a 4-tuple `(term_count, nqubits, 7-tuple-of-PyArray1, 5-tuple-of-PyArray1)` — a doubly-nested tuple of flat arrays. The stub is `materialize(self) -> tuple[Any, ...]`, giving zero type information. The wrapper `_hybrid_arrays` must two-level unpack and rebuild 12 Python lists. All four sibling `materialize` methods (pauli convert.rs:13, fermion structured.rs:423, boson structured.rs:1747, majorana majorana.rs:12) return a flat tuple of flat `PyArray1` arrays — the canonical pattern. Hybrid is the sole outlier.

> *Magnitude caveat: the "12 Python-list rebuild" is the shared Python API contract (returning `List[List[...]]`) present in `_fermion_arrays` and `_boson_arrays` too; flattening the hybrid tuple would NOT eliminate it. The genuine win is ~10 lines of two-level tuple unpacking (structured.py:2106-2116) + an explicit stub, plus tuple-shape consistency with the four flat siblings. Not a 40-line / 12-list removal as the finder framed it. The arrays inside the nested tuples are already flat numpy arrays, so AGENTS.md's "deeply nested Python sequences" rule is only weakly violated by the tuple-grouping shape. Zero correctness/perf impact (one-shot read-back, not a hot loop).*

**Failure scenario:** A caller materializing a `NativeHybridOperatorHandle` and unpacking per the stub gets no signal that element 2 is a 7-tuple and element 3 is a 5-tuple.

**Suggested fix:** Flatten the hybrid `materialize` return into a single tuple of flat `PyArray1` arrays and make the stub an explicit 12-tuple of `object`, mirroring fermion/boson `materialize`.

### 3.2 [REAL, low] `pauli_backend_plan_handle` return stub misrepresents the last element as `Sequence[complex]`; Rust returns a numpy array
**File:** `python/tencirpauli/_native.pyi:769-771`; Rust `crates/tencirpauli-native/src/hamiltonian.rs:22-29, 119-135`

The Rust function returns `BackendPlanOutput = (u8, usize, usize, PyArray1<u64>, PyArray1<u64>, PyArray1<Complex64>)` — the last element is a numpy complex128 ndarray. The stub declares `tuple[int, int, int, Sequence[int], Sequence[int], Sequence[complex]]`. The wrapper (pauli.py:920-933) does `np.asarray(coefficients, dtype=np.complex128)` — a no-op view on the `PyArray`. Sibling stubs `pauli_csr_handle` (758-760) and `pauli_coo_handle` (755-757) use `object` for their array elements, so the backend_plan stub is the lone outlier.

**Failure scenario:** A caller trusting `Sequence[complex]` iterates `[c for c in coeffs]` and materializes Python `complex` objects one per term (slow). Conversely if Rust ever changed to return a real `Vec<Complex64>` (a Python list), the wrapper's `np.asarray` would silently switch from view to copy.

**Benefit / dominant cost:** One stub line to correct; positions 4, 5, 6 should all become `object` for full consistency. Pure stub-accuracy fix with zero runtime effect (stubs do not affect runtime; `backend_mvp_plan` is compile-once, not a hot path). The "backend-plan hot path / zero-copy ambiguity" framing is overstated — the only in-repo caller (pauli.py:921) already works correctly.

**Suggested fix:** Change `_native.pyi:771` array elements (positions 4, 5, 6) from `Sequence[int]`/`Sequence[complex]` to `object`, consistent with `pauli_csr_handle`/`pauli_coo_handle`.

### 3.3 [REAL, low] Sibling CSR read-back methods disagree on arity (dimension prefix) and `max_bytes`
**File:** `crates/tencirpauli-native/src/symmetry.rs:212-222` (canonical 4-tuple); `crates/tencir-pauli-core/src/structured.rs` `StructuredEagerMvpPlan.csr` (returns 3-tuple, no `max_bytes`); `crates/tencirpauli-native/src/charge_sector.rs:653` `NativeChargeEagerMvpPlan.csr` (3-tuple, no `max_bytes`); canonical siblings `pauli_csr_handle` (hamiltonian.rs:66), `structured_csr_handle` (structured.rs:2410)

The dimension-prefixed, `max_bytes`-bounded canonical form is: `pauli_csr_handle`, `pauli_coo_handle`, `structured_csr_handle`, `NativeU1RestrictedOperator.csr(max_bytes)`. The dimension-less, unbounded form is: `StructuredEagerMvpPlan.csr()` (3-tuple, no `max_bytes`) and `NativeChargeEagerMvpPlan.csr()` (3-tuple, no `max_bytes`). The stubs (`_native.pyi:46, 360` vs `520`) reflect the same inconsistency. Strengthening the case: on `NativeChargeEagerMvpPlan` itself, the sibling `coo(max_bytes)` (charge_sector.rs:663-671) **does** enforce `max_bytes` (raises `PyMemoryError` on overflow) while `csr()` on the same class does not — an intra-class inconsistency.

**Failure scenario:** A caller writes `dim, indptr, cols, vals = handle.csr(max_bytes)` assuming the canonical shape, then calls the same on an eager plan and gets `ValueError` for too few values to unpack (3 vs 4). The eager `csr()` has no `max_bytes`, so a pathologically large eager plan can exceed a caller's budget on read-back while every sibling CSR method is bounded.

**Benefit / dominant cost:** Align one live native method (`NativeChargeEagerMvpPlan.csr`, charge_sector.rs:653 — live via charge.py:1006) by adding `max_bytes` + dimension prefix to match `coo()`/canonical `csr`. The second method (`StructuredEagerMvpPlan.csr`, structured.rs) has no Python caller (the facade uses `structured_csr_handle` instead). OOM framing is overstated: eager plans are constructed with `max_bytes` enforcement, so the CSR arrays are already resident and bounded; `csr()` read-back only transiently ~doubles an already-bounded allocation, not an unbounded path.

**New vs ledger:** New. R4-D1..D4 cover `charge_sector.rs:252-302` dead FFI, not these `csr` methods.

**Suggested fix:** Add `max_bytes` and a dimension prefix to `StructuredEagerMvpPlan.csr` and `NativeChargeEagerMvpPlan.csr` to match `pauli_csr_handle`/`structured_csr_handle`/`coo()`, or document/stub the eager variants as intentionally cached/unbounded.

### 3.4 [REAL, low] `charge_sector_plan_compact` stubs nested i128 array parameters as bare `object`
**File:** `python/tencirpauli/_native.pyi:369-379`; Rust `crates/tencirpauli-native/src/charge_sector.rs:802-813`; wrapper `python/tencirpauli/charge.py:648-657`

The Rust signature is `charge_sector_plan_compact(..., fermion_weights: Vec<Vec<i128>>, boson_weights: Vec<Vec<i128>>, qubit_levels: Vec<Vec<(i128, i128)>>, qudit_weights: Vec<Vec<i128>>, target: Vec<i128>, max_bytes: u128)`. The stub declares the four nested-sequence parameters as bare `object`. The wrapper passes lists of lists (correct at runtime). The same stub file uses `Sequence[Sequence[int]]` and `Sequence[Sequence[tuple[int,int]]]` extensively (`pauli_group_handle`, `factors` at 417/444, `structured_*_handle` at 385-408), so the bare `object` here is an internal inconsistency.

**Failure scenario:** A caller passes `fermion_weights=np.array([[1,0],[0,1]])` (a 2D numpy array) trusting the `object` stub. PyO3 expects `Vec<Vec<i128>>`; the numpy array is not a Python sequence-of-sequences and conversion fails with an opaque `TypeError`. A typed stub would have guided the caller to `.tolist()`.

**Benefit / dominant cost:** 4 parameters to type correctly. Pure type-stub accuracy improvement; no runtime impact on the supported wrapper path.

**Suggested fix:** Type `fermion_weights`, `boson_weights`, `qudit_weights` as `Sequence[Sequence[int]]` and `qubit_levels` as `Sequence[Sequence[tuple[int, int]]]` in `_native.pyi:373-376`.

### 3.5 [REAL, low] `structured_hybrid_canonicalize` / `structured_hybrid_jordan_wigner` accept 7-field/12-field nested tuples stubbed as bare `object`
**File:** `python/tencirpauli/_native.pyi:449-466`; Rust `crates/tencirpauli-native/src/structured.rs:56-69` (`HybridInput`, 12-field), `70-78` (`HybridRawInput`, 7-field); wrappers `python/tencirpauli/structured.py:799-819, 3043-3062`

`structured_hybrid_canonicalize` takes `input: HybridRawInput` (a 7-field nested tuple) and `structured_hybrid_jordan_wigner` takes `input: HybridInput` (a 12-field nested tuple). Both stubs declare `input: object`. The wrapper hand-builds these tuples (6 live call sites: structured.py:793, 2900, 2940, 3037, 3066, 3199). No handle-accepting canonicalize variant exists (`NativeHybridOperatorHandle` exposes only `jordan_wigner`, which operates on already-canonicalized handles). The sibling canonicalize functions (`structured_fermion_canonicalize:415`, `structured_boson_canonicalize:442`, `majorana_canonicalize:467`) all use fully-typed flat `Sequence[Sequence[tuple[...]]]` inputs; the two hybrid functions are the ONLY canonicalize entries with opaque `object` input.

**Failure scenario:** A caller constructing a hybrid operator via the FFI directly (bypassing the wrapper) must guess the 7/12-field tuple order from Rust source. Misordering same-typed pairs (e.g. `fermion_factors` ↔ `boson_factors`, or `coefficients_re` ↔ `coefficients_im`) yields a silently wrong canonicalization if the inner types happen to align; misordering differently-typed pairs yields an opaque PyO3 conversion error with no field-name context.

**Benefit / dominant cost:** 2 opaque entry points; realistic low-cost fix = explicit `TypedDict`/`namedtuple` alias in the stub (the handle-accepting-variant alternative is a larger refactor requiring new incremental raw-factor construction FFI). Caveats lowering (not negating) severity: `_native` is private; the AGENTS.md "deeply nested sequences" rule targets read-back round-trips, but this is a one-shot Python→Rust input; blast radius is limited to unsupported direct-FFI usage.

**Suggested fix:** Prefer adding a handle-accepting canonicalize variant. If the tuple entry points must remain, replace `input: object` with an explicit `TypedDict`/`namedtuple` alias so the stub documents the 7/12-field contract.

---

## Dimension 4 — Slow-fallback bugs & perf bottlenecks

Ordered by severity.

### 4.1 [REAL, medium] `StructuredEagerMvpPlan::apply_into` runs a serial CSR SpMV while the equivalent charge/Pauli kernels parallelize via rayon
**File:** `crates/tencir-pauli-core/src/structured.rs:3030-3079` (finder cited 2897-2945, stale — verified at 3030-3079 in the current tree)

`StructuredEagerMvpPlan::apply_into` iterates `output.iter_mut().enumerate()` with a plain serial `for (row, output_value) in output.iter_mut().enumerate()` loop (3058) over `indptr[row]..indptr[row+1]` doing `value += self.values[index] * state[column]` (inner loop at 3066). It never calls rayon: `structured.rs` has zero `use rayon`/`par_` references anywhere. The semantically identical CSR SpMV in the charge module, `apply_charge_csr_into` (charge.rs:174-218), parallelizes the same row-major accumulation with `indptr.par_windows(2).zip(output.par_iter_mut())` once `values.len() >= CSR_PARALLEL_TRANSITION_THRESHOLD (1<<19)` (charge.rs:18, 209-212). The Pauli `MvpPlan` (hamiltonian.rs:228) also parallelizes for `work >= 1<<16`. So the structured eager executor is the only CSR/matvec kernel left serial, despite holding the exact same `(indptr, columns, values)` CSR layout the charge kernel does (with `u64` index types instead of `usize`).

This is the eager path taken by `BosonOperator`/`QuditWeylOperator`/`HybridOperator` compiled with `storage="eager"` (routed through `_compile_finite` → `_native.structured_eager_plan_handle` in structured.py:3472-3475 → `crates/tencirpauli-native/src/structured.rs:2466-2487` → core apply at 3030). It is exercised by `benchmarks/manual/representation_ab.py`'s `cartesian_shift_stride` capability class (`operator.compile("native_mvp", storage="eager")` then `plan.apply_into`). The dispatch is correct; the problem is the executor itself lacks the parallelism its sibling kernels have.

**Failure scenario:** Compile a non-charge-restricted structured operator (e.g. `BosonOperator` with high cutoffs, or a large qudit/hybrid operator) to `native_mvp` with `storage="eager"`; each `apply_into` runs the full CSR SpMV single-threaded. For nnz above ~524288 the charge kernel would already engage 4-8 rayon workers, but the structured eager kernel stays serial, leaving the multi-core speedup entirely on the table.

**Benefit / dominant cost:** Path dominant cost is the CSR SpMV itself (O(nnz) complex FMAs); `apply_into` does only validation + fill + SpMV, so the fix owns ~100% of the hot path. Row-parallel SpMV is memory-bound so sublinear, but ~4-6x on an 8-core machine above the ~524288 nnz threshold (matching what the charge eager kernel already realizes) is the right order. Negligible for the small benchmarked dims (cutoff 3 → dim 9); material for large non-charge structured eager plans. Not off by orders of magnitude. Implementation note: `u64` indptr/columns require a `usize` cast to reuse `apply_charge_csr_into`.

**New vs ledger:** New. None of SR1-SR5, R3-*, R4-* cover the structured eager apply kernel's lack of rayon; phase-10.5 reviews cover construction-builder duplication and re-validation, not apply parallelism.

**Suggested fix:** Reuse `apply_charge_csr_into` (or inline a rayon `par_windows` branch gated by `CSR_PARALLEL_TRANSITION_THRESHOLD`) inside `StructuredEagerMvpPlan::apply_into`, after casting the `u64` indptr/columns to `usize` slices. This mirrors the charge eager kernel and the Pauli `MvpPlan` parallelism threshold.

### 4.2 [REAL, low] `U1RestrictedOperator._ensure_eager_for_target` re-clones the CSR via `mvp_plan` on every `.csr()`/`.coo()`/`.dense()` warm call even when the eager operator is cached
**File:** `python/tencirpauli/symmetry.py:366-380`; Rust `crates/tencirpauli-native/src/symmetry.rs:186-191` → `crates/tencir-pauli-core/src/sector.rs:462-464` → `sector.rs:741-759` (`clone_with_budget` ends with `Ok(self.clone())`)

When `self._native_operator` is already cached (non-`None`) and a target is requested (`dense`/`coo`/`csr`), `_ensure_eager_for_target` unconditionally calls `native.mvp_plan(_effective_max_bytes(max_bytes))` (line 372) solely to read `fixed.transition_count` for the `_u1_materialization_bytes` preflight check (line 375), then discards the resulting plan (line 380). `native.mvp_plan` deep-clones the entire CSR graph (`indptr + columns + values`) on every call — `U1MvpPlan::clone_with_budget` (sector.rs:741-759) ends with `Ok(self.clone())` at 758. `transition_count` itself is just `self.values.len()` (sector.rs:673-675), an O(1) read; the core `U1RestrictedOperator` simply does not expose it without building a plan. The cold path (lines 420-421) already reads `fixed.transition_count` and discards `fixed`, so the value is trivially cacheable.

**Failure scenario:** Calling `restricted.csr()` (or `.coo()`/`.dense()`) twice on a `U1RestrictedOperator` that already built its eager cache: the first call builds the cache; every subsequent call re-clones the full CSR via `mvp_plan` just to compute the byte budget, then re-materializes. `benchmarks/python/test_symmetry_benchmark.py:110-135` calls `restricted.csr()`/`.dense()`/`.coo()` once to build `expected`, then `benchmark(restricted.csr)` runs the warm path repeatedly — every timed iteration pays the redundant `mvp_plan` clone.

**Benefit / dominant cost:** Redundant CSR deep-clone of O(D + T) on every cached `.csr()`/`.coo()`/`.dense()` call, where D = restricted-sector dimension and T = transition count. Comparable to the actual `csr`/`coo` materialization cost (also O(D+T)); dominated by the O(D²) allocation for `.dense()`. Not on the matvec hot loop (`apply`/`apply_into` bypass this path entirely). Same category as the open R3 clone deferrals.

**New vs ledger:** New. R3-11/12/14/16/23 cover scratch allocs in `charge.rs::apply_into`, `sector.rs::apply_into`, `structured.rs:2411`, `propagation.rs` PackedKey clone, `charge.rs:1602` — none cover this U1 eager-cached warm-path `mvp_plan` clone. The round-2 dropped finding #1 (symmetry.py:380, "re-canonicalizes and re-compiles terms") was about term re-canonicalization on the lazy→eager cold path, a different mechanism and code path.

**Suggested fix:** After the eager native operator is first built, cache its `transition_count` (read once via a non-cloning getter, or via the `mvp_plan` result already produced during `_ensure_eager_for_target`'s cold path) and use that cached value for `_u1_materialization_bytes` on subsequent calls, skipping the `fixed = native.mvp_plan(...)` clone.

---

## Method statement

**Dimensions scanned (5 parallel finder tracks):** (1) dead-code/dead-branches, (2) verbose-residue/think-narration, (3) py-api-consistency, (4) py-rust-abi, (5) slow-fallback-bugs. Each track produced candidate findings that were then adversarially re-read against the current tree.

**Adversarial re-read verification:** Each finding was independently re-read at the cited file/line in the current repository (commit `ceab218`, 2026-08-06 file state). The verifier confirmed the cited code is genuine, that line numbers anchor to the current tree (with stale-number corrections noted in-line where the verifier relocated the code — e.g. finding 1.4 relocated from 3598-3602 to 3802-3806; finding 4.1 relocated from 2897-2945 to 3030-3079; finding 1.5 relocated struct 248→260, field 234→245, construction 953→1042, accessor 509→585). Each finding was checked against the open/closed ledger (SR1-SR5, R3-11/12/14/16/23, R4-N1, R4-D1..D4, R4-I1, R4-N2) and confirmed new. Magnitudes were checked against the "off by orders of magnitude" rejection bar; two findings had magnitude framing corrected by the verifier (1.1 O(n⁴) duplication: ~15-25% not ~50%; 3.1 hybrid materialize: ~10 lines not ~40), and one finding had a magnitude correction applied by the synthesizer during this round (1.6 narration: the cited `FastProductMvpPlan::apply_into` 5-line block at 2652-2656 was not located in the current tree — lines 2616-2660 hold `product_schedule_key`, and a repo-wide grep for "traversed by factor component" / "Factor-local actions are therefore" / "public-sector ranker is only used" returns zero matches; only the fermion narration block at charge.rs:1578-1579 survives, so the magnitude is ~2 lines, not ~7).

**No source modified.** This is a report only; the user's machine-level instructions and the `audit-report-only-no-code-changes` memory prohibit source changes during the deep audit. All line numbers were re-resolved to the current tree by `grep -n` / `sed -n` against the working directory at `/Users/shixin/Nutstore Files/newwork/quantum-information/codebases/TenCirPauli`.
