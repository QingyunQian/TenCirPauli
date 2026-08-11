# Phase 10.5 Structured-Operator Execution and Sparse-Materialization Specification

Status: owner-approved implementation contract; technical acceptance is closed in the current worktree. The Qudit-charge, buffered-CSR, native execution-family metadata, graph-free lazy Product, grouped factor-local Product schedules, active-axis Structured schedules, shared multiword FermionMask, actual packed-Pauli Structured routing, fixed-width Generic aggregation, compiled charge schedules, and Fermion-factor×Cartesian-Boson Product slice are implemented and release-verified. CPU-native MVP uses raw Fermion descriptors uniformly by owner decision; eligible one-word contiguous descriptors execute through construction-time public-index masks with no per-call `O(D)` packed-source cache, while explicit mappings remain available without a runtime raw/JW selector. The seven-site lazy QuSpin result remains a matched win, the six-site eager/explicit result is accepted as parity under the owner-approved 10% ceiling, and the complete directly comparable QuSpin capability matrix has passed. Clean-commit provenance is intentionally outside this technical acceptance decision.

This specification reconciles the owner decisions recorded in `phase-10.5-owner-decision-log.md`. It supersedes earlier Phase 10.5 draft proposals and the Phase 8.5 physical-layout requirement where they conflict. In particular, optimized Block execution is deferred, factor tables are not the universal lazy design, there is one buffered CSR builder rather than two production builders, and eager storage may retain an approved lossless compressed complete graph instead of literal CSR.

## 1. Objective and measured problem

The objective is one maintainable CPU-native execution substrate for Pauli, Fermion, Boson, finite Qudit/Weyl, and mixed Hybrid operators that preserves exact public semantics while making the common Cartesian and independent-sector Product paths competitive with the best directly comparable QuSpin implementation. Correctness remains the first gate. After correctness is established, representative end-to-end performance and memory are mandatory acceptance criteria rather than optional follow-up work.

The primary seven-site spinful Holstein Product sector has selected dimension `D = 20,070,400`, compact Fermion dimension `F = 1,225`, Boson Cartesian dimension `B = 16,384`, `T = 66` canonical terms, approximately `R = 449.4 million` valid raw contributions before cross-term aggregation, and approximately `E = 299.6 million` final nonzero transitions. One complex128 vector occupies `306.25 MiB`; caller-owned input and output occupy `612.5 MiB`. After factor-local action reuse outside the spectator loop, the final seven-sample one-thread lazy median is about `1.53 s` versus about `1.57 s` for the matched QuSpin matrix-free path.

The corresponding six-site eager workload has `D = 1,638,400`, `E = 22,527,980`, and an exact destination-major CSR size of approximately `528 MiB`. The current eager build takes about `6.5 s`. The seven-site exact CSR would occupy approximately `6.85 GiB` before construction scratch or solver vectors, so the six-site case is the primary explicit eager/CSR acceptance workload.

## 2. Complexity notation

The specification uses the following symbols consistently:

| Symbol | Meaning |
| --- | --- |
| `D` | Selected full or charge-restricted basis dimension. |
| `d_f` | Dimension of one independent Product factor `f`. |
| `T` | Canonical source operator term count. |
| `G` | Count of compiled net-transition groups after structural grouping. |
| `R` | Count of valid raw term contributions before equal-transition aggregation. |
| `E` | Count of final nonzero duplicate-aggregated transitions. |
| `p` | Worker thread count. |

Lazy retained storage must remain independent of `E`. Its apply work is proportional to the valid grouped action that must actually be evaluated and is written as `O(R)` when no stronger workload-specific bound is available. A literal CSR eager graph retains `8(D + 1) + 24E` bytes with `u64` row pointers and source columns plus complex128 values. A qualifying compact eager graph may retain less by encoding repeated destinations, amplitudes, or spectator segments losslessly.

## 3. Required scope

Phase 10.5 must implement the following capabilities:

- Dedicated low-overhead local actions for Pauli, Fermion, Boson, and Qudit/Weyl domains.
- Fast unrestricted Cartesian execution for every supported finite single-domain and mixed-domain layout.
- Fast Product execution when additive constraints split into independent factors, including Cartesian spectators.
- Public-basis-preserving handling of interleaved axes, including the standard interleaved spinful Holstein sector.
- Raw CPU-native Fermion execution without mandatory Jordan–Wigner expansion under default Jordan–Wigner basis semantics.
- Canonical spin-z additive charge for Qudit axes, including charge analysis, sectors, Product classification, lazy execution, eager execution, and explicit sparse materialization.
- Compact lazy execution, complete eager execution, deterministic CSR materialization, exact duplicate aggregation, and strict leakage behavior.
- One correct Generic fallback for constraints or actions that are not Cartesian or Product-factorable.
- Development-time A/B selection of one eager retained representation per mathematical capability class.
- Small exhaustive mixed-domain correctness coverage, bounded routine capability benchmarks, and separate large manual QuSpin acceptance tooling.

## 4. Explicit non-goals

The following work is outside Phase 10.5:

- No optimized Block-factorized execution. A combined charge such as `N_f + N_b` that couples factors remains correct through Generic execution. The future Block design is recorded in `feature-incubator.md`.
- No Holstein-, Hubbard-, Jaynes–Cummings-, or other model-name dispatch.
- No third public CPU-native storage mode beyond `lazy` and `eager`.
- No public strategy selector, cache-size knob, tile-size knob, builder selector, or runtime compact-versus-CSR switch.
- No arbitrary user-defined Qudit per-level charge table. Only canonical integer-normalized spin-z is promoted; arbitrary diagonal level charges remain incubated.
- No universal factor action table as the lazy architecture and no speculative Table/Sparse/Analytic variant family.
- No per-worker full output vector, atomic complex scatter, global transition hash map, intermediate global COO graph, or global sparse-entry sort.
- No compatibility-only private handles, dead builder branches, duplicate rankers, or retained superseded execution paths after their replacements pass correctness and performance gates.

## 5. Public semantics and API

### 5.1 Storage

There are exactly two public CPU-native storage values:

```python
operator.compile("native_mvp", storage="lazy")
operator.compile("native_mvp", storage="eager")
```

`storage="lazy"` retains compiled operator actions, basis indices, bounded local constants, and bounded public/internal mappings but does not retain a complete selected-space transition graph. Every MVP still evaluates the grouped operator action.

`storage="eager"` computes and retains the complete final duplicate-aggregated transition graph. Its retained bytes may be literal CSR or an approved lossless compact graph. Compact eager storage is not lazy: all cross-term duplicate aggregation and exact cancellation represented by the graph are already resolved, and direct MVP does not reevaluate the original operator or general sector logic.

A fixed plan never changes storage or retained format. An explicit `.csr()` request may cause a restricted facade to cache a CSR materialization under its existing cache lifecycle, but a previously returned fixed compact plan does not silently grow another retained graph.

### 5.2 Basis order

Public state vectors, rank/unrank, dense matrices, COO/CSR output, and all user-visible results preserve the existing filtered axis-0-most-significant mixed-radix order. Internal factor order may differ only through a bounded factor-level permutation or a direct public-order ranker. A full `D`-entry public/internal permutation is not permitted for Product MVP.

A factorable workload must not enter Generic merely because spin, species, or domain axes are interleaved. The standard site-interleaved spinful Holstein sector is mandatory Product fast-path coverage.

### 5.3 Minimal plan metadata

Every public CPU-native MVP plan exposes the following read-only diagnostics:

```python
plan.storage
plan.execution_family
plan.retained_format
plan.source_term_count
plan.schedule_count
plan.estimated_retained_bytes
plan.estimated_construction_peak_bytes
plan.transition_count
```

The allowed values are:

| Field | Values or meaning |
| --- | --- |
| `storage` | `"lazy"` or `"eager"`. |
| `execution_family` | `"pauli"`, `"cartesian"`, `"product"`, or `"generic"`. |
| `retained_format` | `"compiled_actions"`, `"compact_graph"`, or `"csr"`. |
| `transition_count` | Exact final nonzero count for eager; `None` for lazy. |

Existing basis-ordering, finite-cutoff, mapping, and Weyl-convention metadata required to interpret a plan remain available. Internal mask layouts, compression payload types, tile sizes, Rayon thresholds, chunk sizes, and builder details are not public API. The old combined `strategy` string is removed after tests and callers migrate; no compatibility alias is retained.

## 6. Qudit spin-z charge

`AdditiveCharge` accepts Qudit site weights through:

```python
AdditiveCharge(space, qudits={site: integer_weight})
```

Local basis state `|0>` is the highest-spin state. For local dimension `d`, define the exact integer spectrum by:

```text
s_d(m) = (d - 1) / 2 - m    for odd d
s_d(m) = d - 1 - 2m        for even d
```

Thus `d=2` uses `(1, -1)`, `d=3` uses `(1, 0, -1)`, `d=4` uses `(3, 1, -1, -3)`, and `d=5` uses `(2, 1, 0, -1, -2)`. The current Qudit API starts at `d=3`; the `d=2` example documents the shared normalization convention with a qubit. A site contributes `integer_weight * s_d(m)`. Odd dimensions use physical spin-z eigenvalues; even dimensions use twice physical spin-z so all charge arithmetic remains exact integer arithmetic. Negative site weights are valid.

Charge equality, hashing, sector construction, rank/unrank, Product classification, conservation analysis, and native execution include these contributions. A compact charged-Qudit factor uses a public-order suffix-count index over reachable exact charge values: each suffix count is the sum over its `d` possible local levels, and rank/unrank walks levels in public mixed-radix order using those counts. The plan stores only reachable suffix-charge entries under the ordinary `max_bytes` policy rather than allocating across the numerical interval between the minimum and maximum weighted charge. A Weyl shift maps `m` to `(m + a) mod d`; its exact charge change is `s_d((m + a) mod d) - s_d(m)` and may depend on `m`, especially at the wrap boundary. No compiler path may assign one constant spin-z delta to a general cyclic `X^a` action.

`AdditiveCharge.as_operator()` materializes each charged Qudit spin-z term through the existing Weyl clock basis. With `omega = exp(2 pi i / d)`, the coefficient of `Z^b` is:

```text
c_b = integer_weight * (1 / d) * sum_m s_d(m) * omega^(-b m),    b = 1, ..., d - 1
```

The zero-trace spectrum has no identity coefficient. A charged site therefore contributes at most `d - 1` ordinary Weyl terms. This explicit materialization is subject to `max_bytes`; normal charge analysis, sector construction, and MVP consume compact integer metadata directly and never call `as_operator()`.

## 7. Native semantic lowering and execution families

All scalable lowering occurs once in GIL-released Rust. Python performs friendly input normalization and one coarse native call. The Rust core owns canonical terms, validated layout, charge metadata, basis indices, net-transition grouping, retained-memory estimates, and hot-loop descriptors.

One semantic lowering contract is shared by lazy execution, eager graph construction, and CSR expansion. It defines authoritative Fermion signs, Boson ladder amplitudes, Pauli phases, Qudit Weyl phases, cutoff behavior, duplicate aggregation, and leakage semantics. This shared contract is an input IR, not a universal hot-loop object. Each execution family consumes it into flat specialized descriptors with one enum dispatch outside the inner loop.

The implemented execution families are:

```text
Pauli       packed binary Pauli executor
Cartesian   unrestricted finite mixed-radix executor
Product     independent restricted factors plus Cartesian spectators
Generic     correct fallback for nonfactorable constraints or actions
```

There is no Phase 10.5 Block variant or placeholder. Hybrid is a composition of local actions inside Cartesian or Product, not a fifth model-specific execution family.

## 8. Domain-specialized local actions

### 8.1 PauliMask

Pauli actions use packed X/Z masks, XOR destinations, parity/phase arithmetic, and grouping by net X transformation. No per-qubit object traversal belongs in an MVP loop.

### 8.2 FermionMask

A canonical Fermion monomial is lowered once to packed descriptors containing required occupied modes, required empty modes, destination flip mask, source-parity mask, fixed sign, and coefficient. For a valid source it produces at most one destination.

Pure unrestricted CPU-native Fermion compilation uses FermionMask directly by default. It does not first expand to Pauli terms or select a mapped path from topology-dependent runtime heuristics. The public Fock order remains the Jordan–Wigner computational-basis order. Explicit `.map_fermions(...)`, parity or Bravyi–Kitaev encodings, Pauli/backend targets, and a user who maps first continue to use the requested mapping.

The same FermionMask lowering and action code is reused by unrestricted, compact-sector, Product, and Hybrid paths. There is no second unrestricted Fermion sign implementation.

### 8.3 BosonLadder

Boson actions use mixed-radix strides, projected finite-Fock boundary tests, ladder powers, and small precomputed square-root products indexed by the local occupation level. They do not clone or decode a complete occupation vector for every term and source.

### 8.4 WeylShift

Qudit actions use modular destination shifts, mixed-radix strides, and small local Weyl phase tables. Charged Qudit execution evaluates the exact source-level-dependent spin-z change. It does not route a simple Qudit operator through the generic full-digit-copy loop.

Small domain metadata such as masks, strides, local square roots, local phase tables, combinatorial coefficients, and bounded factor-order mappings is allowed in lazy plans. A universal table containing destination and amplitude for every action and every factor source is not the general lazy representation.

## 9. Cartesian execution

Cartesian applies to unrestricted finite layouts, including mixed-domain layouts. Plan construction compiles net transformations and public mixed-radix strides once. Destination-to-source inversion uses Pauli XOR, Fermion masks, Boson inverse ladder ranges, and Qudit inverse modular shifts directly.

The inner loop operates on integer indices, masks, strides, and local tables. It must not allocate a digit vector, rescan term structure, revalidate handles, or cross Python. Terms acting on a subset of axes use contiguous public-order segments wherever the mixed-radix layout provides them.

## 10. Product execution

### 10.1 Eligibility and index

Product applies when constraints reduce exactly to independent factors over disjoint axis subsets, with optional Cartesian spectators. Required examples include fixed Fermion particle number with unrestricted Bosons, independent spin-up/spin-down counts with unrestricted Bosons, fixed Boson number with unrestricted Fermions, Qubit Hamming weight with spectators, charged Qudit spin-z factors with spectators, and separate constraints over multiple domains.

The compiler recognizes algebraically equivalent fixed-spin constraints, including total particle number plus spin balance, validates target parity, and produces disjoint compact factors even when physical modes are interleaved by site. This is constraint algebra, not a Holstein branch.

Compact Fermion factors reuse one combinatorial rank/unrank implementation and optional bounded rank tables. Fixed-number Boson factors use bounded-composition rank/unrank. Qubit factors use packed Hamming-weight indices. Charged Qudit factors use the exact additive-sector index. Unrestricted axes remain mixed-radix Cartesian spectators.

### 10.2 Net-transition schedules

Each canonical tensor monomial is lowered to a schedule containing only its active factors and their specialized local actions. Schedules are grouped by the complete coefficient-free net basis transformation across factors: Pauli flips, Fermion occupation flips, Boson occupation shifts, Qudit modular shifts, and their tensor products. Members retain canonical accumulation order.

For a destination configuration, one group derives at most one source configuration. It evaluates member amplitudes, aggregates them in the fixed order, removes only an exact complex zero, and then emits one contribution. Source-dependent ladder/Weyl amplitudes may use a bounded per-source compact-key aggregate; no state-sized transition table is retained.

### 10.3 Loop structure and reuse

The serial baseline owns a destination range and processes grouped schedules in deterministic order. Loop nesting follows the active-factor set, public mixed-radix strides, valid local ranges, and contiguous public-order segments. It is not fixed to Fermion-outside/Boson-inside and is not selected merely by which factor has the larger dimension.

For an action `A_f tensor I_s`, local action work is evaluated at factor granularity and reused across every spectator state. On the primary workload, a unique Fermion action is evaluated over the `1,225` compact Fermion states rather than recomputed across all `20,070,400` Fermion–Boson states; the resulting transition applies over contiguous Boson spectator segments. If the Fermion factor is larger and the spectator dimension is small, the same algorithm remains correct and obtains the smaller available reuse factor. A term acting on every factor may have no spectator reuse, but it still uses specialized masks and strides rather than Generic occupation-vector execution.

The plan retains `O(T)` grouped analytic descriptors plus bounded factor indices and small local constants. It does not retain the complete `E` transitions. Apply performs the unavoidable valid grouped contributions and is `O(R)` in the general case, while moving local semantic lowering, parity checks, square roots, and phase construction outside repeated term interpretation; the selected-sector ranker remains the public-order boundary for each source and surviving destination.

### 10.4 Exact cancellation and leakage

Restriction accepts only the final aggregated action of an exactly conserved operator. An individually leaking term must not be rejected before another term with the same source and destination can cancel it. Conversely, an out-of-sector term must not be discarded individually as a silent projection.

For example, `XX` and `YY` separately map `|00>` outside the zero-excitation sector, while their amplitudes in `XX + YY` cancel exactly. Product groups them by the same net flip, sums the amplitudes, removes the exact zero, and only then applies leakage rules.

Product uses grouped analytic descriptors for this operation. Generic retains a bounded reusable per-source accumulator when the transition structure cannot be represented by Product grouping. No coefficient-magnitude cutoff is permitted.

## 11. Lazy parallelism

The optimized serial Product gather is mandatory and is implemented first. Candidate parallelism splits caller-owned output into disjoint contiguous destination ranges. Each worker processes all schedules for its range and reads the shared immutable input. It does not allocate a worker output vector and does not use atomic complex scatter.

Parallel workspace is limited to `O(p)` bounded iterator/rank scratch in addition to caller-owned input and output. `O(pD)` storage is prohibited. For the primary `D = 20,070,400` case, parallel execution must not replicate the `306.25 MiB` output per worker.

Rayon Product parallelism remains in production only if a same-machine release A/B shows at least `1.5x` steady `apply_into` speedup over the optimized serial kernel on the primary large Product workload. The A/B reports first apply, steady apply, `1/2/4/physical-core` thread settings, peak RSS, and one additional mixed Product shape with materially different factor sizes. If the gate fails, delete the production parallel branch. If it wins only above a measured work crossover, retain one internal work threshold; there is no public parallel strategy option.

Each output element accumulates schedules in the same canonical order regardless of task scheduling. Thread-count changes must preserve the documented numerical determinism contract.

## 12. Eager complete graphs

### 12.1 Semantic contract

Eager retains the complete final duplicate-aggregated graph but may encode it as literal CSR or a lossless compact graph. A compact graph qualifies only if deterministic CSR can be reconstructed through lightweight expansion of retained final information.

Allowed expansion work is direct mask/XOR or mixed-radix stride arithmetic, modular shifts, bounded lookup into already evaluated local or factor amplitude/index data, and repetition over retained contiguous spectator segments. Expansion may not traverse original terms, redo CAR/CCR/Weyl lowering, run general charge DP or repeated rank/unrank, perform leakage analysis, or aggregate operator contributions. A symbolic schedule or factor-local cache that still performs these operations every MVP remains lazy.

### 12.2 Development-time representation selection

Each coarse mathematical capability class has one production eager representation selected during Phase 10.5 development by release A/B against exact CSR. The production compiler does not build both candidates, select by per-instance density or memory, or expose a selector. Representative capability classes include Pauli XOR graphs, regular Cartesian shift/stride graphs, compact-sector mask graphs, Product spectator-segment graphs, and irregular Generic graphs.

The existing Pauli X-mask diagonal is an approved compact graph. It retains one fully evaluated complex diagonal per distinct X permutation mask, reconstructs `destination = source XOR x_mask`, and has already aggregated all terms sharing the mask. It executes directly and expands exact nonzeros to CSR without reevaluating Pauli terms.

A new compact candidate is retained only when its measured retained bytes are strictly less than CSR and its representative steady `apply_into` median is no more than `10%` slower than CSR under matched semantics and thread count. Construction, first apply, CSR expansion, peak RSS, and scaling are also reported. After selection, delete the losing production executor for that capability class. If later representative evidence invalidates the choice, reopen the design rather than adding a runtime switch.

Generic irregular graphs use CSR unless a separately demonstrated compact graph satisfies the same strict decompression and A/B contract. Phase acceptance records the final winner for every implemented capability class.

### 12.3 Direct eager MVP and CSR materialization

A compact eager plan executes directly without first allocating CSR. An explicit `.csr()` request allocates exact arrays and performs one deterministic linear expansion. The fixed compact plan retains only its compact graph. A restricted facade may cache the explicitly requested CSR under the existing thread-safe cache lifecycle.

A CSR eager plan executes destination-major row gather directly. COO and dense materializations derive from the authoritative eager graph and do not rebuild transitions. Flat NumPy read-back shares or moves native arrays where ownership permits and never returns deeply nested Python sequences.

## 13. One-pass buffered CSR builder

Any capability class whose fixed eager representation is CSR, and any explicit materialization path that cannot write directly from a compact graph, uses one production builder. The preferred producer traverses destination rows; a source-major sector producer is also conforming when it performs bounded per-source aggregation and feeds the same deterministic sink, provided it preserves post-aggregation leakage semantics and does not add an inverse semantic implementation. D25 records the charge-sector owner approval for this equivalent producer.

1. Traverse destination rows in deterministic public order, or traverse source columns once when the capability's inverse lowering is not shared and bounded source aggregation is the correctness-preserving equivalent.
2. Generate candidate sources through the shared plan-family descriptors.
3. Aggregate equal source columns and exact cancellations in bounded row-local scratch.
4. Append finalized `(source_column, value)` entries to temporary chunked storage and record row counts.
5. Allocate exact final `indptr`, `columns`, and `values` after `E` is known.
6. Emit finalized entries linearly into the final CSR arrays, finalize pointers, and release every chunk before returning.

The builder performs one transition-generation pass, per-chunk sorting where required, and deterministic merge/final emission. Source-major chunk finalization is `O(E log K)` for `K` chunks; row-ordered finalization is linear after row-local generation. The implementation does not promise an exact `2x` construction speedup over a two-pass design because sorting, heap merge, and final-array bandwidth remain measurable costs, but it avoids a second expensive action/rank traversal.

Peak major storage is approximately `8(D + 1) + 48E` bytes plus bounded row/chunk metadata. Returned retained CSR is approximately `8(D + 1) + 24E` bytes. Dropped chunks return their memory to the Rust allocator for reuse; operating-system RSS is not guaranteed to fall immediately.

The builder includes final and temporary major buffers in checked best-effort `max_bytes` preflight using a conservative contribution bound and checks chunk growth. It fails rather than switching to lazy, switching to a two-pass builder, or publishing a partial graph. No second production CSR builder is retained.

A global `(row, column)` hash, global COO row vector, per-transition occupation key, and global entry sort are prohibited. The old paths are deleted once correctness and performance gates pass. D29 records the 1/8/64/512 target-chunk sweep and the fresh six-site phase profile; the retained specialized pure-Fermion executor is an explicit measured exception to universal Product replacement because its matched A/B remains materially faster and smaller.

## 14. Generic fallback

Generic handles compatible additive constraints or actions that cannot be represented by Cartesian or independent Product indices, including the coupled-charge cases deferred from optimized Block execution. It remains semantically complete and deterministic.

Generic lowering is still flat and native. It reuses source/destination occupation scratch, avoids per-transition heap allocation, uses direct mixed-radix updates when possible, and retains a bounded per-source accumulator for exact cancellation and leakage. It does not acquire model-specific branches or advertise a high-performance Product label.

The plan metadata must make Generic selection visible. A simple Cartesian or Product case entering Generic is a capability failure and blocks acceptance. A genuinely coupled charge entering Generic is expected and receives correctness and no-regression gates rather than a QuSpin-beating performance claim.

## 15. Memory, failure, and determinism

All dimension products, index conversions, charge sums, transition counts, row pointers, buffer lengths, and major allocations use checked arithmetic. `max_bytes` remains a best-effort guard for cheaply estimated major retained output and workspace, not an exact allocator or peak-RSS promise.

A fixed representation that exceeds `max_bytes` fails clearly. It does not switch to another eager representation. The user may request lazy storage or an explicit sparse target separately. Compact-versus-CSR winners are fixed at development time, not chosen from a runtime memory heuristic.

Long-running lowering, plan construction, MVP, CSR expansion, and sparse construction release the GIL before scalable conversion or iteration begins. Trusted native handles are not repeatedly rescanned for hypothetical invalid state.

Canonical term order, schedule grouping, exact-zero handling, CSR row/column order, and public outputs are deterministic. Hash iteration order must not influence serialized terms, sparse arrays, or numerical accumulation order.

## 16. Correctness requirements

Correctness work precedes performance work. Required tests include:

- Independent dense numerical references for every supported local action, including Fermion signs, Boson square-root amplitudes and cutoffs, Pauli phases, Qudit shifts/phases, and canonical Qudit spin-z charge.
- Differential lazy, eager, dense, COO, CSR, and explicit matrix-vector results on small systems.
- Raw FermionMask versus explicit Jordan–Wigner Pauli matrices for hopping, density, general quartic, sign-sensitive ordering, and complex coefficients.
- Product factor rank/unrank round trips and public-order equality for grouped axes and the mandatory interleaved spinful layout.
- Net-transition duplicate aggregation, exact cancellation, and leakage cases, including `XX + YY` in a sector where the individual terms leak.
- Qudit charge conservation and nonconservation across every source level, including Weyl wrap transitions for odd and even `d`.
- Compact eager direct MVP and expanded CSR equality, exact transition counts, retained-memory accounting, and no reevaluation during expansion.
- One-pass CSR equality with independent dense references, deterministic sorted columns, exact zero removal, retry after memory failure where the facade contract permits it, and no retained temporary chunks.
- Generic correctness for coupled charges and simultaneous constraints after optimized Block deferral.
- Numeric tests for every native entry point; shape-only, replay-only, and metadata-only tests do not constitute correctness evidence.

Fast routine correctness coverage includes all eleven nontrivial mixed subsets of Fermion, Boson, Qubit, and Qudit domains: six pairs, four triples, and one four-domain combination. Each uses `D <= 256`, an independently assembled dense reference, terms acting on individual and multiple factors, and assertions on numerical values plus `execution_family` and `retained_format`.

## 17. Benchmark and profiling policy

### 17.1 Common measurement contract

All performance evidence uses optimized release builds and includes public input conversion and the Python/native boundary in the matched measured scope. Every record reports operator construction, sector construction, plan construction, first apply, steady caller-owned `apply_into`, eager transition count, CSR expansion/materialization, retained bytes, estimated construction peak, measured peak RSS, thread settings, numerical error, and residual where an eigensolver is used.

Steady medians use at least two untimed warmups and seven timed repetitions. The two libraries use the same machine, dtype, basis dimension, coefficients, boundary conditions, starting-vector policy, tolerance, and declared thread count. Timing is rejected if dimensions, energies or mapped actions, and residuals do not first agree. A difference within observed run-to-run noise is not accepted as evidence that TenCirPauli is faster.

### 17.2 Three benchmark tiers

The benchmark suite is divided into three tiers:

1. Fast correctness tests cover all eleven mixed-domain subsets with independent dense references and remain in routine pytest/CI.
2. The routine local performance suite is organized by capability rather than all domain subsets. It contains bounded Cartesian, Product, reversed factor-size, FermionMask, BosonLadder, WeylShift, compact-eager, CSR, and Generic cases sized to complete in minutes rather than hours. These benchmarks are local and informational, not wall-time CI gates.
3. Large seven-site lazy, six-site eager, representation A/B, complete thread/RSS, and QuSpin comparisons live under committed manual/development tooling. They run only during performance development and Phase 10.5 acceptance and are excluded from the default benchmark group.

Large manual runners preflight expected live vectors, retained graphs, construction buffers, sparse copies, and solver workspaces and avoid swap. Acceptance workloads must fit comfortably on the declared machine; a run whose planned major live storage exceeds half of physical RAM is not a valid default acceptance run.

### 17.3 Separate QuSpin Conda environment

QuSpin runs in a dedicated pinned Conda environment separate from the TenCirPauli project environment. The repository provides a capability-named environment file such as `benchmarks/environments/quspin.yml`, committed capability manifests, and environment-specific child runners. TenCirPauli and QuSpin are never imported into the same Python process.

A parent harness accepts or discovers two environment-specific Python executables and launches separate subprocesses. The subprocesses share only a frozen workload manifest, deterministic seed or starting-vector policy, basis and coefficient conventions, explicit BLAS/OpenMP/Rayon thread variables, and flat JSON/NPY/NPZ measurement or result artifacts under ignored benchmark output. No machine-specific environment path, environment contents, credentials, or raw benchmark result directory is committed.

QuSpin comparisons use `quantum_LinearOperator` for matrix-free overlap and an explicit QuSpin sparse path for explicit graph/CSR overlap. Each bounded capability preflight writes an action in canonical occupation order and the parent verifies equality after the documented basis permutation before timing. Every row records construction, first apply, two untimed warmups, seven steady samples, retained bytes, construction peak, RSS, thread settings, numerical error, actual native variant, execution family, and exact child command. TenCirPauli records both caller-owned `apply_into` and allocation-inclusive public `apply`; QuSpin records its allocation-inclusive public `dot` separately because it lacks an equivalent caller-owned output endpoint. Near-threshold rows use fresh-process ABBA order. A callback-heavy `user_basis`, Python custom operator, nonequivalent sector, or nonequivalent Weyl action is not an optimized direct baseline and receives no fabricated speedup claim.

### 17.4 Required development workloads and gates

The primary lazy workload is the seven-site open-boundary spinful Holstein model with local Boson cutoff `3`, `N_up = N_down = 3`, `t = 1.0`, `omega = 0.5`, `g = 0.7`, `u = 1.5`, complex128, and the committed deterministic starting-vector policy. The primary eager and explicit sparse workload is the corresponding six-site system with `N_up = N_down = 3`.

Required development coverage also includes pure Cartesian Fermion and finite-cutoff Boson cases; Qubit/spin execution in full and fixed-Hamming-weight spaces; Fermion–Boson Cartesian; fixed Fermion number with Boson spectators; fixed Boson number with Fermion spectators; independent factorized constraints over two or more factors; one reversed factor-size Product case; and one intentionally Generic coupled-charge case. The QuSpin capability manifests must contain multiple conserved and mixed operator families rather than only a single-model Hamiltonian. Finite Qudit/Weyl and coupled-charge cases without an equivalent optimized QuSpin basis remain in the internal matrix with an explicit no-fair-baseline classification.

For every directly equivalent optimized QuSpin case, TenCirPauli should show a repeatably lower matched median at one thread and at the declared fixed multi-thread setting. A repeatable public-call ratio `TenCirPauli / QuSpin <= 1.10` is accepted as parity when correctness, memory, and scaling remain sound, but it is not a speedup claim; a repeatable ratio above `1.10` is a blocker. The primary seven-site lazy path must beat QuSpin matrix-free. The primary six-site eager/explicit workflow is accepted when its seven-sample matched median satisfies the same 10% parity ceiling after numerical preflight and without a material memory or scaling regression. QuSpin's lightweight matrix-free setup is not an eager comparison. Unsupported or nonequivalent cases use internal before/after, throughput, memory, and scaling evidence instead.

Product Rayon remains only under the `1.5x` serial A/B gate in Section 11. A compact eager representation remains only when it is strictly smaller than CSR and no more than `10%` slower in steady MVP under Section 12.2. These are development-selection gates; losing production implementations are deleted after evidence is recorded.

## 18. Implementation sequence

1. Add benchmark-only profiling around semantic lowering, sector indexing, local actions, rank/unrank, net-group aggregation, CSR row generation, final copy, and direct apply. Confirm the current dominant costs before optimization.
2. Implement the shared native semantic lowering IR, one-time execution-family classifier, minimal plan metadata, and flat plan descriptors. Migrate callers away from the old combined `strategy` string.
3. Implement canonical Qudit spin-z charge end to end, including constructor transport, exact local contributions, conservation, sectors, rank/unrank, Weyl wrap behavior, and `as_operator()` Fourier expansion.
4. Extract one reusable FermionMask and compact Fermion index, then implement and differentially test raw unrestricted Fermion execution against Jordan–Wigner references.
5. Implement Cartesian destination-gather kernels for Fermion, Boson, Qubit/Pauli, Qudit, and mixed layouts using domain-specialized masks, strides, and local tables.
6. Implement Product classification, public-order factor mapping, net-transition grouping, deterministic destination gather, spectator-segment reuse, exact cancellation, and leakage handling. Close the optimized serial primary lazy benchmark before adding Rayon.
7. Implement eager compact-graph candidates by coarse capability class and the one-pass buffered CSR baseline. Run the D17 A/B matrix, document each winner, delete losing executors, and retain only shared CSR expansion where required.
8. Implement the Product destination-range Rayon experiment. Retain it only if the `1.5x` gate passes; otherwise delete it.
9. Route nonfactorable coupled charges through the flattened Generic path, preserve bounded aggregation semantics, and verify no correctness regression without adding a Block placeholder.
10. Delete global COO/hash construction, the two-pass builder if introduced experimentally, duplicate rankers/sign implementations, mandatory native Jordan–Wigner execution, obsolete handles, combined strategy metadata, and compatibility-only branches.
11. Run formatting, linting, type checking, Rust tests, Python tests, routine capability benchmarks, the manual representation A/B matrix, and the separate-environment QuSpin acceptance suite. Record final winners and evidence in a Phase 10.5 acceptance document.

## 19. Acceptance criteria

Phase 10.5 is complete only when all of the following are true:

- Every simple finite Cartesian and independent-sector Product case selects `pauli`, `cartesian`, or `product` rather than `generic` and matches independent numerical references.
- Coupled-charge cases deferred from Block remain correct through Generic and are labeled honestly.
- Public basis order, Qudit spin-z semantics, Fermion signs, Boson amplitudes, Weyl phases, exact cancellation, and leakage behavior pass focused and differential tests.
- Lazy plans retain no complete graph; eager plans retain one complete literal or losslessly compressed graph; compact CSR expansion performs no operator reevaluation or duplicate aggregation.
- Every eager capability class has one documented retained-format winner. No runtime compact/CSR switch or losing production executor remains.
- The one-pass CSR path contains no global sparse hash, COO intermediate, global sort, or retained construction chunks.
- The primary seven-site lazy Product workload beats the matched QuSpin matrix-free path, and the primary six-site eager/explicit workflow either beats the matched QuSpin explicit sparse path or satisfies the owner-approved parity rule in 17.4 (seven matched samples within 10%, numerical preflight passed, and no material memory/scaling regression), with no speedup claim for parity.
- Every other directly equivalent optimized QuSpin development case has either a repeatably lower TenCirPauli median or a repeatable matched public-call ratio no greater than `1.10` under matched semantics and threads. Nonequivalent cases carry an explicit no-fair-baseline classification and make no speedup claim.
- Product parallelism exists only if its measured speedup is at least `1.5x`, uses no `O(pD)` memory, and preserves deterministic accumulation.
- Every compact eager winner is strictly smaller than CSR and no more than `10%` slower in steady MVP.
- The fast correctness matrix covers all eleven mixed-domain subsets, routine benchmarks remain bounded, and large QuSpin work stays in separate-environment manual tooling.
- Superseded global builders, duplicate implementations, dead handles, and compatibility-only branches are deleted after their replacements pass.
- The complete project quality gate passes: Rust formatting, Clippy with warnings denied, Rust workspace tests, Python formatting/lint/type checks, release native build, and Python integration tests.
