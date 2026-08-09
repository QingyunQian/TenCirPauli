# Phase 10.5 Owner Decision Log

Status: historical owner-decision ledger, reconciled into `phase-10.5-spec.md` on 2026-08-07. The formal specification is now the implementation contract; this file preserves the discussion history and superseded alternatives.

## Discussion protocol

1. Discuss one technical decision at a time.
2. Explain the current implementation, concrete workload data, time complexity, retained memory, construction scratch, runtime consequences, maintainability cost, and rejected alternatives before requesting approval.
3. Mark an entry `approved` only after explicit owner confirmation. Questions, tentative agreement, and assistant recommendations remain `pending`.
4. After approval, update this ledger first and report the recorded wording to the owner. Update the implementation specification only when the owner authorizes reconciliation.
5. If later evidence reopens a decision, preserve the earlier decision and append a superseding entry rather than silently rewriting history.

## Approved decisions

### D01 — Eager storage means a complete transition graph

Status: `approved`, with its mandatory-CSR layout narrowed by D15.

Decision: public structured `storage="eager"` computes and retains every final duplicate-aggregated transition. The retained execution representation is a complete destination-major CSR graph. If a user does not want complete transition materialization, the user selects lazy storage. Eager must not alias lazy or retain only bounded local caches while advertising eager storage.

Rationale: eager and lazy are an explicit time/space trade-off. Eager pays construction time and retained memory once to minimize repeated MVP work; lazy avoids the complete graph.

Historical layout requirement: this decision originally required the retained representation itself to be destination-major CSR. D15 supersedes that physical-layout restriction while preserving complete eager materialization semantics. D13 governs cases that retain or explicitly materialize CSR.

### D02 — Internal factor order must not force a generic fallback

Status: `approved`.

Decision: public state vectors, rank/unrank, sparse outputs, and all user-visible results preserve the existing filtered mixed-radix basis order. Internally, Product and Block kernels may use a different factorized order, but they must maintain a bounded mapping or a direct public-order ranker. A mathematically factorable workload must not fall back to Generic merely because its spin, species, or domain axes are interleaved in public order. The standard interleaved spinful Holstein sector is mandatory fast-path coverage.

Rationale: basis ordering is a public semantic contract, while factor ordering is an internal performance choice. The implementation owns the permutation between them.

### D03 — Performance takes priority over universal hot-loop abstraction

Status: `approved`.

Decision: correctness semantics and plan-level interfaces may be shared, but a common abstraction must not impose a material hot-loop performance loss. Execution-family classification happens during plan construction, and Cartesian, Product, Block, and Generic kernels may use specialized data layouts and loops. A universal emitter, trait object, or repeated runtime family switch is rejected when it materially slows representative workloads.

Rationale: maintainability and reuse are required, but reuse belongs at the semantic lowering and plan-contract layers when lower-level unification conflicts with the project's performance purpose.

### D04 — Qudit additive-charge metadata is deferred from Phase 10.5

Status: `approved`.

Decision: Phase 10.5 treats finite qudit/Weyl axes as Cartesian factors or zero-charge spectators. Exact per-level qudit additive-charge metadata is not part of the current implementation scope and is tracked as a near-term item in `feature-incubator.md`.

Reopened: the owner subsequently requested evaluation of bringing diagonal qudit charge metadata back into Phase 10.5. D04 remains historical until a separately approved decision supersedes it; see P04.

### D05 — Superseded implementation paths are deleted

Status: `approved`.

Decision: after replacement paths pass correctness and representative performance gates, delete superseded global COO/hash builders, duplicate rankers, duplicate transition semantics, obsolete private handles, and dead compatibility branches. Do not retain old private implementations solely for source or ABI compatibility.

Rationale: correctness and performance regressions must be prevented by tests and benchmarks, not by carrying parallel production implementations.

### D06 — Domain-specialized lazy kernels composed by basis structure

Status: `approved`.

Decision: each supported local operator domain has a dedicated low-retained-memory lazy action kernel using its natural arithmetic: Pauli bit masks and XOR/phase rules, fermion packed occupation masks and parity with compact-sector rankers when restricted, boson mixed-radix stride and ladder-amplitude rules, and qudit modular stride and Weyl-phase rules. Boson and qudit fast paths must be completed rather than routed through the current full-digit-copy generic loop.

Decision: Cartesian, Product, and Block lazy executors compose these domain kernels according to basis structure and move factor-local computation outside spectator loops. Hybrid is a composition of domain actions, not a model-specific fifth action kernel. Generic occupation-vector/DP execution remains only for constraints or actions that are structurally not reducible to Cartesian, Product, or Block execution.

Decision: the public CPU-native storage choices remain lazy and eager. Product or factor-ordered execution is an internal lazy strategy, not a third public storage value. Lazy means that the complete global transition graph is not retained; it does not prohibit small domain metadata such as strides, masks, local square roots, local Weyl phases, combinatorial coefficients, or bounded public/internal basis mappings.

Rationale: this architecture gives every domain an analogous fast path without forcing binary bit-mask representation onto non-binary boson or qudit axes. It separates local algebra, basis indexing, and storage while keeping domain-specific hot loops available for performance.

### D07 — Qudit charge is canonical integer-normalized spin-z

Status: `approved`; supersedes D04.

Decision: Phase 10.5 brings qudit additive charge into scope using only the canonical local spin-z spectrum. The public declaration is `AdditiveCharge(space, qudits={site: integer_weight})`; it does not accept an arbitrary per-level charge table. Local basis state `|0>` is the highest-spin state. For local dimension `d`, use the smallest integer normalization that preserves exact integer charge storage: odd `d` uses the physical spin-z eigenvalues, while even `d` uses twice the physical spin-z eigenvalues. Examples are `d=2: (1, -1)`, `d=3: (1, 0, -1)`, `d=4: (3, 1, -1, -3)`, and `d=5: (2, 1, 0, -1, -2)`. `integer_weight` multiplies this canonical spectrum and may be negative.

Decision: charge analysis must evaluate the exact source-level-dependent change of a cyclic Weyl shift. In particular, a Weyl `X^a` does not generally have one constant charge delta because its wrap transition has a different spin-z change. This is an internal conservation and transition-analysis rule, not an additional public API mode.

Decision: arbitrary user-defined diagonal qudit level charges remain outside Phase 10.5 and stay in the Feature Incubator.

Rationale: qudit spin-z is concise and unambiguous at the public boundary, preserves exact integer sector arithmetic, and supplies the near-term physical charge semantics without introducing a second general-purpose per-level charge language.

### D08 — Qudit charge materializes through the existing Weyl basis

Status: `approved`.

Decision: `AdditiveCharge.as_operator()` materializes each charged qudit site's canonical integer-normalized spin-z contribution as its exact discrete-Fourier expansion over the existing Weyl clock powers `Z^b`. A charged site of local dimension `d` contributes at most `d - 1` nonidentity Weyl terms because the canonical spin-z spectrum has zero trace. The method returns the ordinary structured operator type and remains subject to its existing best-effort `max_bytes` guard.

Decision: sector construction, charge-conservation analysis, restricted-basis indexing, and normal MVP planning consume compact integer charge metadata directly and must not call `as_operator()` or pay for this Weyl expansion. Phase 10.5 does not add a second `QuditSzWord` algebra representation or reject Qudit charges from `as_operator()` merely to optimize this low-frequency explicit-materialization operation.

Rationale: the Weyl expansion keeps the public operator algebra complete and uniform without spreading a second qudit representation through multiplication, embedding, serialization, sparse construction, and the FFI. Its `O(n_q d)` explicit-materialization cost is isolated from the performance-critical charge and restricted-MVP paths, where `n_q` is the number of charged qudit sites and `d` is their uniform local dimension.

### D09 — CPU-native Jordan–Wigner Fermion execution uses raw Fermion masks

Status: `approved`.

Decision: a pure unrestricted `FermionOperator` compiled for CPU-native execution with the default Jordan–Wigner basis semantics must execute its canonical Fermion words directly. Each word is lowered once to compact bit-string descriptors covering required occupied and empty modes, the destination flip mask, the source-parity mask and fixed sign, and its coefficient. Lazy MVP evaluates these descriptors directly and does not first expand the operator into Pauli words.

Decision: this is an internal implementation choice, not a new public mapping value. The public Fock-basis ordering remains exactly the Jordan–Wigner computational-basis ordering. Explicit `operator.map_fermions(...)`, backend/Pauli targets, and explicitly requested parity or Bravyi–Kitaev encodings continue to perform the requested mapping. A user who explicitly wants Pauli execution may map first and then compile the resulting `PauliOperator`.

Decision: the same Fermion-mask lowering and action semantics must be reused by unrestricted, charge-restricted, Product, Block, and Hybrid CPU-native plans. Do not retain a second raw-Fermion execution implementation for the unrestricted case. Representative hopping, density-density, and general quartic benchmarks must show no performance regression against the existing Jordan–Wigner-to-Pauli CPU path before that path is removed from default native Fermion execution.

Rationale: one canonical Fermion monomial produces at most one transition from a source state, whereas Jordan–Wigner may expand it into several Pauli terms. Direct bit-string validity, parity, and flip operations preserve the same matrix while avoiding mapping expansion and providing the reusable domain kernel already required by the structured Product and Hybrid architecture.

### D10 — Product lazy execution is factor-structural, not model-specific

Status: `approved`.

Decision: a Product lazy plan lowers every canonical tensor monomial to a factor schedule containing only the factors on which that monomial acts and their domain-specialized actions. Fermion masks, boson ladder/stride actions, Pauli masks, and qudit Weyl shifts are evaluated at their factor granularity and reused across every identity spectator factor. The implementation must not dispatch on Holstein, Hubbard, spin-boson, or any other model name.

Decision: loop nesting is derived from the term's active-factor set, the public mixed-radix strides, valid local ranges, and contiguous public-order segments. It is not fixed to Fermion-outside/Boson-inside, is not selected merely by which domain has the larger dimension, and must preserve public basis order. A term acting on every factor may have no spectator reuse, but it still uses the same specialized local actions and Product index rather than full occupation-vector generic execution.

Decision: lazy Product execution owns destination ranges and obtains the unique source of each supported monomial action by its inverse local descriptors. This destination-gather form is the baseline loop contract because Pauli words, canonical Fermion monomials, boson ladder monomials, and Weyl words are partial one-to-one local actions; their tensor products remain partial one-to-one. It permits later race-free parallelism without atomics or one state-sized output buffer per worker. Exact tiling and parallel thresholds remain separate decisions.

Decision: the final Phase 10.5 benchmark matrix must contain representative regression workloads for every supported mixed-domain composition and relevant active-factor shape, not only the Holstein workload. For every case where QuSpin provides a directly equivalent optimized representation under the same basis, sector, operator semantics, dtype, thread count, and measured boundary, TenCirPauli must be faster. Cases without a directly equivalent QuSpin path, including Weyl semantics that would require Python callbacks or a custom slow basis, must be labeled as having no fair direct QuSpin baseline and must instead retain reproducible internal before/after, throughput, memory, and scaling regression evidence. The exact workload and acceptance matrix remains to be approved separately.

Rationale: factor reuse is a property of a tensor-product monomial, not of one physical model or of a fixed relative factor size. Structure-derived schedules provide one reusable implementation while the benchmark matrix prevents the general abstraction from hiding regressions in less prominent mixed-domain combinations.

### D11 — Optimized Block execution is deferred beyond Phase 10.5

Status: `approved`; narrows the Block scope previously included by D06.

Decision: Phase 10.5 implements optimized Cartesian and Product execution but does not implement an optimized Block-factorized plan. Additive charge constraints that couple otherwise separable factors remain supported for correctness through the Generic sector path, without a Phase 10.5 high-performance guarantee. The formal specification must not advertise Block as a required fast family after reconciliation.

Decision: do not add a dormant `BlockPlan` variant, a Jaynes–Cummings-specific branch, unused prefix-segment machinery, or another placeholder implementation in Phase 10.5. The future implementation is tracked in `feature-incubator.md` and should be promoted only for a concrete cross-factor conversion workload such as Jaynes–Cummings/Tavis–Cummings excitation exchange, exciton–photon conversion, atom–molecule Feshbach conversion, or subsystem spin exchange.

Decision: the deferred design target remains a general public-order direct sum of Product blocks with bounded prefix/index metadata and no full selected-space permutation. Promotion requires its own correctness, memory, construction, steady-MVP, eager, and directly comparable performance evidence.

Rationale: the current Holstein target and ordinary independent-sector mixed Hamiltonians require Product, not Block. A general high-performance Block implementation adds cross-block rank/unrank, public-order prefix indexing, inter-block transitions, simultaneous-charge behavior, and source-level-dependent qudit charge changes. Deferring it keeps the current phase focused while preserving correctness through Generic execution.

### D12 — Product parallelism is retained only after a 1.5x A/B win

Status: `approved`.

Decision: the optimized serial destination-gather Product kernel is the required baseline. Destination-range Rayon parallelism is a candidate second-stage optimization, not a required production branch. It may be retained only if a release-mode same-machine A/B shows at least `1.5x` steady `apply_into` speedup on the primary large Product workload, meaning parallel median time is at most two thirds of the serial median, with numerical correctness preserved.

Decision: the A/B must report first apply, steady `apply_into`, fixed `1/2/4/physical-core` thread settings, and peak RSS. It must include at least one additional mixed Product workload with a materially different factor-size and active-factor shape, and must show no material representative regression. Destination ownership must preserve a fixed per-output accumulation order across thread counts.

Decision: production parallelism must use disjoint destination ranges with caller-owned output. Per-worker full output vectors, atomic complex scatter, or any `O(p D)` workspace are prohibited, where `p` is the thread count and `D` is the selected-space dimension. Only `O(p)` bounded iterator/rank scratch is permitted in addition to the caller-owned input and output. If the `1.5x` gate is not met, delete the production parallel branch and retain only the serial kernel and benchmark coverage. If the gain exists only above a measured work crossover, retain an internal work-based serial/parallel threshold rather than a public strategy option.

Rationale: Product MVP has unchanged `O(R)` total work and may become memory-bandwidth-bound after factor-local arithmetic is hoisted, so useful multicore scaling is plausible but not guaranteed. A high adoption threshold prevents Rayon scheduling, threshold logic, and testing burden from surviving for a marginal gain.

### D13 — Eager CSR uses one transition pass with a temporary row buffer

Status: `approved`.

Decision: eager construction generates each candidate transition only once. It traverses destination rows in deterministic public order, aggregates duplicate source columns and exact cancellations in bounded row-local scratch, appends finalized `(source_column, value)` entries to temporary chunked storage, and records row counts. After the exact final entry count `E` is known, it allocates the exact destination-major CSR arrays, copies the buffered entries linearly, finalizes `indptr`, and releases all chunk buffers before returning the plan or explicit matrix.

Decision: the returned eager representation retains only the exact CSR arrays and therefore has the same steady retained memory as a two-pass builder: approximately `8(D + 1) + 24E` bytes for `u64` row pointers/columns and complex128 values. Construction peak is approximately `8(D + 1) + 48E` bytes plus bounded row/chunk metadata. The temporary entry storage is deallocated before return; the process allocator may reuse those pages even if operating-system RSS does not immediately fall.

Decision: the major final and temporary buffers are included in eager construction's best-effort `max_bytes` preflight using a checked conservative contribution bound before work begins, and checked again as chunks grow. Construction fails rather than silently switching to lazy, a two-pass builder, or an incomplete graph. Phase 10.5 has one production eager builder; it does not add an automatic one-pass/two-pass policy.

Decision: the single-pass builder must not use a global `(row, column)` hash map, retain global COO row indices, or perform a global entry sort. The shared destination-row generator and row-local duplicate aggregation define eager semantics. Construction benchmarks report transition generation, final linear copy, peak memory, and total time separately; one generation pass is expected to reduce the dominant work but is not advertised as an unconditional exact `2x` end-to-end speedup.

Rationale: eager is already the high-memory choice and returns a graph large enough to justify repeated use. Paying roughly one additional temporary copy of its entries during construction is acceptable in exchange for avoiding a second expensive rank/action/amplitude traversal, while releasing the temporary storage keeps steady plan memory equal to exact CSR storage. D15 subsequently narrows this decision to eager families without an approved lossless compressed representation and to explicit CSR materialization.

### D14 — Lazy Product groups equal net transitions before leakage handling

Status: `approved`.

Decision: Product lazy schedules are grouped by their exact coefficient-free net basis transformation across all factors: Pauli flip masks, Fermion occupation flip masks, boson occupation shifts, qudit modular shifts, and their tensor products. Terms in one group retain deterministic canonical accumulation order. For a destination configuration, the group derives at most one source configuration, evaluates the member amplitudes, aggregates them, removes only an exact complex zero, and only then accepts an in-sector contribution or reports nonzero sector leakage.

Decision: neither lazy nor eager execution may reject an individually leaking term before contributions with the same source and destination have had the opportunity to cancel. Conversely, no path may discard an out-of-sector contribution term by term and thereby silently project a nonconserving operator. Restriction construction continues to reject an operator whose final aggregated action does not conserve the requested charge.

Decision: Product uses `O(T)` factor-local analytic action descriptors and no state-sized transition table, where `T` is the canonical schedule count. A bounded per-source destination aggregate remains permitted for source-dependent ladder/Weyl amplitudes and exact cancellation; it must contain only the current source's compact factor/spectator keys. The Generic fallback retains the same bounded accumulator for structures that cannot be represented by Product descriptors. Eager construction uses the shared semantic lowering in destination-row scratch before emitting final CSR entries. No coefficient-magnitude cutoff is permitted.

Rationale: distinct canonical words can induce the same transition on a selected state. For example, `XX` and `YY` separately leave the zero-excitation sector from `|00>`, but their amplitudes in `XX + YY` cancel exactly before leakage is assessed. Grouping also reduces repeated input loads, complex multiplies, and output writes; the measured seven-site workload has about `449.4 million` raw valid contributions but about `299.6 million` final CSR entries after cross-term aggregation.

### D15 — Eager may retain a lossless compressed complete graph

Status: `approved`; supersedes D01's mandatory retained-CSR layout and narrows D13.

Decision: public `storage="eager"` means that the complete duplicate-aggregated transition graph has been computed and retained, but it does not require the retained bytes to be literal `indptr`, `columns`, and `values` arrays. A family-specific lossless compressed representation is preferred when it retains less memory and executes MVP at least as fast as materialized CSR on representative release benchmarks.

Decision: a representation qualifies as eager compression only if standard deterministic CSR can be reconstructed by linear decompression or expansion of retained final amplitudes and structural destination metadata. Reconstruction must not rerun canonical operator terms, factor actions, charge-sector rank/unrank, leakage analysis, or duplicate aggregation. A factor-local cache or symbolic schedule that still performs those operations during every MVP remains lazy regardless of its size.

Decision: the existing pure-Pauli X-mask diagonal representation is an approved eager compression. It retains one fully evaluated complex diagonal over all source states for each distinct X permutation mask, reconstructs each destination as `source XOR x_mask`, and has already aggregated all Pauli terms sharing that mask. Explicit `.csr()` scans these retained values, removes exact zeros, and emits deterministic CSR without reevaluating the Pauli operator.

Historical scope note: D15 initially approved only the existing Pauli X-mask representation and left Structured families on D13's CSR baseline. D16 subsequently generalizes lossless compressed eager storage across all supported operator families under one strict qualification contract.

Rationale: eager is a time/space choice, not a demand to retain redundant destination indices. Algebraic destination formulas such as Pauli XOR can encode the same complete graph with less memory and fewer MVP loads. Requiring literal CSR would sacrifice performance and memory for representation uniformity without improving correctness or public semantics.

### D16 — Lossless compressed eager storage applies to every operator family

Status: `approved`; broadens D15 beyond pure Pauli and further narrows D13.

Decision: Cartesian, Product, Fermion, Boson, Qudit, Pauli, and Hybrid eager plans may retain a compact lossless representation of their complete final transition graph instead of materialized CSR whenever the representation can directly execute MVP and reconstruct deterministic CSR through lightweight expansion. Representation selection is by mathematical capability class, never by model name. D17 subsequently fixes selection at development time rather than keeping per-instance runtime alternatives.

Decision: allowed expansion work consists of direct mask/XOR or mixed-radix stride arithmetic, modular shifts, bounded lookup into already evaluated local or factor amplitude/index data, and repetition over encoded contiguous spectator segments. Expansion may not traverse the original canonical term list, redo CAR/CCR/Weyl lowering, invoke general charge-sector DP or repeated rank/unrank, perform leakage analysis, or aggregate duplicate operator contributions. All cross-term duplicate amplitudes and exact cancellations represented by the compressed graph must already be resolved by its retained groups or evaluated tables.

Decision: a fixed eager plan retains only its selected compressed graph. Direct eager MVP executes that representation without first allocating CSR. An explicit `.csr()` request allocates and expands standard deterministic CSR on demand; a restricted facade may cache that explicitly requested CSR under its existing eager-cache lifecycle, but the fixed compact plan itself does not silently grow a second retained graph. D13's one-pass row-buffered builder remains the fallback for noncompressible eager graphs and the materialization path when compressed expansion cannot write exact CSR directly.

Decision: the shared abstraction is the complete-graph/decompression contract, not one universal payload layout. Domain-specific masks, strides, local tables, and segment descriptors remain available where required for performance. A symbolic operator schedule or factor-local lazy cache that still reevaluates terms and aggregates transitions on every MVP does not qualify as eager merely because it is compact.

Rationale: CSR contains redundant structure for regular transition families. A lossless compiled graph can retain final amplitudes and reconstruct destinations from compact masks, shifts, evaluated factor actions, and spectator repetition, reducing steady memory and often index traffic without weakening eager semantics. The qualification boundary prevents ordinary lazy operator evaluation from being relabeled as eager.

### D17 — Eager representation winners are fixed after development-time A/B

Status: `approved`; fixes the selection policy for D15–D16.

Decision: each coarse mathematical capability class has one production retained eager representation chosen during development by release-mode A/B against exact destination-major CSR. Production plan construction does not build or retain both compact and CSR candidates, does not choose per operator instance from a runtime density or memory heuristic, and does not expose a representation-selection option. Capability classes and their fixed winners are documented and observable through plan metadata.

Decision: a compressed eager candidate is eligible only when its measured retained bytes are strictly less than CSR and its representative steady `apply_into` median is no more than `10%` slower than CSR under matched thread count and semantics. Faster direct MVP and larger memory reductions are preferred. Construction time, first apply, CSR expansion time, peak RSS, and scaling are reported as supporting evidence even though retained memory and steady MVP are the primary gate.

Decision: after the A/B choice, delete the losing production executor and all compatibility-only branches for that capability class. Shared deterministic CSR expansion remains when required by the public `.csr()` target and is not a second eager execution strategy. If a future representative workload invalidates the fixed choice, reopen the design with new evidence rather than silently adding an adaptive branch.

Rationale: runtime representation selection would require parallel implementations, estimates, thresholds, metadata states, and regression coverage on every future change. A development-time decision captures the useful memory/performance trade-off while keeping one maintainable execution path per mathematical structure.

### D18 — Public plan metadata is minimal and orthogonal

Status: `approved`.

Decision: every public CPU-native MVP plan exposes one small read-only diagnostic surface with the following orthogonal fields: `storage` (`"lazy" | "eager"`), `execution_family` (`"pauli" | "cartesian" | "product" | "generic"`), `retained_format` (`"compiled_actions" | "compact_graph" | "csr"`), `source_term_count`, `schedule_count`, `estimated_retained_bytes`, `estimated_construction_peak_bytes`, and `transition_count`. `transition_count` is the exact final nonzero transition count for eager plans and `None` for lazy plans.

Decision: basis-ordering and operator-semantic metadata already required to interpret a plan remain available, but internal mask layouts, local table variants, tile sizes, Rayon thresholds, builder chunk sizes, and compression payload names are not public API. Benchmarks may use internal counters behind test or benchmark instrumentation without promoting them to public fields.

Decision: remove the current combined `strategy` string surface after callers and tests migrate to the orthogonal fields. Do not retain compatibility aliases or parallel metadata representations solely for old private or development-stage values.

Rationale: most users only need to apply a plan. The minimal fields are sufficient to verify lazy/eager semantics, prove that a factorable workload did not fall into Generic, identify compact graph versus CSR retention, and account for major resources without freezing replaceable hot-loop details into the API.

### D19 — Benchmark coverage is tiered and QuSpin runs in a separate Conda environment

Status: `approved`.

Decision: fast correctness coverage includes every nontrivial mixed subset of Fermion, Boson, Qubit, and Qudit domains: six pairwise, four three-domain, and one four-domain combination. These tests use small independently assembled dense references, assert numerical values and execution-family metadata, and remain suitable for routine pytest/CI. They are correctness and dispatch tests, not timing gates.

Decision: the routine local performance suite is organized by reusable capability rather than mechanically repeating all eleven domain subsets. It contains bounded Cartesian, Product, reversed factor-size, FermionMask, BosonLadder, WeylShift, compact-eager, CSR, and Generic-fallback cases sized to finish in minutes rather than hours. Performance remains local and informational; no wall-time CI gate or external benchmark service is introduced.

Decision: large seven-site lazy, six-site eager, representation-selection A/B, full thread/RSS matrices, and directly comparable QuSpin measurements live under committed manual/development benchmark tooling and are run only during performance development and Phase 10.5 acceptance. Their workload definitions and runners remain reproducible, but large cases are excluded from the default benchmark group. Development-time A/B results select one production implementation under D12 and D17; losing production executors are deleted rather than retained for recurring comparisons.

Decision: QuSpin runs in a dedicated pinned Conda environment separate from the TenCirPauli project environment. The two libraries are not imported into one Python process. A parent harness invokes the two environment-specific Python executables on the same machine and shares only a frozen workload manifest, deterministic starting-vector policy or seed, coefficient/basis conventions, thread environment, and flat result/measurement records. The repository commits the QuSpin environment specification and runner but no machine-specific environment path, environment contents, credentials, or ignored raw benchmark results.

Decision: cross-environment comparisons pin equivalent Python/NumPy/SciPy/QuSpin versions where relevant, declare BLAS/OpenMP/Rayon thread variables explicitly, include each library's public input conversion and language boundary in the matching measured scope, and validate dimensions, energies or mapped operator actions, and residuals before accepting timing data. Cases requiring QuSpin Python callbacks, `user_basis`, or nonequivalent Weyl semantics are reported as lacking a fair optimized QuSpin baseline.

Rationale: exhaustive small numerical coverage catches domain-composition errors cheaply, while capability-level routine benchmarks catch hot-kernel regressions without making normal development prohibitively slow. Environment isolation prevents QuSpin dependency constraints from contaminating the project runtime and makes cross-library measurements reproducible and auditable.

## Resolved discussion proposals

### D20 — Phase 10.5 first-round remediation closure (reopened)

Status: `reopened; partial implementation only`.

Decision: the reviewed implementation closes only the evidenced subset: authoritative native family selection, factor-component Product lowering with factor-level public-order prefix tables for disjoint compact constraints, construction-time net-transition fingerprints and local net-action descriptors for non-Fermion structured schedules, the interleaved Fermion-factor × Cartesian-Boson slice, direct native CSR read-back, and a documented trusted-handle validation boundary. Coupled charge constraints remain `generic` when no valid disjoint factor decomposition exists. The formal spec requirements for complete multiword FermionMask reuse, a fully specialized Cartesian executor for mixed/raw schedules, a destination-row/chunk CSR builder, complete capability-class A/B evidence, and the literal six-site eager gate remain open.

Evidence: `phase-10.5-acceptance-evidence.md`, `phase-10.5-trusted-handle-audit.md`, the dense Product differential suite, the richer eleven-subset dense matrix, the structured net-schedule unit test, the fixed-capacity sink release A/B rejection, and the previous release quality gate. These are partial-closure evidence, not evidence that all formal acceptance criteria are satisfied.

### D21 — Formal spec gap audit after first-round remediation

Status: `open; implementation in progress`.

Decision: until each item below has an implementation or a representative release A/B plus a correctness-preserving owner decision, Phase 10.5 remains open: (1) lower raw Fermion monomials into required-occupied, required-empty, flip, source-parity, and fixed-sign descriptors reused by all families; (2) replace the per-term Cartesian interpreter with flat net-transition descriptors and specialized destination-gather kernels for every supported family; (3) replace the global `BTreeMap` CSR sink with the formal destination-row buffered builder; (4) complete the eager capability-class A/B matrix; (5) either beat the literal six-site eager QuSpin gate or amend the owner-approved spec with evidence-backed acceptance wording. The Product public-order gap is implemented by factor-level prefix-completion rank tables in `FastProductMvpPlan`; focused charge differentials cover leading/interleaved spectators, independent factors, and `max_bytes=0` caller-owned apply. The eleven-subset numerical action matrix is covered by `test_all_mixed_domain_subsets_match_independent_dense_reference`, including sign-sensitive Fermion, finite-cutoff Boson, Pauli X/Y, Weyl phase/wrap, cross-domain terms, numerical values, and family/format assertions.

Evidence to date and known boundaries are recorded in `phase-10.5-acceptance-evidence.md` and `phase-10.5-trusted-handle-audit.md`; no item is closed by documentation wording alone.

### D22 — Owner-approved eager parity interpretation

Status: `resolved by owner approval; structural capability items remain open`.

Decision: the literal requirement that the six-site eager/explicit workflow beat QuSpin is amended for this acceptance round. A directly matched seven-sample single-thread median within 5% of QuSpin, after numerical preflight passes and with no material memory or scaling regression, is an accepted parity result. It closes the six-site performance gate as parity only and must not be reported as a TenCirPauli speedup. The latest measurements (`21.70 ms` TenCirPauli versus `21.16 ms` QuSpin, approximately `2.6%` slower) satisfy this rule. This decision does not close the independent implementation/evidence requirements for complete FermionMask lowering, Cartesian specialization, the destination-row/chunk CSR builder, or the eager capability-class A/B matrix.

Evidence: `phase-10.5-acceptance-evidence.md` records the seven-sample protocol, numerical preflight, retained-memory and peak-RSS measurements, and the parity calculation. Any later result outside the 5% band or showing a material regression reopens this performance decision with a superseding entry.

### D23 — Structural remediation progress after D22

Status: `implementation in progress; not an acceptance closure`.

The raw structured lowering now has a multiword FermionMask descriptor, lazy Product no longer retains its complete component graph, and the shared CSR sink now uses fixed-capacity temporary chunks with deterministic k-way merge. Structured eager generation traverses destination rows with bounded row-local aggregation. These changes address the corresponding implementation direction in D21 and pass release Rust/focused differential verification. Charge-sector producers still use source-major generation, supported Product classes still require the complete capability-class A/B matrix, and the legacy operation interpreter remains only for explicitly Generic or descriptor-ineligible schedules. No wording in this entry marks those remaining items closed.

### D24 — Eager retained-format winners by capability class

Status: `resolved by owner approval`.

Decision: the bounded release matrix is sufficient to select one retained eager format per capability class without inventing a second production graph. Pauli XOR retains the measured compact graph because it is strictly smaller than CSR and numerically identical. Cartesian, Qubit/Qudit/Boson Product, independent-factor Product, compact-Fermion Product, and Generic irregular cases retain direct native CSR: the graph-free lazy descriptor is not an eager candidate because it intentionally does not retain a complete graph, and no other compliant compact graph payload exists in the current semantic IR. The representation runner covers every class with two warmups, seven steady samples, construction/CSR timing, retained bytes, peak RSS, and numerical error versus the authoritative CSR. This closes the capability-class winner requirement by explicit owner selection and removes the need for a runtime compact/CSR switch; a future compact eager candidate requires a superseding owner decision with a real payload and A/B record.

Evidence: `benchmarks/manual/representation_ab.py` latest records: `pauli_xor` uses `compact_graph`; Cartesian/Product/Generic records use `csr`; all nine records have zero or machine-precision CSR error. The compact-Fermion Product record now reports `execution_family="product"` after the total-plus-spin-balance sector was added.

### D25 — Charge-sector CSR producer order

Status: `resolved by owner approval`.

Decision: the formal “destination-row” wording is amended to accept an equivalent deterministic bounded producer. Structured eager uses destination-row traversal. Charge-sector eager uses source-major sector unranking with one bounded per-source destination aggregate, then publishes through the same fixed-capacity chunk sink and deterministic row/column merge. This is the only path that preserves post-aggregation leakage detection for arbitrary charge terms without introducing a second inverse semantic implementation: a destination-row charge producer would need a separate inverse for every Fermion/Boson/Pauli/Qudit combination and would either re-evaluate the operator or enumerate candidate sources. The shared sink still guarantees no global row map, global COO, global sort, or retained temporary chunks.

Evidence: charge-sector exact-cancellation/leakage tests, direct CSR/COO equality, 160 focused Python regressions, 47 Rust tests, and the six-site eager construction/RSS record. A future inverse charge IR can supersede this decision, but adding a second producer without that IR is explicitly rejected as duplicate semantic lowering.

### D26 — Bounded charge Product lazy lowering (superseded)

Status: `superseded by D27`.

Decision: the charge Product lazy path initially used the immutable, grouped `ChargeTransitionTerm` list as a bounded semantic IR. That decision is retained only as the construction-time lowering input; D27 replaces its per-source interpretation with factor-local action descriptors. Unsupported coupled constraints remain explicitly `generic`.

Evidence: the historical graph-free retention and cancellation tests remain valid; the replacement evidence is recorded in D27.

### D27 — Product lazy factor-local action descriptors

Status: `implemented and release-verified`.

Decision: Product construction lowers each grouped charge term once into flat factor-rank transition tables, Cartesian spectator digit tables, and prevalidated Fermion parity steps. Lazy apply performs only source rank/unrank, descriptor table lookup, bounded destination aggregation, and post-aggregation sector ranking. It does not call `apply_charge_term()`, retain a Product transition graph, or re-check raw mode/index/domain semantics per source. Exact cancellation is removed before destination ranking, so leakage is reported only for a nonzero aggregate. Charge-sector CSR remains source-major as approved by D25.

Evidence: `FastProductMvpPlan` action descriptors in `crates/tencir-pauli-core/src/charge.rs`, independent full-dense Product differential, exact-cancellation/leakage regressions, retained-byte/`transition_count=None` checks, 47 Rust tests, the focused charge suite, and `benchmarks/manual/product_lazy_precompile_ab.py` release records.

### D28 — Second-round implementation closure for D20/D21

Status: `technically resolved and release-verified in the current worktree; immutable commit provenance pending repository commit`.

Decision: D28 supersedes the still-open implementation wording in D20 and D21 for the reviewed code paths. The width-generic `FermionMaskDescriptor` in `crates/tencir-pauli-core/src/fermion_mask.rs` is now the shared lowering and application substrate for Structured one-word and multiword actions, compact charge Product actions, Generic/eager compiled charge schedules, and the primary compact Fermion path. Structured consumers use one caller-owned word scratch per apply; the descriptor no longer exposes the former `u128` apply/flip methods, and the duplicate `WideFermionMaskDescriptor`, `FastFermionTerm`, `apply_fast_fermion_term`, and Structured word-index adapters are deleted.

Decision: pure binary structured inputs are proven against the actual private native variant. The packed Pauli plan is selected during native compilation or through the one native handle conversion used by the qubit-only facade, and the private `NativeMvpPlan.native_variant` getter reports `"pauli"`; the Python facade contains no family-string override. Non-Pauli Structured plans report the compiled variant from their native plan enum.

Decision: supported independent-factor Product plans retain factor-local transition descriptors, evaluate factor actions outside the Cartesian spectator loop, aggregate into scalar mixed-radix destination keys, and rank only after exact cancellation. Generic and charge eager paths consume construction-time `CompiledChargeSchedule` descriptors; their wide-layout vector accumulator is only the checked fallback when a scalar full-radix key cannot be represented. The obsolete global transition builder and its export are deleted.

Decision: Structured eager uses ordered destination-row append, charge eager uses the approved bounded source-major producer, and the shared chunk sink performs deterministic heap k-way merge with cross-chunk coalescing and exact-zero removal. The old linear chunk scan and dead `FastProductLayout` decode/destination helpers are deleted. The multi-chunk sink regression covers duplicate cancellation, empty rows, sorted columns, and repeat equality.

Evidence: release `cargo check`, `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `cargo test --workspace` pass with 50 Rust tests; `maturin develop --release --locked --skip-install` succeeds in the local environment; the full `.conda` Python run passes 416 tests with 57 existing deprecation warnings. `tests/test_native_mvp_resources.py` asserts the actual private native variant and numerical MVP output, while `crates/tencir-pauli-core/src/csr.rs` asserts deterministic multi-chunk CSR output. `rg` scans show no `compile_charge_transitions`, `apply_u128`, `flip_u128`, `FastFermionTerm`, `apply_fast_fermion_term`, `WideFermionMaskDescriptor`, or Python `mapped_binary` family override.

Scope note: coupled charge constraints remain intentionally Generic, D22 remains an owner-approved eager parity interpretation rather than a speedup claim, and the current working tree has not been committed by the agent. The next clean repository commit should record this technical closure as immutable provenance; it is not evidence of an additional code gap.

### D29 — Third-round Product, Cartesian, Generic, and CSR closure

Status: `technically resolved and release-verified in the current worktree; clean-commit provenance excluded by owner direction`.

Decision: the third-round implementation gaps are closed by the current native paths and evidence. `FastProductMvpPlan` compiles coefficient-free `ProductScheduleGroup` fingerprints, preserves canonical member order, evaluates factor actions outside the spectator traversal, applies contiguous suffix segments when the public layout permits it, and uses reusable scalar or fixed-width destination accumulators. Its counters distinguish factor-action evaluations, spectator-segment applications, destination-key allocations, and fallback invocations; the 16-mode U1 release diagnostic reports `(32760, 6552, 0, 0)` for the Product plan and zero numerical error against the Generic reference.

Decision: Structured compiled plans group by active axes and suffix stride, and supported Fermion schedules use the shared width-generic `FermionMaskDescriptor` with caller-owned scratch. The old raw Fermion operation interpreter is absent from the production path. Pure binary Structured plans report the packed native Pauli variant from the native plan. The measured pure-Fermion Product-versus-specialized A/B remains materially in favor of the specialized executor (`approximately 9.11 ms / 8.25 MiB` versus `0.317 ms / 105 KiB`), so that executor is retained explicitly; mixed Fermion×Boson sectors use the shared Product path. This is a measured convergence decision, not evidence that the general Product label covers every compact plan.

Decision: scalar and wide Generic/eager charge paths use construction-time `CompiledChargeSchedule` values and reusable accumulators. The wide accumulator stores destination digits in a preallocated fixed-width buffer, so destination-key heap allocation remains zero after construction. The CSR sink retains deterministic row-ordered append for Structured and heap k-way merge for source-major charge production, with cross-run coalescing, exact-zero removal, and exposed diagnostic phase statistics. The chunk sweep records 1, 8, 63, and 374 actual chunks for a 1,120-transition workload; the fresh six-site profile records 1,375 chunks, 22,527,980 sorted/merged entries, 6.577 s generation, 0.335 s sorting, 0.811 s merge, 11.8 us final-array allocation timing, and approximately 1.45 GB peak RSS.

Evidence: `benchmarks/manual/product_u1_16_ab.py`, `benchmarks/manual/csr_chunk_scaling.py`, `examples/research/holstein/run_tencirpauli.py`, `tests/test_charge.py`, `tests/test_structured_algebra.py`, `crates/tencir-pauli-core/src/charge.rs`, `structured.rs`, `fermion_mask.rs`, and `csr.rs`; release verification passes with 52 Rust tests, 416 Python tests, 8 doctests, 353 bounded Python benchmark cases, Clippy, Black, Ruff, strict mypy, and the complete smoke workflow. Coupled constraints remain intentionally Generic, D22 remains an owner-approved eager parity interpretation rather than a speedup claim, and repository commit creation remains a separate owner operation.

### D30 — Fourth-round selector, Product-index, allocation, and evidence remediation

Status: `technically resolved in the current worktree; clean-commit provenance excluded by owner direction`.

Decision: D29 is historical third-round evidence and does not authorize Product to displace a faster capability-class executor. The selector now retains Qubit, Qudit, Boson, and balanced spinful Fermion winners whenever they are eligible, including balanced Fermion factors with Cartesian Boson spectators. General Product remains constructible for its algebraic envelope and is measured through a forced-variant harness; it becomes a public default for a class only after a reverse-order release A/B shows numerical equality and no material regression.

Decision: Product public-order indexing uses factor-local prefix-completion offsets and counts, not `D`-entry public/component permutations or the global sector ranker during apply. The plan owns only factor descriptors, bounded per-factor metadata, spectator dimensions, and flat local transitions. The destination-gather apply counters record actual factor/group/aggregate work; forbidden Product fallback, ranker, and owned destination-key events remain zero only alongside positive intended-path activity.

Decision: allocator-backed tests are required evidence for the shared FermionMask and Generic accumulators. One-word and three-word mask loops allocate zero; compiled raw-Fermion apply has a constant one scratch allocation per public call rather than per source or term; repeated scalar/wide accumulator cycles allocate zero after construction. A batched native phase diagnostic, kept outside normal MVP execution, provides a reproducible raw-Fermion cost breakdown without per-contribution clock overhead.

Decision: the manual evidence set uses actual native variants, complete commands, explicit thread variables, workload manifest versions, and commit-plus-dirty fingerprints. The CSR chunk runner launches each capacity in a fresh process and verifies both CSR arrays and MVP action against a one-chunk reference; its final-array timing is named allocation time, not copy time.

Evidence: `tests/test_execution_invariants.py`; `crates/tencir-pauli-core/src/charge.rs`, `charge_sector.rs`, `fermion_mask.rs`, and `structured.rs`; `benchmarks/manual/product_variant_matrix.py`, `fermion_mask_phase_profile.py`, `generic_wide_profile.py`, `product_thread_ab.py`, and `csr_chunk_scaling.py`. Current release records show the five-site public Holstein route selecting `fermion` at 5.00/2.91/1.69 ms for 1/2/4 threads and the seven-site public route selecting `fermion` at 1.534 s median, versus the matched isolated QuSpin matrix-free 1.730 s median. The acceptance evidence records the numerical and resource fields and the current regression matrix.

### P01 — Exact lazy representation for factorable workloads

Status: `resolved by D06`.

Question: should every Cartesian, Product, and Block lazy plan use one factorized compiled representation with reusable factor-action tables, eliminating the current generic term-direct lazy implementation from all structurally factorable workloads?

Current measured seven-site Holstein facts:

- Full selected dimension: `20,070,400`.
- Canonical terms: `66`.
- Compact fermion factor dimension: `1,225`.
- Boson Cartesian factor dimension: `16,384`, composed of seven local four-level axes.
- Unique non-identity fermion actions: `45`.
- Unique non-identity boson actions: `21`.
- Current direct-lazy retained estimate: `59,080` bytes.
- Proposed simple factor-table retained estimate using one `u64` destination plus one complex128 amplitude per factor source/action: approximately `1.4 MiB`.
- Complete eager CSR retained estimate from approximately `299,622,260` final entries: approximately `6.85 GiB`.
- One complex128 state vector: `306.25 MiB`; caller-owned input plus output: `612.5 MiB`.

Earlier proposal considered: use one maintainable `FactorActionTable` representation for the required factorable workloads, with no Table/Sparse/Analytic variant family in the first implementation. If required factor tables exceed `max_bytes`, fail instead of falling back to the old generic direct-lazy loop. Generic remains only for constraints or actions that are structurally not factorable.

The earlier proposal would have required separate decisions on table fields, index widths, Cartesian-table behavior, action interning, cache thresholds, and maximum retained table budget.

Resolution: the table-centric proposal is rejected as the general architecture. D06 instead uses domain-specialized direct arithmetic and factor-ordered loop composition. Only small domain metadata is assumed; any larger retained table requires a separate profile-backed owner decision.

### P02 — Public exposure of the factorized lazy strategy

Status: `resolved by D06`.

Question: should the public API retain only `storage="lazy" | "eager"`, with factorized execution exposed through read-only strategy metadata, or should factorized storage become a third public choice?

Dependency: decide P01 and its exact memory/performance contract first.

Resolution: factor/product execution remains an internal lazy strategy. The public storage values remain lazy and eager.

### P03 — Eager CSR construction algorithm

Status: `resolved by D13`.

Question: should the first implementation contain only the low-memory two-pass direct-CSR builder, or also a buffered one-emission builder that trades an additional `O(D + E)` construction buffer for avoiding a second transition traversal?

Resolution: D13 selects one production destination-row buffered builder. It performs one transition traversal, retains temporary finalized row entries until exact CSR allocation, then copies once and releases the temporary storage. The low-memory two-pass builder and automatic builder selection are not retained.

### P04 — Bring diagonal qudit charges into Phase 10.5

Status: `resolved by D07`; D07 supersedes D04.

Question: should Phase 10.5 extend `AdditiveCharge` with exact integer charge values for every local qudit basis level and implement qudit-aware conservation, `ChargeSector` rank/unrank, Product/Block classification, restricted lazy execution, and eager CSR?

Semantic issue requiring owner approval: the current Weyl `X` is a cyclic shift, not a physical non-cyclic spin raising/lowering operator. For a diagonal charge table `q_m`, the transition `m -> (m+a) mod d` has charge difference `q_((m+a) mod d) - q_m`, which may depend on `m` and changes at the wrap boundary. Therefore the implementation must use exact per-level charge tables and must not assume that one Weyl `X^a` has one constant `S_z` delta.

Rejected proposal: accepting `qudits={site: (q_0, q_1, ..., q_(d-1))}` as a general per-level charge table. Resolution: D07 exposes only a site weight multiplying the canonical integer-normalized spin-z spectrum. Convenience factories and symbolic `as_operator()` materialization remain separate decisions.

## Reconciliation note

All approved decisions D01–D19 were reconciled into `phase-10.5-spec.md` on 2026-08-07. The rewritten specification removes optimized Block scope, the universal factor-table proposal, dual CSR builders, runtime compact/CSR selection, mandatory retained CSR, deferred canonical Qudit charge, and other superseded draft text. Entries P01–P04 remain only as historical records of resolved questions.
