# Phase 14.5 qLDPC Static Stabilizer-Code Research Example Specification

Status: specified, not implemented. This phase adds one manual research example comparing TenCirPauli with qLDPC on their shared static qubit stabilizer-code analysis tasks. qLDPC is the only external performance comparator for this phase; QuantumClifford.jl, Stim, Qiskit-QEC, qecsim, decoder packages, and circuit simulators are not additional benchmark anchors.

## 1. Purpose and binding decisions

Phase 14.5 answers one narrow research question: for the static qubit stabilizer-code operations that TenCirPauli and qLDPC both support, what are their end-to-end runtime, throughput, retained/output memory, and scaling on the same code matrices and error batches?

The deliverable is a manual Research Example under `examples/research/`, not a new public API, runtime dependency, CI timing gate, or production fallback. It must compare public or documented package-level workflows, preserve exact mathematical semantics, separate setup/first/steady/materialization boundaries, and report any unmatched capability instead of filling the gap with a large custom baseline implementation.

The following decisions are binding:

1. The comparator is qLDPC, pinned initially to `qldpc==0.3.3`. The final report records the exact qLDPC, Python, NumPy, SciPy, `galois`, Stim, TenCirPauli, Rust, and native-extension versions and paths actually used.

2. QuantumClifford.jl is not a secondary performance anchor. It is a high-performance stabilizer/tableau kernel library, but its `MixedDestabilizer` needs a custom QEC workflow adapter and does not provide qLDPC's code-centered package surface. Adding it would weaken the one-question product comparison and is outside this phase.

3. Stim is not a static-code baseline. It is a circuit/sampling and low-level stabilizer engine; completing underconstrained stabilizers into a pure tableau is not the same operation as retaining a `[[n,k]]` code space with original-check syndrome coordinates.

4. qLDPC distance, distance bounds, logical-operator weight reduction, decoder execution, logical-error-rate estimation, circuit generation, and external-tool calls are excluded from all matched timing rows because Phase 14 does not implement those operations.

5. The main comparison uses only qubit stabilizer codes over GF(2). qLDPC qudit, subsystem/gauge, GAP, GUAVA, MAGMA, QDistRnd, Stim/Sinter circuit, and decoder capabilities are documented as broader package scope but are not benchmarked against unsupported TenCirPauli functionality.

6. qLDPC's canonical logical representatives need not equal TenCirPauli's canonical representatives. Correctness compares stabilizer/normalizer quotient semantics and an explicit logical-basis transformation, not raw logical bit arrays.

7. No performance result is publishable while a known correctness defect affects the measured operation. The Phase 14 lexicographic error-set defect does not block construction/syndrome/logical measurements, but classification/correction results are published only after the relevant Phase 14 acceptance remediation and retained correctness matrix pass.

8. Machine-specific JSON, logs, profiles, environment paths, and benchmark outputs remain local and untracked. The tracked repository contains workload definitions, runners, comparison logic, correctness gates, methodology, and a concise result table once an accepted run exists.

## 2. What qLDPC adds beyond the shared static-analysis layer

qLDPC is a broader QEC research workbench, but it is not substantially richer than Phase 14 within the narrow operation set of arbitrary qubit stabilizer-code static algebra. On that shared layer, both libraries construct codes from check matrices, compute rank and encoded dimension, construct physical logical X/Z representatives, evaluate syndromes, and can classify residuals from stabilizer/logical commutation data.

The main additional core analysis in qLDPC is code-distance work: exact distance for supported sizes or known families, randomized or external upper bounds, and decoder-assisted reduction of logical representatives. These are genuine algorithms and are outside Phase 14 by explicit design.

Other qLDPC capabilities fall into three categories:

### 2.1 Additional code semantics and code construction

- Built-in classical and quantum code families, including repetition, Hamming, Reed–Muller, Steane, surface/toric, hypergraph-product, bivariate-bicycle, lifted-product, Tanner, concatenated, and related code constructors.
- Code composition and transformation such as stacking, concatenation, deformation, chain-complex construction, and selected transversal/automorphism analysis.
- Qudit stabilizer codes over finite fields and subsystem/gauge-code semantics.

These are substantive capabilities, but most produce or transform a check matrix before the static algebra measured in this phase. Workload generation is therefore outside the timed matched analysis unless a row is explicitly labeled as qLDPC-only code-family construction.

### 2.2 Distance, decoding, and logical-performance workflows

- Exact or bounded code-distance computation and minimum-weight logical searches.
- Decoder selection and integration, including BP-OSD, BP-LSD, belief-find, PyMatching, lookup, relay-BP, sliding-window, sequential-window, and custom decoder interfaces.
- Code-capacity logical-error-rate estimation, decoder failure/post-selection accounting, and logical-operator weight reduction through decoder calls.

Distance/search logic is a genuine qLDPC capability. Many decoder entries are adapters over external packages, while qLDPC owns the common workflow, data conversion, sampling, and result aggregation. None is evidence that qLDPC performs the matched static algebra faster, so none enters the Phase 14.5 headline timings.

### 2.3 Circuit and external-tool orchestration

- Stim/Sinter memory experiments, detector/observable arrays, state-preparation diagnostics, encoding circuits, and Pauli noise-model construction.
- GAP/GUAVA, QDistRnd, MAGMA, GroupNames, and related external algebra or distance integrations.
- Tanner graphs, chain complexes, abstract groups/rings/modules, geometry-aware code objects, and visualization/helper workflows.

These capabilities make qLDPC a broader QEC environment. A substantial part is interface, adapter, and orchestration code around specialized external engines; other parts, such as chain-complex/code-family construction and circuit synthesis, are implemented qLDPC functionality. They remain outside the static qubit code-analysis comparison.

### 2.4 Concrete qLDPC-owned capabilities absent from TenCirPauli

The following capabilities are not merely adapters. qLDPC owns the mathematical construction, transformation, search, scheduling, or aggregation logic even when a result is later executed by Stim or a decoder backend.

| Capability | Concrete qLDPC surface | qLDPC-owned work | TenCirPauli status |
| --- | --- | --- | --- |
| Exact and bounded code distance | `get_distance()`, `get_distance_exact()`, `get_distance_bound()`, CSS X/Z distance paths, known-distance code families | Enumerates or searches nontrivial logical representatives, maintains exact/bound semantics and witnesses/known formulas, and dispatches optional external bounds where requested. | No distance API or search by Phase 14 design. |
| Large built-in code-family catalog | `FiveQubitCode`, `SteaneCode`, `SurfaceCode`, `ToricCode`, `HGPCode`, `TBCode`, `BBCode`, `QCCode`, `LPCode`, `SLPCode`, `SHPCode`, `SHYPSCode`, `QTCode`, and others | Constructs parity-check matrices from classical codes, Kronecker products, group rings, lifts, geometry, and chain complexes; freezes family-specific parameters and conventions. | Accepts caller-supplied general/CSS checks but does not generate named QEC code families. |
| Qudit and subsystem/gauge codes | `QuditCode`, `CSSCode(..., is_subsystem_code=True)`, gauge-aware standard form and logical construction | Performs finite-field standard-form algebra, separates stabilizer/gauge/logical sectors, and constructs paired logical operators for subsystem qudit codes. | Supports qubit stabilizer codes only; subsystem/gauge and generalized-Weyl QEC are explicit non-goals. |
| Code composition and homological products | `stack()`, `concatenate()`, classical dual/tensor operations, HGP/lifted/subsystem products, chain-complex tensor products | Expands logical operators through inner/outer codes, builds combined checks, preserves field/order conventions, and constructs product-code matrices. | No code-composition, concatenation, or chain-complex API. |
| Code transformations and transversal analysis | `conjugated()`, `deformed()`, `get_transversal_ops()`, `get_transversal_circuit()` and automorphism-based helpers | Transforms checks/logicals under Clifford/permutation actions and searches selected code automorphisms or transversal logical actions. Some searches are intentionally expensive. | No physical-Clifford logical-action, deformation, automorphism, or transversal-gate analysis. |
| Encoding and syndrome-circuit construction | `get_encoding_circuit()`, memory/state-preparation helpers, `get_syndrome_subgraphs()`, syndrome-measurement scheduling | Synthesizes encoding or measurement structure, orders Tanner-graph interactions, applies hook-error-aware schedules, and emits annotated circuits. Stim executes or analyzes the resulting circuit, but qLDPC owns the construction. | No QEC circuit IR, encoder synthesis, detector timeline, reset/measurement semantics, or fault model. |
| Logical-error-rate workflow | `get_logical_error_rate_func()` and code-capacity helpers | Allocates samples by error weight, generates physical errors, computes syndromes/logical failures, aggregates failure/discard statistics, exposes uncertainty and truncation bounds, and calls the selected decoder through a common interface. | `analyze_corrections()` verifies supplied pairs only; it does not sample noise, run a decoder, or estimate logical rates. |
| Tanner, chain-complex, and algebraic construction infrastructure | Tanner graph conversion, `ChainComplex`, Cayley-complex/group/ring/module helpers, polynomial/group-algebra lifts | Implements graph/matrix conversion, boundary operators, tensor products, algebraic lifts, incidence structure, and geometry used by advanced code families. | Exposes check weights and qubit degrees only; it is not a public graph, chain-complex, or abstract-algebra package. |

The adapter-only or primarily external-engine list is separate: BP-OSD/BP-LSD/belief-find and related decoders come from decoder packages; PyMatching supplies MWPM; Stim/Sinter supplies circuit simulation, detector-error-model execution, and sampling; GAP/GUAVA, QDistRnd, MAGMA, and GroupNames supply external algebra, code catalogs, or distance routines. qLDPC still owns validation, conversion, configuration, and result orchestration around these tools, but Phase 14.5 does not describe their underlying algorithms as qLDPC-native performance.

For the specific Phase 14 static-analysis intersection, the most important qLDPC-owned capability missing from TenCirPauli remains distance/search. The other qLDPC-owned items above broaden the input-construction, algebraic domain, code transformation, or circuit/workflow layers rather than replacing the same syndrome/classification kernels measured by this phase.

### 2.5 qLDPC implementation size

The qLDPC 0.3.3 source distribution is not a small wrapper package. A local source survey found 64 Python files under `src/qldpc`. Excluding `*_test.py` and `conftest.py`, 35 production files contain 20,210 physical lines; `cloc` classifies those as 11,155 code lines, 5,457 comment lines, and 3,598 blank lines. The 29 test or test-support files add 4,566 `cloc` code lines. These counts are version-specific descriptive evidence, not a performance metric.

The breadth is concentrated in substantial infrastructure rather than one compact stabilizer algorithm. The largest production modules cover general code objects and finite-field standard forms, quantum-code families, circuit/noise workflows, groups/rings/modules, custom decoder orchestration, transversal analysis, and external backends. qLDPC can therefore expose much of a QEC workflow because it combines roughly eleven thousand logical production lines with specialized external engines; it should not be interpreted as obtaining the same breadth from a few thin adapters.

### 2.6 Small, high-value follow-up candidates for TenCirPauli

The following prioritization is a product recommendation, not additional Phase 14.5 scope. Phase 14.5 remains a research example and must not add these APIs while constructing the comparison.

| Priority | Candidate | Common user value | Why the scope is manageable here | Required implementation boundary |
| --- | --- | --- | --- | --- |
| A1 | Focused common-code constructors | Users repeatedly need standard fixtures and recognizable research workloads without manually transcribing checks. | Five-qubit, Steane, repetition, rotated surface, toric, and hypergraph-product checks can all terminate in the existing `StabilizerCode.from_symplectic()` or `from_css()` path; they need no new algebra object. | Add a small qubit-only constructor module, fixed qubit/check-order conventions, known-parameter fixtures, and matrix-level differential tests. Do not copy qLDPC's classical-code catalog, Tanner/chain-complex hierarchy, qudit fields, or geometry framework. Start with five-qubit, Steane, and repetition; then add rotated surface/toric and matrix-form HGP. |
| A2 | Bounded distance with a physical witness | `[[n,k,d]]`, CSS `d_X/d_Z`, and a concrete minimum-weight or best-known logical are among the first questions users ask about a code. | Existing packed checks, stabilizer membership, logical coordinates, popcount, and batch classification provide most verification primitives. A useful first implementation can be exact only when a bounded search completes instead of claiming scalable general QLDPC distance. | Return lower bound, upper bound, completion status, physical Pauli witness, logical coordinates, and search provenance. Implement native weight-ordered bounded search, with CSS X/Z specialization first and a small general-code path second. Never infer distance from the lightest canonical logical row, and never return an incomplete upper bound as exact. |
| B1 | Physical-Clifford code transformation and logical-action verification | Users need to verify that a proposed physical Clifford preserves a code and determine the induced logical Clifford, especially for transversal gates and compiled gadgets. | The Rust core already has parameter-free Clifford tape operations, packed Pauli conjugation, canonical stabilizer span data, and logical coordinates. This avoids adding Stim or a new circuit language. | Accept an existing parameter-free `GateTape`, conjugate checks and logicals in one native operation, verify the signed stabilizer group, and return either a `2k x 2k` binary symplectic logical map or a concrete failure witness. Support explicit transformation/verification only; do not add automorphism-group or transversal-gate search. |
| B2 | Direct sum and explicit qubit permutation | Combining independent code blocks and normalizing qubit order are useful for fixtures, concatenated experiments, and benchmark construction. | These are deterministic block placement and bit-permutation operations over existing packed rows. | Add only `direct_sum`/`stack` and explicit permutation with preserved signs and original-check ordering. Defer general concatenation until logical-to-physical mapping conventions have a separate specification. |

The recommended order is constructors first, then bounded distance, then physical-Clifford logical action. Constructors deliver immediate usability at low risk; distance is the most important missing analysis and the strongest candidate for a later qLDPC comparison; logical-action verification is valuable and well matched to TenCirPauli, but it is less universal than distance. Direct sum and permutation are acceptable as small utilities when a concrete caller needs them, not as a reason to create a general code-composition framework.

The following qLDPC areas should not be pulled into the near-term TenCirPauli scope: built-in decoders, subsystem/qudit codes, syndrome-extraction scheduling, detector/reset/measurement semantics, logical-error Monte Carlo orchestration, automorphism/transversal search, chain complexes, group/ring/module algebra, or a large named-code catalog. Each requires a new semantic layer, a specialist external engine, or both; none is a small extension of the completed static qubit-code object. Encoder verification can follow physical-Clifford logical action, but encoder synthesis should remain separate because source/ancilla conventions and circuit correctness make it materially larger.

## 3. Capability intersection and exclusions

| Capability | TenCirPauli Phase 14 | qLDPC | Phase 14.5 treatment |
| --- | --- | --- | --- |
| General qubit symplectic checks | `from_symplectic()` | `QuditCode` over GF(2) | Matched construction and metadata row. |
| CSS X/Z checks | `from_css()` | `CSSCode` | Matched construction and metadata row. |
| Rank and encoded dimension | `rank`, `nlogical` | `rank`, `dimension` | Exact equality required. |
| Physical logical X/Z basis | `logical_operator_arrays()` | `get_logical_ops()` | Matched end-to-end construction/materialization; compare quotient basis after transformation. |
| Original-order batch syndrome | `syndrome_many()` | GF(2) symplectic product over retained check matrix | Thin qLDPC adapter permitted and included in timing. |
| Batch error classification | `classify_many()` | Syndrome plus logical commutation from `matrix` and `get_logical_ops()` | Thin adapter permitted; exact kind and transformed coordinates required. |
| Correction residual analysis | `analyze_corrections()` | XOR/product plus the same syndrome/logical computation | Thin adapter permitted and included in timing. |
| Signed eigenvalue checks | Supported, including `-I` consistency | Check-matrix code objects are primarily phase-free | Correctness documented; excluded from matched timing inputs, which use `+1` checks. |
| Redundant-check relations | Original rows and relation basis exposed | Redundant matrices may be canonicalized, but no matching public relation-basis result | Redundant inputs may be timed; relation read-back is TenCirPauli-only. |
| Error-set correctability witness | Native grouped analysis with deterministic pair | No matching package-level API | Excluded from headline performance; no large custom qLDPC baseline. |
| Structural weights/degrees | Native original-check diagnostics | Weight/Tanner graph helpers | Descriptive only; not a headline row because graph construction is a different boundary. |
| Code distance and bounds | Explicitly out of scope | Supported | qLDPC-only capability, never included in matched timing. |
| Decoder/circuit/subsystem/qudit workflows | Explicitly out of scope | Supported directly or through integrations | Capability discussion only. |

The qLDPC adapter is allowed to normalize shared I/X/Y/Z code arrays, evaluate the documented symplectic products, map logical coordinates between bases, and build the same small result masks used by the comparison. It must not implement a new Gaussian-elimination, canonicalization, distance, decoding, or error-set engine and then attribute that implementation's performance to qLDPC.

## 4. Research-example layout

The tracked deliverable is:

```text
examples/research/stabilizer_code_qldpc/
├── README.md
├── compare.py
├── requirements.txt
├── run_qldpc.py
├── run_tencirpauli.py
└── workloads.py
```

`requirements.txt` pins only research-side dependencies and must not modify the package's runtime requirements. The initial external pin is `qldpc==0.3.3`; transitive dependencies remain resolved by the environment, and every resolved version is printed by the runner.

`workloads.py` is the single owner of deterministic check matrices, error batches, correction batches, expected metadata, and content hashes. Workload generation is performed before timing. Both runners receive or reconstruct identical arrays and print their hashes before correctness or performance execution.

`run_tencirpauli.py` uses only the public `StabilizerCode` API for headline rows and records `tencirpauli.__file__` plus `tencirpauli._native.__file__`. Private native endpoints may be used only in a separately labeled diagnostic that cannot replace the public result.

`run_qldpc.py` uses documented qLDPC code objects, matrices, properties, and `get_logical_ops()`. The thin syndrome/classification/correction adapter remains in this file so readers can see exactly which work belongs to qLDPC data structures and which work is comparison glue.

`compare.py` runs each implementation in a fresh subprocess, validates structured `RESULT_JSON` output, compares correctness fields, and prints one matched result table. It accepts explicit Python executable arguments so TenCirPauli and qLDPC may live in isolated environments without storing machine-specific paths in the repository.

## 5. Workload matrix

The first version remains small enough to audit but covers low-rate, finite-rate, CSS, non-CSS, sparse-check, redundant-check, and 1,024-qubit-scale behavior.

### 5.1 CSS repetition family

Use Z-check repetition matrices with adjacent weight-two rows at `n = 64, 256, 1024`. These encode one logical qubit and isolate sparse CSS construction, original-order syndrome throughput, and the large-rank/small-`k` logical path.

### 5.2 Hypergraph-product family generated through qLDPC

Use `HGPCode(RepetitionCode(L), RepetitionCode(L))` or the exact equivalent matrices for `L = 8, 16, 24`, giving block lengths `n = L**2 + (L-1)**2 = 113, 481, 1105`. qLDPC generates the matrices outside the matched analysis timer; the same extracted `H_X/H_Z` arrays are then supplied to fresh qLDPC and TenCirPauli code objects.

This row tests a recognizable QLDPC-style sparse CSS workload while preventing qLDPC's code-family constructor from being confused with static check analysis. An optional separately labeled qLDPC code-family setup time may be reported but has no TenCirPauli speed ratio.

### 5.3 General non-CSS finite-rate family

Build deterministic direct sums of the five-qubit `[[5,1,3]]` checks with 16, 64, and 256 blocks, giving `n = 80, 320, 1280` and rate `k/n = 1/5`. Apply a fixed qubit permutation across blocks so the input is not accidentally optimized by contiguous block order while remaining exactly reproducible.

This row exercises general symplectic checks, non-CSS logical construction, and `k=O(n)` logical-coordinate work. It is the required counterweight to repetition/HGP cases with `k=1`.

### 5.4 Redundant-check variants

For the medium repetition and finite-rate cases, append deterministic nonidentity XOR combinations equal to 10% of the independent row count. Keep all signs `+1`. Both libraries receive the same ordered complete matrix. Metadata compares independent rank/dimension; syndrome compares the complete original rows. If qLDPC canonicalizes away dependent rows internally, the qLDPC syndrome adapter must retain the original input matrix separately for this row and report that boundary explicitly.

### 5.5 Error and correction batches

For every code, generate deterministic batches of sizes `1`, `256`, and `4096` where memory permits. Include three distributions:

- local weight-one and weight-two errors, dominated by nonzero syndromes;
- zero-syndrome stabilizer/logical mixtures constructed from each implementation-independent reference basis;
- decoder-like error/correction pairs containing successful stabilizer residuals, logical residuals, and residual nonzero syndromes.

The seed, batch composition counts, input bytes, and content hashes are reported. Random generation is never inside a timed closure.

## 6. Correctness contract

No timing is accepted until both runners pass the same correctness package.

1. Input matrix hashes, qubit count, check count, rank, encoded dimension, and redundant-row count agree exactly.

2. Every reported logical row commutes with every check; paired logical X/Z rows have the canonical symplectic form within each library; the combined stabilizer-plus-logical spans agree between libraries.

3. Compute the invertible `2k × 2k` GF(2) symplectic change-of-basis matrix between qLDPC and TenCirPauli logical representatives. Use it to compare logical coordinates. Raw logical arrays are not expected to match.

4. Original-order unpacked syndromes agree bitwise for every error. Packed TenCirPauli syndromes round-trip to the same bits.

5. Classification kinds agree for every row. Zero-syndrome logical coordinates agree after the explicit basis transformation; syndrome-visible rows expose no fabricated coordinates.

6. Correction residual kinds, syndromes, zero-syndrome indices, and transformed coordinates agree with explicit test-side XOR/product results.

7. Small repetition, five-qubit, and Steane fixtures additionally agree with fixed known syndrome and logical-action vectors independent of either package.

8. Distance, decoder success, code-family metadata not derivable from the supplied checks, and circuit results are not correctness oracles for this comparison.

If a mismatch is found, the study stops before timing that operation. The README records the smallest reproducer and identifies whether the issue is TenCirPauli, qLDPC, adapter convention, or unresolved; it does not discard the row or silently adjust qubit/check order.

## 7. Timing and memory contract

Every runner separates the following boundaries:

1. Import/startup time, reported but excluded from operation speed ratios.

2. Dependency/JIT warmup, including the first `galois` finite-field compilation. It is reported separately and never hidden inside a steady median.

3. Fresh construction to forced metadata: create a new code object and force check count, rank, and encoded dimension.

4. Fresh construction to materialized logicals: create a new code object, force metadata, compute logical X/Z, and materialize the complete numeric result. This is the primary matched construction row because qLDPC logicals are lazy/cached while the current TenCirPauli implementation constructs them eagerly.

5. First and steady batch syndrome, classification, and correction calls on a preconstructed code. Output arrays are fully consumed through a deterministic checksum inside the measured package boundary so lazy work cannot escape timing.

6. Peak resident memory in a fresh process for one workload/operation, plus exact input/output payload bytes and package-reported retained estimates where available.

The primary comparison fixes one execution thread using `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and the applicable BLAS/thread controls. An optional default-thread result may be reported separately only when both runner thread counts are printed. A one-thread qLDPC result is never compared as the denominator of a default-multithread TenCirPauli headline ratio.

Construction measurements use fresh objects for every round so cached qLDPC `rank`, `canonicalized`, or `_logical_ops` results cannot become a false steady construction speed. Repeated logical read-back from the same cached object is a separately labeled materialization/cache row and not a substitute for fresh construction.

Use release-built TenCirPauli from the source-tree native extension. Use sufficient warmups and rounds for a stable median while capping large cases to a documented total duration. Report all round counts, median, minimum, maximum, coefficient of variation, and whether garbage collection was enabled or explicitly controlled. Do not hand-select a favorable minimum as the headline value.

Input normalization and adapter conversion are part of public end-to-end syndrome/classification/correction timings. A separately labeled kernel-ready diagnostic may isolate qLDPC GF(2) multiplication and TenCirPauli native execution, but private/native results cannot replace the public comparison.

## 8. Result interpretation

The accepted result may support claims such as “TenCirPauli is X times faster than qLDPC 0.3.3 for this matched static construction/logical/syndrome workload on this machine.” It must not claim to be faster than Stim, QuantumClifford.jl, all QEC software, qLDPC distance search, qLDPC decoders, or qLDPC circuit workflows.

The README must state that qLDPC is a broader package while the timed intersection is narrow. A TenCirPauli speed advantage would establish the value of its Rust packed-data implementation for shared static code algebra; it would not establish broader QEC feature superiority. Conversely, a qLDPC advantage must be reported directly and profiled rather than hidden by changing the workload or timing boundary.

If qLDPC cannot complete a required large row within the local 16 GiB or a documented practical time cap, preserve the largest common completed row, report the qLDPC scaling boundary, and keep the TenCirPauli-only larger row without a fabricated ratio. Do not silently resize only the slower implementation.

## 9. Deliverables and documentation

Acceptance requires:

1. The five tracked files under `examples/research/stabilizer_code_qldpc/`, formatted and linted by the normal project tooling.

2. A README explaining qLDPC's broader capability categories, the exact shared operation boundary, versions, environment, workload matrices, qubit/symplectic conventions, logical-basis transformation, correctness results, timing boundaries, memory method, and limitations.

3. One compact result table covering fresh construction-to-metadata, fresh construction-to-logicals, syndrome, classification, and correction analysis for the accepted common rows.

4. Structured `RESULT_JSON` from both runners with schema version, implementation, versions/paths, thread settings, workload hash, metadata, correctness checksums, timing statistics, payload bytes, and peak-memory observation.

5. An entry in `examples/research/README.md`. After an accepted run exists, add only the durable methodology and representative result to the appropriate performance documentation; keep machine-specific raw outputs untracked.

6. No changes to TenCirPauli runtime dependencies, public APIs, production fallback behavior, or CI wall-time gates.

## 10. Implementation order and acceptance

1. Freeze the shared workload schema and small correctness fixtures.

2. Implement the qLDPC runner and keep its syndrome/classification/correction adapter small, visible, and independently tested.

3. Implement the TenCirPauli runner through public APIs and record exact source-tree module/native paths.

4. Implement cross-basis logical-coordinate comparison and stop-on-mismatch correctness mode.

5. Scaffold timing boundaries and record an exploratory pre-remediation baseline if useful for Phase 14 profiling. Label it internal and do not publish a performance conclusion.

6. Complete the relevant Phase 14 correctness, packed-path, construction, and resource remediation before the final comparison record.

7. Run the one-thread accepted matrix, triage scaling/noise, and run an optional matched default-thread matrix only if it adds useful information.

8. Write the final Research Example README and update the research index and durable performance summary.

Phase 14.5 is complete when the tracked example is reproducible from documented environments, correctness passes before timing, qLDPC is the sole external comparator, the result table covers the shared static-analysis boundary without distance/decoder/circuit contamination, raw machine outputs remain untracked, and every performance statement is scoped to the exact recorded workload and version.
