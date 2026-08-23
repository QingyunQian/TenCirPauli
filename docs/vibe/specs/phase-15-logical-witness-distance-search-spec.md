# Phase 15 Logical-Witness Search and Distance-Upper-Bound Specification

Status: implemented; performance remediation evidence recorded 2026-08-23. This specification defines a static qubit-stabilizer search that returns a validated low-weight nontrivial logical Pauli and therefore an upper bound on code distance. It does not define an exact or exhaustive distance solver, a certified lower-bound solver, a decoder, a noise model, a syndrome-extraction circuit, or a detector timeline.

## 1. Purpose and scope

Phase 15 adds an anytime search over an existing immutable `StabilizerCode`. The result is always a concrete physical Pauli witness `P` with `P in N(S) \ S`; if its qubit weight is `w`, the result certifies only `d <= w`. The implementation initializes the incumbent from deterministic logical representatives before randomized trials, so every valid code with `k > 0` has an upper-bound result even when the initial incumbent already satisfies the stopping condition. Increasing the trial budget may improve that upper bound, but failure to improve it is not evidence of a distance lower bound.

The implementation covers arbitrary qubit stabilizer codes represented by the Phase 14 binary-symplectic core and provides CSS-specific X and Z search paths when the stabilizer group is algebraically CSS, independent of which public constructor or equivalent generator presentation produced it. It reuses the existing packed checks, logical quotient, logical coordinates, qubit ordering, and native-handle ownership. Search setup derives or retains one independent canonical stabilizer basis and, for CSS groups, deterministic pure-X and pure-Z stabilizer and logical bases; it never repeats this setup inside a trial.

The first public slice is CPU-native and dependency-free beyond the existing TenCirPauli runtime. It does not add GAP, `gappy`, `ldpc`, CUDA, PyTorch, Stim, PyMatching, a generic SAT/ILP solver, or a Python reference search kernel. A future optional backend may use an external GPU estimator, but external backends are not part of this contract.

The search is intentionally separate from `StabilizerState`, decoder-output verification, and circuit execution. A returned logical witness is a static code property and is not a correction, a decoder decision, a sampled error, or a logical-error-rate estimate.

### Why this phase has no exhaustive mode

Physical-qubit count alone does not define a useful exact-search boundary. Exhausting all phase-free Paulis costs `4^n`; exhausting the normalizer of an `[[n,k]]` stabilizer code costs `2^(n+k)`. Weight-ordered enumeration is useful only when the actual distance is very small: to certify that a generic 90-qubit code has no logical below weight 9 requires considering up to `sum_{j=1}^8 3^j * binom(90,j)`, approximately `5.25e14`, general Pauli candidates. Even a CSS pure-sector scan has `sum_{j=1}^8 binom(90,j)`, approximately `8.57e10`, supports. More advanced meet-in-the-middle, information-set, SAT, ILP, or family-specific methods can solve selected instances but do not provide a predictable generic exact boundary and are separate solver projects.

Conversely, exhaustive search on a nine-physical-qubit fixture is trivial but adds no product capability needed by the intended tens-to-thousands-of-qubits workload. Phase 15 therefore has no physical-size dispatch to exact search and no exhaustive test helper. Independently established distances of standard fixtures supply expected test values, while every returned witness is validated directly.

### Reference project and design provenance

This specification was initiated after a source-level comparison with the [a7b/yarn](https://github.com/a7b/yarn) qLDPC processor-discovery toolkit. The comparison snapshot was the repository's then-current `main` commit [`82fb695`](https://github.com/a7b/yarn/tree/82fb695), inspected on 2026-08-22. The reference is recorded here so that later implementation and review work can recover the original motivation without relying on conversation history.

The directly relevant reference surfaces are:

| yarn reference | Design idea carried into this specification |
| --- | --- |
| [`sqetch/README.md`](https://github.com/a7b/yarn/blob/82fb695/sqetch/README.md) and [`sqetch/src/sqetch/recover.py`](https://github.com/a7b/yarn/blob/82fb695/sqetch/src/sqetch/recover.py) | Treat randomized low-weight logical search as an anytime upper-bound procedure and optionally return the concrete witness rather than only a scalar. |
| [`code_search/core/dist/quantum_bposd.py`](https://github.com/a7b/yarn/blob/82fb695/code_search/core/dist/quantum_bposd.py) and [`quantum_sqetch.py`](https://github.com/a7b/yarn/blob/82fb695/code_search/core/dist/quantum_sqetch.py) | Preserve CSS cross-type augmentation, strict `< target` early stopping, and explicit trial budgets. Their decoder-specific no-witness state is replaced here by a guaranteed canonical incumbent. |
| [`code_search/core/dist/stab_symplectic.py`](https://github.com/a7b/yarn/blob/82fb695/code_search/core/dist/stab_symplectic.py) | Use centralizer search, quotient/logical tags, weight-preserving local symplectic scrambles, packed elimination, and bounded candidate refinement for general non-CSS codes. |
| [`code_search/logical_basis/logical_basis.py`](https://github.com/a7b/yarn/blob/82fb695/code_search/logical_basis/logical_basis.py) | Keep logical-basis construction separate from minimum-weight witness search and preserve the distinction between an algebraic basis and an optimized representative. |
| [`code_search/search/README.md`](https://github.com/a7b/yarn/blob/82fb695/code_search/search/README.md) | Record configuration, seed, estimator, trial count, and rerun semantics so randomized research results remain auditable. |

These files are algorithmic and workflow references only. Phase 15 does not import yarn, GAP, `gappy`, `ldpc`, CUDA, or PyTorch; it does not copy a Python fallback into production; and its native packed ABI, memory guards, deterministic semantics, and public result types remain governed by TenCirPauli's architecture and current source.

### SQetch comparison and adopted rationale

SQetch's important performance choice is not merely CUDA: it computes the directional null space once, then assigns one independent fixed-size sketch to each GPU block. A block keeps its packed sketch in shared memory, applies a random physical-column permutation, performs collaborative RREF, filters rows by cross-type logical overlap, and contributes only its minimum weight through a global reduction. This changes per-trial work from full-null-space elimination to bounded `k_sub`-row elimination and exposes enough independent trials to saturate a GPU. `k_sub` is therefore both a performance parameter and a search-quality parameter and is recorded with research results.

Phase 15 adopts the reusable structure: one-time packed search-space construction, a bounded row sketch, random information-set order, many independent trials, packed popcount, strict threshold stopping, and explicit recording of the sketch size. It does not copy SQetch's GPU-specific implementation choices. In particular, SQetch samples rows of one null-space basis with replacement, whereas Phase 15 samples linear combinations of a canonical tagged basis so the candidate distribution is not limited to subsets of particular basis rows. Phase 15 also always retains a physical witness, validates it against the trusted code handle, starts from a deterministic incumbent, and uses indexed random streams and deterministic reduction instead of atomic race order.

| Concern | SQetch GPU implementation | Phase 15 native CPU implementation |
| --- | --- | --- |
| Supported code | One CSS direction per call | CSS X/Z directions and general stabilizer codes |
| Trial matrix | `k_sub` rows sampled with replacement from `ker(H_check)` basis | `sketch_rows` random linear combinations of a canonical tagged search basis, with at least one nonzero logical tag |
| Logical test | Dot product of each reduced row with the cross-type logical basis | Logical tags propagated through every row operation, followed by independent final classification |
| Parallel unit | One CUDA block per trial; shared-memory RREF; atomic minimum | One Rayon task per indexed trial inside a fixed-size host chunk; thread-local reusable scratch; deterministic ordered reduction |
| Witness | Separate recovery kernel and device output path | Always retained because a few packed CPU words are negligible beside trial elimination |
| Empty result | Possible when a sketch yields no reported logical row | Impossible for an accepted `k > 0` call because initialization supplies an incumbent and every trial sketch is forced to contain a nonzero tag |
| Reproducibility | Depends on device batching and atomic winner behavior | Trial-indexed counter RNG, fixed algorithm identifier, and deterministic tie-breaking |
| Exactness | Random upper bound only | Random upper bound only |

The fixed-size sketch is a quality-throughput tradeoff, not an accuracy tolerance. Larger sketches expose more row combinations per trial but reduce trials per second and multiply per-worker scratch memory. Consequently benchmark acceptance compares end-to-end best-weight-versus-time curves across representative `sketch_rows` values rather than selecting a default from isolated RREF throughput.

## 2. Binding decisions

1. The public operation is named `search_logical_witness`; it is not named `distance()` and never returns an unlabeled exact distance scalar.
2. Every returned witness is validated inside the native call before it is published: it has zero syndrome, nonzero logical quotient coordinates, and weight equal to the packed support popcount.
3. The randomized search result is an upper-bound result. `best_weight=w` means that a valid witness of weight `w` was found and therefore `d <= w`; it does not mean `d == w`.
4. For every accepted call on a code with `k > 0`, `best` and `best_weight` are non-optional. Failure to improve the deterministic incumbent is a normal upper-bound result, not a no-witness result.
5. Early stopping is strict: `stop_below_weight=t` stops only after obtaining a witness with `weight < t`; a witness with `weight == t` does not stop the search. The name encodes the strict comparison.
6. The first slice accepts an explicit positive `max_trials` budget. A wall-clock timeout, adaptive confidence claim, and cross-process scheduling contract are out of scope.
7. `max_trials` is the total number of randomized trials in one public call. For CSS `sector="auto"`, trials alternate deterministically between X and Z, beginning with X; the deterministic initialization evaluates both directions before trial zero.
8. `sketch_rows` is an explicit positive search-quality parameter and defaults to 64. Each direction resolves it to the smaller of the requested value and that direction's full tagged search-space dimension. It is not hidden inside an automatic hardware heuristic.
9. An explicit random seed gives deterministic mathematical output for identical code, sector, sketch size, stopping threshold, trial budget, and implementation version. `seed=None` is resolved in the Python facade with the package's existing unsigned-64-bit seed normalization and the resolved seed is returned.
10. Trial `i` uses the existing counter-based native random stream with a dedicated logical-search domain, direction, and trial index as counters. Therefore the first `T` trials are a prefix of the first `T' > T` trials, and the best upper bound is monotone non-increasing as a seeded run is extended.
11. Trials execute through Rayon in fixed index chunks of at most 256. Each worker owns and reuses its scratch buffers and retains only its local best candidate. Worker results are reduced by the total ordering `(weight, q0-first Pauli code)`, so discovery and scheduling order cannot change the winner. Threshold stopping occurs only between chunks, so `trials_run` includes every trial actually evaluated and may exceed the index of the first qualifying witness by at most 255. Rayon thread count may change performance and scratch memory but not the seeded result for the evaluated trial prefix.
12. The scalable native ABI accepts the trusted code handle and returns flat packed arrays. The Python wrapper does not materialize all checks, stabilizers, logical representatives, or per-trial candidates merely to pass them back into Rust.
13. The canonical logical basis remains an algebraic basis, not a minimum-weight, geometric, local, or hardware-optimal basis. The search may return a different physical representative and must expose that representative explicitly.
14. Exact distance, exhaustive Pauli or normalizer enumeration, certified bounded lower bounds, and exact test oracles are not part of Phase 15. Tests use independently known code distances and explicit expected witnesses or weights.

## 3. Mathematical contract

### 3.1 Stabilizer code and weight

For `n` qubits, a phase-free Pauli is represented as `p = (x | z)` in `GF(2)^(2n)`, with `I=(0,0)`, `X=(1,0)`, `Y=(1,1)`, and `Z=(0,1)` on each qubit. The physical qubit weight is

```text
weight(p) = popcount(x OR z)
```

The stabilizer group `S` is the signed-check span modulo irrelevant global phase, and its normalizer/centralizer is

```text
N(S) = {p : <p, s>_s = 0 for every s in S}.
```

For `k > 0`, the code distance is

```text
d = min { weight(p) : p in N(S) \ S }.
```

The randomized search does not enumerate the normalizer or stabilizer group. It must retain enough quotient provenance to distinguish a nontrivial logical row from a stabilizer row during native elimination. A bounded deterministic initialization may enumerate products of the canonical logical basis when the logical quotient is small; this enumerates upper-bound candidates only and is not an exact distance computation because it does not exhaust each candidate's stabilizer coset.

### 3.2 Witness validity

A `LogicalWitness` is valid exactly when all of the following hold:

- its packed X/Z words have the code's `nqubits` width and zero unused high bits;
- its syndrome against the complete original ordered check list is zero;
- its logical quotient coordinates are not both zero;
- its reported `weight` equals `popcount(x_words OR z_words)` after masking unused high bits;
- its logical coordinates classify it as `LOGICAL`, not `STABILIZER`.

The witness may be a product of several canonical logical basis elements and may be mixed X/Z even when the code has a useful CSS representation. The phase of the physical Pauli is not returned because distance and logical quotient membership are phase-free. Signed check eigenvalues continue to govern code construction but do not alter the commutation-based witness search.

### 3.3 CSS directions

For CSS checks `Hx` and `Hz`, the search exposes optional directional modes. An X-type physical logical is searched in

```text
ker(Hz) \ rowspace(Hx)
```

and a Z-type physical logical is searched in

```text
ker(Hx) \ rowspace(Hz).
```

The implementation must preserve the cross-type logical constraint. For an X-direction search, a candidate may be forced outside `rowspace(Hx)` using the Z-logical witnesses; for a Z-direction search, it may be forced outside `rowspace(Hz)` using the X-logical witnesses. Same-type augmentation is forbidden because it can select a low-weight stabilizer and misreport it as a logical.

The public `sector` values are `"auto"`, `"x"`, `"z"`, and `"general"`. `"auto"` uses the CSS directional paths when the stabilizer group is algebraically CSS and otherwise uses the general stabilizer path. `"x"` and `"z"` are accepted only for an algebraically CSS group; `"general"` is accepted for every code and deliberately bypasses the CSS specialization. A CSS code's total distance upper bound is the lighter of the X- and Z-direction incumbents. Because `k > 0` gives a nontrivial logical in each CSS direction, both directional initial incumbents exist.

CSS recognition is a property of the stabilizer row space, not of the constructor used. Search setup derives the pure-X subgroup `{(x,0) in S}` and pure-Z subgroup `{(0,z) in S}` and recognizes the group as CSS exactly when the sum of their ranks equals `rank(S)`. This keeps `sector="auto"` invariant under reordered, redundant, or mixed equivalent generator presentations.

## 4. Search algorithm

### 4.1 General native path

The general path builds a packed tagged basis for the centralizer from the independent canonical stabilizer rows followed by the existing `2k` paired logical representatives. Stabilizer basis rows have tag zero and logical basis rows have one distinct bit in a packed `2k` logical quotient tag. Physical words and tags undergo exactly the same combinations and row operations. No separate stabilizer-span provenance is needed because a zero tag is exactly the stabilizer coset for a centralizer row.

Before trial zero, the implementation evaluates every individual canonical logical representative. When `2^(2k) - 1 <= 1_048_575`, it additionally enumerates every nonzero product of the `2k` canonical logical representatives once, using Gray-code XOR updates and allocation-free packed popcount. This small-logical-space initialization is only an incumbent search over one representative per logical product; it does not enumerate stabilizer multiples and never claims exactness.

Each `randomized_information_set_v2` general trial performs the following operations:

1. Resolve `q = min(sketch_rows, n + k)`. If `q` equals the full centralizer dimension, copy the complete canonical tagged basis. Otherwise materialize `q` random linear combinations of that basis directly into reusable packed scratch storage. The first coefficient vector is sampled with a uniformly random nonzero `2k` logical portion and an independent uniformly random stabilizer portion; later vectors are uniform over the full centralizer. This guarantees at least one nonzero logical tag while allowing every member of `N(S) \ S` a nonzero probability of appearing directly.
2. Independently choose one of the six single-qubit binary symplectic maps for every physical qubit from the trial's counter-based random stream and apply it to every sketched physical row. This is a local-Clifford scramble and preserves physical Pauli weight.
3. Generate a Fisher-Yates permutation of all `2n` binary X/Z elimination columns from the same trial stream and row-reduce the tagged sketch in that order.
4. Select up to the four lowest-weight rows with nonzero logical tags and nonzero physical weight, resolving equal weights by the public Pauli tie order.
5. Give each selected row at most two greedy refinement sweeps. A sweep evaluates XOR with every reduced row, rejects zero-tag or zero-weight results, takes the lowest-weight valid result with deterministic tie-breaking, and stops that candidate when no strict weight improvement exists.
6. Undo the scramble, recompute the logical coordinates, and update the global best witness by weight. Equal-weight candidates are ordered lexicographically by their q0-first per-qubit Pauli codes using `I < X < Y < Z`, making the winner independent of hash iteration or discovery order.

The tag is XORed together with every row operation. A nonzero tag is the native quotient test; the hot path must not repeatedly reconstruct a complete stabilizer group or call back into Python for candidate classification.

These choices define `randomized_information_set_v2`. The CSS specialization intentionally omits the general local symplectic scramble and uses an `n`-column sector matrix. Changing the scramble distribution, random-draw order, candidate count, refinement sweeps, or elimination order requires a new algorithm identifier. Internal packed kernels may change without a new identifier only when seeded outputs remain bit-for-bit identical.

### 4.2 CSS native path

The CSS path operates on one binary sector at a time. Search setup constructs a deterministic packed basis for the directional search space: pure stabilizer rows followed by `k` pure logical representatives with packed `k`-bit quotient tags. For X search this basis spans `ker(Hz)` with quotient `rowspace(Hx)`; for Z search it spans `ker(Hx)` with quotient `rowspace(Hz)`.

A CSS trial resolves `q` to the smaller of `sketch_rows` and the directional search-space dimension, constructs a tagged random-combination sketch with the first row forced to have a uniformly random nonzero logical tag, generates a Fisher-Yates permutation of the `n` physical columns, and performs tagged packed RREF. It then selects at most four candidates and applies at most two improving sweeps exactly as in the general path. The returned candidate is pure X or pure Z and is validated against both the original syndrome and the cross-type logical quotient before publication.

Before randomized trials, each direction evaluates its canonical pure logical rows. When `2^k - 1 <= 1_048_575`, it additionally enumerates all nonzero products of those rows once using Gray-code updates. Larger logical spaces skip exhaustive logical-product initialization. Randomized information-set elimination and stabilizer-row refinement remain necessary in both cases because logical-product enumeration does not minimize within stabilizer cosets.

### 4.3 Budget and anytime behavior

`max_trials` counts complete native randomized trials, not deterministic initialization candidates, Python calls, candidate rows, or successful witnesses. For `sector="auto"` on CSS codes, trial index parity selects X then Z; for every other accepted sector each index runs that one path. After deterministic initialization and after each parallel chunk, the native loop may stop under the strict threshold rule. If initialization already finds a witness below the threshold, the result has `trials_run=0` and `early_stopped=True`.

The best upper bound is monotone non-increasing as the same seeded search is extended with additional trials. More trials improve coverage but do not provide a universal probability-of-optimality or probability-of-correctness statement. The API must not fabricate a confidence interval from the trial count alone because candidate-discovery probabilities depend on code structure, logical dimension, weight distribution, algorithm, and random stream.

The result records enough information to replay a run against the same code: requested sector, algorithm identifier, stopping threshold, resolved seed, requested and completed trials, early-stop flag, and best witness. The native setup and initialization counters are observability metadata for benchmark records; first compilation, Python normalization, code-handle construction, and complete public-call timing remain separate benchmark boundaries.

## 5. Public Python API

### 5.1 Logical witness record

```python
@dataclass(frozen=True, slots=True)
class LogicalWitness:
    x_words: np.ndarray
    z_words: np.ndarray
    logical_x_words: np.ndarray
    logical_z_words: np.ndarray
    weight: int
    sector: Literal["x", "z", "mixed"]
    nqubits: int
    qubit_order: Literal["q0_first"]
```

All arrays are owned, C-contiguous, read-only `uint64` arrays. `x_words` and `z_words` have shape `(ceil(nqubits/64),)`. Logical coordinate arrays have shape `(ceil(nlogical/64),)` and use logical coordinate zero as the least-significant bit of word zero. The record does not materialize a `PauliWord` automatically; a future explicit helper may construct one when requested.

### 5.2 Search result record

```python
@dataclass(frozen=True, slots=True)
class LogicalWitnessSearchResult:
    best: LogicalWitness
    best_weight: int
    sector: Literal["auto", "x", "z", "general"]
    algorithm: Literal["randomized_information_set_v2"]
    sketch_rows: int
    max_trials: int
    trials_run: int
    stop_below_weight: int | None
    seed: int
    early_stopped: bool
    initial_incumbent_weight: int
    initialization_candidates: int
    resolved_sketch_rows: int
    setup_nanos: int
    initialization_nanos: int
```

`best_weight` always equals `best.weight`. The record has upper-bound semantics and must document that `best_weight=w` means only `d <= w`. It has no `lower_bound`, `exact`, `complete`, `confidence`, or no-witness state. `initial_incumbent_weight` and `initialization_candidates` describe the deterministic pre-trial incumbent search; `resolved_sketch_rows` is the actual cap used by the selected path (the maximum of the two directions for `auto`); and the nanosecond fields expose native setup and initialization costs for benchmark provenance. `algorithm` versions the seeded trial mapping; an implementation that changes that mapping must use a new identifier rather than silently changing replay for `randomized_information_set_v2`.

### 5.3 Search entry point

```python
code.search_logical_witness(
    *,
    sector: Literal["auto", "x", "z", "general"] = "auto",
    max_trials: int,
    sketch_rows: int = 64,
    stop_below_weight: int | None = None,
    seed: int | None = None,
    max_bytes: int | None = DEFAULT_MAX_BYTES,
) -> LogicalWitnessSearchResult
```

`max_trials` and `sketch_rows` must be positive. `sketch_rows` records the requested cap; each path internally uses the smaller of this value and its full tagged search-space dimension. `stop_below_weight`, when supplied, must be positive. `seed` accepts an unsigned 64-bit integer; `None` is resolved by the same Python helper used by the existing stochastic APIs and the resolved value is recorded in the result. `max_bytes` guards the prepared bases, retained result, and one reusable trial workspace per Rayon worker using the repository's best-effort memory contract; checked arithmetic and major-allocation rejection remain mandatory.

The method raises for `k=0`, invalid threshold, trial, or sketch values, `sector="x"` or `"z"` on a non-CSS stabilizer group, overflow, and an estimated workspace above `max_bytes`. A valid accepted call cannot return without a witness.

### 5.4 Explicit Pauli materialization

An explicit helper may materialize `LogicalWitness` as a `PauliWord` only when the caller requests it. The search result itself must stay on the flat packed ABI. No new nested-list read-back path is allowed.

## 6. Native and FFI design

The PyO3 method accepts the existing `NativeStabilizerCodeHandle` and scalar search metadata. The complete search-basis setup, optional bounded logical-product initialization, quotient tagging, randomized trial loop, packed RREF, candidate refinement, validation, and output preparation execute inside one GIL-released native call.

The Rust core owns prepared canonical bases, scratch buffers, random streams, packed row storage, tag storage, and the winning witness. Rayon workers use `map_init` or an equivalent worker-local mechanism so scratch matrices and candidate buffers are allocated once per worker rather than once per trial. Candidate rows are never sent across PyO3 individually. A returned witness uses one flat packed read-back record.

The implementation must reuse the existing packed word convention, symplectic inner product, logical quotient, counter-based random primitives, and checked memory-estimation helpers. The canonical independent stabilizer rows currently used transiently during `StabilizerCode` construction must either be retained on the immutable handle or reconstructed once in search setup; they must not be reconstructed per trial. A separate Python copy of the stabilizer matrix or a Python reference search loop is prohibited in production code.

## 7. Correctness and validation

Every native result is validated before read-back. Tests must cover the following contracts:

- A returned witness has zero syndrome and nonzero logical coordinates.
- The reported weight equals an independent packed support popcount.
- Fixed five-qubit, Steane, Shor, repetition, and small asymmetric CSS fixtures use independently known distances or directional distances. At documented fixed seeds and trial budgets, the search must return witnesses at those known weights; the expected values are fixture facts, not outputs of a second exhaustive implementation.
- For a known distance `d`, a stopping threshold `t > d` may early-stop only after finding a witness with weight `< t`; a witness at exactly `t` does not satisfy the early-stop condition.
- A stabilizer row, identity, and zero logical tag are never returned as witnesses.
- Products of canonical logical basis elements are searched and can be returned; the implementation does not restrict itself to the individual basis rows.
- Small-logical-space initialization covers both sides of its fixed combination-cap boundary and remains explicitly an upper-bound operation.
- CSS X and Z directions use the correct cross-type quotient condition and agree with analytically specified expected witnesses or weights on small asymmetric codes.
- General non-CSS cases include mixed X/Z logical witnesses and redundant signed checks.
- Reordered, redundant, and mixed equivalent generator lists preserve CSS recognition, best witness, and deterministic result under the frozen canonicalization and tie-breaking policy while public syndrome coordinates remain input ordered.
- q0, the final qubit, widths crossing 64-bit words, zero unused high bits, `n=0`, empty batches, and `k=0` rejection are covered.
- Same code, algorithm, seed, threshold, and trial count replay the same packed result; changing the seed is allowed to change the witness.
- For one seed, every longer trial budget has a best weight no greater than the corresponding shorter prefix.
- `sector="auto"` uses one total alternating trial budget, and an initialization hit reports zero completed randomized trials.
- Sketch sizes below, at, and above the full search-space dimension use the documented resolved dimension; changing Rayon thread count does not change the seeded result.
- `max_bytes` rejects an estimated major workspace before allocation and does not alter mathematical results when the budget is sufficient.

Tests may use direct dense Pauli matrices to verify a returned fixture witness's action on a small code space, but they do not search the dense Pauli space or derive an expected distance by exhaustive enumeration. No exhaustive distance implementation exists in production, test helpers, examples, or benchmarks.

## 8. Benchmark and observability contract

Release benchmarks must separate cold build or extension setup, code-handle construction, search setup, first search, steady seeded search, witness materialization, and memory. The primary search measurements report trials per second, completed trials, requested and resolved sketch rows, deterministic-initialization candidate count, initial incumbent weight, final best weight, witness validity, worker count, and peak or retained memory where available.

Representative workloads include small known-distance stabilizer codes, asymmetric CSS codes, redundant-check codes, a roughly 90-physical-qubit CSS or qLDPC-style case, and wide packed sparse codes. The pre-implementation algorithm spike and the retained release benchmark compare at least `sketch_rows=16, 32, 64`, and the full-space value where affordable. Quality reporting must show `best_weight` as a function of both trial count and end-to-end wall time and compare it with the deterministic incumbent; throughput alone is insufficient evidence that the search is useful. The default remains 64 only if the representative quality-versus-time comparison supports it; changing the accepted default before release updates the specification and algorithm identifier. A curated known-distance case must demonstrate that the fixed release profile can reach a known optimum, while larger cases report only witnessed upper bounds unless their distance is independently known from the construction. Benchmark claims include the complete public call and record the actual `tencirpauli.__file__` and `_native.__file__` used by the process. No wall-time CI gate is introduced; benchmark records remain informational.

## 9. Explicit non-goals

This phase does not implement exact distance for codes of any size, exhaustive Pauli or normalizer search, a certified lower-bound distance API, a generic decoder, BP/OSD as a correction decoder, syndrome-to-correction selection, noise channels, Monte Carlo logical-error rates, syndrome-extraction circuits, detector error models, fault propagation, subsystem/gauge codes, qudit stabilizer codes, GAP-backed LP/HGP code construction, hardware placement, or external GPU execution.

The bounded enumeration of canonical logical products is not an exception: it examines neither all stabilizer multiples nor all members of `N(S) \ S` and therefore remains only a deterministic upper-bound initialization.

## 10. Acceptance evidence

Phase 15 is complete only when the Rust core, PyO3 binding, and Python facade expose the same upper-bound and witness semantics; all returned witnesses pass independent numerical validation; the focused known-distance, deterministic-replay, prefix-monotonicity, threshold, CSS-recognition, logical-combination-cap, boundary, redundant-check, general non-CSS, and memory-guard tests pass; release native code is rebuilt in the test environment; and the benchmark source records cold, first, steady, trial-count, incumbent-quality, witness, and memory boundaries.

The implementation review must explicitly answer four questions: where the logical quotient tag is maintained, how the algorithm avoids returning stabilizers, how equivalent generator presentations receive the same CSS classification and seeded search result, and how the public result makes upper-bound-only semantics unmistakable.
