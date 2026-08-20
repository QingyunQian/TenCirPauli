# Phase 13 Classical Shadow Snapshots Specification

Status: owner-approved implementation-ready specification; not implemented. The required P0 is an exact small-system proof and release-mode profile of global-Clifford sampling through commuting-Pauli measurements. This phase does not reopen Phase 6.5 time evolution and does not add a general statevector, MPS, circuit, or hardware simulator.

## 1. Purpose and binding decisions

Phase 13 adds a Rust-native classical-shadow data plane with local Pauli and global Clifford protocols. The main new capability is exact global-Clifford sampling and estimation without applying a Clifford gate sequence to an arbitrary statevector.

For a randomly sampled Clifford `U`, computational-basis measurement after `U` is implemented in the Heisenberg picture by jointly measuring the signed commuting Pauli generators `Q_i = U^dagger Z_i U` on the input state. Rust generates the random symplectic frame, retains the measured `Z` images, and samples their joint eigenvalues. It never evolves an arbitrary statevector through H/S/CNOT gates and does not expose a circuit executor.

The canonical public object is `tencirpauli.shadows.Snapshots`, re-exported as `tencirpauli.Snapshots`. `Snapshots` is one immutable public facade with `protocol="pauli"` or `protocol="clifford"`; Rust retains separate packed variants behind one native handle. `StabilizerState`, `PauliSnapshotArrays`, `CliffordSnapshotArrays`, and `shadow_bound` are also public and re-exported from the package root.

Native sampling always generates random measurement settings internally. Users provide the number of independently randomized settings, the number of repeated outcomes per setting, and an optional seed; they do not provide native-sampling bases or Clifford frames. Explicit bases and frames appear only in external-import constructors.

TensorCircuit's existing local-Pauli shadow functions remain the JAX/TensorFlow-compatible path. TenCirPauli does not place a native handle inside a backend JIT trace. TensorCircuit remains responsible for MPS and other backend-specific state representations, while TenCirPauli owns compact snapshots, estimators, stabilizer sampling, and complete-statevector sampling through specialized Pauli measurement kernels.

All public qubit-indexed arrays use TensorCircuit's q0-first order. TensorCircuit statevector basis index bit `nqubits - 1 - q` corresponds to qubit `q`; packed Pauli storage retains qubit `q` at packed bit `q % 64`.

## 2. Public construction API

~~~python
from tencirpauli.shadows import (
    CliffordSnapshotArrays,
    PauliSnapshotArrays,
    Snapshots,
    StabilizerState,
    shadow_bound,
)
~~~

### 2.1 Importing local-Pauli measurements

~~~python
Snapshots.from_pauli_measurements(
    bases,
    outcomes,
    *,
    qubit_order="q0_first",
    max_bytes=DEFAULT_MAX_BYTES,
)
~~~

The canonical TensorCircuit-compatible setting-major form is:

~~~text
bases:    (nsettings, nqubits), uint8-compatible codes 1=X, 2=Y, 3=Z
outcomes: (nsettings, repeats, nqubits), binary values 0/1
~~~

A flat per-shot form is also accepted as `bases.shape == outcomes.shape == (nsettings, nqubits)` and is stored with `repeats=1`. TensorFlow/JAX values are converted outside a traced function; Python performs one normalization to contiguous NumPy arrays and one coarse native construction call. Floating-point basis or outcome arrays are rejected rather than silently rounded.

This is the interchange format for `tensorcircuit.shadows.shadow_snapshots(..., measurement_only=True)`. TensorCircuit's five-dimensional local snapshot-state output is not accepted.

### 2.2 Importing global-Clifford measurements

~~~python
Snapshots.from_clifford_measurements(
    frame_x_words,
    frame_z_words,
    frame_signs,
    outcomes,
    *,
    nqubits,
    frame_version=1,
    qubit_order="q0_first",
    max_bytes=DEFAULT_MAX_BYTES,
)
~~~

Version 1 represents one measurement frame by `nqubits` independent signed commuting Pauli generators `Q_i`. With `word_count = ceil(nqubits / 64)`, the arrays are:

~~~text
frame_x_words: (nsettings * nqubits, word_count), uint64
frame_z_words: (nsettings * nqubits, word_count), uint64
frame_signs:   (nsettings, nqubits), int8 values in {-1, +1}
outcomes:      (nsettings, repeats, nqubits), uint8 values in {0, 1}
~~~

Generator row `setting * nqubits + i` and `frame_signs[setting, i]` define the signed Hermitian Pauli `Q_i`. Outcome bit `b_i` records eigenvalue `(-1)**b_i` of `Q_i`; equivalently the resulting stabilizer snapshot has signed generator `(-1)**b_i Q_i`.

Construction validates padding bits, Hermiticity, pairwise commutation, independence, shapes, codes, signs, and ordering once. External callers are responsible for supplying frames drawn from the protocol ensemble; validation does not attempt a statistical randomness test.

### 2.3 Native sampling

~~~python
Snapshots.sample(
    state,
    *,
    nsettings,
    protocol="pauli",
    repeats=1,
    seed=None,
    max_bytes=DEFAULT_MAX_BYTES,
)
~~~

`state` is a complete statevector or a `StabilizerState`. `nsettings` is the required positive integer number of independently randomized measurement settings. `repeats` is the positive integer number of conditionally independent outcomes drawn under each retained setting. Total state copies are the checked product `shots = nsettings * repeats`.

The statevector public boundary accepts finite one-dimensional complex arrays safely promotable to contiguous complex128. Its length must be a power of two, its squared norm must differ from one by at most `1e-10`, and the native snapshot normalizes the accepted copy before sampling. Density matrices and batched statevectors are rejected. Stabilizer inputs remain handle-to-handle and are never materialized as statevectors.

For the Pauli protocol, each setting independently selects X, Y, or Z with equal probability on every qubit. For the Clifford protocol, each setting is sampled exactly uniformly from the n-qubit Clifford group modulo global phase; the retained measurement frame is the signed `Z` half of its symplectic action.

`seed` is `None` or an unsigned 64-bit integer. Python resolves `None` once with `secrets.randbits(64)` and passes the resulting seed to the native call; all setting and outcome draws then occur in Rust. An explicit seed gives deterministic replay for the same package version, platform, input, `nsettings`, and `repeats`, independent of Rayon scheduling and thread count; no hidden global RNG state is used.

### 2.4 Metadata and raw access

Every finalized object exposes:

~~~python
snapshots.protocol
snapshots.nqubits
snapshots.nsettings
snapshots.repeats
snapshots.shots
snapshots.qubit_order
~~~

`Snapshots` is immutable. Estimators consume its native handle directly. The public read-back records are frozen slotted dataclasses with exactly these fields:

~~~python
@dataclass(frozen=True, slots=True)
class PauliSnapshotArrays:
    bases: np.ndarray
    outcomes: np.ndarray
    qubit_order: Literal["q0_first"]

@dataclass(frozen=True, slots=True)
class CliffordSnapshotArrays:
    frame_x_words: np.ndarray
    frame_z_words: np.ndarray
    frame_signs: np.ndarray
    outcomes: np.ndarray
    nqubits: int
    frame_version: Literal[1]
    qubit_order: Literal["q0_first"]
~~~

`to_arrays()` is overloaded to return `PauliSnapshotArrays` for the Pauli protocol and `CliffordSnapshotArrays` for the Clifford protocol. Pauli `bases` has shape `(nsettings, nqubits)` and dtype `uint8`; Pauli `outcomes` has shape `(nsettings, repeats, nqubits)` and dtype `uint8`. Clifford array shapes and dtypes are exactly those in Section 2.2. Every returned array is an owned C-contiguous NumPy copy with `writeable=False`, remains valid independently of the native handle, and can be passed directly to the matching import constructor. The scalar fields are the constructor metadata that cannot be inferred safely from the packed Clifford arrays. `nsettings`, `repeats`, and `shots` remain properties of `Snapshots` and are inferred from record array shapes rather than duplicated in the records. Estimators never call `to_arrays()` or require materialization.

## 3. Native sampling algorithms

### 3.1 Exact global-Clifford frames

P0 samples the exact uniform Clifford ensemble; it must not substitute a finite-depth random Clifford circuit, a mixing heuristic, or a user-selected basis list. Modulo global phase, a uniform Clifford consists of a uniform binary symplectic action and uniform Pauli displacement. The retained marginal can be generated by sampling an ordered symplectic basis `(X_0, Z_0, ..., X_{n-1}, Z_{n-1})` uniformly and retaining the `Z_i` vectors as the unsigned frame, then drawing the retained generator signs as independent unbiased bits.

The required constructive sampler recursively operates in the current `2m`-dimensional binary symplectic subspace. It draws `Z` uniformly from all nonzero vectors in that subspace, draws `X` uniformly from the affine hyperplane of vectors satisfying `[X, Z] = 1`, emits the pair, constructs a symplectic basis of `span{X, Z}^perp`, and recurses at dimension `2(m-1)`. Uniform coefficient bits in any fixed basis of the current subspace give the required uniform vector and affine-hyperplane draws; deterministic symplectic Gram-Schmidt only changes coordinates. At every recursion level the number of choices is `(2**(2m) - 1) * 2**(2m - 1)`, so every ordered symplectic basis has equal probability. The retained `Z_i` are therefore an exact uniform ordered independent commuting frame, with the same marginal distribution as `U^dagger Z_i U` for uniform Clifford `U`.

The core implementation uses this recursive construction with packed `u64` rows, deterministic symplectic Gram-Schmidt, and checked GF(2) elimination. The random-bit mapping must not use modulo reduction with bias; bounded integer draws use rejection.

### 3.2 Global-Clifford measurement without circuit simulation

Let `U` be the sampled Clifford, let `Q_i = U^dagger Z_i U`, and let the current normalized state be `|phi>`. Every `Q_i` is a signed Hermitian Pauli word, the `Q_i` commute pairwise, and they are independent.

For one generator `Q_i`, define `m_i = Re <phi|Q_i|phi>`. Outcome bit `b_i` has eigenvalue `lambda_i = (-1)**b_i` and probability

~~~text
p(b_i | previous outcomes) = (1 + lambda_i * m_i) / 2.
~~~

After sampling the outcome, the normalized conditional state is

~~~text
|phi'> = (I + lambda_i Q_i)|phi> / sqrt(2 * (1 + lambda_i * m_i)).
~~~

Sequentially applying this rule to all `nqubits` commuting generators samples the exact joint distribution

~~~text
p(b) = <psi| product_i [(I + (-1)**b_i Q_i) / 2] |psi>.
~~~

Each repeat restarts from the original input state. The implementation does not sample the generators independently without projection, because that would lose their joint correlations.

For a dense statevector, one signed Pauli application is an XOR permutation plus a parity-dependent phase over the `2**nqubits` amplitudes. Expectation is a conjugate dot reduction; projection is a scaled vector addition. No dense matrix, eigensolver, BLAS, LAPACK, `ndarray`, or `nalgebra` dependency is required. The existing Hamiltonian `MvpPlan` already implements the same q0/MSB mask, fixed-Y phase, XOR, and parity conventions; Phase 13 extracts one internal `StatevectorPauliAction` carrying the compiled `usize` X mask, Z mask, and fixed canonical-Y/sign phase. It provides an allocation-free fused `apply_into_and_expectation(input, scratch)` operation and `project_from_action_in_place(state, scratch, eigenvalue)`. For each generator the sampler applies `Q` exactly once into scratch, obtains the real Hermitian expectation during that same pass, and consumes that same `Q|phi>` scratch in the projector update; it never recomputes the action after sampling the eigenvalue. The sampler never constructs a `PauliOperator` or `MvpPlan` per generator.

Within one repeat, generators are processed in retained row order. The Hermitian expectation uses a fixed indexed reduction order; its roundoff result is clamped to `[-1, 1]`, `p0 = (1 + m_i) / 2`, and outcome zero is selected exactly when the stream draw satisfies `u < p0`. The selected projection reuses one full-size scratch vector and the normalization in the displayed formula; it does not perform an additional norm pass after every generator. Nonfinite expectations are errors rather than alternate sampling paths.

Dense global-Clifford sampling costs `O(nsettings * repeats * nqubits * 2**nqubits)` time and retains the input plus bounded per-worker state/action scratch. It is a specialized joint-Pauli measurement kernel, not a gate simulator. Stabilizer inputs instead use tableau measurement and polynomial storage/time.

The P0 implementation must verify exact frame distribution for exhaustive or enumerated one- and two-qubit cases, exact equivalence to dense `U`-then-Z probabilities on small systems, frame signs, conditional normalization, q0 ordering, and seeded replay before the public Clifford sampler is exposed.

### 3.3 Local-Pauli sampling

For a dense statevector and one sampled Pauli setting, Rust copies the input once, applies the tensor-product single-qubit basis change in place, constructs the `2**nqubits` computational-basis probabilities and one cumulative distribution once, and draws all `repeats` full bitstrings from that distribution. X measurement uses the existing Hadamard convention; Y measurement uses the existing `Sdg`-then-Hadamard convention; Z performs no rotation. The final cumulative entry is set to one after normalization, each draw uses binary search, and the sampled basis index is decoded in q0/MSB order. This samples the complete correlated bitstring, not independent one-qubit marginals.

Consequently dense Pauli sampling costs `O(nsettings * nqubits * 2**nqubits + nsettings * repeats * nqubits)` time, plus `O(nsettings * repeats * log(2**nqubits))` categorical-search work, and `O(2**nqubits)` state/probability scratch per active worker. Repeats amortize the basis transform and probability construction for this protocol, although Section 4's variance argument still makes `repeats=1` the statistical default when state copies, rather than setting changes, are the constrained resource. This fixed tensor-product rotation kernel is not exposed as a general gate executor.

For a `StabilizerState`, the same sampled single-qubit Pauli setting is measured through tableau updates. Every repeat starts from the original immutable tableau, sequentially measures the commuting local generators, and retains the resulting correlated outcomes; no statevector or dense probability vector is constructed.

## 4. Settings, repeats, and practical shot allocation

The independent statistical unit is one randomized setting, not one repeated outcome. Estimators first average all repeats belonging to a setting and only then form means, blocks, or cross-setting U-statistics. Results from one setting are never split across median-of-means blocks.

For a linear per-outcome estimator `Y`, write `V_setting = Var_setting(E[Y | setting])` and `V_outcome = E_setting(Var[Y | setting])`. With `M = nsettings`, `R = repeats`, and fixed total copies `N = M R`, the grand-mean variance is

~~~text
Var(mean Y) = V_setting / M + V_outcome / (M R)
            = (R * V_setting + V_outcome) / N.
~~~

Therefore `repeats=1` weakly minimizes variance at fixed total copies whenever randomized-setting variance is nonzero. It is the required default and the recommended choice for linear expectations, energies, RDMs, shadow-U-statistic Rényi-2, and global-Clifford sampling. Global-Clifford repeats also do not amortize dense sampling because every repeat requires a fresh sequence of conditional projections from the original state.

Repeated outcomes remain useful when experimental setting changes are expensive or when using the randomized-measurement Rényi-2 estimator, which requires within-setting outcome pairs. The public sampler does not choose an automatic repeat count because the optimum depends on subsystem size, state, setup cost, and total budget.

Practical non-binding guidance is:

- For linear quantities and shadow Rényi-2, use `repeats=1` and spend the complete copy budget on `nsettings`.
- For randomized-measurement Rényi-2 on a `k`-qubit subsystem, require `repeats >= 2`. With `d_A = 2**k`, the order-one exact-collision scale solves `R(R-1)/(2 d_A) ~= 1`, namely `R_0 = ceil((1 + sqrt(1 + 8 d_A))/2) = Theta(sqrt(d_A))`. This is a birthday-collision pilot scale, not an accuracy theorem; compare `R_0`, `2 R_0`, and `4 R_0` while separately doubling settings to identify the dominant error source.
- For global Clifford on a dense statevector, start with `repeats=1`; increasing settings improves ensemble coverage, while increasing repeats multiplies the full conditional-projection cost.
- For global-Clifford low-rank targets such as stabilizer fidelity, a practical first pilot is `256–1024` settings with `repeats=1`; inspect `5–10` returned block values and double the setting count until the estimate and block spread stabilize. This is a convergence workflow, not a rigorous bound.
- Do not use global Clifford as a presumed faster estimator for local Pauli observables. For a traceless Pauli word `P` under exact uniform global Clifford, the one-setting estimator has second moment `2**nqubits + 1` and variance `2**nqubits + 1 - <P>**2`; the local-Pauli estimator for weight `w` has second moment `3**w` and variance `3**w - <P>**2`. Global Clifford can be better for genuinely full-width/global targets, but it is exponentially less sample-efficient when `w` is small compared with `nqubits`; this is an actual estimator-variance difference, not merely a loose bound. Even at full width, the global-Clifford Pauli variance remains exponential (`2**nqubits`): the dimension-independent stabilizer-fidelity result in Section 6.6 is target-specific and does not make every global observable polynomial. Local/low-weight observables should ordinarily use the Pauli protocol and `shadow_bound()`.
- No heuristic is presented as a rigorous sample bound. Users needing a Pauli linear-observable guarantee use `shadow_bound()`.

## 5. StabilizerState

`StabilizerState` represents a pure stabilizer state, not a density matrix or mixture.

~~~python
StabilizerState.zero_state(nqubits)

StabilizerState.from_generators(
    generators,  # sequence of (PauliWord, sign), sign in {+1, -1}
)

StabilizerState.from_clifford_tape(tape)  # existing advanced.GateTape

StabilizerState.from_tableau(
    x_words,
    z_words,
    signs,
)

state.to_tableau()
~~~

Every `StabilizerState` is immutable and exposes `state.nqubits`.

The generator form contains exactly `nqubits` independent commuting signed Pauli generators. Construction reuses `PauliWord` and validates width, count, independence, and commutation.

`from_clifford_tape()` starts from `|0...0>`, accepts the existing `GateTape` object, and executes its operations once in append/Schrodinger order. The tape must have `nparameters == 0`, and every operation must be one of X/Y/Z/H/S/Sdg/CNOT/CZ/SWAP; static or runtime rotations, channels, and every other operation are rejected even if a particular angle would mathematically be Clifford. It reuses the tape's existing native handle and operation conventions and does not introduce a second Python operation grammar.

For `word_count = ceil(nqubits / 64)`, tableau `x_words` and `z_words` have shape `(nqubits, word_count)` and dtype `uint64`, while `signs` has shape `(nqubits,)`, dtype `int8`, and values in `{-1,+1}`. `to_tableau()` returns the exact three-tuple `(x_words, z_words, signs)` as owned C-contiguous NumPy copies with `writeable=False`. Rust deterministically completes imported stabilizer generators to a compatible full tableau and retains the destabilizer rows required by measurement; those rows are not part of the public interchange format.

## 6. Estimators and block semantics

### 6.1 Mean and median of block means

~~~python
snapshots.expectation(observable, *, blocks=1, return_blocks=False)
snapshots.estimate_many(observables, *, blocks=1, return_blocks=False)
~~~

Supported inputs are `PauliWord`, real-coefficient `PauliTerm`, and exactly Hermitian `PauliOperator`; `estimate_many` accepts a sequence of these values. Non-Hermitian targets are rejected, so scalar and array outputs are real float64.

Every target must have `nqubits == snapshots.nqubits`. `energy()` and `energy_variance()` require an exactly Hermitian `PauliOperator`; callers use `expectation()` for a single word or term. The equality and Hermiticity checks reuse the existing canonical native operator rather than materializing terms in Python.

`blocks=1` returns the ordinary arithmetic mean over setting-level estimates and is the practical default for lowest ordinary Monte Carlo variance. For `blocks > 1`, settings are partitioned in retained order into balanced contiguous blocks: the first `nsettings % blocks` blocks contain `floor(nsettings / blocks) + 1` settings and the remainder contain `floor(nsettings / blocks)`. Each block returns the arithmetic mean of its complete setting-level values, and the public estimate is the elementwise median of block means. Median uses the usual NumPy convention: the middle value for an odd count and the arithmetic mean of the two middle values for an even count.

For example, `nsettings=1000`, `repeats=4`, and `blocks=5` produces five disjoint groups of 200 settings. Each setting first averages its four outcomes, each block then averages its 200 setting values, and the reported estimator is the median of the five block means. `blocks` is purely a robust post-processing choice; it is not a measurement basis count and does not change the stored snapshots.

For an operator sum, terms are combined into the complete operator value within each setting before block averaging or the median. Taking a separate median for each term and then summing is forbidden because median is nonlinear.

`return_blocks=False` returns the aggregate scalar or shape `(nobservables,)`. `return_blocks=True` returns block means instead, with shape `(blocks,)` for one target and `(blocks, nobservables)` for multiple targets. Linear estimation requires `1 <= blocks <= nsettings`.

Median-of-means is recommended when a high-probability guarantee, many simultaneous targets, or heavy-tailed shadow values matter. If the data are ordinary i.i.d. samples and only a point estimate is wanted, `blocks=1` is preferred. When `shadow_bound()` is used, its returned block count is the matching theoretical choice; otherwise `5–25` blocks with at least about `20` settings per block is a practical diagnostic range rather than a theorem.

### 6.2 Protocol-specific linear kernels

For a local-Pauli snapshot, a weight-`w` Pauli word contributes only when every non-identity letter matches the recorded local basis. The per-outcome value is `3**w` times the product of measured eigenvalues on its support; Rust evaluates support matching, parity, repeated-outcome averaging, block accumulation, and coefficient combination without per-term or per-shot FFI.

For a global-Clifford outcome, the postmeasurement stabilizer state `|s>` is defined by generators `(-1)**b_i Q_i`, and the inverse-channel snapshot is

~~~text
rho_hat = (2**nqubits + 1) |s><s| - I.
~~~

For a non-identity Pauli word `P`, its outcome value is `(2**nqubits + 1) <s|P|s>`, where the stabilizer expectation is `+1` or `-1` if `P` belongs to the signed stabilizer group and zero otherwise. Identity has value one. Batched membership and sign evaluation use packed GF(2) plans and share repeated words across observables.

`energy(hamiltonian, *, blocks=1, return_blocks=False)` is an alias of the Hermitian `expectation()` path with identical block and return-shape semantics.

### 6.3 RDM

~~~python
snapshots.rdm(
    *,
    subsystem_to_keep=None,
    subsystems_to_trace_out=None,
    max_bytes=DEFAULT_MAX_BYTES,
)
~~~

The two subsystem forms are mutually exclusive when either is supplied. If both are `None`, all qubits are kept in q0-first order. User order is preserved for `subsystem_to_keep`; complement order is q0-first for `subsystems_to_trace_out`. For `k` retained qubits, the first retained qubit is the MSB of the returned C-contiguous complex128 matrix with shape `(2**k, 2**k)`. Empty and full subsystems are valid: the empty RDM is the shape `(1, 1)` matrix `[[1]]`. `rdm()` returns the arithmetic mean of reduced snapshot estimators, preserves Hermiticity up to floating-point roundoff, and does not project to positive semidefinite form or clip eigenvalues. It materializes only the requested subsystem and rejects oversized output/workspace estimates.

For a local-Pauli outcome with measured one-qubit eigenstate `|s_q>`, the full snapshot is `tensor_product_q (3 |s_q><s_q| - I_q)`. Tracing out a qubit contributes the scalar trace one, so for retained subsystem `A` the exact per-outcome reduced estimator is

~~~text
A_hat = tensor_product_{q in A, in requested output order} (3 |s_q><s_q| - I_q).
~~~

For a global-Clifford outcome stabilizer state `|s>`, let `B` be the traced subsystem and `d_B = 2**|B|`. The exact reduced estimator is

~~~text
A_hat = Tr_B[(2**nqubits + 1) |s><s| - I_AB]
      = (2**nqubits + 1) * rho_A(s) - d_B * I_A,
rho_A(s) = Tr_B |s><s|.
~~~

Rust obtains `rho_A(s)` from the restricted stabilizer group and materializes only the requested `2**|A|` square result. Repeats are averaged within their setting before the final arithmetic mean. P0 `rdm()` deliberately exposes no `blocks` or `return_blocks` option; robust nonlinear processing belongs to the dedicated estimators below.

### 6.4 Rényi-2 entropy

~~~python
snapshots.renyi_entropy(
    *,
    subsystem_to_keep=None,
    subsystems_to_trace_out=None,
    alpha=2,
    method="shadow",
    blocks=1,
)
~~~

Phase 13 supports exactly `alpha=2`. Other alpha values fail explicitly instead of silently materializing an RDM and selecting an unrelated finite-sample estimator.

For `method="shadow"`, let `A_s` be the requested-subsystem snapshot estimator averaged over repeats within setting `s`. Within each block containing `m >= 2` settings, the unbiased purity U-statistic is

~~~text
purity = [sum_{s != t} Tr(A_s A_t)] / [m * (m - 1)].
~~~

The kernel must use aggregate identities, packed stabilizer overlaps, or bounded dense subsystem accumulators rather than materializing an `m * m` pair table. Block purities are combined by their median, then entropy is `-log(purity)`. A negative final purity returns `NaN`; an exact zero returns positive infinity. `method="shadow"` is supported by both protocols and requires `1 <= blocks` and `nsettings >= 2 * blocks`, so every balanced block has at least two settings.

For `method="randomized_measurement"`, only local-Pauli setting-major snapshots are supported and `repeats >= 2` is required. On a `k`-qubit subsystem, setting `s` has the unbiased distinct-repeat estimator

~~~text
purity_s = 2**k / [R * (R - 1)]
           * sum_{r != r'} (-2)**(-Hamming(b[s,r], b[s,r'])).
~~~

Setting purities are averaged inside blocks and combined by their median before applying `-log`. Self-pairs are excluded; this intentionally differs from TensorCircuit's current finite-repeat plug-in expression, while agreeing with its randomized-measurement target asymptotically. This method requires `1 <= blocks <= nsettings`; unlike the cross-setting U-statistic, a block may contain one setting because each setting already contains an unbiased distinct-repeat purity estimate. Differential tests compare the exact kernel on fixed outcomes and compare convergence to the same dense purity, not per-sample equality with TensorCircuit.

`renyi_entropy()` returns only the final scalar entropy in P0. It does not expose block purities or block entropies: median aggregation occurs in purity space and only the aggregate purity is transformed by `-log`, so a generic `return_blocks` flag would have ambiguous units.

### 6.5 Energy variance

~~~python
snapshots.energy_variance(hamiltonian, *, blocks=1, max_bytes=DEFAULT_MAX_BYTES)
~~~

For every block, `Tr(rho H**2)` is estimated linearly after Rust-side Pauli multiplication and duplicate aggregation under the memory guard. If `x_s` is the complete setting-level estimate of `<H>` and the block has `m >= 2` settings, the unbiased square is

~~~text
<H>**2_hat = [(∑_s x_s)**2 - ∑_s x_s**2] / [m * (m - 1)].
~~~

The block variance is the block mean estimate of `<H**2>` minus this cross-setting square. Block variances are combined by their median. The call requires `1 <= blocks` and `nsettings >= 2 * blocks`; finite-sample negative results are returned unchanged and never clipped. Term explosion and workspace overflow fail before a major allocation.

`energy_variance()` returns only the final scalar in P0 and has no `return_blocks` option. Its block values mix a linear `<H**2>` estimate with a cross-setting U-statistic and are kept as an internal robust-reduction detail.

### 6.6 Fidelity

~~~python
snapshots.fidelity(target, *, blocks=1, return_blocks=False)
~~~

Global-Clifford snapshots require a same-width `StabilizerState` target in Phase 13. One outcome contributes `(2**nqubits + 1) * |<target|s>|**2 - 1`, evaluated by a native stabilizer-overlap kernel. Arbitrary statevector targets and implicit Pauli expansions are not supported. `fidelity()` is unsupported on local-Pauli snapshots; a caller with an explicit Hermitian Pauli expansion evaluates that expansion through `expectation()`.

Fidelity is a linear setting-level estimator, so `blocks` and `return_blocks` have exactly the semantics and return shapes in Section 6.1.

For `d = 2**nqubits` and true fidelity `F`, exact uniform global Clifford gives one-setting second moment `d * (1 + 2 F) / (d + 2)` and variance `d * (1 + 2 F) / (d + 2) - F**2 < 2`. Consequently the number of independent settings needed for one or many fixed stabilizer fidelities scales as `O(epsilon**-2 * log(L / delta))` with no exponential dimension factor. Stabilizer-overlap evaluation is polynomial in `nqubits`, so post-processing and the complete stabilizer-input pipeline are polynomial; generating the same snapshots from an arbitrary dense statevector still costs `O(nqubits * 2**nqubits)` per setting and is not polynomial in `nqubits`.

### 6.7 Pauli shadow bound

~~~python
shadow_bound(observables, epsilon, delta=0.01)
~~~

`observables` is one `PauliWord`, a nonempty sequence of same-width `PauliWord` values, or a nonempty TensorCircuit-compatible integer code array. A one-dimensional code array is one word and a two-dimensional array is a batch; codes are `0=I, 1=X, 2=Y, 3=Z`, and floating arrays are rejected. `PauliOperator` is not accepted because the simple locality formula is a simultaneous termwise bound, not a coefficient-aware Hamiltonian-energy bound. `epsilon` must be finite and strictly positive, and `delta` must be finite with `0 < delta < 1`.

For `L` targets and maximum non-identity weight `w`, the TensorCircuit-compatible values are

~~~text
blocks = ceil(2 * log(2 * L / delta))
settings_per_block = ceil(34 * 3**w / epsilon**2)
nsettings = blocks * settings_per_block.
~~~

The block count is the median-of-means confidence amplifier: one block mean is made accurate with a fixed constant success probability, while the median of `blocks` independent block means suppresses failure exponentially. The factor `log(2 L / delta)` includes a union bound over `L` simultaneous targets. Therefore `blocks` depends on target count and failure probability, while `epsilon` and the Pauli shadow norm `3**w` determine settings per block. Constants `2` and `34` follow the TensorCircuit/Huang–Kueng–Preskill convention rather than an empirical tuning rule.

The return is `(nsettings, blocks)`. It assumes independently randomized settings and is used with `repeats=1`; extra repeats do not reduce the required setting count. Automatic global-Clifford bounds are not included in the first helper contract even though stabilizer fidelity has dimension-independent variance; adding that case requires freezing the exact advertised high-probability constant rather than reusing the local-Pauli constant silently.

## 7. Ownership, FFI, and dependencies

Python owns public types, friendly array normalization, subsystem argument resolution, small immutable metadata, and explicit materialization. Rust owns frame generation, packed storage, statevector and tableau sampling, support/membership kernels, block reduction, U-statistics, Pauli multiplication, overlap calculations, and memory checks.

The implementation uses the following authoritative file ownership. Splitting a listed Rust module into private children is allowed when it improves readability, but moving scalable work into Python or creating a second public representation is not.

| File | Required ownership |
| --- | --- |
| `crates/tencir-pauli-core/src/pauli_action.rs` | Shared compiled single-Pauli statevector action, expectation reduction, and projector update extracted from `hamiltonian::MvpPlan` conventions; both MVP and shadows consume this primitive. |
| `crates/tencir-pauli-core/src/random.rs` | Private counter-based `u64` generation extracted from SPPS, domain separation, and unbiased bounded-integer draws; existing seeded SPPS behavior remains unchanged. |
| `crates/tencir-pauli-core/src/shadows.rs` | Packed protocol storage, exact symplectic-frame generation, dense/tableau sampling, stabilizer state/tableau algorithms, estimators, block reduction, and core validation. |
| `crates/tencir-pauli-core/src/lib.rs` | Core module declarations and the minimal public Rust re-exports needed by the native crate. |
| `crates/tencirpauli-native/src/shadows.rs` | `NativeSnapshotsHandle` and `NativeStabilizerStateHandle`, flat NumPy construction/read-back, coarse sampling and estimator calls, error mapping, and GIL release. A Clifford-frame Python class is not exposed. |
| `crates/tencirpauli-native/src/lib.rs` | Registration of the two private native handle classes and no shadow algorithms. |
| `python/tencirpauli/shadows.py` | `Snapshots`, `StabilizerState`, `PauliSnapshotArrays`, `CliffordSnapshotArrays`, `shadow_bound`, friendly normalization, public type hints, and docstrings. |
| `python/tencirpauli/_native.pyi` | Exact private handle method signatures and flat-array return types. |
| `python/tencirpauli/__init__.py` | Root re-exports of the five public shadow names; no wrapper logic. |
| `tests/test_shadows.py` | Public/core numerical semantics, import/read-back round trips, sampling, estimators, invalid inputs, memory, concurrency, and docstring contracts. |
| `tests/test_shadow_tensorcircuit.py` | TensorCircuit measurement-only interchange and bounded dense differential references. |
| `crates/tencir-pauli-core/benches/shadows.rs` | Criterion kernels for frame generation, Pauli action/projection, tableau operations, and estimator kernels, with the matching Cargo bench entry. |
| `benchmarks/python/test_shadows_benchmark.py` | End-to-end sampling, FFI, retained-memory, estimator, and TensorCircuit comparisons integrated with the existing benchmark runner. |

PyO3 performs one coarse construction, sampling, or estimation call. Work scaling with settings, repeats, qubits, amplitudes, terms, frames, blocks, or subsystem basis states executes with the GIL released. Native handles are immutable; state/action scratch, accumulators, and RNG counters are call-local.

`NativeSnapshotsHandle` owns a private protocol enum with separate packed Pauli and Clifford variants. `NativeStabilizerStateHandle` owns the tableau. Python `Snapshots` and `StabilizerState` each retain exactly one corresponding native handle plus small immutable scalar metadata; they do not retain duplicate arrays, frames, generators, or reference kernels. Generated Clifford frames never cross FFI during normal sampling or estimation. They are read back only through the versioned packed arrays returned by `to_arrays()`.

The Rust core continues to depend only on `num-complex`, `rayon`, and `rustc-hash`; Phase 13 adds neither a general linear-algebra dependency nor an RNG crate. The existing deterministic hash/counter random primitive in `spps.rs` is extracted into a private shared `random.rs` helper that can return a complete pseudorandom `u64` while preserving existing SPPS draws. Shadows use domain-separated counters keyed by seed, protocol, setting, repeat, and local draw index, so parallel scheduling cannot change results. Uniform bounded integers use rejection from `u64`, local Pauli basis selection has no modulo bias, and rejection inside one stream advances only that stream's local draw index.

Public docstrings are part of the Phase 13 contract. `Snapshots.sample` must explain `nsettings`, `repeats`, derived `shots`, the `repeats=1` recommendation, and protocol-specific sampling cost. Estimator docstrings must explain setting-level averaging, `blocks=1` mean versus `blocks>1` median-of-means, return shapes, and minimum settings per block. `renyi_entropy` must distinguish the shadow U-statistic from randomized measurement and state the latter's distinct-repeat requirement and collision-scale guidance. Linear-estimator docstrings must state the local-Pauli versus global-Clifford variance tradeoff rather than presenting global Clifford as uniformly superior.

The implementation must reuse or extract the existing q0/MSB Pauli action convention from `hamiltonian::MvpPlan`, existing `PauliWord`/`PauliOperator` handles, Clifford gate enums and conjugation rules, GF(2) elimination where applicable, best-effort memory guards, flat NumPy read-back, and error mapping. It must not retain a second Python reference execution path after the native implementation exists.

## 8. Validation and memory

Construction rejects incompatible shapes, invalid subsystem indices, unsupported dtypes, invalid codes or signs, non-power-of-two or materially unnormalized statevectors, invalid generators/tableaux, noncommuting or dependent Clifford measurement generators, incompatible qubit counts, unsupported protocol/target combinations, checked arithmetic overflow, and enforced memory-limit failures.

Public integer counts (`nsettings`, `repeats`, and `blocks`) reject booleans and must be strictly positive and representable by the native index type; `seed` likewise rejects booleans and must fit `uint64`. The only accepted protocol strings are `"pauli"` and `"clifford"`, the only accepted public qubit order is `"q0_first"`, and the only accepted Clifford frame version is `1`. `estimate_many()` rejects an empty observable sequence. `shadow_bound()` applies the nonempty-target, `epsilon`, and `delta` checks in Section 6.7. Subsystem selectors contain unique in-range integer qubits and use exactly one of the two selector forms. Method-specific minimum settings and repeats are the contracts in Sections 6.4 and 6.5; linear methods require `blocks <= nsettings`.

Friendly Python inputs are copied or promoted once into canonical contiguous arrays. Private native array endpoints require their documented dtype and contiguity. Wrong object kinds raise `TypeError`; invalid values or incompatible domains raise `ValueError`; dimension arithmetic raises `OverflowError`; enforced allocation limits raise `MemoryError`.

Packed settings, frames, and outcomes, dense state/action scratch, active workers, RDM output, stabilizer overlap workspace, H-squared expansion, and block accumulators participate in the existing best-effort `max_bytes` policy. `max_bytes=None` disables the estimate guard while checked arithmetic remains mandatory. The guard is not an exact peak-RSS guarantee.

Parallel dense sampling is over independent settings or repeats; the statevector operation within one copy retains a fixed serial reduction order so an explicit seed does not depend on Rayon scheduling. For dimension `d`, the shared dense input contributes `16d` bytes, each active global-Clifford worker is budgeted at least `32d` bytes for current state plus Pauli-action scratch, and each active local-Pauli worker is budgeted at least `24d` bytes for transformed state plus cumulative probabilities. Packed output and Rayon/task overhead are added under the existing best-effort policy. Worker count is capped before launching work; the implementation must not allocate one full scratch set per Rayon thread when the requested budget supports fewer active workers.

## 9. Correctness tests

Required tests include:

1. Pauli packed round trips, q0-first ordering, setting-major repeats, flat per-shot import, and TensorCircuit measurement-only interchange.
2. Exact one- and two-qubit global-Clifford frame distribution checks, including 3 and 90 unsigned ordered frames or 6 and 360 signed retained frames respectively, constant symplectic-completion multiplicity, symplectic validity, signed `U^dagger Z_i U` references, unbiased bounded draws, and deterministic seed replay across thread counts.
3. Dense joint probabilities from sequential commuting-Pauli projection versus explicit small Clifford matrices, including nonlocal generators, frame signs, outcome correlations, and asymmetric first/last qubits.
4. Statevector and stabilizer sampling for product, Bell, GHZ, random non-stabilizer, and toric-code fixtures.
5. Numerical local-Pauli and global-Clifford expectations for Pauli words, Hamiltonians, multiple observables, repeats, means, block means, and median-of-means.
6. RDMs and shadow Rényi-2 against independent dense references for both protocols.
7. Randomized-measurement Rényi-2 fixed-outcome formula tests, explicit self-pair exclusion, and convergence tests distinct from TensorCircuit's finite-repeat plug-in estimator.
8. Energy variance against dense `H` and `H**2` references, including the cross-setting square, duplicate aggregation, finite-sample negative results, and term-explosion guards.
9. Stabilizer generator/tableau/tape round trips and global-Clifford stabilizer-target fidelity.
10. Concurrent sampling and estimation with no hidden global RNG, coarse FFI, GIL release, memory guards, and unsupported-input errors.
11. Public docstring assertions or documentation tests covering `nsettings`, `repeats`, derived shots, block construction, both Rényi-2 methods, and the local-Pauli/global-Clifford variance tradeoff.

Every numerical native entry point must have at least one fixed known-correct value assertion; shape, replay, or convergence-only tests are insufficient.

## 10. Performance and practical benchmarks

Benchmarks are release-mode and separate frame generation, frame compilation, external import, Pauli statevector preparation, global-Clifford single-Pauli action, expectation reduction, projector update, complete sampling, estimator-only execution, first call, steady calls, retained memory, and numerical error.

Required dense global-Clifford workloads include `nqubits = 10, 14, 18`, `nsettings = 64, 256, 1024` where the local 16 GiB policy permits them, and `repeats = 1` plus one small repeated control. Records report amplitudes processed per second, Pauli projections per second, frame time, sampling time, peak estimate, active workers, and agreement with a small dense reference.

Required stabilizer workloads include at least 64, 256, and 1024 qubits with setting and repeat sweeps. Required estimator workloads include many local Pauli terms, shared words across observables, global-Clifford membership queries, stabilizer fidelity, subsystem RDM size sweeps, both Rényi-2 methods, and repeated estimation over one immutable handle.

The Pauli sampling baseline is TensorCircuit's measurement-only JAX path with synchronized first and warm execution. Global-Clifford dense sampling is compared against the best correct TensorCircuit construction available for bounded sizes, but a changing-circuit compile cost is reported separately rather than used as the sole performance claim.

The first profile must identify the dominant share among Pauli action, conjugate reduction, projector update, frame generation, state copying, and allocation. Optimizations follow the dominant cost; no specialized hashing, SIMD, scratch cache, or parallel branch is retained merely because it improves an isolated microbenchmark.

The P0 profile is a go/no-go gate, with same-machine release A/B evidence, on these practical criteria:

1. Dense complete-sampler time and peak estimate follow the specified `O(nsettings * repeats * nqubits * 2**nqubits)` and worker-scratch scaling; doubling settings or repeats must not expose recompilation, pairwise-setting work, or retained per-call growth beyond the expected output.
2. At `nqubits >= 14` with at least 64 settings, frame generation, frame compilation, scheduling, and allocation together account for no more than 20% of complete global-Clifford sampling time. Otherwise the avoidable dominant component is fixed and reprofiled before the public sampler proceeds.
3. The extracted single-Pauli action is within 5% of the pre-extraction direct one-term MVP kernel on representative dimensions, and the existing many-term MVP benchmark shows no material regression beyond 5%. A larger change requires a measured explanation and owner review rather than being hidden by the new feature's total speed.
4. Hot-loop allocation counters show zero allocation per generator and per amplitude. State/action and CDF buffers are allocated once per active worker and reused across assigned settings or repeats.
5. The reported TensorCircuit comparisons include input conversion, synchronization, first execution, and warm execution. They are evidence, not an acceptance requirement that local-Pauli Rust beat the fixed-structure JAX path.

Performance records remain local and informational under repository policy, but acceptance requires representative end-to-end evidence showing that the global-Clifford path is usable and that the new shared Pauli action does not regress existing Hamiltonian MVP behavior.

## 11. Non-goals, implementation order, and acceptance

This phase does not add a public measurement plan, user-supplied native sampling settings, arbitrary gate application, a statevector output terminal, general circuit ingestion, arbitrary matrix observables, MPS/tensor-network execution, hardware compilation, automatic TensorCircuit Circuit extraction, global-Clifford-on-MPS simulation, general entropy orders, non-Hermitian median semantics, or a coefficient-aware Hamiltonian shadow bound.

Recommended implementation order:

1. Record the pre-extraction one-term and many-term MVP release baselines, then extract and differentially verify one compiled single-Pauli statevector action from existing MVP conventions.
2. Complete the P0 exact-uniform symplectic frame generator and dense sequential commuting-Pauli sampling spike, including small dense differentials and the first release profile.
3. Implement packed Pauli and Clifford snapshot handles, external import, metadata, and flat read-back.
4. Implement local-Pauli and global-Clifford linear estimators with setting-level repeats and block semantics.
5. Implement `StabilizerState`, tableau sampling, and stabilizer overlap/membership kernels.
6. Implement RDM, both Rényi-2 estimators, energy variance, stabilizer fidelity, and `shadow_bound()`.
7. Complete TensorCircuit interchange, full correctness gates, release benchmarks, public docs, and examples.

Acceptance requires both protocols, arbitrary complete-statevector global-Clifford sampling through joint Pauli measurements, scalable stabilizer sampling, exact numerical estimators, the two distinct Rényi-2 methods, TensorCircuit Pauli interchange, versioned Clifford-frame interchange, immutable native ownership, no general linear-algebra dependency, the complete quality gate, and representative release-mode evidence.
