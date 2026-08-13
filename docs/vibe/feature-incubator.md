# Feature Incubator

Status: living design ledger for ideas that are not mature enough to enter a frozen phase specification.

## Purpose and maintenance policy

This document preserves feature ideas, partial designs, unresolved questions, and deferred directions that arise during TenCirPauli development but do not yet have a committed implementation phase. It prevents useful reasoning from being lost and prevents speculative features from leaking into an active specification.

Every new concept should enter this document before implementation unless it is already covered by a frozen phase spec. Each entry records its status, motivation, current decisions, unresolved questions, dependencies, and promotion criteria.

Allowed statuses are:

| Status | Meaning |
| --- | --- |
| `seed` | A potentially useful idea with little design work. |
| `discussing` | Active design work; semantics or scope remains open. |
| `deferred` | Understood well enough to postpone intentionally. |
| `promoted` | Moved into a named phase spec; this entry remains as historical context and links to that spec. |
| `rejected` | Deliberately excluded, with the reason retained. |

When an idea is promoted, do not delete its entry. Mark it `promoted`, summarize the owner decision, and link to the authoritative spec. The phase spec becomes the source of truth; this ledger should not duplicate changing implementation details.

## Promoted ideas

### Majorana algebra, parity/BK mappings, and additive-charge sectors

Status: `promoted` to the frozen implementation contract `phase-7.5-spec.md`.

The promoted scope includes public Majorana words/operators, fermion-to-Majorana conversion, reusable Jordan–Wigner/parity/Bravyi–Kitaev mapping plans, exact integer-valued additive charges, simultaneous commuting-charge sectors, zero-charge qudit spectators, symmetry validation, and guarded Hamiltonian-space reduction.

## Deferred ideas

### Public finite-boson algebra

Status: `deferred`.

Motivation: support exact algebra after choosing finite occupation cutoffs, make projected boson operators first-class, and provide a natural input to boson-to-qubit encodings.

Current design: do not reuse infinite-Fock CCR power blocks as the canonical finite algebra. For a one-mode cutoff `N`, use matrix units `E_mn = |m><n|` with `E_mn E_pq = delta_np E_mq`; multimode words are tensor products of local matrix units. This is exact, finite, and closed but is mathematically a structured finite local-matrix algebra rather than a second CCR algebra.

Reason for deferral: Phase 7 already supports explicit projected finite targets, while no current user workflow requires symbolic multiplication of projected operators. Public exposure would add a broad local-operator abstraction and potentially large term counts without an immediate consumer.

Promotion criteria: a concrete boson-to-qubit, finite-boson circuit, or restricted finite-boson workflow that benefits from reusable finite algebra rather than direct compilation.

### Non-orthogonal subspace projection

Status: `seed`.

Motivation: support variational quantum-subspace, quantum chemistry, and reduced-model workflows in which the supplied basis states are not mutually orthogonal. Given operator (H) and ordered basis states ({\lvert \phi_i\rangle}), return the projected operator (H_{ij}=\langle\phi_i\rvert H\lvert\phi_j\rangle) together with the overlap matrix (S_{ij}=\langle\phi_i\vert\phi_j\rangle), so the caller can solve the generalized eigenproblem (Hc=ESc).

Current design: expose projection as an operator/state algebra result, not as an implicit eigensolver. The input basis may contain sparse computational-basis states or native-backed linear combinations; the output is a deterministic dense (H) matrix and (S) matrix with explicit ordering, dtype, endian, and memory metadata. Orthogonal projection is the special case (S=I), but the incubated feature specifically targets the non-orthogonal contract. The implementation must validate compatible operator spaces, basis dimensions, finite coefficients, and the requested output budget; it must not silently orthogonalize the basis or discard a singular/ill-conditioned overlap matrix.

Dependencies: native operator-on-state and inner-product kernels, a batched matrix-element path, checked dense-output estimates, and a numerical policy for zero or nearly dependent basis vectors. SciPy or another generalized eigensolver may be used by the caller or an optional integration layer; the Rust core should return (H) and (S) without depending on an eigensolver.

Open questions: whether the first API accepts only one operator family or any structured operator with a finite target; whether complex non-Hermitian (H) is supported or only Hermitian projected problems; whether overlap conditioning is reported only or also checked against a caller-provided tolerance; and whether a reusable native projection plan is needed for repeated operators or repeated basis states.

Promotion criteria: one concrete chemistry or quantum-subspace workload; dense reference tests for orthogonal, non-orthogonal, singular, and nearly dependent bases; complex-valued matrix-element differentials; explicit ordering and conditioning semantics; representative setup/steady-runtime and memory measurements; and a decision that the result contract is useful without adding a built-in generalized eigensolver.

### Boson-to-qubit encodings

Status: `deferred`.

Motivation: compile finite boson modes to Pauli operators for qubit hardware, VQE, and TensorCircuit qubit backends.

Current design: support explicit binary, Gray, and unary/one-hot encodings. Return both the Pauli operator and immutable encoding metadata, including the code-space projector and the action on unused computational states. Non-power-of-two compact encodings must define illegal-state behavior explicitly; penalty terms are opt-in and never hidden.

Dependencies: a frozen finite-boson transition representation, exact isometry convention, analytic or sparse matrix-unit-to-Pauli decomposition, and representative term-count/Pauli-weight benchmarks.

Promotion criteria: an identified qubit-simulation workload and owner decisions on encoding set, illegal-state semantics, projector/penalty API, and acceptable term-growth boundaries.

### Lie closure for Pauli-polynomial generators

Status: `deferred`.

> **Example candidate**: mark for a future `examples/research/` algebra-simplification demo built from existing primitives (`PauliOperator.commutator`, `PauliWord.multiply`, `commutes_with`, `symplectic_inner_product`). Scope to the easy packed word-level closure plus a small Pauli-sum case, producing a structure-constant table and verifying it numerically against the Jacobi identity. This demonstrates native commutator speed and term explosion without advancing the deferred universal polynomial engine.

Motivation: analyze dynamical Lie algebras, controllability, reachable operator spaces, and symmetry-induced reductions from user-supplied Hamiltonian generators.

Current design: closure of individual Pauli words is comparatively simple because a nonzero commutator produces one Pauli word up to scalar, allowing a packed symplectic set closure. That restricted feature is not the desired endpoint. The target use case accepts Pauli sums or general operator polynomials, for which commutators are sparse vectors and new generators must be tested for linear independence in the real anti-Hermitian Lie algebra.

Main difficulties: sparse-vector fill-in, exact versus numerical rank, real versus complex scalar conventions, coefficient tolerance, deterministic basis selection, exponential dimension up to `4**n - 1`, and the fact that a `max_dimension` cap yields an incomplete rather than closed result.

Promotion criteria: a concrete control workload, a frozen coefficient/rank contract, a deterministic sparse elimination design, explicit completeness/truncation semantics, and bounded benchmark cases with known Lie dimensions.

### Arbitrary-order BCH and formal operator series

Status: `deferred`.

> **Example candidate**: mark for a future `examples/research/` demo of fixed-order (order ≤ 4) truncated BCH using only the existing `commutator`/`add`/`scale` primitives. Validate by exponential-error convergence against `exp(A) exp(B)` and contrast against a plain-Python Pauli simplifier to expose native speed. Fixed-order tables deliberately avoid the deferred arbitrary-order free-Lie work; take care with the exact BCH coefficients (e.g. the fourth-order term has two contributions).

Motivation: effective Hamiltonians, composition of exponentials, canonical transformations, Floquet/Magnus calculations, and symbolic perturbation theory.

Current design: treat BCH as a formal truncated Lie series, not as a numerical approximation with an automatic error bound. Arbitrary order requires a free-Lie representation such as Hall/Lyndon bases or another audited coefficient generator, exact rational universal coefficients, and explicit `order`, `max_terms`, and `max_bytes` limits. The universal BCH term growth is compounded by term growth and CAR/CCR contractions in the evaluated operator algebra.

Related easier feature: truncated adjoint action `exp(A) B exp(-A) = sum_k ad_A**k(B)/k!` follows one commutator chain and may be worth promoting independently if a concrete canonical-transformation use case appears.

Reason for deferral: BCH is not required by the Phase 7.5 Majorana/mapping/symmetry workflows and would pressure the project into a premature universal operator protocol.

Promotion criteria: a concrete effective-Hamiltonian or formal-series workload, a decision between bounded fixed-order tables and arbitrary-order generation, a coefficient-exactness contract, and representative term-growth evidence.

### Reference normal ordering and Wick expansion

Status: `deferred`.

Motivation: many-body perturbation theory, Gaussian fermion expectations, reference-dependent normal ordering, diagrammatic contractions, and reduced-density calculations.

Current design: ordinary vacuum CAR/CCR normal ordering is already implemented and is not a missing feature. A future API must distinguish reference-normal-ordered objects from the current vacuum-canonical `FermionOperator`/`BosonOperator`. Fermionic Gaussian expectations should preferably use Majorana covariance matrices and Pfaffians rather than enumerate all pairings. Explicit Wick expansion, if required, must retain reference metadata and contraction provenance.

Promotion criteria: a concrete Gaussian/reference-state workload and owner decisions on Slater versus general Gaussian references, expectation-only versus explicit expansion, anomalous contractions, and result representation.

### Qudit symmetry and sector reduction

Status: `deferred`.

Motivation: discover generalized Pauli/Weyl symmetries and reduce uniform-qudit Hamiltonians.

Current design: prime local dimension permits symplectic linear algebra over a finite field. Composite dimensions such as `d=4` or `d=6` require module methods over `Z_d`, potentially Smith normal form, and cannot reuse prime-field Gaussian elimination unchanged.

Reason for deferral: Phase 10.5 promotes canonical spin-z additive charge on Qudit axes but does not implement generalized Weyl/stabilizer symmetry discovery. The current Weyl API supports both prime and composite dimensions, and a partial implementation that silently treats every `Z_d` as a field would be incorrect.

Promotion criteria: a target qudit workload, a decision on prime-only versus general composite support, a frozen stabilizer/sector convention, and independent small-dimension references.

### Arbitrary diagonal additive charges on qudit axes

Status: `deferred`; canonical spin-z charge was promoted to Phase 10.5.

Motivation: extend the canonical Qudit spin-z charge with user-defined exact integer charge values for every local level. This would support occupation-like, model-specific, or otherwise non-spin spectra without requiring generalized Weyl/stabilizer symmetry discovery.

Current design direction: a future API may accept explicit per-level metadata such as `qudits={site: (q_0, q_1, ..., q_(d-1))}` with exactly one integer per local basis level. A Weyl `X^a` shift then has the source-dependent charge change `q_((m+a) mod d) - q_m`; conservation and transition compilation must use the exact local table rather than assume one constant delta. This API is deliberately separate from Phase 10.5's concise `qudits={site: integer_weight}` canonical spin-z contract.

Resolved boundary: Phase 10.5 defines `|0>` as highest spin-z, uses the smallest exact integer normalization, and expands `AdditiveCharge.as_operator()` through Weyl `Z` powers. Those decisions do not imply that arbitrary per-level tables differing by a constant offset are equivalent or canonicalized.

Open questions: whether table offsets are semantic or canonicalized, whether heterogeneous local Qudit dimensions should be introduced at the same time, how arbitrary spectra participate in public charge-homogeneous decomposition, and whether a general `charge_deltas` API is justified.

Promotion criteria: a concrete workload that cannot use canonical spin-z; owner approval of table and offset semantics; flat native transport; independent dense conservation and rank/unrank references; and release evidence that the added generality does not force ordinary canonical spin-z sectors through Generic execution.

### Occupation-parity sectors and simultaneous U(1)+Z2 restrictions

Status: `deferred`.

Motivation: support practical restricted-Hilbert-space workflows in which a user selects a sector by an integer weighted occupation parity, such as particle parity, spin parity, or another conserved mod-2 charge, and in some cases combines that parity with an exact U(1) charge. The desired public behavior is analogous to `U1Sector`: construct a deterministic selected basis, validate that an operator preserves the selected sector, and expose restricted dense/COO/CSR and matrix-free MVP targets.

Semantic boundary: this is an occupation-space parity sector, not the existing Pauli `Z2SymmetryAnalysis`/`Z2TaperingPlan`. Pauli Z2 tapering finds Pauli generators, applies a Clifford transform, and removes qubits; an occupation-parity sector filters and indexes basis states. The two features may share small GF(2) row-reduction utilities, but they must not share one high-level executor or one ambiguous public `Z2Sector` type.

Current design direction: represent parity weights modulo two and solve the selected basis as a finite constraint system over exact integer equations and GF(2) equations. A narrow first implementation could use a separate internal parity indexer for one binary parity constraint and expose it through the common restricted-plan contract. Pure-binary parity can use direct free-bit/pivot-bit rank and unrank; multiple parity constraints require deterministic GF(2) reduction. If U(1) and parity are combined, the internal constraint group is conceptually `Z × (Z2)^r`, with redundant, inconsistent, and genuinely independent constraints canonicalized before plan selection.

Execution boundary: once a selected basis has been compiled into one validated transition graph or CSR representation, the numerical MVP apply can reuse the common restricted executor. Lazy MVP is not completely representation-independent: it must compute destination membership and rank/unrank on the fly, so the parity or mixed-constraint indexer remains part of the lazy hot path. Eager CSR therefore has a simpler reuse story, while a performant lazy path needs a dedicated indexer or a benchmark-proven generic constraint indexer.

Reason for deferral: adding a modulus flag directly to `AdditiveCharge` would spread exact-versus-modular branches through sector dimension, rank/unrank, factorized/Product/Generic classification, conservation validation, transition compilation, Python types, PyO3 handles, and the private ABI. Simultaneous U(1)+parity also creates a new constraint-combination dimension, including cases where parity is already implied by total particle number. The feature is realistic, but its main complexity is sector compilation and basis indexing rather than the final CSR SpMV arithmetic; an unisolated implementation would increase the current symmetry-path entropy substantially.

Open questions: whether the first public form supports only one parity constraint or several independent constraints; whether parity contributions are defined for boson and qudit levels as well as binary occupations; whether conservation means global commutation or only invariance of the selected sector; whether combined U(1)+parity is required in the first release; how redundant constraints are reported; and which public mixed-radix ordering and memory contract apply to the selected basis.

Dependencies: a reusable exact-plus-GF(2) constraint model; deterministic row reduction and redundancy detection; direct rank/unrank or bounded fallback indexing for mixed-radix spaces; a common restricted-plan/CSR execution contract; flat native transport; and differential references for basis ordering, conservation, leakage, dense matrices, lazy MVP, eager CSR, and wide systems.

Promotion criteria: one concrete workload that cannot be expressed adequately with the current U(1) or Pauli-tapering APIs; an owner decision separating parity-sector restriction from Pauli Z2 tapering; a bounded first scope, preferably one parity constraint before arbitrary simultaneous constraints; exact small-system and wide-system references; proof that ordinary U(1) paths do not regress; and release-mode end-to-end evidence for setup, lazy steady MVP, eager CSR/steady MVP, memory, and numerical agreement. Promotion must introduce a separate parity-aware indexer or constraint-plan owner and must not expand every existing charge kernel into an explicit U(1)×Z2 variant.

### Fast Block-factorized charge sectors

Status: `deferred` from Phase 10.5 by owner decision D11. Phase 11 implemented compact factorized execution for provably independent charge components; the coupled cross-factor Block fast path described here remains deferred.

Motivation: accelerate finite sectors whose conserved additive charge couples otherwise separable factors and whose Hamiltonian transfers charge between them. Representative applications include Jaynes–Cummings and Tavis–Cummings excitation exchange, exciton–photon polaritons and cavity QED, atom–molecule Feshbach conversion with charges such as `N_f + 2 N_b`, and exchange terms preserving only a combined subsystem spin projection.

Current design direction: represent a coupled sector as a public-order direct sum of Product blocks. Retain bounded prefix offsets over reachable outer-factor states or another bounded factor-level index, reuse the existing FermionMask, BosonLadder, PauliMask, and WeylShift actions inside each block, and compile charge-preserving terms as direct source-block to destination-block schedules. Do not materialize or apply a permutation with one entry per selected product state, and do not introduce model-name dispatch.

Reason for deferral: the current Holstein performance target and ordinary mixed Hamiltonians with independent species charges are Product cases. A complete Block fast path additionally requires cross-block rank/unrank, public mixed-radix ordering, inter-block lazy gather, eager construction, simultaneous constraints, leakage semantics, and source-level-dependent qudit spin-z changes. The existing Generic sector path preserves correctness for these cases while optimized Block execution is absent.

Dependencies: stable Cartesian and Product kernels, shared domain-action lowering, a frozen public-order segmented index, direct-sum dimension and memory accounting, and benchmark workloads whose combined sector materially reduces a real calculation.

Promotion criteria: at least one concrete high-frequency cross-factor conversion workload; owner approval of the exact block index and supported simultaneous-constraint envelope; dense and Generic differentials for within-block and cross-block terms; deterministic public-order rank/unrank tests; lazy, eager, construction-memory, and threading benchmarks; and a faster result than every directly equivalent optimized QuSpin path under matched semantics. If no fair QuSpin representation exists, promotion requires documented internal before/after, throughput, memory, and scaling evidence instead.

### Graph-dependent fermion-to-qubit mappings

Status: `deferred`.

Motivation: reduce Pauli weight or improve locality beyond JW/parity/BK for structured interaction graphs.

Candidates include Bravyi–Kitaev superfast and auxiliary-fermion mappings. These may change qubit count, introduce stabilizer constraints, and depend on graph orientation or auxiliary choices, so they do not fit the Phase 7.5 invertible occupation-encoding plan contract.

Promotion criteria: a representative interaction graph, a target metric, explicit stabilizer/code-space semantics, and evidence that JW/parity/BK are insufficient.

## Seed ideas

### JAX-compatible gradients for parameterized PTMs

Status: `deferred after Phase 12 scope decision`; implementation not started.

Priority: workload-gated follow-up; implementation effort is intentionally not estimated before a concrete use case.

Motivation: allow a PTM whose entries are produced by an upstream JAX parameterization to participate in the same native value-and-gradient contract as parameterized Pauli rotations. This covers differentiable unitary-derived PTMs, parameterized noise channels, and user-defined local linear maps without making the Rust core depend on JAX.

Current decision: Phase 12 retains the existing static one- and two-qubit `PropagationCircuit.ptm()` forward path but does not add runtime PTM values or PTM-entry gradients. Phase 12 instead implements TensorCircuit-compatible common channels and gradients with respect to their small scalar parameter sets. Rust retains one flat scalar-gradient contract, while the high-level circuit facade presents separate angle and channel-gradient vectors. This avoids 16- or 256-entry gradients for every PTM occurrence.

Dependencies for any later reconsideration remain a runtime PTM payload, native PTM-entry VJP read-back, a coarse JAX callback carrying PTM values, explicit frozen-support behavior for exact-zero entries, memory accounting, and independent one-/two-qubit dense references.

Open questions: whether a real channel-learning or system-identification workload needs entry-wise PTM gradients; whether its runtime layout is dense or fixed-sparse; and whether its zero-entry derivative target should follow the current frozen-support contract or a distinct fixed-basis contract.

Promotion criteria: a concrete workload not covered by the Phase 12 scalar-channel API, an owner decision on zero-entry semantics, exact support-stable finite-difference and dense-VJP tests for one- and two-qubit maps, deterministic JAX/concurrency evidence, and a release-mode end-to-end benchmark showing that the additional boundary and gradient size are justified.

### Specialized quadratic-algebra analysis

Status: `seed`.

Quadratic Majorana Hamiltonians correspond to real antisymmetric matrices and admit specialized symmetry, canonical-form, and Lie-closure algorithms. Quadratic boson Hamiltonians similarly connect to symplectic linear algebra. These specialized paths may be more useful and much safer than a universal polynomial Lie engine, but no concrete user workflow has yet selected the required outputs.

### Formal selection rules and operator grading

Status: `seed`.

Additive-charge analysis can later expose operator homogeneity and selection rules, not only Hamiltonian conservation. The frozen Phase 7.5 contract intentionally excludes a public `charge_deltas` API. Possible future outputs include a charge-homogeneous decomposition and rectangular maps between different sectors.

### Model-library layer

Status: `seed`.

Reusable Hubbard, Bose-Hubbard, Holstein, spin-boson, and related model factories could demonstrate the structured algebra and symmetry APIs. The current project deliberately prioritizes lower-level operator infrastructure; model factories should be promoted only when their conventions, boundary conditions, and parameter schemas can be kept small and stable.
