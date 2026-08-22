# Phase 14 Stabilizer-Code Analysis and Decoder-Output Verification Specification

Status: core implementation and review remediation are complete; eager logical construction is selected by owner decision and the eager-versus-lazy A/B is waived for the intended tens-to-thousands-of-qubits workload. The product scope and two central decisions are approved. The approved decisions are that public syndromes preserve the caller's original check order, including redundant checks, and that the first public slice includes batched decoder-correction analysis. This specification does not approve distance search, a decoder, a QEC circuit IR, or subsystem-code semantics.

## 1. Purpose and scope

Phase 14 adds a Rust-native static qubit stabilizer-code analysis layer centered on one immutable public `StabilizerCode`. It covers signed commuting-check validation, deterministic canonicalization, `[[n,k]]` parameters, redundant-check relations, deterministic physical logical X/Z representatives, single and batched syndrome calculation, Pauli-error classification, error-set correctability analysis, and batched verification of caller-supplied decoder corrections.

The intended user workflow is complete but deliberately static: construct a code from Pauli generators or CSS check matrices, inspect its algebraic parameters and one deterministic logical basis, classify physical Pauli errors, and determine whether an external decoder's corrections leave a stabilizer, a logical Pauli, or a residual syndrome. The implementation does not simulate syndrome-extraction circuits and does not choose corrections.

The scope includes asymmetric repetition codes, CSS codes derived from classical binary parity-check matrices, general qubit stabilizer codes such as the five-qubit code, and sparse-in-spirit QLDPC instances supplied as ordinary binary check arrays. It does not add a separate classical encoder/decoder API: a classical parity-check matrix enters as the X-check or Z-check half of a CSS stabilizer code.

The core remains independent of Python, NumPy, TensorCircuit, Stim, PyMatching, and every decoder package. Python owns friendly input normalization, small result records, and explicit materialization; Rust owns all work scaling with checks, qubits, errors, logical rows, or check relations.

## 2. Binding decisions

1. `StabilizerCode` represents a qubit stabilizer code space, not a pure stabilizer state. It allows stabilizer rank `r < n` and is not a subclass, mode, or optional branch of `StabilizerState`.
2. The code retains the caller's complete ordered signed check list, including valid dependent checks. `syndrome()` and every batched syndrome result use exactly that order and therefore have width `nchecks`, not `rank`.
3. Rust separately constructs a deterministic canonical independent basis for rank, span membership, logical quotient, and classification. Internal elimination never replaces or reorders the public check coordinates.
4. A deterministic physical logical X/Z basis is required. It is algebraically correct and stable under the frozen canonicalization policy, but it is not described as minimum-weight, local, geometry-aware, decoder-optimal, or hardware-friendly.
5. `classify()` and `classify_many()` have identical mathematical semantics. The singular method returns one ergonomic record; the batched method performs one coarse native call and returns arrays rather than one Python object per error.
6. A single Pauli error is never labeled `correctable` without context. Correctability is exposed only for a supplied error set or for supplied error/correction pairs.
7. `analyze_corrections()` is part of this phase. It verifies corrections produced elsewhere but never calls, embeds, selects, or configures a decoder.
8. Logical coordinates are defined only for zero-syndrome Paulis. Batched results do not invent decoder-independent logical coordinates for syndrome-visible errors.
9. Public qubit order remains q0-first. A binary check-matrix column `q` and a Pauli code-array column `q` refer to qubit `q`; packed storage keeps qubit `q` at bit `q % 64`.
10. All public results are deterministic. Parallel scheduling, hash iteration, dependent input rows, and equivalent generator orderings must not leak nondeterminism into canonical generators, check relations, logical pairs, conflict witnesses, or batch results.

## 3. Mathematical contract

### 3.1 Signed checks and code parameters

A signed check is `G_i = s_i P_i`, where `P_i` is a phase-free Hermitian `PauliWord` and `s_i` is `+1` or `-1`. The code space is the simultaneous `+1` eigenspace of all supplied `G_i`.

Ignoring global phase, a Pauli word is the binary symplectic vector

```text
p = (x_0, ..., x_{n-1} | z_0, ..., z_{n-1}) in GF(2)^(2n).
```

Checks must commute pairwise. Dependent checks are accepted only when their signed product is `+I`; any supplied or implied `-I` makes the code space empty and construction fails. Identity rows are rejected as invalid checks rather than retained as permanently zero syndrome coordinates.

For `m = nchecks` supplied checks and independent stabilizer rank `r`, the public parameters are

```text
nrelations = m - r
nlogical = k = n - r
code-space dimension = 2**k
```

The rank of a valid commuting qubit stabilizer group cannot exceed `n`. Construction validates this rather than relying on a later null-space failure.

### 3.2 Original check coordinates and relations

For physical Pauli error `E`, the public syndrome is

```text
s_i(E) = <E, P_i>_s,  i = 0, ..., nchecks - 1.
```

Signs do not change commutation and therefore do not change syndrome bits. The bit at public position `i` always corresponds to the caller's input check `i`, including dependent checks.

A check relation is a binary vector `c` of length `nchecks` whose selected signed checks multiply to `+I`. The implementation returns a deterministic basis of the relation space. Every valid public syndrome satisfies the corresponding parity constraints

```text
c dot s(E) = 0 mod 2.
```

For CSS construction, input X-check rows come first and input Z-check rows follow. Syndrome output preserves this concatenated order exactly.

### 3.3 Canonical stabilizer basis and logical pairs

The internal canonical basis uses binary columns ordered as `x_0, ..., x_{n-1}, z_0, ..., z_{n-1}`, ascending pivot search, reduced row-echelon normalization, and deterministic tie-breaking by the complete packed row. Provenance bits track signed products of original checks so canonical signs and relations are exact.

The normalizer is the symplectic centralizer

```text
N(S) = {P : <P, G>_s = 0 for every G in S}.
```

The logical Pauli space is `N(S) / S` and has binary dimension `2k`. Rust computes a deterministic quotient basis and symplectically pairs it into phase-free physical representatives

```text
logical_x[j], logical_z[j],  j = 0, ..., k - 1
```

satisfying

```text
<logical_x[i], logical_z[j]>_s = delta_ij
<logical_x[i], logical_x[j]>_s = 0
<logical_z[i], logical_z[j]>_s = 0.
```

Every logical representative commutes with every stabilizer. Two representatives differing by a stabilizer have the same logical action. Canonical output is required to be invariant under permutation of equivalent input generator rows and under valid stabilizer row operations, even though public syndrome coordinates continue to reflect the original input order.

### 3.4 Error classification and logical coordinates

Every phase-free physical Pauli error has exactly one public classification:

| Condition | `StabilizerErrorKind` | Meaning |
| --- | --- | --- |
| Syndrome is nonzero | `SYNDROME_VISIBLE` | The error moves the code space to another syndrome sector |
| Syndrome is zero and the Pauli belongs to `S` | `STABILIZER` | The error is logically harmless up to global phase |
| Syndrome is zero and the Pauli belongs to `N(S) \ S` | `LOGICAL` | The error is an undetectable nontrivial logical Pauli |

For a zero-syndrome Pauli `R`, logical coordinates `(a,b)` are defined by

```text
R = stabilizer * product_j logical_x[j]**a[j] * logical_z[j]**b[j]
```

up to global phase. Equivalently, `a[j] = <R, logical_z[j]>_s` and `b[j] = <R, logical_x[j]>_s`. A stabilizer has all-zero logical coordinates. No logical coordinates are exposed for `SYNDROME_VISIBLE` rows. This spelling is intentional: the broader QEC notion of error detectability may also treat a zero-syndrome stabilizer action as harmlessly detectable, whereas this enum value means exactly that a syndrome bit is nonzero.

### 3.5 Error-set correctability

For a supplied finite error set `{E_a}`, `analyze_error_set()` applies the stabilizer Knill-Laflamme condition. The set is correctable exactly when every pair satisfies one of:

```text
E_a^dagger E_b belongs to S
```

or

```text
E_a^dagger E_b has nonzero syndrome.
```

If two errors have the same syndrome and differ by a nontrivial logical Pauli, the set is not jointly correctable. The result returns the lexicographically smallest conflicting input pair `(a,b)` with `a < b` and the logical coordinates of `E_a^dagger E_b`. This is a property of the supplied set, not a distance claim and not a statement about errors omitted from the set.

### 3.6 Decoder-correction analysis

For physical errors `E_i` and caller-supplied corrections `C_i`, Rust computes the residual Pauli

```text
R_i = C_i E_i
```

up to irrelevant global phase and classifies `R_i` by the same three-way contract. For Pauli corrections the reverse product order differs at most by global phase and therefore has the same syndrome and logical class, but the public convention is fixed as `correction * error`.

The residual meanings are:

| Residual kind | Decoder interpretation |
| --- | --- |
| `STABILIZER` | Successful correction; the residual is identity on the logical code space |
| `LOGICAL` | Syndrome has been cleared but a logical Pauli remains |
| `SYNDROME_VISIBLE` | The supplied correction leaves a nonzero residual syndrome |

The method verifies decoder output only. It does not infer a correction from a syndrome, score a decoder, sample a noise model, or estimate logical error probability.

## 4. Public Python API

### 4.1 Module and exports

The authoritative module is `tencirpauli.qec`. `StabilizerCode` and `StabilizerErrorKind` are re-exported from the package root. Result records remain importable from `tencirpauli.qec` but are not required as additional root-level convenience exports.

```python
from tencirpauli.qec import (
    CorrectionAnalysis,
    ErrorClassification,
    ErrorClassificationBatch,
    ErrorSetAnalysis,
    LogicalOperatorArrays,
    StabilizerCheckArrays,
    StabilizerCode,
    StabilizerErrorKind,
)
```

### 4.2 Construction from Pauli generators

```python
StabilizerCode.from_generators(
    nqubits,
    generators,
    *,
    signs=None,
    max_bytes=DEFAULT_MAX_BYTES,
)
```

`generators` is a sequence of same-width `PauliWord` values. Explicit `nqubits` permits the valid empty-check code `[[n,n]]` and prevents width inference from becoming a special case. Every generator must have exactly that width. `signs=None` means all `+1`; otherwise `signs` is an integer sequence of equal length containing only `-1` and `+1`.

Construction accepts valid dependent checks and retains their original order. It rejects identity checks, noncommuting rows, incompatible widths, inconsistent signs, and every dependency producing `-I`.

### 4.3 Construction from binary symplectic arrays

```python
StabilizerCode.from_symplectic(
    x,
    z,
    *,
    signs=None,
    max_bytes=DEFAULT_MAX_BYTES,
)
```

`x` and `z` are binary integer or boolean arrays with the same shape `(nchecks, nqubits)`. Row `i` represents the phase-free Pauli with symplectic bits `(x[i], z[i])`; `(1,1)` at one site is Y. Inputs are normalized once to C-contiguous binary arrays and transferred through one native construction call. Floating values are rejected rather than rounded.

The empty-check code is represented by arrays with explicit shape `(0,nqubits)`. Arrays with shape `(0,0)` construct the zero-qubit empty code.

### 4.4 CSS construction and classical repetition checks

```python
StabilizerCode.from_css(
    x_checks,
    z_checks,
    *,
    x_signs=None,
    z_signs=None,
    max_bytes=DEFAULT_MAX_BYTES,
)
```

`x_checks` has shape `(nx,nqubits)` and creates X-only stabilizers; `z_checks` has shape `(nz,nqubits)` and creates Z-only stabilizers. Both are binary integer or boolean arrays and must have the same column count. Construction validates `x_checks @ z_checks.T == 0 mod 2` and then applies the same signed dependency validation as the general constructor.

Input order is frozen as all X-check rows followed by all Z-check rows. The same order is used by `check_arrays()`, `check_relations()`, `syndrome()`, `syndrome_many()`, and classification results.

The three-bit bit-flip repetition code is expressed as

```python
h = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
code = StabilizerCode.from_css(
    x_checks=np.empty((0, 3), dtype=np.uint8),
    z_checks=h,
)
```

This imports the classical repetition parity checks as Z stabilizers and detects X-type bit flips. It does not claim to correct arbitrary single-qubit Pauli errors and does not create a classical decoding object. The phase-flip repetition analogue places `h` in `x_checks` instead.

### 4.5 Metadata and explicit read-back

Every code exposes immutable scalar metadata:

```python
code.nqubits
code.nchecks
code.rank
code.nlogical
code.nrelations
code.qubit_order       # always "q0_first"
code.syndrome_order    # always "input_checks"
```

Two explicit low-cost structural diagnostics describe the supplied check system rather than an optimized or canonical generator set:

```python
code.check_weights() -> np.ndarray
code.qubit_degrees() -> np.ndarray
```

`check_weights()` returns a read-only platform-`uintp` array with shape `(nchecks,)`; entry `i` is the Pauli weight of original input check `i`. `qubit_degrees()` returns a read-only platform-`uintp` array with shape `(nqubits,)`; entry `q` counts the original input checks acting nontrivially on qubit `q`. Valid redundant checks contribute to both diagnostics because they remain part of the caller's supplied check system. Empty axes return correctly shaped empty or all-zero arrays. These methods make no LDPC, locality, geometry, or Tanner-graph claim.

The public packed check read-back is:

```python
@dataclass(frozen=True, slots=True)
class StabilizerCheckArrays:
    x_words: np.ndarray
    z_words: np.ndarray
    signs: np.ndarray
    nqubits: int
    qubit_order: Literal["q0_first"]

code.check_arrays() -> StabilizerCheckArrays
```

With `word_count = ceil(nqubits / 64)`, `x_words` and `z_words` have shape `(nchecks,word_count)` and dtype `uint64`; `signs` has shape `(nchecks,)`, dtype `int8`, and values in `{-1,+1}`. Arrays are owned, C-contiguous, read-only copies and preserve original check order.

The relation basis is explicitly materialized by

```python
code.check_relations(*, packed=False) -> np.ndarray
```

For `packed=False`, the result is read-only `uint8` with shape `(nrelations,nchecks)`. For `packed=True`, it is read-only `uint64` with shape `(nrelations,ceil(nchecks/64))`, with original check zero at the least-significant bit of packed word zero.

Logical read-back is:

```python
logical_x, logical_z = code.logical_operators()
arrays = code.logical_operator_arrays()
```

`logical_operators()` explicitly materializes two tuples of `PauliWord`, each of length `nlogical`. It is intended for ordinary small and medium codes. The scalable flat read-back record is

```python
@dataclass(frozen=True, slots=True)
class LogicalOperatorArrays:
    logical_x_x_words: np.ndarray
    logical_x_z_words: np.ndarray
    logical_z_x_words: np.ndarray
    logical_z_z_words: np.ndarray
    nqubits: int
    qubit_order: Literal["q0_first"]
```

Each packed array has shape `(nlogical,word_count)`, dtype `uint64`, owned C-contiguous storage, and `writeable=False`. Native read-back uses these flat arrays; `logical_operators()` builds Python `PauliWord` values only after the caller explicitly requests them.

### 4.6 Syndrome API

```python
code.syndrome(error: PauliWord) -> tuple[int, ...]

code.syndrome_many(
    errors,
    *,
    packed=False,
    max_bytes=DEFAULT_MAX_BYTES,
) -> np.ndarray
```

The singular result always has length `nchecks` in original input order. `syndrome_many()` accepts either a sequence of same-width `PauliWord` values or a two-dimensional binary Pauli-code array with shape `(batch,nqubits)` and entries `0=I, 1=X, 2=Y, 3=Z`. A NumPy batch is the preferred scalable input; invalid dtypes are rejected rather than silently rounded.

For `packed=False`, the batch result has shape `(batch,nchecks)` and dtype `uint8`. For `packed=True`, it has shape `(batch,ceil(nchecks/64))` and dtype `uint64`, with check zero at the least-significant bit of word zero. Empty batches retain the corresponding two-dimensional shape. Returned arrays are owned, C-contiguous, and read-only.

### 4.7 Error classification API

```python
@enum.unique
class StabilizerErrorKind(enum.IntEnum):
    SYNDROME_VISIBLE = 0
    STABILIZER = 1
    LOGICAL = 2
```

```python
@dataclass(frozen=True, slots=True)
class ErrorClassification:
    syndrome: tuple[int, ...]
    kind: StabilizerErrorKind
    logical_x: tuple[int, ...] | None
    logical_z: tuple[int, ...] | None
```

```python
code.classify(error: PauliWord) -> ErrorClassification
```

For `SYNDROME_VISIBLE`, both logical fields are `None`. For `STABILIZER` or `LOGICAL`, they are binary tuples of length `nlogical`; stabilizers have all-zero tuples.

The batched path is

```python
code.classify_many(
    errors,
    *,
    packed_syndromes=False,
    max_bytes=DEFAULT_MAX_BYTES,
) -> ErrorClassificationBatch
```

```python
@dataclass(frozen=True, slots=True)
class ErrorClassificationBatch:
    syndromes: np.ndarray
    kinds: np.ndarray
    zero_syndrome_indices: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    syndromes_packed: bool
```

`syndromes` follows the shapes and dtypes of `syndrome_many()`. `kinds` is read-only `uint8[batch]` using the enum codes. `zero_syndrome_indices` is an ascending read-only platform-`uintp` vector containing exactly the batch rows whose syndrome is zero. `logical_x` and `logical_z` are read-only `uint8` arrays with shape `(len(zero_syndrome_indices),nlogical)` and give coordinates for those rows in the same order. This indexed representation avoids placeholder coordinates for syndrome-visible errors.

The singular method reuses the same native classification semantics; there is no independent Python reference implementation. The batch endpoint performs one native call and never creates one Python result object per error.

### 4.8 Error-set analysis

```python
@dataclass(frozen=True, slots=True)
class ErrorSetAnalysis:
    is_correctable: bool
    conflicting_pair: tuple[int, int] | None
    logical_x: tuple[int, ...] | None
    logical_z: tuple[int, ...] | None
```

```python
code.analyze_error_set(
    errors,
    *,
    max_bytes=DEFAULT_MAX_BYTES,
) -> ErrorSetAnalysis
```

For a correctable set, the witness fields are `None`. For an uncorrectable set, `conflicting_pair` is the deterministic lexicographically smallest conflicting pair and `logical_x/logical_z` describe their nontrivial logical difference. Empty and singleton sets are correctable.

The implementation groups errors by original-coordinate syndrome and compares logical cosets within each group; it must not perform a naive Python `O(batch**2)` pair loop. Any native worst-case work and retained hash tables remain subject to checked allocation and `max_bytes`.

### 4.9 Decoder-correction analysis

```python
code.analyze_corrections(
    errors,
    corrections,
    *,
    packed_syndromes=False,
    max_bytes=DEFAULT_MAX_BYTES,
) -> CorrectionAnalysis
```

`errors` and `corrections` accept the same batch forms as `classify_many()`, must have equal batch size, and must match the code width. One coarse native call multiplies each pair modulo global phase, computes residual syndromes, classifies residuals, and computes logical coordinates for zero-syndrome residuals.

```python
@dataclass(frozen=True, slots=True)
class CorrectionAnalysis:
    residual_syndromes: np.ndarray
    kinds: np.ndarray
    zero_syndrome_indices: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    syndromes_packed: bool
```

The array contracts match `ErrorClassificationBatch`, with `zero_syndrome_indices` selecting zero-syndrome residual rows. Callers derive success, logical-failure, and residual-syndrome masks by comparing `kinds` with the three `StabilizerErrorKind` values; no duplicate native mask outputs are returned.

For the three-bit repetition code, `error=IXX` and `correction=XII` have the same syndrome but leave residual `XXX`; the required result is `LOGICAL` with logical-X coordinate one. `error=XII` and `correction=XII` leave identity and return `STABILIZER` with all-zero logical coordinates.

## 5. Core representation and algorithms

### 5.1 Shared packed GF(2) primitives

Before implementing code-specific algorithms, the Rust core must identify and reuse the elimination, null-space, symplectic, and span-solving primitives already present in `symmetry.rs` and the stabilizer portions of `shadows.rs`. The implementation must not add a third silently divergent row-reduction convention.

A narrow internal packed-GF(2) module may be extracted when the contracts genuinely match. At minimum, Phase 14 migrates the compatible Z2-symmetry helpers to that shared primitive and proves unchanged symmetry results. Specialized shadow tableau or sampling routines remain separate when their signed-row, coefficient-provenance, or hot-loop representation differs; this phase does not refactor them merely to reduce line count.

The shared primitive must support checked packed dimensions, deterministic pivot order, rank, reduced basis construction, null-space basis, row-space solve with coefficients, provenance tracking, and stable relation output. It is an internal implementation detail rather than a public general-purpose binary-matrix API.

### 5.2 Module ownership

The implementation uses the following responsibility split. Exact private type and function names may follow Rust conventions and do not require a second specification, but ownership must not drift across these boundaries.

| Module | Authoritative responsibility |
| --- | --- |
| `crates/tencir-pauli-core/src/gf2.rs` or an equivalently narrow private module | Packed GF(2) row reduction, null space, relation basis, provenance, and reusable span solving shared by compatible symmetry and QEC paths |
| `crates/tencir-pauli-core/src/stabilizer_code.rs` | Signed-check validation, original/canonical check representations, logical quotient and pairing, syndrome, classification, error-set analysis, correction analysis, and structural check statistics |
| `crates/tencirpauli-native/src/qec.rs` | One immutable native code handle, coarse PyO3 construction/execution calls, GIL release, checked flat-array conversion, and NumPy read-back |
| `python/tencirpauli/qec.py` | Public `StabilizerCode`, enums and frozen result records, friendly scalar/shape validation, and explicit materialization only |
| `tests/test_qec.py`, Rust core tests, and QEC benchmark sources | Independent references, public/native ABI contracts, deterministic regression vectors, and release-mode performance evidence |

`symmetry.rs` consumes the shared packed-GF(2) primitives where its contract matches. `shadows.rs` keeps its specialized tableau, signed-span, and sampling representation unless profiling and correctness evidence show that a particular primitive can be shared without changing its hot path. No QEC semantics are added to `StabilizerState`, `GateTape`, propagation, or symmetry public types.

### 5.3 Key algorithm flow

Construction first packs and validates the original ordered signed checks, then performs deterministic GF(2) elimination with provenance. Elimination produces rank, a canonical independent stabilizer basis, signed dependency validation, and a relation basis in original-check coordinates.

Logical construction forms the symplectic centralizer null space, quotients out the canonical stabilizer row space, and applies deterministic symplectic pairing to obtain `nlogical` logical X/Z pairs. The fixed pivot and tie-breaking conventions in Section 3.3, together with the required regression vectors, define the public deterministic output; the implementation need not expose elimination steps or a general binary-matrix API.

Batch syndrome evaluates packed symplectic parities against original checks. Classification first separates nonzero syndromes, then uses packed logical coordinates for zero-syndrome rows; the quotient invariant makes an all-zero coordinate vector equivalent to stabilizer membership, so no retained stabilizer span solver is needed on the ordinary classification path. Correction analysis XOR-multiplies each correction/error pair modulo global phase and feeds the residual into the same classification kernel. Error-set analysis groups by original-coordinate syndrome and compares stabilizer/logical cosets within each group, rather than enumerating every pair in Python.

### 5.4 `StabilizerCode` native state

The immutable core object retains only one authoritative representation of each required runtime concept:

- Original ordered signed checks in packed symplectic form.
- The checked rank of the canonical independent signed stabilizer basis; signed canonical rows and provenance are construction workspaces, not retained speculative state.
- A deterministic relation basis over original checks.
- Deterministic logical X/Z pairs and the packed data required for logical coordinates.

Construction validates the complete untrusted input once. Trusted native handles are not rescanned on every method call. The current implementation selects eager logical construction: classification and explicit logical read-back use one packed logical representation, and the post-remediation construction profile showed that removing repeated RREF and pair-span work materially reduced the finite-rate logical-basis cost without adding a synchronized runtime cache. The owner waives a matched eager-versus-lazy A/B for the intended tens-to-thousands-of-qubits static workflow because lazy construction would add synchronization and first-access state without a demonstrated product need; a lazy alternative is not retained as a runtime mode. In either design, the handle must never retain two independently computed logical bases.

### 5.5 Batch kernels

Syndrome computation is a packed symplectic matrix-by-batch parity kernel. Classification adds a zero-syndrome test and packed logical coordinate evaluation; an all-zero coordinate vector is the stabilizer case by the quotient invariant. Correction analysis fuses pairwise Pauli XOR with the same classification path and does not materialize residual `PauliWord` objects in Python.

Batch work is parallelized only after release-mode profiling identifies a representative gain. If Rayon is used, output row `i` always corresponds to input row `i`, zero-syndrome indices are produced in ascending input order, and results are bitwise identical across worker counts.

`analyze_error_set()` uses deterministic syndrome grouping and logical-coset comparison. It reports the lexicographically smallest conflict independently of hash iteration or worker scheduling.

## 6. FFI, ownership, and memory

`StabilizerCode` is a thin immutable Python facade over one private native handle. No parallel Python generator list, canonical basis, or logical basis is retained after native construction. Scalar metadata is cached only when it is cheap and immutable.

Every construction, batch syndrome, batch classification, error-set analysis, and correction analysis uses one coarse native call. Work scaling with check rows, input rows, qubits, relations, or logicals runs while the GIL is released, including Rust-side packing or validation of already normalized NumPy buffers where safe borrowing permits it.

Native read-back returns only flat NumPy arrays and scalar metadata. Python may build `PauliWord` values from flat logical arrays only for the explicit `logical_operators()` request. No nested generator lists cross the native ABI, and Python never materializes a native result merely to feed it back into another native method.

Returned arrays are owned C-contiguous copies with `writeable=False` and remain valid after the code handle is released. Native handles do not retain caller-owned NumPy buffers. Code handles and cached immutable analysis are safe for concurrent calls; all batch output and scratch are call-local.

All dimensions, packed word counts, batch products, relation storage, logical storage, and output byte estimates use checked arithmetic. `max_bytes` is an advisory best-effort guard: cheaply known retained payloads and selected outputs are checked, difficult transient allocations may be underestimated, and ordinary system-memory exhaustion remains possible. The implementation must not sum incompatible lifetime maxima or add costly accounting that rejects otherwise runnable work merely because of conservative overestimation.

## 7. Validation and failure behavior

- `nqubits`, row counts, batch counts, packed word counts, and allocation sizes use checked arithmetic.
- Qubit counts and array dimensions must be non-negative integers; booleans are not accepted as integers.
- Binary check arrays accept boolean or integer entries exactly equal to zero or one. Pauli-code batches accept integer entries exactly in `0..3`. Floating arrays are rejected rather than rounded.
- Generator widths and batch widths must exactly equal `code.nqubits`.
- Signs must be integer `-1` or `+1`; zero, booleans, and other values fail.
- Identity checks fail. Valid dependent nonidentity checks are retained; signed dependencies producing `-I` fail.
- General checks must commute pairwise. CSS construction additionally reports an explicit X/Z orthogonality error when `H_X H_Z^T != 0 mod 2`.
- Batch error and correction lengths must agree. Empty batches are valid and return correctly shaped empty arrays.
- `packed=True` and `packed_syndromes=True` control output representation only; they do not alter semantics.
- No method accepts a `StabilizerState` as though it were a code, and no `k>0` code is silently converted to a state.

Wrong Python object kinds and wrong array dtypes raise `TypeError`; invalid signs, codes, shapes, widths, commutation, identity checks, dependency consistency, or paired batch lengths raise `ValueError`; checked arithmetic overflow raises `OverflowError`; enforced resource limits raise `MemoryError`. Tests assert exception class and a stable message fragment.

## 8. Correctness tests

Required independent and deterministic tests include:

- Three-bit bit-flip repetition code from both Pauli generators and a CSS Z-check matrix: original-order syndromes, redundant-check relation, the three single-X syndromes, stabilizer classification, logical X/Z classification, error-set ambiguity, and correction residuals.
- Three-bit phase-flip repetition code from CSS X checks, with X/Z roles exchanged.
- Five-qubit perfect code: non-CSS commutation, rank four, one logical qubit, canonical logical pairing, all weight-one Pauli errors syndrome-visible, and known distance-three witnesses used only as fixed classifications rather than a distance claim.
- Steane code: CSS orthogonality, rank six, one logical qubit, deterministic logical pairing, dependent-check variants, and classification of representative X-, Z-, and Y-type errors.
- Shor code or another degenerate code: a nonidentity stabilizer error is classified as `STABILIZER`, not `LOGICAL`.
- Empty-check `[[n,n]]` and full-rank `[[n,0]]` boundary cases, including zero qubits.
- Reordered and row-operated representations of the same stabilizer group: canonical logical arrays are identical, while public syndrome columns follow each input's own check order.
- Signed dependent rows: products yielding `+I` construct successfully and products yielding `-I` fail.
- `check_weights()` and `qubit_degrees()` match direct counts on general, CSS, redundant-check, empty-check, and widths-crossing-64-bit examples; the values refer to original checks rather than canonical generators.
- Singular `syndrome()`/`classify()` agree exactly with every corresponding row of the batched methods for random valid small codes and errors.
- Unpacked and packed syndrome outputs round-trip exactly, including widths crossing 64-bit boundaries.
- `analyze_error_set()` agrees with an independent pairwise small-system oracle and returns the required lexicographically smallest witness.
- `analyze_corrections()` agrees with explicit `PauliWord.multiply()` followed by singular classification for random batches.
- Every reported logical pair commutes with stabilizers and satisfies the canonical symplectic relations. Multiplying a zero-syndrome error by any stabilizer leaves its logical coordinates unchanged.
- A dense small-system projector oracle checks signed code-space consistency and representative stabilizer/logical action without using the production GF(2) routines.
- Serial and parallel worker counts produce bitwise-identical arrays, relation bases, logical pairs, and conflict witnesses.

Numeric tests must assert known syndrome bits, kind codes, relation bits, and logical coordinates; shape-only or replay-only assertions do not satisfy correctness coverage.

## 9. Performance benchmarks

Release-mode benchmarks cover code construction, canonicalization, logical-basis construction, relation and structural-statistics extraction, batch syndrome, batch classification, error-set grouping, and correction analysis. The committed QEC acceptance matrix is deliberately compact: it uses representative `n={64,256,1024}` points for small, medium, and large behavior, includes independent, CSS, general, and redundantly checked inputs, and does not repeat every operation for every family and width. The complete Rust plus Python/FFI QEC record is budgeted and measured below ten minutes; the default profile uses bounded per-row sampling, while `--quick` provides a shorter daily regression profile with the same semantic coverage.

Python integration benchmarks include input normalization and the Python/Rust boundary. They record and verify `tencirpauli.__file__` and `tencirpauli._native.__file__` before making any performance claim. Representative rows retain exact output equality or `max_abs_error=0` metadata outside the timed region. A kernel-only Rust benchmark does not establish end-to-end speed.

For each workload, record setup time, first call, steady call, errors per second, check-error parity throughput where applicable, cheaply known output bytes, advisory resource-limit inputs, output representation, worker count, and output materialization cost. Exact retained bytes, exact peak workspace, and peak RSS are not acceptance conditions because `max_bytes` is advisory and difficult transient allocations may be underestimated. Packed and unpacked syndrome output are measured separately because unpacking may dominate narrow algebra kernels. The eager-versus-lazy logical-basis comparison is waived by owner decision for this static target scale and is not a permanent user-facing mode.

The comparison baseline is an independent, vectorized NumPy/GF(2) reference for small and medium cases and any applicable external code-algebra routine with exactly matched check order and classification semantics. Stim circuit-sampling speed and decoder thresholds are not applicable baselines for this static code-algebra slice.

No wall-time CI gate is added. Benchmarks remain local and informational, but every material optimization retains a stable Rust microbenchmark and a Python/FFI integration benchmark.

## 10. Documentation and examples

Public documentation must distinguish a `StabilizerCode` from `StabilizerState`, explain that the former represents a `2**k`-dimensional code space, and state that canonical logical representatives are not minimum-weight representatives.

The first example uses the three-bit bit-flip repetition code to show CSS construction, ordered syndromes, `classify()` versus `classify_many()`, an uncorrectable supplied error pair with the same syndrome, and the decoder failure `IXX` corrected by `XII`, leaving logical `XXX`.

A second example uses a non-CSS five-qubit code and verifies the returned logical pair. A third compact example shows an external decoder boundary: the decoder receives `syndrome_many()` output and returns Pauli corrections, while `analyze_corrections()` evaluates residual logical outcomes without importing the decoder into Rust.

Documentation must say that a classical parity-check matrix can be used as one CSS half but that TenCirPauli does not provide a general classical code or classical decoder API in this phase.

## 11. Non-goals

This phase does not include:

- Exact, exhaustive, bounded, heuristic, CSS-specialized, or meet-in-the-middle code-distance search; check weights and qubit degrees are structural diagnostics and must not be presented as distance evidence.
- A `distance` property inferred from the returned canonical logical rows.
- Minimum-weight, local, geometric, homological, hardware-routed, or decoder-optimal logical representatives.
- Syndrome-to-correction lookup tables, MWPM, union-find, BP, OSD, neural, or any other decoder.
- Noise-model enumeration, logical-channel aggregation, Monte Carlo logical error rates, or confidence intervals.
- Encoder/decoder Clifford circuit synthesis, encoder verification, or physical Clifford logical-action analysis.
- Reset, measurement, detector, observable, repeat-round, fault-location, or classical-control circuit semantics.
- Stim detector error models, PyMatching graph construction, or external decoder serialization formats.
- Subsystem/gauge codes, bare or dressed logicals, gauge fixing, Bacon-Shor gauge semantics, or lattice surgery.
- Non-additive, approximate, bosonic, GKP, qudit, or generalized-Weyl QEC.
- A standalone public GF(2), classical linear-code, Tanner-graph, or graph-optimization library.
- Automatic conversion between `StabilizerCode` and `StabilizerState`.

Any later distance API must return lower bound, upper bound, completion status, witness, and search provenance. Any later circuit/fault API requires a separate narrow QEC circuit IR and must not extend the existing propagation `GateTape` with dynamic measurement semantics.

## 12. Implementation order and acceptance

Implementation proceeds in dependency order without retaining abandoned parallel representations:

1. Freeze independent Python reference vectors for repetition, five-qubit, Steane, signed dependencies, error-set conflicts, and correction residuals.
2. Extract or expose the minimal shared packed-GF(2) primitives and migrate the compatible Z2 symmetry code with unchanged tests and release performance.
3. Implement the core immutable code representation, validation, relations, canonical logical pairs, and singular algebra tests.
4. Implement fused batch syndrome, classification, error-set, and correction kernels with deterministic outputs and resource guards.
5. Add one private PyO3 handle, flat NumPy ABI, exact `_native.pyi` types, and GIL-released coarse calls.
6. Add `python/tencirpauli/qec.py`, root exports, public docstrings, examples, benchmarks, and documentation.
7. Rebuild the source-tree native extension in the exact test environment and run the complete quality and integration matrix.

Acceptance requires:

- Every constructor, property, return type, dtype, shape, ownership rule, error class, and ordering contract in this specification is covered by focused tests.
- The independent numerical/algebraic matrix in Section 8 passes, including dense projector references and batch-versus-singular differential tests.
- Existing Z2 symmetry and classical-shadow behavior remains unchanged; no performance claim is made until the rebuilt source-tree extension paths are recorded.
- `cargo fmt --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `cargo test --workspace` pass.
- `black --check python tests benchmarks scripts examples`, `ruff check python tests benchmarks scripts examples`, strict `mypy`, and the complete `pytest` suite pass.
- `python scripts/check.py` passes in the project environment.
- Release-mode benchmark sources cover the representative construction, syndrome, classification, and correction workloads, with no unexplained material regression in existing symmetry paths touched by GF(2) reuse.

The public/core capability is implemented and the second-round remediation evidence is recorded in `docs/vibe/reports/phase-14-second-round-remediation-2026-08-22.md`. The owner-selected eager design is accepted for this scope; the capability is not an unimplemented specification.
