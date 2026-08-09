"""Private structured-operator implementation module."""

from __future__ import annotations

# The modules share the native-handle boundary in ``base``.  Some imports are
# intentionally consumed by dynamically resolved cross-domain methods after
# the package has completed its acyclic construction order.
# ruff: noqa: F401, I001

import cmath
import math
import numbers
import struct
from dataclasses import dataclass, replace
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from .. import _native
from .._validation import normalize_pauli_code, validate_nonnegative_int
from ..hamiltonian import (
    _PLAN_FACTORY_TOKEN,
    DEFAULT_MAX_BYTES,
    DIRECT_WEYL_BASIS_ORDERING,
    MIXED_RADIX_BASIS_ORDERING,
    BackendMVPPlan,
    CompileResult,
    COOMatrix,
    CSRMatrix,
    NativeMVPPlan,
    _check_allocation,
    _effective_max_bytes,
    _validate_max_bytes,
)
from ..pauli import _PAULI_CHAR_TO_CODE, PauliOperator
from .base import (
    MappedOperator,
    StructuredNativeHandle,
    _IDENTITY_CODES,
    _PAULI_PRODUCT,
    _U32_MAX,
    _action,
    _add_native_handles,
    _adjoint_native_handle,
    _boson_arrays,
    _boson_from_native,
    _complex_sort_key,
    _estimate_terms,
    _factor,
    _fermion_arrays,
    _fermion_from_integral_blocks,
    _fermion_from_native,
    _finite_complex,
    _guard_terms,
    _hybrid_arrays,
    _hybrid_from_native,
    _native_boson_raw,
    _native_fermion_raw,
    _native_result_class,
    _nonnegative_int,
    _positive_mode,
    _prepare_integral_array,
    _prepare_integral_constant,
    _scale_native_handle,
    _terms_from_native_handle,
    _StructuredOperator,
    _Term,
)
from .words import (
    BosonTerm,
    BosonWord,
    FermionTerm,
    FermionWord,
    QuditProduct,
    QuditWeylTerm,
    QuditWeylWord,
)

if TYPE_CHECKING:
    from ..charge import (
        AdditiveCharge,
        AdditiveSymmetryAnalysis,
        ChargeRestrictedOperator,
        ChargeSector,
        ChargeStorage,
    )
    from ..majorana import MajoranaOperator
    from ..mapping import FermionQubitMapping

from .space import OperatorSpace, _Axis

if TYPE_CHECKING:
    from .boson import BosonOperator
    from .fermion import FermionOperator
    from .hybrid import HybridOperator


def _replace(term: _Term, **changes: Any) -> _Term:
    values: Dict[str, Any] = {
        "fermion": term.fermion,
        "boson": term.boson,
        "qubit": term.qubit,
        "qudit": term.qudit,
        "mapped_fermion": term.mapped_fermion,
        "coefficient": term.coefficient,
    }
    values.update(changes)
    return _Term(
        cast(Optional[FermionWord], values["fermion"]),
        cast(Optional[BosonWord], values["boson"]),
        cast(Tuple[int, ...], values["qubit"]),
        cast(Optional[QuditWeylWord], values["qudit"]),
        cast(Optional[Tuple[int, ...]], values["mapped_fermion"]),
        cast(complex, values["coefficient"]),
    )


def _with_plan_metadata(
    plan: Union[NativeMVPPlan, BackendMVPPlan],
    *,
    mapping: Optional[str],
    source_term_count: int,
) -> CompileResult:
    """Copy a reusable plan while preserving its private native handle."""
    if isinstance(plan, NativeMVPPlan):
        return NativeMVPPlan(
            plan.nqubits,
            plan.term_count,
            plan._native_plan,
            storage=plan.storage,
            local_dimensions=plan.local_dimensions,
            basis_ordering=plan.basis_ordering,
            estimated_bytes=plan.estimated_bytes,
            schema_version=plan.schema_version,
            target=plan.target,
            source_term_count=source_term_count,
            plan_term_count=plan.plan_term_count,
            mapping=mapping,
            boson_cutoffs=plan.boson_cutoffs,
            boson_boundary=plan.boson_boundary,
            qudit_dimension=plan.qudit_dimension,
            weyl_convention=plan.weyl_convention,
            execution_family=plan.execution_family,
            retained_format=plan.retained_format,
            schedule_count=plan.schedule_count,
            estimated_retained_bytes=plan.estimated_retained_bytes,
            estimated_construction_peak_bytes=plan.estimated_construction_peak_bytes,
            transition_count=plan.transition_count,
            _factory_token=_PLAN_FACTORY_TOKEN,
        )
    return replace(
        plan,
        mapping=mapping,
        source_term_count=source_term_count,
    )


def _mapping_name(mapping: Union[str, "FermionQubitMapping"]) -> str:
    if isinstance(mapping, str):
        return mapping
    name = mapping.name
    if not isinstance(name, str):
        raise TypeError("mapping plan has an invalid name")
    return name


def _make_operator(
    space: OperatorSpace, terms: Iterable[_Term], max_bytes: Optional[int]
) -> "_StructuredOperator":
    # A domain-specific object is retained for low-level one-domain spaces.
    terms = tuple(terms)
    if space.fermions and space.bosons == space.qubits == len(space.qudits) == 0:
        return FermionOperator._from_terms(space, terms, max_bytes)
    if space.bosons and space.fermions == space.qubits == len(space.qudits) == 0:
        return BosonOperator._from_terms(space, terms, max_bytes)
    if len(space.qudits) and space.fermions == space.bosons == space.qubits == 0:
        return QuditWeylOperator._from_terms(space, terms, max_bytes)
    return HybridOperator._from_terms(space, terms, max_bytes)


def _combine_nonexpanding_terms(
    left: _Term, right: _Term, space: OperatorSpace
) -> _Term:
    coefficient = left.coefficient * right.coefficient
    qubit = []
    for a, b in zip(left.qubit, right.qubit):
        code, phase = _PAULI_PRODUCT[a][b]
        qubit.append(code)
        coefficient *= phase
    fword: Optional[FermionWord] = None
    if left.fermion is not None or right.fermion is not None:
        lf = left.fermion or FermionWord(space.fermions)
        rf = right.fermion or FermionWord(space.fermions)
        fword = FermionWord(
            space.fermions,
            lf.creation_modes + rf.creation_modes,
            rf.annihilation_modes + lf.annihilation_modes,
        )
        if (len(lf.annihilation_modes) * rf.parity) & 1:
            coefficient = -coefficient
    bword: Optional[BosonWord] = None
    if left.boson is not None or right.boson is not None:
        lb = left.boson or BosonWord(space.bosons)
        rb = right.boson or BosonWord(space.bosons)
        bword = BosonWord(space.bosons, lb.blocks + rb.blocks)
    qudit = left.qudit or right.qudit
    if left.qudit is not None and right.qudit is not None:
        assert left.qudit is not None and right.qudit is not None
        product_result = left.qudit.multiply(right.qudit)
        qudit = product_result.word
        coefficient *= cmath.exp(
            2j * math.pi * product_result.phase_exponent / product_result.word.dimension
        )
    mapped, mapped_phase = _multiply_mapped(left.mapped_fermion, right.mapped_fermion)[
        0
    ]
    coefficient *= mapped_phase
    return _Term(fword, bword, tuple(qubit), qudit, mapped, coefficient)


def _shift_term(
    term: _Term,
    offsets: Mapping[str, int],
    space: OperatorSpace,
    source_space: OperatorSpace,
) -> _Term:
    fermion = (
        None
        if term.fermion is None
        else FermionWord(
            space.fermions,
            tuple(mode + offsets["fermion"] for mode in term.fermion.creation_modes),
            tuple(
                mode + offsets["fermion"] for mode in term.fermion.annihilation_modes
            ),
        )
    )
    boson = (
        None
        if term.boson is None
        else BosonWord(
            space.bosons,
            tuple(
                (mode + offsets["boson"], create, annihilate)
                for mode, create, annihilate in term.boson.blocks
            ),
        )
    )
    qubit_values = [0] * space.qubits
    for index, code in enumerate(term.qubit):
        if code:
            qubit_values[index + offsets["qubit"]] = code
    qubit = tuple(qubit_values)
    qudit = (
        None
        if term.qudit is None
        else QuditWeylWord(
            term.qudit.dimension,
            tuple((site + offsets["qudit"], a, b) for site, a, b in term.qudit.triples),
        )
    )
    if term.mapped_fermion is None:
        mapped = None
    else:
        mapped_values = [0] * space.fermions
        for index, code in enumerate(term.mapped_fermion):
            if code:
                mapped_values[index + offsets["fermion"]] = code
        mapped = tuple(mapped_values)
    return _Term(fermion, boson, qubit, qudit, mapped, term.coefficient)


def _multiply_mapped(
    left: Optional[Tuple[int, ...]], right: Optional[Tuple[int, ...]]
) -> Tuple[Tuple[Optional[Tuple[int, ...]], complex], ...]:
    if left is None and right is None:
        return ((None, 1.0 + 0j),)
    if left is None:
        return ((right, 1.0 + 0j),)
    if right is None:
        return ((left, 1.0 + 0j),)
    codes = []
    phase = 1.0 + 0j
    for a, b in zip(left, right):
        code, local_phase = _PAULI_PRODUCT[a][b]
        codes.append(code)
        phase *= local_phase
    return ((tuple(codes), phase),)


def _term_has_raw_and_mapped_fermions(term: _Term) -> bool:
    """Whether a term has an ambiguous raw/mapped fermion product."""
    return term.fermion is not None and term.mapped_fermion is not None


def _requires_eager_fermion_mapping(
    left: "HybridOperator", right: "HybridOperator"
) -> bool:
    """Detect a product whose operands do not share one fermion encoding."""
    left_handle = left._native_handle
    right_handle = right._native_handle
    if not isinstance(
        left_handle, _native.NativeHybridOperatorHandle
    ) or not isinstance(right_handle, _native.NativeHybridOperatorHandle):
        raise RuntimeError("hybrid operators must retain native handles")
    return (left_handle.has_raw_fermions() and right_handle.has_mapped_fermions()) or (
        left_handle.has_mapped_fermions() and right_handle.has_raw_fermions()
    )


def _tensor_mapped_fermion_terms(
    left: _Term,
    right: _Term,
    space: OperatorSpace,
    left_fermion_count: int,
    right_is_mapped: bool,
) -> Tuple[_Term, ...]:
    """Map a fermionic tensor product using the combined global JW order.

    A raw right factor is mapped with its already shifted global mode indices,
    which naturally supplies the parity string over all left modes.  A mapped
    right factor has a local representation, so the same parity string is
    inserted explicitly according to its fermionic parity.
    """
    left_expansion = (
        _jordan_wigner_word(left.fermion)
        if left.fermion is not None
        else ((left.mapped_fermion or (0,) * space.fermions, 1.0 + 0j),)
    )
    right_expansion = (
        _jordan_wigner_word(right.fermion)
        if right.fermion is not None
        else ((right.mapped_fermion or (0,) * space.fermions, 1.0 + 0j),)
    )
    left_nonfermion = _replace(left, fermion=None, mapped_fermion=None)
    right_nonfermion = _replace(right, fermion=None, mapped_fermion=None)
    base = _combine_nonexpanding_terms(left_nonfermion, right_nonfermion, space)
    output: List[_Term] = []
    for left_codes, left_coefficient in left_expansion:
        for right_codes, right_coefficient in right_expansion:
            corrected_right = list(right_codes)
            if right_is_mapped and _pauli_fermion_parity(right_codes):
                for mode in range(left_fermion_count):
                    code, phase = _PAULI_PRODUCT[3][corrected_right[mode]]
                    corrected_right[mode] = code
                    right_coefficient *= phase
            mapped_codes, phase = _multiply_mapped(
                tuple(left_codes), tuple(corrected_right)
            )[0]
            assert mapped_codes is not None
            output.append(
                _replace(
                    base,
                    mapped_fermion=mapped_codes,
                    coefficient=base.coefficient
                    * left_coefficient
                    * right_coefficient
                    * phase,
                )
            )
    return tuple(output)


def _pauli_fermion_parity(codes: Sequence[int]) -> int:
    """Return the fermionic parity encoded by a mapped Pauli word."""
    return sum(code in (1, 2) for code in codes) & 1


def _jordan_wigner_word(
    word: FermionWord,
) -> Tuple[Tuple[Tuple[int, ...], complex], ...]:
    # Keep this tensor-product adapter aligned with Rust's
    # `jordan_wigner_word_expansion`: X=1/2, Y=+i/2 for annihilation and
    # -i/2 for creation, with Z on modes below the active mode.
    current: Dict[Tuple[int, ...], complex] = {(0,) * word.n_modes: 1.0 + 0j}
    for mode, action in word.factors:
        factor: Dict[Tuple[int, ...], complex] = {}
        for code, coefficient in (
            (1, 0.5),
            (2, 0.5j if action == "annihilate" else -0.5j),
        ):
            codes = [0] * word.n_modes
            for lower in range(mode):
                codes[lower] = 3
            codes[mode] = code
            factor[tuple(codes)] = coefficient
        updated: Dict[Tuple[int, ...], complex] = {}
        for left, left_value in current.items():
            for right, right_value in factor.items():
                result = [0] * word.n_modes
                phase = 1.0 + 0j
                for index, (a, b) in enumerate(zip(left, right)):
                    result[index], local_phase = _PAULI_PRODUCT[a][b]
                    phase *= local_phase
                key = tuple(result)
                updated[key] = updated.get(key, 0j) + left_value * right_value * phase
        current = {
            key: value
            for key, value in updated.items()
            if value.real != 0 or value.imag != 0
        }
    return tuple(sorted(current.items()))


def _pauli_axis_descriptor(space: OperatorSpace) -> List[Tuple[int, int]]:
    """Encode ordered fermion/qubit axes for native Hybrid-to-Pauli lowering."""
    domains = {"fermion": 0, "qubit": 1}
    if any(axis.domain not in domains for axis in space._axes):
        raise ValueError("Pauli projection requires only fermion and qubit axes")
    return [(domains[axis.domain], axis.index) for axis in space._axes]


class QuditWeylOperator(_StructuredOperator):
    """Immutable uniform-dimension direct-convention Weyl operator.

    Factors use ``X^a Z^b`` with exponents reduced modulo the common local
    dimension. Matrix targets use deterministic qudit-zero-MSB mixed-radix
    ordering.
    """

    _domain = "qudit"

    @classmethod
    def from_terms(
        cls,
        dimension: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, int, int]], complex]],
        *,
        n_sites: Optional[int] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "QuditWeylOperator":
        """Construct a modular ``X^a Z^b`` operator and aggregate phases."""
        dimension = _nonnegative_int(dimension, "dimension")
        if not 3 <= dimension <= _U32_MAX:
            raise ValueError("qudit dimension must satisfy 3 <= dimension <= 2**32-1")
        input_terms = tuple(terms)
        raw_triples: List[List[Tuple[int, int, int]]] = []
        raw_coefficients: List[complex] = []
        max_site = -1
        for factors, coefficient in input_terms:
            value = _finite_complex(coefficient)
            current: Dict[int, Tuple[int, int]] = {}
            phase = 0
            for factor in factors:
                if not isinstance(factor, (tuple, list)) or len(factor) != 3:
                    raise ValueError("Weyl factors must be (site, a, b) triples")
                site, a, b = (
                    _nonnegative_int(factor[0], "site"),
                    _nonnegative_int(factor[1], "a"),
                    _nonnegative_int(factor[2], "b"),
                )
                max_site = max(max_site, site)
                a %= dimension
                b %= dimension
                previous_a, previous_b = current.get(site, (0, 0))
                phase = (phase + previous_b * a) % dimension
                combined_a = (previous_a + a) % dimension
                combined_b = (previous_b + b) % dimension
                if combined_a or combined_b:
                    current[site] = (combined_a, combined_b)
                else:
                    current.pop(site, None)
            raw_triples.append(
                [(site, a, b) for site, (a, b) in sorted(current.items())]
            )
            raw_coefficients.append(value * cmath.exp(2j * math.pi * phase / dimension))
        if n_sites is not None:
            n_sites = _nonnegative_int(n_sites, "n_sites")
            if n_sites <= max_site:
                raise ValueError("n_sites is smaller than a supplied Weyl site")
        else:
            n_sites = max_site + 1
        space = OperatorSpace._from_axes(
            tuple(_Axis("qudit", i, dimension) for i in range(n_sites))
        )
        result = _native.structured_hybrid_canonicalize(
            0,
            0,
            0,
            n_sites,
            dimension,
            (
                [[] for _ in raw_triples],
                [[] for _ in raw_triples],
                [[] for _ in raw_triples],
                [True for _ in raw_triples],
                raw_triples,
                [value.real for value in raw_coefficients],
                [value.imag for value in raw_coefficients],
            ),
            _effective_max_bytes(max_bytes),
        )
        return cast("QuditWeylOperator", _hybrid_from_native(space, result, cls))

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "QuditWeylOperator":
        raw_terms = tuple(terms)
        triples: List[List[Tuple[int, int, int]]] = []
        coefficients: List[complex] = []
        for term in raw_terms:
            if (
                term.qudit is None
                or term.fermion is not None
                or term.boson is not None
                or term.qubit
                or term.mapped_fermion is not None
            ):
                raise ValueError("qudit terms must contain only qudit factors")
            triples.append(list(term.qudit.triples))
            coefficients.append(term.coefficient)
        result = _native.structured_hybrid_canonicalize(
            0,
            0,
            0,
            len(space.qudits),
            space.qudits[0] if space.qudits else 0,
            (
                [[] for _ in raw_terms],
                [[] for _ in raw_terms],
                [[] for _ in raw_terms],
                [bool(word) for word in triples],
                triples,
                [value.real for value in coefficients],
                [value.imag for value in coefficients],
            ),
            _effective_max_bytes(max_bytes),
        )
        return cast("QuditWeylOperator", _hybrid_from_native(space, result, cls))

    def compile(  # type: ignore[override]
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a uniform-dimension Weyl operator."""
        return super().compile(target, storage=storage, max_bytes=max_bytes)
