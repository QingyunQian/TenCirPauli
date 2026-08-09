"""Structured fermion, boson, Weyl, and hybrid operator algebra.

The symbolic kernels in this module deliberately keep their representations
small: fermion words are two mode tuples, boson words are power blocks, and
Weyl words are modular site triples.  Public operations canonicalize a whole
operator at a time and finite targets use one mixed-radix basis convention.
"""

# This module is the private common/native boundary and legacy compatibility
# namespace; concrete implementations live in sibling modules and are imported
# only after the shared storage primitives are defined.
# ruff: noqa: F401, I001

from __future__ import annotations

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


_U32_MAX = 2**32 - 1
_IDENTITY_CODES = _PAULI_CHAR_TO_CODE
_PAULI_PRODUCT: Tuple[Tuple[Tuple[int, complex], ...], ...] = (
    ((0, 1), (1, 1), (2, 1), (3, 1)),
    ((1, 1), (0, 1), (3, 1j), (2, -1j)),
    ((2, 1), (3, -1j), (0, 1), (1, 1j)),
    ((3, 1), (2, 1j), (1, -1j), (0, 1)),
)

MappedOperator = Union["_StructuredOperator", PauliOperator]


def _finite_complex(value: object, name: str = "coefficient") -> complex:
    if isinstance(value, (bool, np.ndarray)) or not isinstance(
        value, (numbers.Real, numbers.Complex)
    ):
        raise ValueError(f"{name} must be a finite real or complex scalar")
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real or complex scalar") from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


_nonnegative_int = validate_nonnegative_int


def _positive_mode(mode: object, count: int, name: str = "mode") -> int:
    value = _nonnegative_int(mode, name)
    if value >= count:
        raise ValueError(f"{name} {value} is outside 0..{count}")
    return value


def _action(action: object) -> str:
    if action not in ("create", "annihilate"):
        raise ValueError("action must be exactly 'create' or 'annihilate'")
    return str(action)


def _factor(value: object, count: int) -> Tuple[int, str]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("fermion and boson factors must be (mode, action) pairs")
    return _positive_mode(value[0], count), _action(value[1])


def _estimate_terms(term_count: int, bytes_per_term: int = 160) -> int:
    return max(1, int(term_count)) * bytes_per_term


def _guard_terms(term_count: int, max_bytes: Optional[int], context: str) -> None:
    _validate_max_bytes(max_bytes)
    if max_bytes is not None:
        _check_allocation(_estimate_terms(term_count), max_bytes, context)


def _complex_sort_key(value: complex) -> Tuple[bytes, bytes]:
    return struct.pack("!d", value.real), struct.pack("!d", value.imag)


def _prepare_integral_array(
    value: object,
    shape: Tuple[int, ...],
    name: str,
) -> np.ndarray[Any, Any]:
    """Validate one contiguous numerical integral array before the native call."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if value.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise TypeError(f"{name} must have dtype float64 or complex128")
    return value


def _prepare_integral_constant(value: object) -> complex:
    constant = _finite_complex(value, "constant")
    if constant.imag != 0.0:
        raise ValueError("constant must be real")
    return complex(constant.real, 0.0)


def _fermion_from_integral_blocks(
    cls: Any,
    n_spatial: int,
    one_alpha: object,
    one_beta: object,
    eri_aa: object,
    eri_ab: object,
    eri_ba: object,
    eri_bb: object,
    *,
    spin_ordering: str,
    constant: object,
    max_bytes: Optional[int],
) -> "FermionOperator":
    """Call the compact native chemistry ingestion path."""
    n_spatial = _nonnegative_int(n_spatial, "n_spatial")
    _validate_max_bytes(max_bytes)
    shape_one = (n_spatial, n_spatial)
    shape_eri = (n_spatial, n_spatial, n_spatial, n_spatial)
    alpha = _prepare_integral_array(one_alpha, shape_one, "one_alpha")
    beta = _prepare_integral_array(one_beta, shape_one, "one_beta")
    eri_aa, eri_ab, eri_ba, eri_bb = tuple(
        _prepare_integral_array(value, shape_eri, name)
        for name, value in (
            ("eri_aa", eri_aa),
            ("eri_ab", eri_ab),
            ("eri_ba", eri_ba),
            ("eri_bb", eri_bb),
        )
    )
    if spin_ordering == "interleaved":
        ordering_code = 0
    elif spin_ordering == "alpha_then_beta":
        ordering_code = 1
    else:
        raise ValueError(
            "spin_ordering must be exactly 'interleaved' or 'alpha_then_beta'"
        )
    constant_value = _prepare_integral_constant(constant)
    result = _native.structured_fermion_integral_blocks(
        n_spatial,
        alpha,
        beta,
        eri_aa,
        eri_ab,
        eri_ba,
        eri_bb,
        ordering_code,
        constant_value.real,
        constant_value.imag,
        _effective_max_bytes(max_bytes),
    )
    return _fermion_from_native(cls, 2 * n_spatial, result)


from .words import (  # noqa: E402  # Depends on the validation helpers above.
    BosonTerm,
    BosonWord,
    FermionTerm,
    FermionWord,
    QuditProduct,
    QuditWeylTerm,
    QuditWeylWord,
)


@dataclass(frozen=True)
class _Term:
    """Canonical hybrid term; exposed as :class:`HybridTerm` in ``terms``."""

    fermion: Optional[FermionWord]
    boson: Optional[BosonWord]
    qubit: Tuple[int, ...]
    qudit: Optional[QuditWeylWord]
    mapped_fermion: Optional[Tuple[int, ...]]
    coefficient: complex

    def key(self) -> Tuple[Any, ...]:
        return (
            self.fermion.creation_modes if self.fermion else (),
            self.fermion.annihilation_modes if self.fermion else (),
            self.boson.blocks if self.boson else (),
            self.qubit,
            self.mapped_fermion or (),
            self.qudit.triples if self.qudit else (),
        )


HybridTerm = _Term

StructuredNativeHandle = Union[
    _native.NativeFermionOperatorHandle,
    _native.NativeBosonOperatorHandle,
    _native.NativeHybridOperatorHandle,
]


class _StructuredOperator:
    _domain = "hybrid"
    space: OperatorSpace
    _terms: Optional[Tuple[_Term, ...]]
    _native_handle: Optional[StructuredNativeHandle]
    _locked: bool

    def __init__(
        self,
        space: OperatorSpace,
        terms: Optional[Sequence[_Term]] = None,
        native_handle: Optional[StructuredNativeHandle] = None,
    ) -> None:
        if not isinstance(space, OperatorSpace):
            raise TypeError("space must be an OperatorSpace")
        if terms is not None or native_handle is None:
            raise RuntimeError("structured operators require native storage")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "_terms", None)
        object.__setattr__(self, "_native_handle", native_handle)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("operators are immutable")
        object.__setattr__(self, name, value)

    @property
    def terms(
        self,
    ) -> Tuple[Union[FermionTerm, BosonTerm, QuditWeylTerm, HybridTerm], ...]:
        """Return immutable typed canonical terms in deterministic order."""
        terms = self._materialized_terms()
        result: List[Union[FermionTerm, BosonTerm, QuditWeylTerm, HybridTerm]] = []
        for term in terms:
            if self._domain == "fermion":
                assert term.fermion is not None
                result.append(FermionTerm(term.fermion, term.coefficient))
            elif self._domain == "boson":
                assert term.boson is not None
                result.append(BosonTerm(term.boson, term.coefficient))
            elif self._domain == "qudit":
                assert term.qudit is not None
                result.append(QuditWeylTerm(term.qudit, term.coefficient))
            else:
                result.append(term)
        return tuple(result)

    @property
    def term_count(self) -> int:
        """Return the number of nonzero canonical terms."""
        handle = self._native_handle
        if handle is None:
            raise RuntimeError("structured operators must retain native handles")
        return handle.term_count

    def __len__(self) -> int:
        return self.term_count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _StructuredOperator):
            return NotImplemented
        if type(self) is not type(other) or self.space != other.space:
            return False
        left = self._native_handle
        right = other._native_handle
        if left is None or right is None:
            raise RuntimeError("structured operators must retain native handles")
        if isinstance(left, _native.NativeFermionOperatorHandle):
            return isinstance(right, _native.NativeFermionOperatorHandle) and bool(
                left.content_eq(right)
            )
        if isinstance(left, _native.NativeBosonOperatorHandle):
            return isinstance(right, _native.NativeBosonOperatorHandle) and bool(
                left.content_eq(right)
            )
        return isinstance(right, _native.NativeHybridOperatorHandle) and bool(
            left.content_eq(right)
        )

    def __hash__(self) -> int:
        handle = self._native_handle
        if handle is None:
            raise RuntimeError("structured operators must retain native handles")
        return hash((type(self), self.space, int(handle.content_hash())))

    def _check_other(self, other: object) -> "_StructuredOperator":
        if not isinstance(other, _StructuredOperator):
            raise TypeError(f"expected structured operator, got {type(other).__name__}")
        if self.space != other.space:
            raise ValueError("operators use incompatible OperatorSpace layouts")
        return other

    def _materialized_terms(self) -> Tuple[_Term, ...]:
        cached = self._terms
        if cached is not None:
            return cached
        handle = self._native_handle
        if handle is None:
            raise RuntimeError("structured operators must retain native handles")
        cached = _terms_from_native_handle(self.space, self._domain, handle)
        object.__setattr__(self, "_terms", cached)
        return cached

    def to_dict(self) -> Dict[Any, complex]:
        """Return plain canonical word data without constructing typed terms."""
        if self._domain == "fermion":
            creation, annihilation, real, imaginary = _fermion_arrays(
                cast(FermionOperator, self)
            )
            return {
                tuple(
                    [(int(mode), "create") for mode in left]
                    + [(int(mode), "annihilate") for mode in right]
                ): complex(re, im)
                for left, right, re, im in zip(creation, annihilation, real, imaginary)
            }
        if self._domain == "boson":
            blocks, real, imaginary = _boson_arrays(cast(BosonOperator, self))
            return {
                tuple(tuple(block) for block in word): complex(re, im)
                for word, re, im in zip(blocks, real, imaginary)
            }
        data = _hybrid_arrays(self)
        real, imaginary = data[-2], data[-1]
        if self._domain == "qudit":
            return {
                tuple(tuple(triple) for triple in data[9][index]): complex(re, im)
                for index, (re, im) in enumerate(zip(real, imaginary))
            }
        result: Dict[Any, complex] = {}
        for index, (re, im) in enumerate(zip(real, imaginary)):
            key = (
                tuple(data[1][index]) if data[0][index] else (),
                tuple(data[2][index]) if data[0][index] else (),
                tuple(data[4][index]) if data[3][index] else (),
                tuple(data[5][index]),
                tuple(data[7][index]) if data[6][index] else (),
                tuple(data[9][index]) if data[8][index] else (),
            )
            result[key] = complex(re, im)
        return result

    def add(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        """Return the exact structural sum of two compatible operators."""
        other = self._check_other(other)
        if self._native_handle is None or other._native_handle is None:
            raise RuntimeError("structured operators must retain native handles")
        _guard_terms(
            self.term_count + other.term_count,
            max_bytes,
            "operator addition",
        )
        return _add_native_handles(self, other, max_bytes)

    def scale(
        self, coefficient: complex, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "_StructuredOperator":
        """Return a copy with every coefficient multiplied by ``coefficient``."""
        scalar = _finite_complex(coefficient, "scale")
        if self._native_handle is None:
            raise RuntimeError("structured operators must retain native handles")
        _guard_terms(self.term_count, max_bytes, "operator scaling")
        return _scale_native_handle(self, scalar)

    def multiply(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> MappedOperator:
        """Return the exact ordered product of compatible operators."""
        other = self._check_other(other)
        if isinstance(
            self._native_handle, _native.NativeHybridOperatorHandle
        ) and isinstance(other._native_handle, _native.NativeHybridOperatorHandle):
            if (
                isinstance(self, HybridOperator)
                and isinstance(other, HybridOperator)
                and _requires_eager_fermion_mapping(self, other)
            ):
                mapped_left = self.map_fermions(max_bytes=max_bytes)
                mapped_right = other.map_fermions(max_bytes=max_bytes)
                if isinstance(mapped_left, PauliOperator):
                    if not isinstance(mapped_right, PauliOperator):
                        raise TypeError(
                            "mapped fermion operands have incompatible types"
                        )
                    return mapped_left.multiply(mapped_right, max_bytes=max_bytes)
                if not isinstance(mapped_right, _StructuredOperator):
                    raise TypeError("mapped fermion operands have incompatible types")
                return mapped_left.multiply(mapped_right, max_bytes=max_bytes)
            return _hybrid_from_native(
                self.space,
                self._native_handle.multiply(
                    other._native_handle, _effective_max_bytes(max_bytes)
                ),
                _native_result_class(self, other),
            )
        raise RuntimeError("structured operators must retain compatible native handles")

    def commutator(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> MappedOperator:
        """Return ``self * other - other * self``."""
        other = self._check_other(other)
        if self._native_handle is not None and other._native_handle is not None:
            if isinstance(
                self._native_handle, _native.NativeFermionOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeFermionOperatorHandle):
                return _fermion_from_native(
                    FermionOperator,
                    self.space.fermions,
                    self._native_handle.commutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeBosonOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeBosonOperatorHandle):
                return _boson_from_native(
                    BosonOperator,
                    self.space.bosons,
                    self._native_handle.commutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeHybridOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeHybridOperatorHandle):
                if not (
                    isinstance(self, HybridOperator)
                    and isinstance(other, HybridOperator)
                    and _requires_eager_fermion_mapping(self, other)
                ):
                    return _hybrid_from_native(
                        self.space,
                        self._native_handle.commutator(
                            other._native_handle, _effective_max_bytes(max_bytes)
                        ),
                        _native_result_class(self, other),
                    )
        left = self.multiply(other, max_bytes=max_bytes)
        right = other.multiply(self, max_bytes=max_bytes)
        if isinstance(left, PauliOperator):
            if not isinstance(right, PauliOperator):
                raise TypeError("commutator operands have incompatible mapped types")
            return left.add(right.scale(-1, max_bytes=max_bytes), max_bytes=max_bytes)
        if not isinstance(right, _StructuredOperator):
            raise TypeError("commutator operands have incompatible mapped types")
        return left.add(
            right.scale(-1, max_bytes=max_bytes),
            max_bytes=max_bytes,
        )

    def anticommutator(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> MappedOperator:
        """Return ``self * other + other * self``."""
        other = self._check_other(other)
        if self._native_handle is not None and other._native_handle is not None:
            if isinstance(
                self._native_handle, _native.NativeFermionOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeFermionOperatorHandle):
                return _fermion_from_native(
                    FermionOperator,
                    self.space.fermions,
                    self._native_handle.anticommutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeBosonOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeBosonOperatorHandle):
                return _boson_from_native(
                    BosonOperator,
                    self.space.bosons,
                    self._native_handle.anticommutator(
                        other._native_handle, _effective_max_bytes(max_bytes)
                    ),
                )
            if isinstance(
                self._native_handle, _native.NativeHybridOperatorHandle
            ) and isinstance(other._native_handle, _native.NativeHybridOperatorHandle):
                if not (
                    isinstance(self, HybridOperator)
                    and isinstance(other, HybridOperator)
                    and _requires_eager_fermion_mapping(self, other)
                ):
                    return _hybrid_from_native(
                        self.space,
                        self._native_handle.anticommutator(
                            other._native_handle, _effective_max_bytes(max_bytes)
                        ),
                        _native_result_class(self, other),
                    )
        left = self.multiply(other, max_bytes=max_bytes)
        right = other.multiply(self, max_bytes=max_bytes)
        if isinstance(left, PauliOperator):
            if not isinstance(right, PauliOperator):
                raise TypeError(
                    "anticommutator operands have incompatible mapped types"
                )
            return left.add(right, max_bytes=max_bytes)
        if not isinstance(right, _StructuredOperator):
            raise TypeError("anticommutator operands have incompatible mapped types")
        return left.add(right, max_bytes=max_bytes)

    def adjoint(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "_StructuredOperator":
        """Return the coefficient-conjugated structural adjoint."""
        if self._native_handle is None:
            raise RuntimeError("structured operators must retain native handles")
        return _adjoint_native_handle(self, max_bytes)

    def is_hermitian(self, tolerance: float = 0.0) -> bool:
        """Check adjoint equality with an explicit coefficient tolerance."""
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError("Hermiticity tolerance must be finite and non-negative")
        handle = self._native_handle
        if handle is None:
            raise RuntimeError("structured operators must retain native handles")
        return bool(handle.is_hermitian(float(tolerance)))

    def analyze_charge(
        self,
        charge: "AdditiveCharge",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "AdditiveSymmetryAnalysis":
        """Analyze an exact additive charge using the complete commutator."""
        from ..charge import analyze_charge

        return analyze_charge(self, charge, max_bytes=max_bytes)

    def conserves(
        self,
        charge: "AdditiveCharge",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> bool:
        """Return whether this operator exactly conserves ``charge``."""
        return self.analyze_charge(charge, max_bytes=max_bytes).is_conserved

    def restrict_charge(
        self,
        sector: "ChargeSector",
        *,
        storage: "ChargeStorage" = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "ChargeRestrictedOperator":
        """Restrict an exactly conserved operator to a charge-sector MVP.

        CPU-native restricted plans default to ``storage="lazy"``. An explicit
        eager request or later materialization may populate a retained cache.
        """
        from ..symmetry import _canonical_u1_sector, _restrict_u1

        canonical_u1 = _canonical_u1_sector(sector)
        if canonical_u1 is not None and all(
            axis.domain == "qubit" for axis in self.space._axes
        ):
            if not isinstance(self._native_handle, _native.NativeHybridOperatorHandle):
                raise RuntimeError(
                    "qubit structured operators require a native hybrid handle"
                )
            as_pauli = PauliOperator._from_native_handle(
                self._native_handle.to_pauli(
                    _pauli_axis_descriptor(self.space),
                    _effective_max_bytes(max_bytes),
                )
            )
            return _restrict_u1(
                as_pauli,
                canonical_u1,
                max_bytes,
                term_count=self.term_count,
                storage=storage,
            )  # type: ignore[return-value]
        from ..charge import restrict_charge

        return restrict_charge(self, sector, storage=storage, max_bytes=max_bytes)

    def tensor_product(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "HybridOperator":
        """Combine independent spaces using ordinary or graded tensor rules."""
        if not isinstance(other, _StructuredOperator):
            raise TypeError("tensor_product expects a structured operator")
        if (
            self.space.qudits
            and other.space.qudits
            and self.space.qudits[0] != other.space.qudits[0]
        ):
            raise ValueError("mixed-dimension tensor products are not supported")
        offsets = {
            "fermion": self.space.fermions,
            "boson": self.space.bosons,
            "qubit": self.space.qubits,
            "qudit": len(self.space.qudits),
        }
        axes = self.space._axes + tuple(
            _Axis(axis.domain, axis.index + offsets[axis.domain], axis.dimension)
            for axis in other.space._axes
        )
        space = OperatorSpace._from_axes(axes)
        terms: List[_Term] = []
        for left, right in product(
            self._materialized_terms(), other._materialized_terms()
        ):
            left_shifted = _shift_term(
                left,
                {"fermion": 0, "boson": 0, "qubit": 0, "qudit": 0},
                space,
                self.space,
            )
            right_shifted = _shift_term(right, offsets, space, other.space)
            if _term_has_raw_and_mapped_fermions(
                left_shifted
            ) or _term_has_raw_and_mapped_fermions(right_shifted):
                raise ValueError(
                    "tensor_product cannot operate on a term containing both "
                    "raw and mapped fermion factors"
                )
            if (
                left_shifted.mapped_fermion is not None
                or right_shifted.mapped_fermion is not None
            ):
                terms.extend(
                    _tensor_mapped_fermion_terms(
                        left_shifted,
                        right_shifted,
                        space,
                        self.space.fermions,
                        right_shifted.fermion is None
                        and right_shifted.mapped_fermion is not None,
                    )
                )
            else:
                terms.append(
                    _combine_nonexpanding_terms(left_shifted, right_shifted, space)
                )
        return HybridOperator._from_terms(space, terms, max_bytes=max_bytes)

    def map_fermions(
        self,
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> MappedOperator:
        """Map every raw fermion factor with the structured JW convention."""
        if not isinstance(mapping, str):
            from ..mapping import FermionQubitMapping

            if not isinstance(mapping, FermionQubitMapping):
                raise TypeError(
                    "mapping must be a supported name or FermionQubitMapping"
                )
            if self.space.fermions == 0:
                if mapping.n_modes != 0:
                    raise ValueError(
                        "mapping plan and fermion axis counts are incompatible"
                    )
                return self
            if not isinstance(self, HybridOperator):
                raise TypeError(
                    "reusable mappings require a hybrid or fermion operator"
                )
            return mapping.map_hybrid(self, max_bytes=max_bytes)
        if mapping != "jordan_wigner":
            from ..mapping import FermionQubitMapping

            plan = FermionQubitMapping.from_name(
                mapping, self.space.fermions, max_bytes=max_bytes
            )
            if self.space.fermions == 0:
                return self
            if not isinstance(self, HybridOperator):
                raise TypeError(
                    "reusable mappings require a hybrid or fermion operator"
                )
            return plan.map_hybrid(self, max_bytes=max_bytes)
        if self.space.fermions == 0:
            return self
        handle = self._native_handle
        if isinstance(handle, _native.NativeHybridOperatorHandle):
            if handle.has_mixed_fermion_roles():
                raise ValueError(
                    "cannot map a term containing both raw and mapped fermion factors"
                )
            if isinstance(self, HybridOperator):
                result = handle.jordan_wigner(_effective_max_bytes(max_bytes))
                if all(
                    axis.domain in ("fermion", "qubit") for axis in self.space._axes
                ):
                    return PauliOperator._from_native_handle(
                        result.to_pauli(
                            _pauli_axis_descriptor(self.space),
                            _effective_max_bytes(max_bytes),
                        )
                    )
                return _hybrid_from_native(self.space, result)
        if isinstance(handle, _native.NativeFermionOperatorHandle):
            return PauliOperator._from_native_handle(
                handle.jordan_wigner(_effective_max_bytes(max_bytes))
            )
        raise TypeError(
            "raw fermion mapping requires a FermionOperator or HybridOperator"
        )

    def compile(
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        boson_cutoffs: Optional[Mapping[object, object]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile into a dense, sparse, native, or backend MVP target.

        Raw Fermion factors remain raw descriptors for the default
        ``compile("native_mvp")`` path. Its ``mapping="jordan_wigner"``
        default denotes the public occupation-basis convention; it does not
        force Pauli expansion. Call :meth:`map_fermions` before compiling to
        explicitly request mapped-Pauli execution.
        """
        if target not in {"dense", "coo", "csr", "native_mvp", "backend_mvp"}:
            raise ValueError(
                "target must be one of 'dense', 'coo', 'csr', 'native_mvp', or 'backend_mvp'"
            )
        use_raw_native_fermions = (
            target == "native_mvp"
            and mapping == "jordan_wigner"
            and isinstance(self._native_handle, _native.NativeHybridOperatorHandle)
            and bool(self._native_handle.has_raw_fermions())
            and not bool(self._native_handle.has_mapped_fermions())
        )
        mapped = (
            self
            if use_raw_native_fermions
            else self.map_fermions(mapping, max_bytes=max_bytes)
        )
        if (
            target != "backend_mvp"
            and isinstance(mapped, _StructuredOperator)
            and all(axis.domain == "qubit" for axis in mapped.space._axes)
            and isinstance(mapped._native_handle, _native.NativeHybridOperatorHandle)
        ):
            packed = PauliOperator._from_native_handle(
                mapped._native_handle.to_pauli(
                    _pauli_axis_descriptor(mapped.space),
                    _effective_max_bytes(max_bytes),
                )
            )
            packed_result = packed.compile(target, storage=storage, max_bytes=max_bytes)
            if isinstance(packed_result, (NativeMVPPlan, BackendMVPPlan)):
                return _with_plan_metadata(
                    packed_result,
                    mapping=(_mapping_name(mapping) if self.space.fermions else None),
                    source_term_count=self.term_count,
                )
            return packed_result
        if isinstance(mapped, PauliOperator):
            result: CompileResult = mapped.compile(
                target, storage=storage, max_bytes=max_bytes
            )
            if isinstance(result, (NativeMVPPlan, BackendMVPPlan)):
                return _with_plan_metadata(
                    result,
                    mapping=(_mapping_name(mapping) if self.space.fermions else None),
                    source_term_count=self.term_count,
                )
            return result
        cutoffs = boson_cutoffs
        if mapped.space.bosons:
            if not isinstance(cutoffs, Mapping):
                raise ValueError(
                    "boson_cutoffs is required for every finite boson target"
                )
            normalized_cutoffs = _validate_cutoffs(mapped.space, cutoffs)
        else:
            if cutoffs is not None:
                raise ValueError(
                    "boson_cutoffs is only valid for spaces containing bosons"
                )
            normalized_cutoffs = {}
        if target == "backend_mvp" and mapped.space.bosons:
            raise NotImplementedError(
                "backend_mvp is deferred for finite boson and mixed hybrid layouts"
            )
        if target == "backend_mvp" and mapped.space.qudits:
            if not all(axis.domain == "qudit" for axis in mapped.space._axes):
                raise NotImplementedError(
                    "backend_mvp is deferred for mixed local-dimension layouts"
                )
            return _direct_weyl_backend_plan(mapped, max_bytes)
        return _compile_finite(
            mapped,
            target,
            normalized_cutoffs,
            max_bytes,
            storage=storage,
            mapping=(_mapping_name(mapping) if self.space.fermions else None),
        )

    def __add__(self, other: object) -> "_StructuredOperator":
        if not isinstance(other, _StructuredOperator):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> "_StructuredOperator":
        if not isinstance(other, _StructuredOperator):
            return NotImplemented
        return self.add(other.scale(-1))

    def __neg__(self) -> "_StructuredOperator":
        return self.scale(-1)

    def __mul__(self, other: object) -> MappedOperator:
        if isinstance(other, _StructuredOperator):
            return self.multiply(other)
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.scale(complex(other))
        return NotImplemented

    def __rmul__(self, other: object) -> MappedOperator:
        return self * other


def _native_fermion_raw(
    n_modes: int,
    terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
    max_bytes: Optional[int],
) -> _native.NativeFermionOperatorHandle:
    n_modes = _nonnegative_int(n_modes, "n_modes")
    factors: List[List[Tuple[int, int]]] = []
    coefficients: List[complex] = []
    for raw_factors, coefficient in terms:
        normalized = []
        for factor in raw_factors:
            mode, action = _factor(factor, n_modes)
            normalized.append((mode, int(action == "annihilate")))
        factors.append(normalized)
        coefficients.append(_finite_complex(coefficient))
    result = _native.structured_fermion_canonicalize(
        n_modes,
        factors,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
        _effective_max_bytes(max_bytes),
    )
    return result


def _native_boson_raw(
    n_modes: int,
    terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
    max_bytes: Optional[int],
) -> _native.NativeBosonOperatorHandle:
    n_modes = _nonnegative_int(n_modes, "n_modes")
    factors: List[List[Tuple[int, int]]] = []
    coefficients: List[complex] = []
    for raw_factors, coefficient in terms:
        normalized = []
        for factor in raw_factors:
            mode, action = _factor(factor, n_modes)
            normalized.append((mode, int(action == "annihilate")))
        factors.append(normalized)
        coefficients.append(_finite_complex(coefficient))
    return _native.structured_boson_canonicalize(
        n_modes,
        factors,
        [value.real for value in coefficients],
        [value.imag for value in coefficients],
        _effective_max_bytes(max_bytes),
    )


def _fermion_arrays(
    operator: "FermionOperator",
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    handle = operator._native_handle
    if not isinstance(handle, _native.NativeFermionOperatorHandle):
        raise RuntimeError("fermion operators must retain native handles")
    (
        _,
        creation_flat,
        creation_offsets,
        annihilation_flat,
        annihilation_offsets,
        coefficients,
    ) = handle.materialize()
    creation_values = np.asarray(creation_flat, dtype=np.uint32)
    annihilation_values = np.asarray(annihilation_flat, dtype=np.uint32)
    creation_stops = np.asarray(creation_offsets, dtype=np.uintp)
    annihilation_stops = np.asarray(annihilation_offsets, dtype=np.uintp)
    creation_rows = [
        [int(mode) for mode in creation_values[start:stop]]
        for start, stop in zip(creation_stops[:-1], creation_stops[1:])
    ]
    annihilation_rows = [
        [int(mode) for mode in annihilation_values[start:stop]]
        for start, stop in zip(annihilation_stops[:-1], annihilation_stops[1:])
    ]
    values = np.asarray(coefficients, dtype=np.complex128)
    return (
        creation_rows,
        annihilation_rows,
        [float(value.real) for value in values],
        [float(value.imag) for value in values],
    )


def _boson_arrays(
    operator: "BosonOperator",
) -> Tuple[List[List[Tuple[int, int, int]]], List[float], List[float]]:
    handle = operator._native_handle
    if not isinstance(handle, _native.NativeBosonOperatorHandle):
        raise RuntimeError("boson operators must retain native handles")
    _, blocks_flat, block_offsets, coefficients = handle.materialize()
    blocks_values = np.asarray(blocks_flat, dtype=np.uint32).reshape((-1, 3))
    offsets = np.asarray(block_offsets, dtype=np.uintp)
    block_rows: List[List[Tuple[int, int, int]]] = [
        [
            cast(Tuple[int, int, int], tuple(int(value) for value in block))
            for block in blocks_values[start:stop]
        ]
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    values = np.asarray(coefficients, dtype=np.complex128)
    return (
        block_rows,
        [float(value.real) for value in values],
        [float(value.imag) for value in values],
    )


def _hybrid_arrays(
    operator: "_StructuredOperator",
) -> Tuple[
    List[bool],
    List[List[int]],
    List[List[int]],
    List[bool],
    List[List[Tuple[int, int, int]]],
    List[List[int]],
    List[bool],
    List[List[int]],
    List[bool],
    List[List[Tuple[int, int, int]]],
    List[float],
    List[float],
]:
    handle = operator._native_handle
    if not isinstance(handle, _native.NativeHybridOperatorHandle):
        raise RuntimeError("hybrid operators must retain native handles")
    term_count, nqubits, structural_arrays, fixed_arrays = handle.materialize()
    (
        flags,
        fermion_creation,
        fermion_creation_offsets,
        fermion_annihilation,
        fermion_annihilation_offsets,
        boson_blocks,
        boson_offsets,
    ) = structural_arrays
    qubit_codes, mapped_codes, qudit_triples, qudit_offsets, coefficients = fixed_arrays
    flags_array = np.asarray(flags, dtype=np.uint8).reshape((int(term_count), 4))
    f_creation_values = np.asarray(fermion_creation, dtype=np.uint32)
    f_creation_stops = np.asarray(fermion_creation_offsets, dtype=np.uintp)
    f_annihilation_values = np.asarray(fermion_annihilation, dtype=np.uint32)
    f_annihilation_stops = np.asarray(fermion_annihilation_offsets, dtype=np.uintp)
    boson_values = np.asarray(boson_blocks, dtype=np.uint32).reshape((-1, 3))
    boson_stops = np.asarray(boson_offsets, dtype=np.uintp)
    qudit_values = np.asarray(qudit_triples, dtype=np.uint32).reshape((-1, 3))
    qudit_stops = np.asarray(qudit_offsets, dtype=np.uintp)
    f_creation_rows = [
        [int(value) for value in f_creation_values[start:stop]]
        for start, stop in zip(f_creation_stops[:-1], f_creation_stops[1:])
    ]
    f_annihilation_rows = [
        [int(value) for value in f_annihilation_values[start:stop]]
        for start, stop in zip(f_annihilation_stops[:-1], f_annihilation_stops[1:])
    ]
    boson_rows: List[List[Tuple[int, int, int]]] = [
        [
            cast(Tuple[int, int, int], tuple(int(value) for value in block))
            for block in boson_values[start:stop]
        ]
        for start, stop in zip(boson_stops[:-1], boson_stops[1:])
    ]
    qudit_rows: List[List[Tuple[int, int, int]]] = [
        [
            cast(Tuple[int, int, int], tuple(int(value) for value in triple))
            for triple in qudit_values[start:stop]
        ]
        for start, stop in zip(qudit_stops[:-1], qudit_stops[1:])
    ]
    values = np.asarray(coefficients, dtype=np.complex128)
    qubit_values = np.asarray(qubit_codes, dtype=np.uint8).reshape(
        (int(term_count), int(nqubits))
    )
    mapped_values = np.asarray(mapped_codes, dtype=np.uint8).reshape(
        (int(term_count), int(operator.space.fermions))
    )
    return (
        [bool(row[0]) for row in flags_array],
        f_creation_rows,
        f_annihilation_rows,
        [bool(row[1]) for row in flags_array],
        boson_rows,
        [list(row) for row in qubit_values],
        [bool(row[2]) for row in flags_array],
        [list(row) for row in mapped_values],
        [bool(row[3]) for row in flags_array],
        qudit_rows,
        [float(value.real) for value in values],
        [float(value.imag) for value in values],
    )


def _terms_from_native_handle(
    space: OperatorSpace, domain: str, handle: StructuredNativeHandle
) -> Tuple[_Term, ...]:
    """Materialize typed structural words at the explicit ``terms`` boundary."""
    native: Any = handle
    if domain == "fermion":
        (
            _term_count,
            creation_flat,
            creation_offsets,
            annihilation_flat,
            annihilation_offsets,
            coefficients,
        ) = native.materialize()
        creation_values = np.asarray(creation_flat, dtype=np.uint32)
        creation_stops = np.asarray(creation_offsets, dtype=np.uintp)
        annihilation_values = np.asarray(annihilation_flat, dtype=np.uint32)
        annihilation_stops = np.asarray(annihilation_offsets, dtype=np.uintp)
        values = np.asarray(coefficients, dtype=np.complex128)
        return tuple(
            _Term(
                FermionWord(
                    space.fermions,
                    tuple(int(value) for value in creation_values[start:stop]),
                    tuple(
                        int(value) for value in annihilation_values[ann_start:ann_stop]
                    ),
                ),
                None,
                (),
                None,
                None,
                complex(value),
            )
            for (start, stop), (ann_start, ann_stop), value in zip(
                zip(creation_stops[:-1], creation_stops[1:]),
                zip(annihilation_stops[:-1], annihilation_stops[1:]),
                values,
            )
        )
    if domain == "boson":
        _term_count, blocks_flat, block_offsets, coefficients = native.materialize()
        blocks_values = np.asarray(blocks_flat, dtype=np.uint32).reshape((-1, 3))
        offsets = np.asarray(block_offsets, dtype=np.uintp)
        values = np.asarray(coefficients, dtype=np.complex128)
        return tuple(
            _Term(
                None,
                BosonWord(
                    space.bosons,
                    tuple(
                        cast(Tuple[int, int, int], tuple(int(value) for value in block))
                        for block in blocks_values[start:stop]
                    ),
                ),
                (),
                None,
                None,
                complex(value),
            )
            for (start, stop), value in zip(zip(offsets[:-1], offsets[1:]), values)
        )
    (
        term_count,
        nqubits,
        structural_arrays,
        fixed_arrays,
    ) = native.materialize()
    (
        flags,
        fermion_creation,
        fermion_creation_offsets,
        fermion_annihilation,
        fermion_annihilation_offsets,
        boson_blocks,
        boson_offsets,
    ) = structural_arrays
    qubit_codes, mapped_codes, qudit_triples, qudit_offsets, coefficients = fixed_arrays
    flags_array = np.asarray(flags, dtype=np.uint8).reshape((int(term_count), 4))
    creation_values = np.asarray(fermion_creation, dtype=np.uint32)
    creation_stops = np.asarray(fermion_creation_offsets, dtype=np.uintp)
    annihilation_values = np.asarray(fermion_annihilation, dtype=np.uint32)
    annihilation_stops = np.asarray(fermion_annihilation_offsets, dtype=np.uintp)
    boson_values = np.asarray(boson_blocks, dtype=np.uint32).reshape((-1, 3))
    boson_stops = np.asarray(boson_offsets, dtype=np.uintp)
    qudit_values = np.asarray(qudit_triples, dtype=np.uint32).reshape((-1, 3))
    qudit_stops = np.asarray(qudit_offsets, dtype=np.uintp)
    qubit_values = np.asarray(qubit_codes, dtype=np.uint8).reshape(
        (int(term_count), int(nqubits))
    )
    mapped_values = np.asarray(mapped_codes, dtype=np.uint8).reshape(
        (int(term_count), int(space.fermions))
    )
    values = np.asarray(coefficients, dtype=np.complex128)
    qudit_dimension = space.qudits[0] if space.qudits else 0
    return tuple(
        _Term(
            (
                FermionWord(
                    space.fermions,
                    tuple(int(value) for value in creation_values[start:stop]),
                    tuple(
                        int(value) for value in annihilation_values[ann_start:ann_stop]
                    ),
                )
                if flags_array[index, 0]
                else None
            ),
            (
                BosonWord(
                    space.bosons,
                    tuple(
                        cast(Tuple[int, int, int], tuple(int(value) for value in block))
                        for block in boson_values[b_start:b_stop]
                    ),
                )
                if flags_array[index, 1]
                else None
            ),
            tuple(int(code) for code in qubit_values[index]),
            (
                QuditWeylWord(
                    qudit_dimension,
                    tuple(
                        cast(
                            Tuple[int, int, int], tuple(int(value) for value in triple)
                        )
                        for triple in qudit_values[q_start:q_stop]
                    ),
                )
                if flags_array[index, 3]
                else None
            ),
            (
                tuple(int(code) for code in mapped_values[index])
                if flags_array[index, 2]
                else None
            ),
            complex(value),
        )
        for index, (
            (start, stop),
            (ann_start, ann_stop),
            (b_start, b_stop),
            (q_start, q_stop),
            value,
        ) in enumerate(
            zip(
                zip(creation_stops[:-1], creation_stops[1:]),
                zip(annihilation_stops[:-1], annihilation_stops[1:]),
                zip(boson_stops[:-1], boson_stops[1:]),
                zip(qudit_stops[:-1], qudit_stops[1:]),
                values,
            )
        )
    )


def _add_native_handles(
    left: _StructuredOperator,
    right: _StructuredOperator,
    max_bytes: Optional[int],
) -> _StructuredOperator:
    left_handle = left._native_handle
    right_handle = right._native_handle
    if left_handle is None or right_handle is None:
        raise RuntimeError("native handle addition requires native handles")
    if isinstance(left_handle, _native.NativeFermionOperatorHandle) and isinstance(
        right_handle, _native.NativeFermionOperatorHandle
    ):
        return _fermion_from_native(
            type(left),
            left.space.fermions,
            left_handle.add(right_handle, _effective_max_bytes(max_bytes)),
        )
    if isinstance(left_handle, _native.NativeBosonOperatorHandle) and isinstance(
        right_handle, _native.NativeBosonOperatorHandle
    ):
        return _boson_from_native(
            type(left),
            left.space.bosons,
            left_handle.add(right_handle, _effective_max_bytes(max_bytes)),
        )
    if isinstance(left_handle, _native.NativeHybridOperatorHandle) and isinstance(
        right_handle, _native.NativeHybridOperatorHandle
    ):
        result = left_handle.add(right_handle, _effective_max_bytes(max_bytes))
        return _hybrid_from_native(
            left.space, result, _native_result_class(left, right)
        )
    if isinstance(left_handle, _native.NativeFermionOperatorHandle) and isinstance(
        right_handle, _native.NativeHybridOperatorHandle
    ):
        result = left_handle.to_hybrid().add(
            right_handle, _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    if isinstance(left_handle, _native.NativeHybridOperatorHandle) and isinstance(
        right_handle, _native.NativeFermionOperatorHandle
    ):
        result = left_handle.add(
            right_handle.to_hybrid(), _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    if isinstance(left_handle, _native.NativeBosonOperatorHandle) and isinstance(
        right_handle, _native.NativeHybridOperatorHandle
    ):
        result = left_handle.to_hybrid().add(
            right_handle, _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    if isinstance(left_handle, _native.NativeHybridOperatorHandle) and isinstance(
        right_handle, _native.NativeBosonOperatorHandle
    ):
        result = left_handle.add(
            right_handle.to_hybrid(), _effective_max_bytes(max_bytes)
        )
        return _hybrid_from_native(left.space, result)
    raise TypeError("native structured handles have incompatible families")


def _scale_native_handle(
    operator: _StructuredOperator, scalar: complex
) -> _StructuredOperator:
    handle = operator._native_handle
    if handle is None:
        raise RuntimeError("native handle scaling requires native storage")
    if isinstance(handle, _native.NativeFermionOperatorHandle):
        return _fermion_from_native(
            type(operator),
            operator.space.fermions,
            handle.scale(scalar.real, scalar.imag),
        )
    if isinstance(handle, _native.NativeBosonOperatorHandle):
        return _boson_from_native(
            type(operator),
            operator.space.bosons,
            handle.scale(scalar.real, scalar.imag),
        )
    return _hybrid_from_native(
        operator.space, handle.scale(scalar.real, scalar.imag), type(operator)
    )


def _adjoint_native_handle(
    operator: _StructuredOperator, max_bytes: Optional[int]
) -> _StructuredOperator:
    handle = operator._native_handle
    if handle is None:
        raise RuntimeError("native handle adjoint requires native storage")
    if isinstance(handle, _native.NativeFermionOperatorHandle):
        return _fermion_from_native(
            type(operator),
            operator.space.fermions,
            handle.adjoint(_effective_max_bytes(max_bytes)),
        )
    if isinstance(handle, _native.NativeBosonOperatorHandle):
        return _boson_from_native(
            type(operator), operator.space.bosons, handle.adjoint()
        )
    return _hybrid_from_native(operator.space, handle.adjoint(), type(operator))


def _fermion_from_native(
    cls: Any,
    n_modes: int,
    result: _native.NativeFermionOperatorHandle,
) -> "FermionOperator":
    instance = object.__new__(cls)
    _StructuredOperator.__init__(
        instance,
        OperatorSpace(fermions=n_modes),
        native_handle=result,
    )
    return cast("FermionOperator", instance)


def _boson_from_native(
    cls: Any,
    n_modes: int,
    result: _native.NativeBosonOperatorHandle,
) -> "BosonOperator":
    instance = object.__new__(cls)
    _StructuredOperator.__init__(
        instance,
        OperatorSpace(bosons=n_modes),
        native_handle=result,
    )
    return cast("BosonOperator", instance)


def _hybrid_from_native(
    space: OperatorSpace,
    result: _native.NativeHybridOperatorHandle,
    cls: Any = None,
) -> "HybridOperator":
    if cls is None:
        cls = HybridOperator
    instance = object.__new__(cls)
    _StructuredOperator.__init__(instance, space, native_handle=result)
    return cast("HybridOperator", instance)


def _native_result_class(left: _StructuredOperator, right: _StructuredOperator) -> Any:
    """Keep pure-family facades while promoting genuinely mixed results."""
    if type(left) is type(right) and isinstance(
        left, (QuditWeylOperator, HybridOperator)
    ):
        return type(left)
    return HybridOperator


# Import after the common native storage boundary is fully defined.  The
# implementation modules import this file for those shared primitives, while
# the stable facade re-exports the public values below.
from .space import (  # noqa: E402
    OperatorSpace,
    _Axis,
    _BosonFactory,
    _Factory,
    _FermionFactory,
    _QubitFactory,
    _QuditFactory,
    _factory_native_operator,
)

from .mapping import (  # noqa: E402
    QuditWeylOperator,
    _combine_nonexpanding_terms,
    _jordan_wigner_word,
    _make_operator,
    _mapping_name,
    _multiply_mapped,
    _pauli_axis_descriptor,
    _pauli_fermion_parity,
    _replace,
    _requires_eager_fermion_mapping,
    _shift_term,
    _tensor_mapped_fermion_terms,
    _term_has_raw_and_mapped_fermions,
    _with_plan_metadata,
)
from .fermion import FermionOperator  # noqa: E402
from .boson import BosonOperator  # noqa: E402
from .hybrid import HybridOperator, OperatorBuilder  # noqa: E402
from .compile import (  # noqa: E402
    _compile_finite,
    _compile_native_hybrid_handle,
    _direct_weyl_backend_plan,
    _native_structured_csr,
    _native_structured_dense,
    _native_structured_mvp_plan,
    _native_structured_sparse,
    _structured_axis_descriptor,
    _validate_cutoffs,
)

# The implementation modules intentionally share the compact native-storage
# primitives above.  Their cross-domain references are resolved once after all
# concrete classes have been imported, rather than through import-time class
# cycles or duplicate compatibility implementations.
from . import boson as _boson_module  # noqa: E402
from . import compile as _compile_module  # noqa: E402
from . import fermion as _fermion_module  # noqa: E402
from . import hybrid as _hybrid_module  # noqa: E402
from . import mapping as _mapping_module  # noqa: E402
from . import space as _space_module  # noqa: E402

for _implementation_module in (
    _space_module,
    _mapping_module,
    _fermion_module,
    _boson_module,
    _hybrid_module,
    _compile_module,
):
    for _name, _value in tuple(globals().items()):
        if not _name.startswith("__"):
            _implementation_module.__dict__.setdefault(_name, _value)

del _implementation_module, _name, _value

__all__ = [
    "BosonOperator",
    "BosonTerm",
    "BosonWord",
    "FermionOperator",
    "FermionTerm",
    "FermionWord",
    "HybridOperator",
    "HybridTerm",
    "OperatorBuilder",
    "OperatorSpace",
    "QuditProduct",
    "QuditWeylOperator",
    "QuditWeylTerm",
    "QuditWeylWord",
]
