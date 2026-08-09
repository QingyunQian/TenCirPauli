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

from .space import OperatorSpace


class BosonOperator(_StructuredOperator):
    """Immutable symbolic boson operator with infinite-Fock CCR semantics.

    Finite matrix and MVP targets require explicit per-mode occupation cutoffs;
    symbolic algebra itself remains cutoff-free.
    """

    _domain = "boson"

    @classmethod
    def _from_words(
        cls, n_modes: int, products: Mapping[BosonWord, int]
    ) -> "BosonOperator":
        return cls._from_terms(
            OperatorSpace(bosons=n_modes),
            (
                _Term(None, word, (), None, None, coefficient)
                for word, coefficient in products.items()
            ),
        )

    @classmethod
    def from_terms(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "BosonOperator":
        """Construct and CCR-canonicalize raw ordered boson factors."""
        return _boson_from_native(
            cls,
            _nonnegative_int(n_modes, "n_modes"),
            _native_boson_raw(n_modes, terms, max_bytes),
        )

    def multiply(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        """Multiply two boson operators while retaining CCR contractions."""
        if not isinstance(other, BosonOperator) or other.space != self.space:
            return cast(
                "_StructuredOperator", super().multiply(other, max_bytes=max_bytes)
            )
        left_handle = self._native_handle
        right_handle = other._native_handle
        if not isinstance(
            left_handle, _native.NativeBosonOperatorHandle
        ) or not isinstance(right_handle, _native.NativeBosonOperatorHandle):
            raise RuntimeError("boson operators must retain native handles")
        result = left_handle.multiply(right_handle, _effective_max_bytes(max_bytes))
        return _boson_from_native(
            BosonOperator,
            self.space.bosons,
            result,
        )

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "BosonOperator":
        raw_terms = []
        for term in terms:
            if (
                term.boson is None
                or any(
                    value is not None
                    for value in (term.fermion, term.qudit, term.mapped_fermion)
                )
                or term.qubit
            ):
                raise ValueError("boson terms must contain only boson factors")
            raw_terms.append(
                (
                    tuple(
                        [
                            (mode, "create")
                            for mode, create, _ in term.boson.blocks
                            for _ in range(create)
                        ]
                        + [
                            (mode, "annihilate")
                            for mode, _, annihilate in term.boson.blocks
                            for _ in range(annihilate)
                        ]
                    ),
                    term.coefficient,
                )
            )
        return _boson_from_native(
            cls,
            space.bosons,
            _native_boson_raw(space.bosons, raw_terms, max_bytes),
        )

    def compile(  # type: ignore[override]
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        boson_cutoffs: Mapping[object, object],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a boson operator with its required finite cutoffs."""
        return super().compile(
            target,
            storage=storage,
            boson_cutoffs=boson_cutoffs,
            max_bytes=max_bytes,
        )
