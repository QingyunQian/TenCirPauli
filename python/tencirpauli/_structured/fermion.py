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

from .compile import _compile_finite
from .mapping import _mapping_name, _with_plan_metadata
from .space import OperatorSpace


class FermionOperator(_StructuredOperator):
    """Immutable canonical fermionic operator governed by CAR.

    Raw ladder products are expanded and aggregated exactly. Use
    :meth:`map_fermions` for a Pauli representation or :meth:`compile` for a
    matrix/MVP target.
    """

    _domain = "fermion"

    def __init__(
        self, n_modes: int, terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]]
    ) -> None:
        result = _native_fermion_raw(n_modes, terms, DEFAULT_MAX_BYTES)
        _StructuredOperator.__init__(
            self,
            OperatorSpace(fermions=_nonnegative_int(n_modes, "n_modes")),
            native_handle=result,
        )

    @classmethod
    def _from_words(
        cls, n_modes: int, products: Mapping[FermionWord, int]
    ) -> "FermionOperator":
        return cls._from_terms(
            OperatorSpace(fermions=n_modes),
            (
                _Term(word, None, (), None, None, coefficient)
                for word, coefficient in products.items()
            ),
        )

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        raw_terms = []
        for term in terms:
            if (
                term.fermion is None
                or any(
                    value is not None
                    for value in (term.boson, term.qudit, term.mapped_fermion)
                )
                or term.qubit
            ):
                raise ValueError("fermion terms must contain only fermion factors")
            raw_terms.append(
                (
                    tuple(
                        [(mode, "create") for mode in term.fermion.creation_modes]
                        + [
                            (mode, "annihilate")
                            for mode in term.fermion.annihilation_modes
                        ]
                    ),
                    term.coefficient,
                )
            )
        return _fermion_from_native(
            cls,
            space.fermions,
            _native_fermion_raw(space.fermions, raw_terms, max_bytes),
        )

    @classmethod
    def from_terms(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        """Construct and CAR-canonicalize raw ordered fermion factors.

        Each term is ``(factors, coefficient)`` with factors such as
        ``(mode, "create")`` and ``(mode, "annihilate")``. Equal canonical
        words are aggregated and exact zero coefficients are removed.

        Examples:
            >>> import tencirpauli as tcp
            >>> number = tcp.FermionOperator.from_terms(
            ...     1, [(((0, "create"), (0, "annihilate")), 1.0)]
            ... )
            >>> number.term_count
            1
        """
        return cls._from_raw(n_modes, terms, max_bytes)

    @classmethod
    def from_integrals(
        cls,
        one_body: np.ndarray[Any, Any],
        two_body: np.ndarray[Any, Any],
        *,
        constant: complex = 0.0,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        """Construct a canonical spin-orbital molecular Hamiltonian.

        ``one_body[p, q]`` and ``two_body[p, q, r, s]`` use the fixed
        convention from ``a†_p a_q`` and ``a†_p a†_q a_s a_r``. The native
        importer applies the two-body factor of one half exactly once.
        """
        _validate_max_bytes(max_bytes)
        if not isinstance(one_body, np.ndarray) or one_body.ndim != 2:
            raise TypeError("one_body must be a two-dimensional NumPy array")
        if one_body.shape[0] != one_body.shape[1]:
            raise ValueError("one_body must have a square shape")
        n_modes = int(one_body.shape[0])
        one = _prepare_integral_array(one_body, (n_modes, n_modes), "one_body")
        two = _prepare_integral_array(
            two_body, (n_modes, n_modes, n_modes, n_modes), "two_body"
        )
        constant_value = _prepare_integral_constant(constant)
        result = _native.structured_fermion_integrals(
            one,
            two,
            constant_value.real,
            constant_value.imag,
            _effective_max_bytes(max_bytes),
        )
        return _fermion_from_native(cls, n_modes, result)

    @classmethod
    def _from_raw(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        max_bytes: Optional[int],
    ) -> "FermionOperator":
        return _fermion_from_native(
            cls,
            _nonnegative_int(n_modes, "n_modes"),
            _native_fermion_raw(n_modes, terms, max_bytes),
        )

    def multiply(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        """Multiply two fermion operators while retaining CAR contractions."""
        if not isinstance(other, FermionOperator) or other.space != self.space:
            return cast(
                "_StructuredOperator", super().multiply(other, max_bytes=max_bytes)
            )
        left_handle = self._native_handle
        right_handle = other._native_handle
        if not isinstance(
            left_handle, _native.NativeFermionOperatorHandle
        ) or not isinstance(right_handle, _native.NativeFermionOperatorHandle):
            raise RuntimeError("fermion operators must retain native handles")
        result = left_handle.multiply(right_handle, _effective_max_bytes(max_bytes))
        return _fermion_from_native(
            FermionOperator,
            self.space.fermions,
            result,
        )

    def map_fermions(
        self,
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> PauliOperator:
        """Map the fermion operator to a canonical Pauli operator.

        ``mapping`` may be a supported mapping name or a reusable
        :class:`FermionQubitMapping`. The result has one qubit per fermion
        mode and includes exact Jordan-Wigner/CNOT conjugation phases.
        """
        if not isinstance(mapping, str):
            from ..mapping import FermionQubitMapping

            if not isinstance(mapping, FermionQubitMapping):
                raise TypeError(
                    "mapping must be a supported name or FermionQubitMapping"
                )
            return mapping.map_fermion_operator(self, max_bytes=max_bytes)
        if mapping != "jordan_wigner":
            from ..mapping import FermionQubitMapping

            return FermionQubitMapping.from_name(
                mapping, self.space.fermions, max_bytes=max_bytes
            ).map_fermion_operator(self, max_bytes=max_bytes)
        handle = self._native_handle
        if not isinstance(handle, _native.NativeFermionOperatorHandle):
            raise RuntimeError("fermion operators must retain native handles")
        return PauliOperator._from_native_handle(
            handle.jordan_wigner(_effective_max_bytes(max_bytes))
        )

    def compile(  # type: ignore[override]
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a Fermion operator, keeping raw masks for native MVP.

        The default native-MVP route uses direct CAR descriptors in the
        Jordan--Wigner occupation-basis order. Call :meth:`map_fermions`
        first to force a Pauli mapping.
        """
        if target == "native_mvp" and mapping == "jordan_wigner":
            result = _compile_finite(
                self,
                target,
                {},
                max_bytes,
                storage=storage,
                mapping="jordan_wigner",
            )
        else:
            result = self.map_fermions(mapping, max_bytes=max_bytes).compile(
                target, storage=storage, max_bytes=max_bytes
            )
        if isinstance(result, (NativeMVPPlan, BackendMVPPlan)):
            return _with_plan_metadata(
                result,
                mapping=_mapping_name(mapping),
                source_term_count=self.term_count,
            )
        return result

    def to_majorana(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> "MajoranaOperator":
        """Convert this canonical fermion operator to the Majorana and charge algebra."""
        from ..majorana import fermion_to_majorana

        return fermion_to_majorana(self, max_bytes=max_bytes)
