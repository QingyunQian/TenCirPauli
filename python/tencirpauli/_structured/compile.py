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


def _validate_cutoffs(
    space: OperatorSpace, cutoffs: Mapping[object, object]
) -> Dict[int, int]:
    if set(cutoffs) != set(range(space.bosons)):
        raise ValueError(
            "boson_cutoffs must contain exactly one inclusive cutoff for every boson mode"
        )
    result = {}
    for mode, cutoff in cutoffs.items():
        result[_positive_mode(mode, space.bosons)] = _nonnegative_int(
            cutoff, "boson cutoff"
        )
    return result


def _compile_finite(
    operator: _StructuredOperator,
    target: str,
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
    storage: Literal["lazy", "eager"] = "lazy",
    mapping: Optional[str] = None,
) -> CompileResult:
    dimensions = tuple(
        cutoffs[axis.index] + 1 if axis.domain == "boson" else axis.dimension
        for axis in operator.space._axes
    )
    if not dimensions:
        dimensions = (1,)
    dimension = math.prod(dimensions)
    if dimension > np.iinfo(np.intp).max:
        raise OverflowError(
            "finite basis dimension cannot be represented by platform indices"
        )
    if target == "native_mvp":
        native_plan = _native_structured_mvp_plan(
            operator, dimensions, cutoffs, max_bytes, storage
        )
        qudit_dimension = operator.space.qudits[0] if operator.space.qudits else None
        basis_ordering = (
            "qubit0_msb_matrix"
            if operator.space.fermions
            and not operator.space.bosons
            and not operator.space.qubits
            and not operator.space.qudits
            else MIXED_RADIX_BASIS_ORDERING
        )
        execution_family = str(getattr(native_plan, "execution_family", "cartesian"))
        retained_format = str(
            getattr(
                native_plan,
                "retained_format",
                "compiled_actions" if storage == "lazy" else "csr",
            )
        )
        schedule_count = int(
            getattr(native_plan, "schedule_count", operator.term_count)
        )
        estimated_retained_bytes = int(native_plan.estimated_bytes)
        estimated_construction_peak_bytes = int(
            getattr(native_plan, "estimated_construction_peak_bytes", 0)
        )
        native_transition_count = getattr(native_plan, "transition_count", None)
        transition_count = (
            int(native_transition_count)
            if native_transition_count is not None
            else None
        )
        return NativeMVPPlan(
            (
                operator.space.fermions
                if not operator.space.bosons
                and not operator.space.qubits
                and not operator.space.qudits
                else 0
            ),
            operator.term_count,
            native_plan,
            storage=storage,
            local_dimensions=dimensions,
            basis_ordering=basis_ordering,
            estimated_bytes=int(native_plan.estimated_bytes),
            mapping=mapping,
            boson_cutoffs=tuple(sorted(cutoffs.items())),
            boson_boundary="projected_fock" if cutoffs else None,
            qudit_dimension=qudit_dimension,
            weyl_convention="X^a Z^b" if qudit_dimension is not None else None,
            execution_family=execution_family,
            retained_format=retained_format,
            schedule_count=schedule_count,
            estimated_retained_bytes=estimated_retained_bytes,
            estimated_construction_peak_bytes=estimated_construction_peak_bytes,
            transition_count=transition_count,
            _factory_token=_PLAN_FACTORY_TOKEN,
        )
    if target == "dense":
        _check_allocation(
            dimension * dimension * 16, max_bytes, "dense structured matrix"
        )
        native_dimension, values = _native_structured_dense(
            operator, dimensions, cutoffs, max_bytes
        )
        return cast(
            np.ndarray[Any, Any],
            np.asarray(values, dtype=np.complex128).reshape(
                (native_dimension, native_dimension)
            ),
        )
    if target == "backend_mvp":
        raise NotImplementedError(
            "backend_mvp is available only for Pauli and uniform-dimension Weyl plans"
        )
    if target == "csr":
        native_dimension, indptr, columns, values = _native_structured_csr(
            operator, dimensions, cutoffs, max_bytes
        )
        return CSRMatrix(indptr, columns, values, (native_dimension, native_dimension))
    native_dimension, rows, columns, values = _native_structured_sparse(
        operator, dimensions, cutoffs, max_bytes
    )
    if target == "coo":
        _check_allocation(
            int(rows.nbytes + columns.nbytes + values.nbytes),
            max_bytes,
            "COO structured matrix",
        )
        return COOMatrix(rows, columns, values, (dimension, dimension))
    raise AssertionError(target)


def _direct_weyl_backend_plan(
    operator: _StructuredOperator, max_bytes: Optional[int]
) -> BackendMVPPlan:
    """Build the compact, versioned plan for a uniform Weyl layout."""
    dimensions = operator.space.local_dimensions
    if (
        not dimensions
        or len(set(dimensions)) != 1
        or any(axis.domain != "qudit" for axis in operator.space._axes)
    ):
        raise NotImplementedError(
            "direct Weyl backend MVP requires a uniform pure-qudit layout"
        )
    # Check the Python-int product before allocating exponent arrays.  NumPy's
    # intp product wraps on overflow and would publish a false plan dimension.
    dimension = math.prod(dimensions)
    if dimension > int(np.iinfo(np.intp).max):
        raise OverflowError(
            "finite basis dimension cannot be represented by platform indices"
        )
    dimension = dimensions[0]
    handle = _compile_native_hybrid_handle(operator)
    term_count = handle.term_count
    a_flat, b_flat, native_coefficients = handle.direct_weyl_flat(
        _effective_max_bytes(max_bytes)
    )
    a_exponents = np.asarray(a_flat, dtype=np.uint32).reshape(
        (term_count, len(dimensions))
    )
    b_exponents = np.asarray(b_flat, dtype=np.uint32).reshape(
        (term_count, len(dimensions))
    )
    coefficients = np.asarray(native_coefficients, dtype=np.complex128)
    estimated_bytes = int(
        a_exponents.nbytes
        + b_exponents.nbytes
        + coefficients.nbytes
        + len(dimensions) * 8
        + term_count * 64
    )
    _check_allocation(estimated_bytes, max_bytes, "direct Weyl backend MVP plan")
    return BackendMVPPlan(
        2,
        0,
        0,
        np.empty((0, 0), dtype=np.uint64),
        np.empty((0, 0), dtype=np.uint64),
        coefficients,
        local_dimensions=dimensions,
        basis_ordering=DIRECT_WEYL_BASIS_ORDERING,
        ordering=DIRECT_WEYL_BASIS_ORDERING,
        estimated_bytes=estimated_bytes,
        plan_kind="direct_weyl",
        qudit_dimension=dimension,
        a_exponents=a_exponents,
        b_exponents=b_exponents,
        required_operations=("broadcast_phase", "cyclic_shift", "multiply", "add"),
        target="backend_mvp",
        source_term_count=term_count,
        plan_term_count=term_count,
        weyl_convention="X^a Z^b",
        _factory_token=_PLAN_FACTORY_TOKEN,
    )


def _native_structured_sparse(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    compile_handle = _compile_native_hybrid_handle(operator)
    if compile_handle is None:
        raise RuntimeError("structured operators require a native hybrid handle")
    native_dimension, rows, columns, values = _native.structured_sparse_handle(
        compile_handle,
        list(dimensions),
        _structured_axis_descriptor(operator.space),
        _effective_max_bytes(max_bytes),
    )
    return (
        native_dimension,
        np.asarray(rows, dtype=np.uint64),
        np.asarray(columns, dtype=np.uint64),
        np.asarray(values, dtype=np.complex128),
    )


def _native_structured_csr(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    del cutoffs
    compile_handle = _compile_native_hybrid_handle(operator)
    if compile_handle is None:
        raise RuntimeError("structured operators require a native hybrid handle")
    native_dimension, indptr, columns, values = _native.structured_csr_handle(
        compile_handle,
        list(dimensions),
        _structured_axis_descriptor(operator.space),
        _effective_max_bytes(max_bytes),
    )
    return (
        native_dimension,
        np.asarray(indptr, dtype=np.uint64),
        np.asarray(columns, dtype=np.uint64),
        np.asarray(values, dtype=np.complex128),
    )


def _native_structured_mvp_plan(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
    storage: Literal["lazy", "eager"],
) -> Any:
    compile_handle = _compile_native_hybrid_handle(operator)
    if compile_handle is None:
        raise RuntimeError("structured operators require a native hybrid handle")
    plan_factory = (
        _native.structured_eager_plan_handle
        if storage == "eager"
        else _native.structured_sparse_plan_handle
    )
    return plan_factory(
        compile_handle,
        list(dimensions),
        _structured_axis_descriptor(operator.space),
        _effective_max_bytes(max_bytes),
    )


def _structured_axis_descriptor(space: OperatorSpace) -> List[Tuple[int, int]]:
    domains = {"fermion": 0, "boson": 1, "qubit": 2, "qudit": 3}
    return [(domains[axis.domain], axis.index) for axis in space._axes]


def _native_structured_dense(
    operator: _StructuredOperator,
    dimensions: Tuple[int, ...],
    cutoffs: Mapping[int, int],
    max_bytes: Optional[int],
) -> Tuple[int, Any]:
    compile_handle = _compile_native_hybrid_handle(operator)
    if compile_handle is None:
        raise RuntimeError("structured operators require a native hybrid handle")
    return _native.structured_dense_handle(
        compile_handle,
        list(dimensions),
        _structured_axis_descriptor(operator.space),
        _effective_max_bytes(max_bytes),
    )


def _compile_native_hybrid_handle(
    operator: _StructuredOperator,
) -> _native.NativeHybridOperatorHandle:
    """Promote specialized native storage without exporting operator terms."""
    handle = operator._native_handle
    if isinstance(handle, _native.NativeHybridOperatorHandle):
        return handle
    if isinstance(handle, _native.NativeFermionOperatorHandle):
        return handle.to_hybrid()
    if isinstance(handle, _native.NativeBosonOperatorHandle):
        return handle.to_hybrid()
    raise RuntimeError("structured operators require a native handle")
