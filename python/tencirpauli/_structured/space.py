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

if TYPE_CHECKING:
    from .boson import BosonOperator
    from .fermion import FermionOperator
    from .hybrid import HybridOperator, OperatorBuilder
    from .mapping import QuditWeylOperator


@dataclass(frozen=True)
class _Axis:
    domain: str
    index: int
    dimension: int


class OperatorSpace:
    """Immutable ordered logical subsystem layout for structured operators.

    Axes are ordered as fermions, bosons, qubits, then uniform-dimension
    qudits. This ordering controls term serialization, mixed-radix basis
    ordering, embedding, tensor products, and finite matrix targets.

    Examples:
        >>> import tencirpauli as tcp
        >>> space = tcp.OperatorSpace(fermions=1, qubits=1)
        >>> operator = space.fermion.create(0) * space.qubit.z(0)
        >>> operator.term_count
        1
    """

    __slots__ = ("_axes", "_locked", "bosons", "fermions", "qubits", "qudits")
    fermions: int
    bosons: int
    qubits: int
    qudits: Tuple[int, ...]
    _axes: Tuple[_Axis, ...]
    _locked: bool

    def __init__(
        self,
        fermions: int = 0,
        bosons: int = 0,
        qubits: int = 0,
        qudits: Sequence[int] = (),
    ) -> None:
        """Create an immutable ordered layout with uniform qudit dimensions."""
        fermions = _nonnegative_int(fermions, "fermions")
        bosons = _nonnegative_int(bosons, "bosons")
        qubits = _nonnegative_int(qubits, "qubits")
        if not isinstance(qudits, (tuple, list)):
            raise ValueError("qudits must be a tuple of local dimensions")
        dimensions = tuple(
            _nonnegative_int(value, "qudit dimension") for value in qudits
        )
        if dimensions and (
            any(value < 3 or value > _U32_MAX for value in dimensions)
            or len(set(dimensions)) != 1
        ):
            raise ValueError("qudits must have one uniform dimension 3 <= d <= 2**32-1")
        axes = tuple(
            [_Axis("fermion", index, 2) for index in range(fermions)]
            + [_Axis("boson", index, 0) for index in range(bosons)]
            + [_Axis("qubit", index, 2) for index in range(qubits)]
            + [_Axis("qudit", index, dimensions[0]) for index in range(len(dimensions))]
        )
        self._initialize(fermions, bosons, qubits, dimensions, axes)

    @classmethod
    def _from_axes(cls, axes: Sequence[_Axis]) -> "OperatorSpace":
        instance = object.__new__(cls)
        counts = {domain: 0 for domain in ("fermion", "boson", "qubit", "qudit")}
        dimensions: List[int] = []
        for axis in axes:
            counts[axis.domain] += 1
            if axis.domain == "qudit":
                dimensions.append(axis.dimension)
        instance._initialize(
            counts["fermion"],
            counts["boson"],
            counts["qubit"],
            tuple(dimensions),
            tuple(axes),
        )
        return instance

    def _initialize(
        self,
        fermions: int,
        bosons: int,
        qubits: int,
        qudits: Tuple[int, ...],
        axes: Tuple[_Axis, ...],
    ) -> None:
        object.__setattr__(self, "fermions", fermions)
        object.__setattr__(self, "bosons", bosons)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "qudits", qudits)
        object.__setattr__(self, "_axes", axes)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("OperatorSpace is immutable")
        object.__setattr__(self, name, value)

    @property
    def axes(self) -> Tuple[Tuple[str, int, int], ...]:
        """Return ordered ``(domain, index, local-dimension)`` descriptors."""
        return tuple((axis.domain, axis.index, axis.dimension) for axis in self._axes)

    @property
    def layout_fingerprint(self) -> Tuple[Tuple[str, int, int], ...]:
        """Return the immutable compatibility fingerprint for this layout."""
        return self.axes

    @property
    def local_dimensions(self) -> Tuple[int, ...]:
        """Return finite local dimensions, rejecting uncut boson axes."""
        if any(axis.domain == "boson" and axis.dimension == 0 for axis in self._axes):
            raise ValueError("boson local dimensions require explicit finite cutoffs")
        return tuple(axis.dimension for axis in self._axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OperatorSpace) and self.axes == other.axes

    def __hash__(self) -> int:
        return hash(self.axes)

    @property
    def fermion(self) -> "_FermionFactory":
        """Return factories for fermion creation and annihilation."""
        return _FermionFactory(self)

    @property
    def boson(self) -> "_BosonFactory":
        """Return factories for boson creation and annihilation."""
        return _BosonFactory(self)

    @property
    def qubit(self) -> "_QubitFactory":
        """Return factories for physical Pauli X, Y, and Z factors."""
        return _QubitFactory(self)

    @property
    def qudit(self) -> "_QuditFactory":
        """Return the direct-convention Weyl factory."""
        return _QuditFactory(self)

    def builder(self) -> "OperatorBuilder":
        """Create a mutable batched builder for this immutable space."""
        return OperatorBuilder(self)

    def embed(
        self, operator: "_StructuredOperator", **maps: object
    ) -> "_StructuredOperator":
        """Embed an operator into this space with explicit domain index maps.

        Each supplied map is a source-to-target index mapping for one of
        ``fermions``, ``bosons``, ``qubits``, or ``qudits``. Unmapped domains
        must be empty; no implicit axis matching is performed.
        """
        if not isinstance(operator, _StructuredOperator):
            raise TypeError("embed expects a structured operator")
        valid_domains = {"fermions", "bosons", "qubits", "qudits"}
        unknown = set(maps) - valid_domains
        if unknown:
            raise TypeError("unexpected embedding maps: " + ", ".join(sorted(unknown)))
        mappings: Dict[str, Dict[int, int]] = {}
        for domain, source_count in (
            ("fermions", operator.space.fermions),
            ("bosons", operator.space.bosons),
            ("qubits", operator.space.qubits),
            ("qudits", len(operator.space.qudits)),
        ):
            has_supplied = domain in maps
            supplied = maps.get(domain)
            if source_count == 0:
                if has_supplied:
                    raise ValueError(f"{domain} embedding must be empty")
                mappings[domain] = {}
                continue
            if domain == "qudits":
                source_dimension = operator.space.qudits[0]
                if not self.qudits or self.qudits[0] != source_dimension:
                    raise ValueError(
                        "qudit embedding requires equal source and target dimensions"
                    )
            if not has_supplied:
                source_value = (
                    len(operator.space.qudits)
                    if domain == "qudits"
                    else getattr(operator.space, domain)
                )
                target_value = (
                    len(self.qudits) if domain == "qudits" else getattr(self, domain)
                )
                if source_value == target_value:
                    mappings[domain] = {index: index for index in range(source_count)}
                else:
                    raise ValueError(f"an explicit {domain} embedding map is required")
            elif supplied is None:
                raise ValueError(f"{domain} embedding must be a mapping or sequence")
            elif isinstance(supplied, Mapping):
                if any(
                    not isinstance(key, int)
                    or isinstance(key, bool)
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    for key, value in supplied.items()
                ):
                    raise ValueError(f"{domain} embedding indices must be integers")
                mappings[domain] = dict(supplied)
            else:
                if not isinstance(supplied, (tuple, list)):
                    raise ValueError(
                        f"{domain} embedding must be a mapping or sequence"
                    )
                values = tuple(supplied)
                if any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in values
                ):
                    raise ValueError(f"{domain} embedding indices must be integers")
                mappings[domain] = {index: value for index, value in enumerate(values)}
            if set(mappings[domain]) != set(range(source_count)):
                raise ValueError(
                    f"{domain} embedding must map every source index exactly once"
                )
            target_count = (
                len(self.qudits) if domain == "qudits" else getattr(self, domain)
            )
            if any(
                value < 0 or value >= target_count
                for value in mappings[domain].values()
            ):
                raise ValueError(f"{domain} embedding indices are outside the target")
            if len(set(mappings[domain].values())) != source_count:
                raise ValueError(f"{domain} embedding targets must be injective")
        if operator.space == self and not any(maps.values()):
            return operator
        native_handle = operator._native_handle
        if native_handle is None:
            raise RuntimeError("structured embedding requires a native handle")
        if isinstance(
            native_handle,
            (_native.NativeFermionOperatorHandle, _native.NativeBosonOperatorHandle),
        ):
            native_handle = native_handle.to_hybrid()
        if not isinstance(native_handle, _native.NativeHybridOperatorHandle):
            raise RuntimeError("structured embedding received an invalid native handle")
        result = native_handle.embed(
            self.fermions,
            self.bosons,
            self.qubits,
            len(self.qudits),
            self.qudits[0] if self.qudits else 0,
            tuple(
                mappings["fermions"][index] for index in range(operator.space.fermions)
            ),
            tuple(mappings["bosons"][index] for index in range(operator.space.bosons)),
            tuple(mappings["qubits"][index] for index in range(operator.space.qubits)),
            tuple(
                mappings["qudits"][index] for index in range(len(operator.space.qudits))
            ),
            _effective_max_bytes(DEFAULT_MAX_BYTES),
        )
        if self.fermions and not self.bosons and not self.qubits and not self.qudits:
            return _fermion_from_native(
                FermionOperator,
                self.fermions,
                result.to_fermion(),
            )
        if self.bosons and not self.fermions and not self.qubits and not self.qudits:
            return _boson_from_native(
                BosonOperator,
                self.bosons,
                result.to_boson(),
            )
        if self.qudits and not self.fermions and not self.bosons and not self.qubits:
            return _hybrid_from_native(self, result, QuditWeylOperator)
        return _hybrid_from_native(self, result)


class _Factory:
    def __init__(self, space: OperatorSpace) -> None:
        self.space = space


def _factory_native_operator(
    space: OperatorSpace,
    *,
    fermion_creation: Sequence[int] = (),
    fermion_annihilation: Sequence[int] = (),
    boson_blocks: Sequence[Tuple[int, int, int]] = (),
    qubit_codes: Optional[Sequence[int]] = None,
    qudit_triples: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> "HybridOperator":
    """Build one factory result from native canonical hybrid arrays."""
    result = _native.structured_hybrid_canonicalize(
        space.fermions,
        space.bosons,
        space.qubits,
        len(space.qudits),
        space.qudits[0] if space.qudits else 0,
        (
            [
                list((mode, 0) for mode in fermion_creation)
                + list((mode, 1) for mode in fermion_annihilation)
            ],
            [
                list(
                    (mode, 0) for mode, create, _ in boson_blocks for _ in range(create)
                )
                + list(
                    (mode, 1)
                    for mode, _, annihilate in boson_blocks
                    for _ in range(annihilate)
                )
            ],
            [list(qubit_codes or (0,) * space.qubits)],
            [qudit_triples is not None],
            [list(qudit_triples or ())],
            [1.0],
            [0.0],
        ),
        _effective_max_bytes(DEFAULT_MAX_BYTES),
    )
    return _hybrid_from_native(space, result)


class _FermionFactory(_Factory):
    @property
    def annihilate(self) -> Callable[[object], "HybridOperator"]:
        """Return the fermion annihilation factory."""
        return lambda mode: self._make(mode, "annihilate")

    @property
    def create(self) -> Callable[[object], "HybridOperator"]:
        """Return the fermion creation factory."""
        return lambda mode: self._make(mode, "create")

    def _make(self, mode: object, action: str) -> "HybridOperator":
        mode_value = _positive_mode(mode, self.space.fermions)
        return _factory_native_operator(
            self.space,
            fermion_creation=(mode_value,) if action == "create" else (),
            fermion_annihilation=(mode_value,) if action == "annihilate" else (),
        )


class _BosonFactory(_Factory):
    @property
    def annihilate(self) -> Callable[[object], "HybridOperator"]:
        """Return the boson annihilation factory."""
        return lambda mode: self._make(mode, "annihilate")

    @property
    def create(self) -> Callable[[object], "HybridOperator"]:
        """Return the boson creation factory."""
        return lambda mode: self._make(mode, "create")

    def _make(self, mode: object, action: str) -> "HybridOperator":
        mode_value = _positive_mode(mode, self.space.bosons)
        return _factory_native_operator(
            self.space,
            boson_blocks=(
                (mode_value, int(action == "create"), int(action == "annihilate")),
            ),
        )


class _QubitFactory(_Factory):
    @property
    def x(self) -> Callable[[object], "HybridOperator"]:
        """Return the Pauli-X factory."""
        return lambda index: self._make(index, 1)

    @property
    def y(self) -> Callable[[object], "HybridOperator"]:
        """Return the Pauli-Y factory."""
        return lambda index: self._make(index, 2)

    @property
    def z(self) -> Callable[[object], "HybridOperator"]:
        """Return the Pauli-Z factory."""
        return lambda index: self._make(index, 3)

    def _make(self, index: object, code: int) -> "HybridOperator":
        index_value = _positive_mode(index, self.space.qubits, "qubit")
        codes = [0] * self.space.qubits
        codes[index_value] = code
        return _factory_native_operator(self.space, qubit_codes=codes)


class _QuditFactory(_Factory):
    @property
    def weyl(self) -> Callable[[object, object, object], "HybridOperator"]:
        """Return the direct-convention Weyl factory."""
        return self._make

    def _make(self, site: object, a: object, b: object) -> "HybridOperator":
        site_value = _positive_mode(site, len(self.space.qudits), "site")
        return _factory_native_operator(
            self.space,
            qudit_triples=(
                (
                    site_value,
                    _nonnegative_int(a, "a") % self.space.qudits[0],
                    _nonnegative_int(b, "b") % self.space.qudits[0],
                ),
            ),
        )
