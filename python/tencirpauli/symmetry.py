"""Z2 symmetry tapering and explicit U(1) sector APIs."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np

from . import _native
from ._native_mvp import NativePlanMetadata, _NativeMvpFacade
from .hamiltonian import (
    DEFAULT_MAX_BYTES,
    COOMatrix,
    CSRMatrix,
    _check_allocation,
    _effective_max_bytes,
    _enforced_max_bytes,
    _validate_apply_into_buffers,
    _validate_max_bytes,
)
from .pauli import PauliOperator, PauliWord


U1Storage = Literal["lazy", "eager"]


@dataclass(frozen=True)
class Z2SymmetryAnalysis:
    """Deterministic, exactly validated Pauli ``Z2`` symmetry analysis.

    The input is a Pauli Hamiltonian. A discovered generator ``S`` satisfies
    ``S**2 = I`` and commutes with every Hamiltonian term, so its eigenvalue
    labels sectors by ``+1`` or ``-1``. Generators may contain ``X`` and ``Y``
    factors and therefore need not be diagonal in the computational or
    occupation basis. For a fermionic source, choose a fermion-to-qubit
    encoding first (Jordan--Wigner, parity, or Bravyi--Kitaev), then call
    ``find_z2_symmetries`` on the resulting :class:`PauliOperator`.

    This is a Pauli stabilizer symmetry, not an additive occupation charge
    and not an occupation-parity sector. ``constraint_rank`` is the GF(2)
    null-space dimension of the symmetry constraint matrix and can exceed the
    number of mutually commuting generators selected for tapering. ``rank``
    is the number of selected commuting generators.
    """

    nqubits: int
    generators: Tuple[PauliWord, ...]
    constraint_rank: int

    @property
    def rank(self) -> int:
        """Return selected commuting-generator count.

        This is ``len(generators)`` and is distinct from ``constraint_rank``
        (the GF(2) null-space dimension) and from ``ChargeSector.rank`` or
        ``U1Sector.rank`` (basis-state indices).
        """
        return len(self.generators)

    def tapering_plan(self, sector: Sequence[int]) -> "Z2TaperingPlan":
        """Build a reusable Clifford plan for selected Pauli eigenvalues.

        ``sector[i]`` selects the ``+1`` or ``-1`` eigenvalue of
        ``generators[i]``. The returned plan applies a Clifford change of
        Pauli basis, substitutes those selected eigenvalues, and removes the
        corresponding qubits. It does not construct an occupation-space
        ``ChargeSector``.
        """
        selected = tuple(_validate_sector_value(value) for value in sector)
        native_plan = _native.pauli_z2_tapering_plan(
            self.nqubits,
            tuple(word.to_codes() for word in self.generators),
            selected,
        )
        return Z2TaperingPlan._from_native(native_plan)


@dataclass(frozen=True, init=False)
class Z2TaperingPlan:
    """Reusable Clifford transform and selected Pauli ``Z2`` sector.

    ``transform_operator`` applies this same transform to compatible Pauli
    Hamiltonians or observables. The plan is intentionally Pauli-specific:
    it does not automatically transform an :class:`AdditiveCharge`, combine
    with a ``ChargeSector``, or implement occupation-parity restriction.
    Such a composition requires checking that both constraints act within the
    same selected subspace and is not implied by having the same word
    ``Z2`` in their names.
    """

    nqubits_before: int
    nqubits_after: int
    generators: Tuple[PauliWord, ...]
    sector: Tuple[int, ...]
    removed_qubits: Tuple[int, ...]
    clifford_operations: np.ndarray[Any, Any]
    _native_plan: Any

    def __init__(self, native_plan: Any) -> None:
        generators = tuple(
            PauliWord.from_codes(codes) for codes in native_plan.generators
        )
        operations_raw = tuple(native_plan.clifford_operations)
        operations = np.asarray(operations_raw, dtype=np.int64).reshape(
            (len(operations_raw), 3)
        )
        operations.flags.writeable = False
        object.__setattr__(self, "nqubits_before", int(native_plan.nqubits_before))
        object.__setattr__(self, "nqubits_after", int(native_plan.nqubits_after))
        object.__setattr__(self, "generators", generators)
        object.__setattr__(
            self, "sector", tuple(int(value) for value in native_plan.sector)
        )
        object.__setattr__(
            self,
            "removed_qubits",
            tuple(int(value) for value in native_plan.removed_qubits),
        )
        object.__setattr__(self, "clifford_operations", operations)
        object.__setattr__(self, "_native_plan", native_plan)

    @classmethod
    def _from_native(cls, native_plan: Any) -> "Z2TaperingPlan":
        return cls(native_plan)

    def transform_operator(self, operator: PauliOperator) -> PauliOperator:
        """Transform a compatible operator and substitute the selected sector."""
        if not isinstance(operator, PauliOperator):
            raise TypeError(f"expected PauliOperator, got {type(operator).__name__}")
        if operator._native_handle is None:
            raise RuntimeError("PauliOperator must retain a native handle")
        result = self._native_plan.transform_operator_handle(operator._native_handle)
        return PauliOperator._from_native_handle(result)


@dataclass(frozen=True)
class U1Sector:
    """Fixed-Hamming-weight basis with TensorCircuit integer ordering."""

    nqubits: int
    particle_number: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.nqubits, int)
            or isinstance(self.nqubits, bool)
            or self.nqubits < 0
        ):
            raise ValueError("nqubits must be a non-negative integer")
        if (
            not isinstance(self.particle_number, int)
            or isinstance(self.particle_number, bool)
            or not 0 <= self.particle_number <= self.nqubits
        ):
            raise ValueError("particle_number must be between 0 and nqubits")
        dimension = math.comb(self.nqubits, self.particle_number)
        if dimension > np.iinfo(np.uint64).max:
            raise OverflowError("U1 sector dimension exceeds uint64 restricted indices")
        if dimension > np.iinfo(np.intp).max:
            raise OverflowError("U1 sector dimension exceeds platform indices")

    @property
    def dimension(self) -> int:
        """Number of basis states, ``C(nqubits, particle_number)``."""
        return math.comb(self.nqubits, self.particle_number)

    def rank(self, bitstring: int | Sequence[int]) -> int:
        """Return the ascending-basis rank without materializing the basis."""
        value = _coerce_bitstring(self.nqubits, bitstring)
        if bin(value).count("1") != self.particle_number:
            raise ValueError("bitstring has the wrong Hamming weight")
        rank = 0
        ones = 0
        for position in range(self.nqubits):
            if (value >> (self.nqubits - 1 - position)) & 1:
                remaining = self.nqubits - position - 1
                rank += math.comb(remaining, self.particle_number - ones)
                ones += 1
        return rank

    def unrank(self, index: int) -> Tuple[int, ...]:
        """Return the occupation bits at a restricted index.

        The tuple is always ordered from qubit zero to qubit ``nqubits - 1``;
        unlike the historical API, its type does not depend on system width.
        """
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < self.dimension
        ):
            raise IndexError("restricted basis index is out of range")
        remaining_index = index
        remaining_ones = self.particle_number
        value = 0
        for position in range(self.nqubits):
            zero_count = math.comb(self.nqubits - position - 1, remaining_ones)
            if remaining_index >= zero_count:
                remaining_index -= zero_count
                value |= 1 << (self.nqubits - 1 - position)
                remaining_ones -= 1
        return tuple(
            (value >> (self.nqubits - 1 - position)) & 1
            for position in range(self.nqubits)
        )

    def basis_states(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Return read-only ``uint8`` basis rows in restricted-sector order."""
        _validate_max_bytes(max_bytes)
        _check_allocation(
            self.dimension * self.nqubits,
            max_bytes,
            "U1 basis states",
        )
        result = np.asarray(
            [self.unrank(index) for index in range(self.dimension)], dtype=np.uint8
        ).reshape((self.dimension, self.nqubits))
        result.flags.writeable = False
        return cast(np.ndarray[Any, Any], result)

    def basis_words_packed(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Return the advanced packed ``uint64`` U(1) basis representation."""
        _validate_max_bytes(max_bytes)
        dimension, word_count, values = _native.u1_basis_words(
            self.nqubits, self.particle_number, _effective_max_bytes(max_bytes)
        )
        result = np.asarray(values, dtype=np.uint64).reshape((dimension, word_count))
        result.flags.writeable = False
        return cast(np.ndarray[Any, Any], result)


@dataclass(frozen=True, init=False)
class U1RestrictedOperator:
    """A validated Pauli operator restricted to one U(1) sector."""

    sector: U1Sector
    dimension: int
    term_count: int
    _native_operator: Any
    _native_lazy_plan: Any
    storage: U1Storage
    _operator: Optional[PauliOperator]
    _lock: Any
    _lazy_estimate: int

    def __init__(
        self,
        sector: U1Sector,
        native_operator: Any,
        term_count: int = 0,
        *,
        operator: Optional[PauliOperator] = None,
        storage: U1Storage = "eager",
        native_lazy_plan: Any = None,
    ) -> None:
        if storage not in {"lazy", "eager"}:
            raise ValueError("storage must be either 'eager' or 'lazy'")
        if storage == "lazy" and operator is None:
            raise ValueError("lazy U1 restriction requires the source operator")
        if native_operator is None and native_lazy_plan is None:
            raise RuntimeError("U1 restriction requires a native plan")
        object.__setattr__(self, "sector", sector)
        object.__setattr__(
            self,
            "dimension",
            int(
                native_operator.dimension
                if native_operator is not None
                else native_lazy_plan.dimension
            ),
        )
        object.__setattr__(self, "term_count", int(term_count))
        object.__setattr__(self, "_native_operator", native_operator)
        object.__setattr__(self, "_native_lazy_plan", native_lazy_plan)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "_operator", operator)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(
            self,
            "_lazy_estimate",
            int(
                native_lazy_plan.estimated_bytes
                if native_lazy_plan is not None
                else sector.dimension * 16
                + (sector.dimension + 1) * 8
                + term_count * (sector.nqubits // 8 + 32)
            ),
        )

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Apply the restricted operator without allocating a full-space state."""
        _validate_max_bytes(max_bytes)
        values = np.asarray(state, dtype=np.complex128)
        if values.ndim != 1 or values.shape[0] != self.dimension:
            raise ValueError(
                f"state must have shape ({self.dimension},), got {values.shape}"
            )
        native = (
            self._native_operator
            if self._native_operator is not None
            else self._native_lazy_plan
        )
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                native.apply(
                    np.ascontiguousarray(values), _effective_max_bytes(max_bytes)
                ),
                dtype=np.complex128,
            ),
        )

    def apply_into(
        self,
        input_state: np.ndarray[Any, Any],
        output_state: np.ndarray[Any, Any],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> None:
        """Apply into strict, non-overlapping caller-owned buffers."""
        _validate_apply_into_buffers(input_state, output_state, self.dimension)
        native = (
            self._native_operator
            if self._native_operator is not None
            else self._native_lazy_plan
        )
        native.apply_into(input_state, output_state, _effective_max_bytes(max_bytes))

    def mvp_plan(
        self,
        *,
        storage: U1Storage = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "U1MvpPlan":
        """Build a fixed reusable restricted matrix-free plan."""
        _validate_max_bytes(max_bytes)
        if storage not in {"lazy", "eager"}:
            raise ValueError("storage must be either 'eager' or 'lazy'")
        if storage == "lazy":
            lazy = self._ensure_lazy(max_bytes)
            _check_allocation(
                int(lazy.estimated_bytes),
                max_bytes,
                "U1 MVP plan",
            )
            return U1MvpPlan(
                self.sector,
                lazy,
                self.term_count,
                storage="lazy",
            )
        native = self._ensure_eager(max_bytes)
        return U1MvpPlan(
            self.sector,
            native.mvp_plan(_effective_max_bytes(max_bytes)),
            self.term_count,
            storage="eager",
        )

    @property
    def estimated_bytes(self) -> int:
        if self._native_operator is not None:
            return int(
                self._lazy_estimate + self._native_operator.transition_count * 32
            )
        return self._lazy_estimate

    def _ensure_eager(self, max_bytes: Optional[int]) -> Any:
        return self._ensure_eager_for_target(max_bytes, None)

    def _ensure_lazy(self, max_bytes: Optional[int]) -> Any:
        native = self._native_lazy_plan
        if native is not None:
            return native
        if self._operator is None or self._operator._native_handle is None:
            raise RuntimeError("U1 lazy cache requires a native Pauli handle")
        with self._lock:
            native = self._native_lazy_plan
            if native is None:
                _check_allocation(self._lazy_estimate, max_bytes, "U1 lazy MVP plan")
                native = _native.pauli_restrict_u1_lazy_handle(
                    self._operator._native_handle,
                    self.sector.particle_number,
                    _effective_max_bytes(max_bytes),
                )
                object.__setattr__(self, "_native_lazy_plan", native)
            return native

    def _ensure_eager_for_target(
        self, max_bytes: Optional[int], target: Optional[str]
    ) -> Any:
        native = self._native_operator
        if native is not None:
            if target is not None:
                _check_allocation(
                    _u1_materialization_bytes(
                        self.dimension, native.transition_count, target
                    ),
                    max_bytes,
                    f"U1 {target} materialization",
                )
            return native
        if self._operator is None:
            raise RuntimeError("U1 eager cache has no source operator")
        with self._lock:
            native = self._native_operator
            if native is None:
                enforced_max_bytes = _enforced_max_bytes(max_bytes)
                if enforced_max_bytes is None:
                    remaining = None
                else:
                    _check_allocation(
                        self._lazy_estimate,
                        max_bytes,
                        "U1 lazy MVP plan",
                    )
                    remaining = enforced_max_bytes - self._lazy_estimate
                target_floor = (
                    0
                    if target is None
                    else _u1_materialization_bytes(self.dimension, 0, target)
                )
                if remaining is not None:
                    _check_allocation(
                        target_floor,
                        remaining,
                        (
                            f"U1 {target} materialization preflight"
                            if target is not None
                            else "U1 eager MVP plan"
                        ),
                    )
                    construction_budget = remaining - target_floor
                else:
                    construction_budget = None
                if self._operator._native_handle is None:
                    raise RuntimeError("U1 eager cache requires a native Pauli handle")
                native = _native.pauli_restrict_u1_handle(
                    self._operator._native_handle,
                    self.sector.particle_number,
                    _effective_max_bytes(construction_budget),
                )
                transition_count = native.transition_count
                retained_bytes = self._lazy_estimate + transition_count * 32
                target_bytes = (
                    0
                    if target is None
                    else _u1_materialization_bytes(
                        self.dimension, transition_count, target
                    )
                )
                _check_allocation(
                    retained_bytes + target_bytes,
                    max_bytes,
                    "U1 eager cache and materialization",
                )
                object.__setattr__(self, "_native_operator", native)
            return native

    def dense(
        self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES
    ) -> np.ndarray[Any, Any]:
        """Materialize the bounded dense matrix in restricted-space ordering."""
        _validate_max_bytes(max_bytes)
        dimension, values = self._ensure_eager_for_target(max_bytes, "dense").dense(
            _effective_max_bytes(max_bytes)
        )
        return cast(
            np.ndarray[Any, Any],
            np.asarray(values, dtype=np.complex128).reshape((dimension, dimension)),
        )

    def coo(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> COOMatrix:
        """Materialize deterministic COO arrays in restricted-space ordering."""
        _validate_max_bytes(max_bytes)
        dimension, rows, columns, values = self._ensure_eager_for_target(
            max_bytes, "coo"
        ).coo(_effective_max_bytes(max_bytes))
        return COOMatrix(
            np.asarray(rows, dtype=np.uint64),
            np.asarray(columns, dtype=np.uint64),
            np.asarray(values, dtype=np.complex128),
            (dimension, dimension),
        )

    def csr(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> CSRMatrix:
        """Materialize bounded CSR arrays in restricted-space ordering."""
        _validate_max_bytes(max_bytes)
        dimension, indptr, indices, values = self._ensure_eager_for_target(
            max_bytes, "csr"
        ).csr(_effective_max_bytes(max_bytes))
        return CSRMatrix(
            np.asarray(indptr, dtype=np.uint64),
            np.asarray(indices, dtype=np.uint64),
            np.asarray(values, dtype=np.complex128),
            (dimension, dimension),
        )

    def compile(
        self,
        target: str,
        *,
        storage: U1Storage = "lazy",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> Union[
        np.ndarray[Any, Any],
        COOMatrix,
        CSRMatrix,
        "U1MvpPlan",
    ]:
        """Compile one supported U(1)-restricted target.

        U(1) restrictions expose dense/COO/CSR materialization and native MVP
        plans; backend MVP is not a supported representation for this family.
        """
        if target == "dense":
            return self.dense(max_bytes=max_bytes)
        if target == "coo":
            return self.coo(max_bytes=max_bytes)
        if target == "csr":
            return self.csr(max_bytes=max_bytes)
        if target == "native_mvp":
            return self.mvp_plan(storage=storage, max_bytes=max_bytes)
        raise ValueError("target must be one of 'dense', 'coo', 'csr', or 'native_mvp'")


@dataclass(frozen=True, init=False)
class U1MvpPlan(_NativeMvpFacade):
    """Reusable matrix-free plan over a fixed-particle-number basis."""

    sector: U1Sector
    _execution: Any = field(repr=False, compare=False)
    _metadata: NativePlanMetadata = field(repr=False, compare=False)

    def __init__(
        self,
        sector: U1Sector,
        native_plan: Any,
        term_count: int = 0,
        *,
        storage: U1Storage = "eager",
    ) -> None:
        if storage not in {"lazy", "eager"}:
            raise ValueError("storage must be either 'eager' or 'lazy'")
        if native_plan is None:
            raise RuntimeError("U1 MVP plan requires a native plan")
        dimension = int(native_plan.dimension)
        eager_transition_count = (
            0 if storage == "lazy" else int(native_plan.transition_count)
        )
        transition_count = None if storage == "lazy" else eager_transition_count
        retained_bytes = int(
            int(native_plan.estimated_bytes)
            if storage == "lazy"
            else dimension * 16 + (dimension + 1) * 8 + eager_transition_count * 32
        )
        object.__setattr__(self, "sector", sector)
        object.__setattr__(self, "_execution", native_plan)
        object.__setattr__(
            self,
            "_metadata",
            NativePlanMetadata(
                dimension=dimension,
                term_count=int(term_count),
                storage=storage,
                basis_ordering="qubit0_msb_matrix",
                target="native_mvp",
                execution_family="product",
                native_variant=str(getattr(native_plan, "native_variant", "qubit")),
                retained_format="compiled_actions" if storage == "lazy" else "csr",
                source_term_count=int(term_count),
                schedule_count=int(term_count),
                estimated_retained_bytes=retained_bytes,
                estimated_construction_peak_bytes=retained_bytes,
                transition_count=transition_count,
            ),
        )

    @property
    def _native_plan(self) -> Any:
        """Compatibility access to the single retained native execution handle."""
        return self._execution


def _validate_sector_value(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1):
        raise ValueError("sector values must be +1 or -1")
    return value


def _coerce_bitstring(nqubits: int, bitstring: int | Sequence[int]) -> int:
    if isinstance(bitstring, bool):
        raise TypeError("bitstring must be an integer or a sequence of bits")
    if isinstance(bitstring, int):
        value = bitstring
    else:
        bits = tuple(bitstring)
        if len(bits) != nqubits or any(bit not in (0, 1) for bit in bits):
            raise ValueError("bitstring must contain exactly nqubits binary values")
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
    if value < 0 or (nqubits < value.bit_length()):
        raise ValueError("bitstring is outside the computational basis")
    return value


def _canonical_u1_sector(sector: object) -> Optional[U1Sector]:
    """Recognize one canonical qubit-number constraint without trusting flags."""
    try:
        from .charge import ChargeSector

        if not isinstance(sector, ChargeSector):
            return None
        if sector.space.fermions or sector.space.bosons or sector.space.qudits:
            return None
        if len(sector.constraints) != 1:
            return None
        charge, requested = sector.constraints[0]
        if charge.offset != 0 or any(
            levels != (0, 1) for levels in charge.qubit_levels
        ):
            return None
        if requested < 0 or requested > sector.space.qubits:
            return None
        candidate = U1Sector(sector.space.qubits, requested)
        if candidate.dimension != sector.dimension:
            return None
        return candidate
    except (TypeError, ValueError, OverflowError):
        return None


def _u1_materialization_bytes(
    dimension: int, transition_count: int, target: str
) -> int:
    if target == "dense":
        return dimension * dimension * 16
    if target == "coo":
        return transition_count * 32
    if target == "csr":
        return (dimension + 1) * 8 + transition_count * 24
    raise ValueError(f"unsupported U1 materialization target: {target}")


def _restrict_u1(
    operator: PauliOperator,
    sector: U1Sector,
    max_bytes: Optional[int],
    *,
    term_count: Optional[int] = None,
    storage: U1Storage = "lazy",
) -> U1RestrictedOperator:
    if storage not in {"lazy", "eager"}:
        raise ValueError("storage must be either 'eager' or 'lazy'")
    from .charge import AdditiveCharge
    from .structured import OperatorSpace

    number = AdditiveCharge(
        OperatorSpace(qubits=sector.nqubits),
        qubits={index: (0, 1) for index in range(sector.nqubits)},
    )
    if (
        storage == "lazy"
        and not operator.analyze_charge(number, max_bytes=max_bytes).is_conserved
    ):
        raise ValueError(
            "selected U1 sector requires an exactly conserved operator; "
            "nonzero U(1) sector leakage was detected"
        )
    native_operator = None
    native_lazy_plan = None
    if operator._native_handle is None:
        raise RuntimeError("U1 restriction requires a native Pauli handle")
    if storage == "eager":
        native_operator = _native.pauli_restrict_u1_handle(
            operator._native_handle,
            sector.particle_number,
            _effective_max_bytes(max_bytes),
        )
    else:
        native_lazy_plan = _native.pauli_restrict_u1_lazy_handle(
            operator._native_handle,
            sector.particle_number,
            _effective_max_bytes(max_bytes),
        )
    return U1RestrictedOperator(
        sector,
        native_operator,
        operator.term_count if term_count is None else term_count,
        operator=operator,
        storage=storage,
        native_lazy_plan=native_lazy_plan,
    )
