"""Rust-native local-Pauli and global-Clifford classical shadows."""

from __future__ import annotations

import math
import operator
import secrets
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np

from . import _native
from .hamiltonian import DEFAULT_MAX_BYTES, _effective_max_bytes, _validate_max_bytes
from .pauli import PauliOperator, PauliTerm, PauliWord


QubitOrder = Literal["q0_first"]
Protocol = Literal["pauli", "clifford"]


@dataclass(frozen=True)
class PauliSnapshotArrays:
    """Owned setting-major local-Pauli snapshot arrays."""

    __slots__ = ("bases", "outcomes", "qubit_order")

    bases: np.ndarray[Any, Any]
    outcomes: np.ndarray[Any, Any]
    qubit_order: QubitOrder


@dataclass(frozen=True)
class CliffordSnapshotArrays:
    """Owned packed global-Clifford frame and outcome arrays."""

    __slots__ = (
        "frame_signs",
        "frame_version",
        "frame_x_words",
        "frame_z_words",
        "nqubits",
        "outcomes",
        "qubit_order",
    )

    frame_x_words: np.ndarray[Any, Any]
    frame_z_words: np.ndarray[Any, Any]
    frame_signs: np.ndarray[Any, Any]
    outcomes: np.ndarray[Any, Any]
    nqubits: int
    frame_version: Literal[1]
    qubit_order: QubitOrder


def _validate_qubit_order(value: object) -> QubitOrder:
    if value != "q0_first":
        raise ValueError("qubit_order must be 'q0_first'")
    return "q0_first"


def _normalize_integer_array(
    value: object, name: str, dtype: np.dtype[Any]
) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise TypeError(
            f"{name} must be an integer array; floating-point arrays are rejected"
        )
    limits = np.iinfo(dtype)
    if np.any(array < limits.min) or np.any(array > limits.max):
        raise ValueError(f"{name} contains values outside {dtype} range")
    return cast(np.ndarray[Any, Any], np.ascontiguousarray(array, dtype=dtype))


def _normalize_bases(value: object) -> np.ndarray[Any, Any]:
    array = _normalize_integer_array(value, "bases", np.dtype(np.uint8))
    if array.ndim != 2:
        raise ValueError("bases must have shape (nsettings, nqubits)")
    if np.any((array < 1) | (array > 3)):
        raise ValueError("bases codes must be 1=X, 2=Y, or 3=Z")
    return array


def _normalize_outcomes(
    value: object, nsettings: int, nqubits: int
) -> np.ndarray[Any, Any]:
    array = _normalize_integer_array(value, "outcomes", np.dtype(np.uint8))
    if array.ndim == 2:
        if array.shape != (nsettings, nqubits):
            raise ValueError("flat outcomes must have shape (nsettings, nqubits)")
        array = np.ascontiguousarray(array[:, None, :], dtype=np.uint8)
    elif array.ndim != 3 or array.shape[0] != nsettings or array.shape[2] != nqubits:
        raise ValueError(
            "outcomes must have shape (nsettings, repeats, nqubits) or "
            "(nsettings, nqubits)"
        )
    if array.shape[1] <= 0:
        raise ValueError("repeats must be positive")
    if np.any(array > 1):
        raise ValueError("outcomes must contain only binary values 0 or 1")
    return cast(np.ndarray[Any, Any], np.ascontiguousarray(array, dtype=np.uint8))


def _normalize_seed(seed: Optional[int]) -> int:
    if seed is None:
        return secrets.randbits(64)
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an unsigned 64-bit integer or None")
    try:
        value = operator.index(seed)
    except TypeError as error:
        raise TypeError("seed must be an unsigned 64-bit integer or None") from error
    if value < 0 or value >= 2**64:
        raise ValueError("seed must fit in uint64")
    return int(value)


def _positive_count(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _normalize_statevector(state: object) -> np.ndarray[Any, Any]:
    array = np.asarray(state)
    if array.ndim != 1:
        raise ValueError("state must be a one-dimensional complete statevector")
    if array.dtype.kind not in "biufc" or array.dtype.kind == "b":
        raise TypeError("state must be a finite real or complex numeric array")
    normalized = np.ascontiguousarray(array, dtype=np.complex128).copy()
    if normalized.size == 0 or not (normalized.size & (normalized.size - 1) == 0):
        raise ValueError("state length must be a positive power of two")
    if not np.isfinite(normalized).all():
        raise ValueError("state entries must be finite")
    norm = float(np.vdot(normalized, normalized).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise ValueError("state norm must be within 1e-10 of one")
    return cast(np.ndarray[Any, Any], normalized)


def _readonly_copy(
    value: object, dtype: Optional[np.dtype[Any]] = None
) -> np.ndarray[Any, Any]:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.flags.writeable = False
    return cast(np.ndarray[Any, Any], array)


def _operator_for_target(
    target: Union[PauliWord, PauliTerm, PauliOperator], nqubits: int
) -> PauliOperator:
    if isinstance(target, PauliWord):
        if target.nqubits != nqubits:
            raise ValueError("observable width must match snapshots.nqubits")
        return PauliOperator.from_terms(nqubits, [(target, 1.0)])
    if isinstance(target, PauliTerm):
        if target.word.nqubits != nqubits:
            raise ValueError("observable width must match snapshots.nqubits")
        coefficient = complex(target.coefficient)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("observable coefficients must be finite")
        if coefficient.imag != 0.0:
            raise ValueError(
                "PauliTerm coefficients must be real for shadow estimators"
            )
        return PauliOperator.from_terms(nqubits, [(target.word, coefficient)])
    if isinstance(target, PauliOperator):
        if target.nqubits != nqubits:
            raise ValueError("observable width must match snapshots.nqubits")
        if not target.is_hermitian(0.0):
            raise ValueError("expectation requires an exactly Hermitian observable")
        return target
    raise TypeError("observable must be a PauliWord, PauliTerm, or PauliOperator")


def _operators_for_targets(targets: object, nqubits: int) -> Tuple[PauliOperator, ...]:
    if isinstance(targets, (PauliWord, PauliTerm, PauliOperator)):
        raise TypeError("estimate_many expects a non-empty sequence of observables")
    try:
        values = tuple(cast(Iterable[object], targets))
    except TypeError as error:
        raise TypeError(
            "estimate_many expects a non-empty sequence of observables"
        ) from error
    if not values:
        raise ValueError("estimate_many requires a non-empty observable sequence")
    return tuple(_operator_for_target(value, nqubits) for value in values)  # type: ignore[arg-type]


def _normalize_subsystem(
    nqubits: int,
    subsystem_to_keep: Optional[Sequence[int]],
    subsystems_to_trace_out: Optional[Sequence[int]],
) -> Tuple[int, ...]:
    if subsystem_to_keep is not None and subsystems_to_trace_out is not None:
        raise ValueError(
            "provide only one of subsystem_to_keep or subsystems_to_trace_out"
        )
    if subsystem_to_keep is None and subsystems_to_trace_out is None:
        return tuple(range(nqubits))

    value = (
        subsystem_to_keep if subsystem_to_keep is not None else subsystems_to_trace_out
    )
    assert value is not None
    normalized: list[int] = []
    seen: set[int] = set()
    for qubit in value:
        if not isinstance(qubit, int) or isinstance(qubit, bool):
            raise TypeError("subsystem qubits must be integers")
        if qubit < 0 or qubit >= nqubits:
            raise ValueError("subsystem qubit is outside snapshots.nqubits")
        if qubit in seen:
            raise ValueError("subsystem qubits must be unique")
        seen.add(qubit)
        normalized.append(int(qubit))
    if subsystem_to_keep is not None:
        return tuple(normalized)
    return tuple(qubit for qubit in range(nqubits) if qubit not in seen)


class Snapshots:
    """Immutable Rust-owned classical-shadow snapshots.

    ``nsettings`` is the number of independently randomized settings and
    ``repeats`` is the number of outcomes retained under each setting;
    ``shots`` is their checked product. The recommended default is
    ``repeats=1`` because randomized-setting variance is not reduced by
    repeating a setting. Local-Pauli dense sampling amortizes one basis
    rotation over repeats, while global-Clifford dense sampling performs a
    fresh conditional Pauli-projection sequence for every repeat.
    """

    __slots__ = ("_native", "_nqubits", "_nsettings", "_protocol", "_repeats")

    _native: Any
    _protocol: str
    _nqubits: int
    _nsettings: int
    _repeats: int

    def __init__(self, native: Any) -> None:
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_protocol", str(native.protocol))
        object.__setattr__(self, "_nqubits", int(native.nqubits))
        object.__setattr__(self, "_nsettings", int(native.nsettings))
        object.__setattr__(self, "_repeats", int(native.repeats))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Snapshots are immutable")

    @classmethod
    def from_pauli_measurements(
        cls,
        bases: object,
        outcomes: object,
        *,
        qubit_order: QubitOrder = "q0_first",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "Snapshots":
        """Import local-Pauli measurements in setting-major form.

        ``bases`` has shape ``(nsettings, nqubits)`` with codes ``1=X``,
        ``2=Y``, ``3=Z``. ``outcomes`` has shape
        ``(nsettings, repeats, nqubits)`` or a flat per-shot shape
        ``(nsettings, nqubits)``. Floating-point arrays are rejected.
        """
        _validate_qubit_order(qubit_order)
        _validate_max_bytes(max_bytes)
        normalized_bases = _normalize_bases(bases)
        normalized_outcomes = _normalize_outcomes(
            outcomes, int(normalized_bases.shape[0]), int(normalized_bases.shape[1])
        )
        native = _native.shadows_from_pauli_measurements(
            normalized_bases,
            normalized_outcomes,
            _effective_max_bytes(max_bytes),
        )
        return cls(native)

    @classmethod
    def from_clifford_measurements(
        cls,
        frame_x_words: object,
        frame_z_words: object,
        frame_signs: object,
        outcomes: object,
        *,
        nqubits: int,
        frame_version: Literal[1] = 1,
        qubit_order: QubitOrder = "q0_first",
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "Snapshots":
        """Import version-1 packed signed commuting Clifford generators.

        Frame rows are ``(nsettings*nqubits, ceil(nqubits/64))`` and signs are
        ``(nsettings, nqubits)``; outcomes are setting-major with shape
        ``(nsettings, repeats, nqubits)``. Construction validates padding,
        commutation, independence, signs, and binary outcomes once.
        """
        _validate_qubit_order(qubit_order)
        _validate_max_bytes(max_bytes)
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        if frame_version != 1:
            raise ValueError("frame_version must be 1")
        x = _normalize_integer_array(
            frame_x_words, "frame_x_words", np.dtype(np.uint64)
        )
        z = _normalize_integer_array(
            frame_z_words, "frame_z_words", np.dtype(np.uint64)
        )
        signs_array = np.asarray(frame_signs)
        if signs_array.dtype.kind not in "iu" or signs_array.dtype.kind == "b":
            raise TypeError("frame_signs must be an integer array")
        if np.any((signs_array != -1) & (signs_array != 1)):
            raise ValueError("frame_signs must contain only -1 or +1")
        signs = np.ascontiguousarray(signs_array, dtype=np.int8)
        if signs.ndim != 2:
            raise ValueError("frame_signs must have shape (nsettings, nqubits)")
        normalized_outcomes = _normalize_outcomes(
            outcomes, int(signs.shape[0]), int(nqubits)
        )
        if x.ndim != 2 or z.ndim != 2:
            raise ValueError("frame word arrays must be two-dimensional")
        native = _native.shadows_from_clifford_measurements(
            x,
            z,
            signs,
            normalized_outcomes,
            int(nqubits),
            int(frame_version),
            _effective_max_bytes(max_bytes),
        )
        return cls(native)

    @classmethod
    def sample(
        cls,
        state: Union[object, "StabilizerState"],
        *,
        nsettings: int,
        protocol: Protocol = "pauli",
        repeats: int = 1,
        seed: Optional[int] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "Snapshots":
        """Sample complete-statevector or stabilizer snapshots natively.

        ``nsettings`` counts independent randomized settings; ``repeats``
        counts conditional outcomes per setting and ``shots`` is
        ``nsettings * repeats``. ``repeats=1`` is recommended for linear
        estimators, shadow Rényi-2, and global Clifford because increasing
        settings reduces ensemble variance. Dense local-Pauli sampling costs
        one basis transform per setting plus outcome draws; dense global
        Clifford costs a fresh ``O(nqubits * 2**nqubits)`` conditional
        projection sequence per outcome. Stabilizer inputs use tableau kernels.
        """
        _validate_max_bytes(max_bytes)
        count = _positive_count(nsettings, "nsettings")
        repeat_count = _positive_count(repeats, "repeats")
        if protocol not in ("pauli", "clifford"):
            raise ValueError("protocol must be 'pauli' or 'clifford'")
        protocol_code = 0 if protocol == "pauli" else 1
        normalized_seed = _normalize_seed(seed)
        if isinstance(state, StabilizerState):
            native = _native.shadows_sample_stabilizer(
                state._native,
                count,
                protocol_code,
                repeat_count,
                normalized_seed,
                _effective_max_bytes(max_bytes),
            )
        else:
            normalized_state = _normalize_statevector(state)
            native = _native.shadows_sample_statevector(
                normalized_state,
                count,
                protocol_code,
                repeat_count,
                normalized_seed,
                _effective_max_bytes(max_bytes),
            )
        return cls(native)

    @property
    def protocol(self) -> Protocol:
        return cast(Protocol, self._protocol)

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @property
    def nsettings(self) -> int:
        return self._nsettings

    @property
    def repeats(self) -> int:
        return self._repeats

    @property
    def shots(self) -> int:
        return self.nsettings * self.repeats

    @property
    def qubit_order(self) -> QubitOrder:
        return "q0_first"

    def to_arrays(self) -> Union[PauliSnapshotArrays, CliffordSnapshotArrays]:
        """Return owned, C-contiguous, read-only raw snapshot arrays."""
        if self.protocol == "pauli":
            bases, outcomes = self._native.pauli_arrays()
            return PauliSnapshotArrays(
                _readonly_copy(bases, np.dtype(np.uint8)),
                _readonly_copy(outcomes, np.dtype(np.uint8)),
                "q0_first",
            )
        x, z, signs, outcomes, nqubits, version = self._native.clifford_arrays()
        return CliffordSnapshotArrays(
            _readonly_copy(x, np.dtype(np.uint64)),
            _readonly_copy(z, np.dtype(np.uint64)),
            _readonly_copy(signs, np.dtype(np.int8)),
            _readonly_copy(outcomes, np.dtype(np.uint8)),
            int(nqubits),
            cast(Literal[1], int(version)),
            "q0_first",
        )

    def expectation(
        self,
        observable: Union[PauliWord, PauliTerm, PauliOperator],
        *,
        blocks: int = 1,
        return_blocks: bool = False,
    ) -> Union[float, np.ndarray[Any, Any]]:
        """Estimate one Hermitian observable by setting-level averaging.

        ``blocks=1`` returns the arithmetic mean; ``blocks>1`` partitions
        complete settings into balanced contiguous blocks and returns the
        median of their block means. Repeats are averaged inside each setting
        first. The default returns a scalar; ``return_blocks=True`` returns a
        float64 array of shape ``(blocks,)``.
        """
        block_count = _positive_count(blocks, "blocks")
        operator_value = _operator_for_target(observable, self.nqubits)
        values = self._native.expectation(
            operator_value._native_handle,
            block_count,
            bool(return_blocks),
        )
        array = np.asarray(values, dtype=np.float64)
        if return_blocks:
            return cast(
                np.ndarray[Any, Any], np.array(array, dtype=np.float64, copy=True)
            )
        return float(array[0])

    def estimate_many(
        self,
        observables: Sequence[Union[PauliWord, PauliTerm, PauliOperator]],
        *,
        blocks: int = 1,
        return_blocks: bool = False,
    ) -> np.ndarray[Any, Any]:
        """Estimate many Hermitian targets in one native call.

        Setting-level values are formed after averaging repeats. ``blocks=1``
        returns an array of shape ``(nobservables,)``; ``blocks>1`` returns
        their elementwise median-of-means. With ``return_blocks=True`` the
        shape is ``(blocks, nobservables)`` and each operator sum is combined
        before block reduction.
        """
        block_count = _positive_count(blocks, "blocks")
        operators = _operators_for_targets(observables, self.nqubits)
        values = self._native.estimate_many(
            [operator._native_handle for operator in operators],
            block_count,
            bool(return_blocks),
        )
        array = np.asarray(values, dtype=np.float64)
        if return_blocks:
            return cast(
                np.ndarray[Any, Any],
                np.array(array.reshape((block_count, len(operators))), copy=True),
            )
        return cast(
            np.ndarray[Any, Any], np.array(array.reshape((len(operators),)), copy=True)
        )

    def energy(
        self,
        hamiltonian: PauliOperator,
        *,
        blocks: int = 1,
        return_blocks: bool = False,
    ) -> Union[float, np.ndarray[Any, Any]]:
        """Alias of :meth:`expectation` for an exactly Hermitian Hamiltonian."""
        if not isinstance(hamiltonian, PauliOperator):
            raise TypeError("hamiltonian must be a PauliOperator")
        operator_value = _operator_for_target(hamiltonian, self.nqubits)
        block_count = _positive_count(blocks, "blocks")
        values = self._native.energy(
            operator_value._native_handle,
            block_count,
            bool(return_blocks),
        )
        array = np.asarray(values, dtype=np.float64)
        return np.array(array, copy=True) if return_blocks else float(array[0])

    def rdm(
        self,
        *,
        subsystem_to_keep: Optional[Sequence[int]] = None,
        subsystems_to_trace_out: Optional[Sequence[int]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Return the arithmetic-mean reduced snapshot estimator.

        User order is preserved for ``subsystem_to_keep``; a trace-out
        selector uses the q0-first complement. Only the requested square
        subsystem is materialized. Empty and full subsystems are valid.
        """
        _validate_max_bytes(max_bytes)
        keep = _normalize_subsystem(
            self.nqubits, subsystem_to_keep, subsystems_to_trace_out
        )
        k, values = self._native.rdm(list(keep), _effective_max_bytes(max_bytes))
        dimension = 1 << int(k)
        return cast(
            np.ndarray[Any, Any],
            np.array(
                np.asarray(values, dtype=np.complex128).reshape((dimension, dimension)),
                copy=True,
            ),
        )

    def renyi_entropy(
        self,
        *,
        subsystem_to_keep: Optional[Sequence[int]] = None,
        subsystems_to_trace_out: Optional[Sequence[int]] = None,
        alpha: int = 2,
        method: Literal["shadow", "randomized_measurement"] = "shadow",
        blocks: int = 1,
    ) -> float:
        """Estimate Rényi-2 using a shadow U-statistic or distinct repeats.

        ``alpha=2`` is the only supported order. ``method='shadow'`` uses
        cross-setting pairs and requires at least two settings per block;
        ``method='randomized_measurement'`` is local-Pauli-only, requires
        ``repeats>=2``, excludes self-pairs, and follows the collision scale
        ``R=Theta(sqrt(2**k))`` as a pilot heuristic. Purity is median-reduced
        before applying ``-log``.
        """
        if alpha != 2:
            raise ValueError("only alpha=2 is supported")
        if method not in ("shadow", "randomized_measurement"):
            raise ValueError("method must be 'shadow' or 'randomized_measurement'")
        keep = _normalize_subsystem(
            self.nqubits, subsystem_to_keep, subsystems_to_trace_out
        )
        block_count = _positive_count(blocks, "blocks")
        method_code = 0 if method == "shadow" else 1
        return float(self._native.renyi_entropy(list(keep), block_count, method_code))

    def energy_variance(
        self,
        hamiltonian: PauliOperator,
        *,
        blocks: int = 1,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> float:
        """Estimate ``<H**2>-<H>**2`` with a cross-setting square.

        Each block requires at least two complete randomized settings. The
        finite-sample result is returned unchanged, including negative values;
        duplicate Pauli products and workspace checks are performed in Rust.
        """
        if not isinstance(hamiltonian, PauliOperator):
            raise TypeError("hamiltonian must be a PauliOperator")
        _validate_max_bytes(max_bytes)
        operator_value = _operator_for_target(hamiltonian, self.nqubits)
        block_count = _positive_count(blocks, "blocks")
        return float(
            self._native.energy_variance(
                operator_value._native_handle,
                block_count,
                _effective_max_bytes(max_bytes),
            )
        )

    def fidelity(
        self,
        target: "StabilizerState",
        *,
        blocks: int = 1,
        return_blocks: bool = False,
    ) -> Union[float, np.ndarray[Any, Any]]:
        """Estimate fidelity to a same-width stabilizer target.

        Fidelity is supported only for global-Clifford snapshots. It is a
        linear setting-level estimator with the same mean, median-of-means,
        and return-shape semantics as :meth:`expectation`.
        """
        if not isinstance(target, StabilizerState):
            raise TypeError("target must be a StabilizerState")
        block_count = _positive_count(blocks, "blocks")
        values = self._native.fidelity(target._native, block_count, bool(return_blocks))
        array = np.asarray(values, dtype=np.float64)
        return np.array(array, copy=True) if return_blocks else float(array[0])


class StabilizerState:
    """Immutable pure stabilizer state backed by a native tableau."""

    __slots__ = ("_native", "_nqubits")

    _native: Any
    _nqubits: int

    def __init__(self, native: Any) -> None:
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_nqubits", int(native.nqubits))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("StabilizerState is immutable")

    @classmethod
    def zero_state(cls, nqubits: int) -> "StabilizerState":
        if not isinstance(nqubits, int) or isinstance(nqubits, bool) or nqubits < 0:
            raise ValueError("nqubits must be a non-negative integer")
        return cls(_native.stabilizer_zero_state(int(nqubits)))

    @classmethod
    def from_generators(
        cls, generators: Sequence[Tuple[PauliWord, int]]
    ) -> "StabilizerState":
        values = tuple(generators)
        if not values:
            raise ValueError("from_generators requires exactly nqubits generators")
        if not all(isinstance(word, PauliWord) for word, _ in values):
            raise TypeError("stabilizer generators must contain PauliWord values")
        nqubits = values[0][0].nqubits
        if len(values) != nqubits:
            raise ValueError("from_generators requires exactly nqubits generators")
        signs: list[int] = []
        x_rows: list[Tuple[int, ...]] = []
        z_rows: list[Tuple[int, ...]] = []
        for word, sign in values:
            if word.nqubits != nqubits:
                raise ValueError("stabilizer generator widths must agree")
            if isinstance(sign, bool) or sign not in (-1, 1):
                raise ValueError("stabilizer signs must be +1 or -1")
            signs.append(int(sign))
            x_rows.append(word.x_words)
            z_rows.append(word.z_words)
        word_count = (nqubits + 63) // 64
        return cls(
            _native.stabilizer_from_tableau(
                np.asarray(x_rows, dtype=np.uint64).reshape((nqubits, word_count)),
                np.asarray(z_rows, dtype=np.uint64).reshape((nqubits, word_count)),
                np.asarray(signs, dtype=np.int8),
            )
        )

    @classmethod
    def from_tableau(
        cls, x_words: object, z_words: object, signs: object
    ) -> "StabilizerState":
        x = _normalize_integer_array(x_words, "x_words", np.dtype(np.uint64))
        z = _normalize_integer_array(z_words, "z_words", np.dtype(np.uint64))
        sign_array = np.asarray(signs)
        if sign_array.dtype.kind not in "iu" or sign_array.dtype.kind == "b":
            raise TypeError("signs must be an integer array")
        if np.any((sign_array != -1) & (sign_array != 1)):
            raise ValueError("signs must contain only -1 or +1")
        normalized_signs = np.ascontiguousarray(sign_array, dtype=np.int8)
        if x.ndim != 2 or z.ndim != 2 or normalized_signs.ndim != 1:
            raise ValueError(
                "tableau arrays must have shapes (nqubits, word_count), (nqubits, word_count), (nqubits,)"
            )
        if x.shape != z.shape or x.shape[0] != normalized_signs.shape[0]:
            raise ValueError("tableau arrays have incompatible shapes")
        return cls(_native.stabilizer_from_tableau(x, z, normalized_signs))

    @classmethod
    def from_clifford_tape(cls, tape: object) -> "StabilizerState":
        from .propagation import GateTape

        if not isinstance(tape, GateTape):
            raise TypeError("tape must be a GateTape")
        return cls(_native.stabilizer_from_tape(tape._native_handle(DEFAULT_MAX_BYTES)))

    @property
    def nqubits(self) -> int:
        return self._nqubits

    def to_tableau(
        self,
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return owned read-only ``(x_words, z_words, signs)`` arrays."""
        x, z, signs = self._native.tableau()
        return (
            _readonly_copy(x, np.dtype(np.uint64)),
            _readonly_copy(z, np.dtype(np.uint64)),
            _readonly_copy(signs, np.dtype(np.int8)),
        )


def shadow_bound(
    observables: Union[PauliWord, Sequence[PauliWord], np.ndarray[Any, Any]],
    epsilon: float,
    delta: float = 0.01,
) -> Tuple[int, int]:
    """Return the TensorCircuit-compatible local-Pauli shadow budget.

    For ``L`` words of maximum weight ``w``, the result is
    ``(nsettings, blocks)`` with ``blocks=ceil(2 log(2L/delta))`` and
    ``settings_per_block=ceil(34*3**w/epsilon**2)``. This coefficient-free
    bound is for simultaneous Pauli words and does not accept a
    ``PauliOperator`` Hamiltonian.
    """
    words: Tuple[PauliWord, ...]
    if isinstance(observables, PauliWord):
        words = (observables,)
    elif isinstance(observables, np.ndarray):
        if observables.dtype.kind not in "iu" or observables.dtype.kind == "b":
            raise TypeError("shadow_bound code arrays must be integer")
        if observables.ndim == 1:
            arrays = observables[None, :]
        elif observables.ndim == 2:
            arrays = observables
        else:
            raise ValueError("shadow_bound code arrays must be one- or two-dimensional")
        if arrays.shape[0] == 0:
            raise ValueError("shadow_bound requires a non-empty target collection")
        words = PauliWord.batch_from_codes(int(arrays.shape[1]), arrays.tolist())
    else:
        try:
            words = tuple(cast(Iterable[PauliWord], observables))
        except TypeError as error:
            raise TypeError(
                "observables must be PauliWord values or integer code arrays"
            ) from error
        if not words or not all(isinstance(word, PauliWord) for word in words):
            raise ValueError("shadow_bound requires a non-empty PauliWord collection")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if not math.isfinite(float(delta)) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be finite with 0 < delta < 1")
    assert words
    nqubits = words[0].nqubits
    if any(word.nqubits != nqubits for word in words):
        raise ValueError("all shadow_bound observables must have the same width")
    target_count = len(words)
    blocks = max(1, math.ceil(2.0 * math.log(2.0 * target_count / delta)))
    settings_per_block = math.ceil(
        34.0 * (3.0 ** max(word.weight for word in words)) / float(epsilon) ** 2
    )
    return settings_per_block * blocks, blocks


__all__ = [
    "CliffordSnapshotArrays",
    "PauliSnapshotArrays",
    "Snapshots",
    "StabilizerState",
    "shadow_bound",
]
