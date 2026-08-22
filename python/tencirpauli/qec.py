"""Static qubit stabilizer-code analysis backed by the Rust core."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np

from . import _native
from ._validation import validate_nonnegative_int
from .hamiltonian import DEFAULT_MAX_BYTES, _validate_max_bytes
from .pauli import PauliWord


QubitOrder = Literal["q0_first"]
SyndromeOrder = Literal["input_checks"]
PauliBatchInput = Union[Sequence[PauliWord], np.ndarray[Any, Any]]


# The public result records use slots to avoid per-instance dictionaries.
@dataclass(frozen=True, slots=True)
class StabilizerCheckArrays:
    """Owned packed signed checks in the caller's original order."""

    x_words: np.ndarray[Any, Any]
    z_words: np.ndarray[Any, Any]
    signs: np.ndarray[Any, Any]
    nqubits: int
    qubit_order: QubitOrder


@dataclass(frozen=True, slots=True)
class LogicalOperatorArrays:
    """Owned packed physical representatives of logical X/Z operators."""

    logical_x_x_words: np.ndarray[Any, Any]
    logical_x_z_words: np.ndarray[Any, Any]
    logical_z_x_words: np.ndarray[Any, Any]
    logical_z_z_words: np.ndarray[Any, Any]
    nqubits: int
    qubit_order: QubitOrder


@dataclass(frozen=True, slots=True)
class ErrorClassification:
    """Classification of one phase-free physical Pauli error."""

    syndrome: Tuple[int, ...]
    kind: "StabilizerErrorKind"
    logical_x: Optional[Tuple[int, ...]]
    logical_z: Optional[Tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class ErrorClassificationBatch:
    """Flat deterministic classification arrays for a batch of errors."""

    syndromes: np.ndarray[Any, Any]
    kinds: np.ndarray[Any, Any]
    zero_syndrome_indices: np.ndarray[Any, Any]
    logical_x: np.ndarray[Any, Any]
    logical_z: np.ndarray[Any, Any]
    syndromes_packed: bool


@dataclass(frozen=True, slots=True)
class ErrorSetAnalysis:
    """Correctability result for a supplied finite error set."""

    is_correctable: bool
    conflicting_pair: Optional[Tuple[int, int]]
    logical_x: Optional[Tuple[int, ...]]
    logical_z: Optional[Tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class CorrectionAnalysis:
    """Classification of residual correction-times-error Paulis."""

    residual_syndromes: np.ndarray[Any, Any]
    kinds: np.ndarray[Any, Any]
    zero_syndrome_indices: np.ndarray[Any, Any]
    logical_x: np.ndarray[Any, Any]
    logical_z: np.ndarray[Any, Any]
    syndromes_packed: bool


class StabilizerErrorKind(IntEnum):
    """Classification codes used by singular and batched analysis."""

    SYNDROME_VISIBLE = 0
    STABILIZER = 1
    LOGICAL = 2


@dataclass(frozen=True)
class _BatchInput:
    kind: Literal["words", "codes"]
    x_words: Optional[np.ndarray[Any, Any]]
    z_words: Optional[np.ndarray[Any, Any]]
    codes: Optional[np.ndarray[Any, Any]]
    batch: int


@dataclass(frozen=True, init=False)
class StabilizerCode:
    """Immutable Rust-native analysis of a qubit stabilizer code space.

    ``StabilizerCode`` represents a ``2**nlogical``-dimensional code space,
    not a pure :class:`StabilizerState`. Canonical logical representatives are
    deterministic algebraic representatives; they are not minimum-weight or
    geometry-aware operators.
    """

    nqubits: int
    nchecks: int
    rank: int
    nlogical: int
    nrelations: int
    _native_handle: _native.NativeStabilizerCodeHandle

    def __init__(self) -> None:
        raise TypeError(
            "StabilizerCode instances must be created by a from_* constructor"
        )

    @classmethod
    def _from_handle(
        cls, handle: _native.NativeStabilizerCodeHandle
    ) -> "StabilizerCode":
        return cls._build(
            int(handle.nqubits),
            int(handle.nchecks),
            int(handle.rank),
            int(handle.nlogical),
            int(handle.nrelations),
            handle,
        )

    @classmethod
    def _build(
        cls,
        nqubits: int,
        nchecks: int,
        rank: int,
        nlogical: int,
        nrelations: int,
        handle: _native.NativeStabilizerCodeHandle,
    ) -> "StabilizerCode":
        instance = object.__new__(cls)
        object.__setattr__(instance, "nqubits", nqubits)
        object.__setattr__(instance, "nchecks", nchecks)
        object.__setattr__(instance, "rank", rank)
        object.__setattr__(instance, "nlogical", nlogical)
        object.__setattr__(instance, "nrelations", nrelations)
        object.__setattr__(instance, "_native_handle", handle)
        return instance

    @classmethod
    def from_generators(
        cls,
        nqubits: int,
        generators: Iterable[PauliWord],
        *,
        signs: Optional[Sequence[int]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "StabilizerCode":
        """Construct from same-width signed phase-free Pauli generators."""
        n = validate_nonnegative_int(nqubits, "nqubits")
        _validate_max_bytes(max_bytes)
        words = tuple(generators)
        for index, word in enumerate(words):
            if not isinstance(word, PauliWord):
                raise TypeError(f"generator at index {index} must be a PauliWord")
            if word.nqubits != n:
                raise ValueError(
                    f"generator at index {index} has width {word.nqubits}, expected {n}"
                )
        normalized_signs = _normalize_signs(signs, len(words), "signs")
        word_count = (n + 63) // 64
        x_words = np.empty((len(words), word_count), dtype=np.uint64)
        z_words = np.empty((len(words), word_count), dtype=np.uint64)
        for index, word in enumerate(words):
            x_words[index, :] = word.x_words
            z_words[index, :] = word.z_words
        handle = _native.stabilizer_code_from_packed(
            n,
            np.ascontiguousarray(x_words),
            np.ascontiguousarray(z_words),
            normalized_signs,
            _qec_max_bytes(max_bytes),
        )
        return cls._from_handle(handle)

    @classmethod
    def from_symplectic(
        cls,
        x: object,
        z: object,
        *,
        signs: Optional[Sequence[int]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "StabilizerCode":
        """Construct from binary symplectic check arrays with shape ``(m, n)``."""
        _validate_max_bytes(max_bytes)
        x_array = _normalize_binary_matrix(x, "x")
        z_array = _normalize_binary_matrix(z, "z")
        if x_array.shape != z_array.shape:
            raise ValueError("x and z must have identical two-dimensional shapes")
        normalized_signs = _normalize_signs(signs, x_array.shape[0], "signs")
        handle = _native.stabilizer_code_from_symplectic(
            x_array,
            z_array,
            normalized_signs,
            _qec_max_bytes(max_bytes),
        )
        return cls._from_handle(handle)

    @classmethod
    def from_css(
        cls,
        x_checks: object,
        z_checks: object,
        *,
        x_signs: Optional[Sequence[int]] = None,
        z_signs: Optional[Sequence[int]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "StabilizerCode":
        """Construct from X-only rows followed by Z-only rows."""
        _validate_max_bytes(max_bytes)
        x_array = _normalize_binary_matrix(x_checks, "x_checks")
        z_array = _normalize_binary_matrix(z_checks, "z_checks")
        if x_array.shape[1] != z_array.shape[1]:
            raise ValueError("x_checks and z_checks must have the same column count")
        x_normalized = _normalize_signs(x_signs, x_array.shape[0], "x_signs")
        z_normalized = _normalize_signs(z_signs, z_array.shape[0], "z_signs")
        handle = _native.stabilizer_code_from_css(
            x_array,
            z_array,
            x_normalized,
            z_normalized,
            _qec_max_bytes(max_bytes),
        )
        return cls._from_handle(handle)

    @property
    def qubit_order(self) -> QubitOrder:
        """Return the fixed public qubit ordering."""
        return "q0_first"

    @property
    def syndrome_order(self) -> SyndromeOrder:
        """Return the original input-check syndrome ordering."""
        return "input_checks"

    def check_weights(self) -> np.ndarray[Any, Any]:
        """Return weights of the original ordered checks."""
        return _readonly_copy(self._native_handle.check_weights(), np.uintp)

    def qubit_degrees(self) -> np.ndarray[Any, Any]:
        """Return original-check incidence counts for each qubit."""
        return _readonly_copy(self._native_handle.qubit_degrees(), np.uintp)

    def check_arrays(self) -> StabilizerCheckArrays:
        """Materialize owned packed signed checks in caller order."""
        x_words, z_words, signs = self._native_handle.check_arrays()
        return StabilizerCheckArrays(
            _readonly_copy(x_words, np.uint64),
            _readonly_copy(z_words, np.uint64),
            _readonly_copy(signs, np.int8),
            self.nqubits,
            "q0_first",
        )

    def check_relations(self, *, packed: bool = False) -> np.ndarray[Any, Any]:
        """Return a deterministic basis of signed-check relations."""
        if not isinstance(packed, bool):
            raise TypeError("packed must be a bool")
        values = (
            self._native_handle.check_relations_packed()
            if packed
            else self._native_handle.check_relations_unpacked()
        )
        return _readonly_copy(values, np.uint64 if packed else np.uint8)

    def logical_operator_arrays(self) -> LogicalOperatorArrays:
        """Materialize flat packed logical X/Z representatives."""
        x_x, x_z, z_x, z_z = self._native_handle.logical_operator_arrays()
        return LogicalOperatorArrays(
            _readonly_copy(x_x, np.uint64),
            _readonly_copy(x_z, np.uint64),
            _readonly_copy(z_x, np.uint64),
            _readonly_copy(z_z, np.uint64),
            self.nqubits,
            "q0_first",
        )

    def logical_operators(self) -> Tuple[Tuple[PauliWord, ...], Tuple[PauliWord, ...]]:
        """Explicitly materialize logical representatives as ``PauliWord`` values."""
        arrays = self.logical_operator_arrays()
        word_count = (self.nqubits + 63) // 64
        logical_x = tuple(
            PauliWord(
                self.nqubits,
                tuple(int(value) for value in x_words),
                tuple(int(value) for value in z_words),
            )
            for x_words, z_words in zip(
                arrays.logical_x_x_words.reshape((self.nlogical, word_count)),
                arrays.logical_x_z_words.reshape((self.nlogical, word_count)),
                strict=True,
            )
        )
        logical_z = tuple(
            PauliWord(
                self.nqubits,
                tuple(int(value) for value in x_words),
                tuple(int(value) for value in z_words),
            )
            for x_words, z_words in zip(
                arrays.logical_z_x_words.reshape((self.nlogical, word_count)),
                arrays.logical_z_z_words.reshape((self.nlogical, word_count)),
                strict=True,
            )
        )
        return logical_x, logical_z

    def syndrome(self, error: PauliWord) -> Tuple[int, ...]:
        """Return the syndrome in the original supplied-check order."""
        word = _validate_word(error, self.nqubits, "error")
        return tuple(
            int(value)
            for value in self._native_handle.syndrome_one(word.x_words, word.z_words)
        )

    def syndrome_many(
        self,
        errors: PauliBatchInput,
        *,
        packed: bool = False,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> np.ndarray[Any, Any]:
        """Return one unpacked or packed syndrome row for every input error."""
        _validate_max_bytes(max_bytes)
        _validate_bool(packed, "packed")
        batch = _normalize_batch(errors, self.nqubits)
        values = _call_syndrome_many(
            self._native_handle,
            batch,
            packed,
            _qec_max_bytes(max_bytes),
        )
        return _readonly_copy(values, np.uint64 if packed else np.uint8)

    def classify(self, error: PauliWord) -> ErrorClassification:
        """Classify one phase-free Pauli error."""
        word = _validate_word(error, self.nqubits, "error")
        syndrome, kind, logical_x, logical_z = self._native_handle.classify_one(
            word.x_words, word.z_words
        )
        return ErrorClassification(
            tuple(int(value) for value in syndrome),
            StabilizerErrorKind(int(kind)),
            _optional_bits(logical_x),
            _optional_bits(logical_z),
        )

    def classify_many(
        self,
        errors: PauliBatchInput,
        *,
        packed_syndromes: bool = False,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> ErrorClassificationBatch:
        """Classify a batch with one coarse native call."""
        _validate_max_bytes(max_bytes)
        _validate_bool(packed_syndromes, "packed_syndromes")
        batch = _normalize_batch(errors, self.nqubits)
        result = _call_classify_many(
            self._native_handle,
            batch,
            packed_syndromes,
            _qec_max_bytes(max_bytes),
        )
        syndromes, kinds, zero_indices, logical_x, logical_z = result
        return ErrorClassificationBatch(
            _readonly_copy(syndromes, np.uint64 if packed_syndromes else np.uint8),
            _readonly_copy(kinds, np.uint8),
            _readonly_copy(zero_indices, np.uintp),
            _readonly_copy(logical_x, np.uint8),
            _readonly_copy(logical_z, np.uint8),
            packed_syndromes,
        )

    def analyze_error_set(
        self,
        errors: PauliBatchInput,
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> ErrorSetAnalysis:
        """Check joint correctability of a supplied finite error set."""
        _validate_max_bytes(max_bytes)
        batch = _normalize_batch(errors, self.nqubits)
        correctable, pair, logical_x, logical_z = _call_error_set(
            self._native_handle, batch, _qec_max_bytes(max_bytes)
        )
        return ErrorSetAnalysis(
            bool(correctable),
            None if pair is None else (int(pair[0]), int(pair[1])),
            _optional_bits(logical_x),
            _optional_bits(logical_z),
        )

    def analyze_corrections(
        self,
        errors: PauliBatchInput,
        corrections: PauliBatchInput,
        *,
        packed_syndromes: bool = False,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CorrectionAnalysis:
        """Classify residuals using the fixed convention ``correction * error``."""
        _validate_max_bytes(max_bytes)
        _validate_bool(packed_syndromes, "packed_syndromes")
        error_batch = _normalize_batch(errors, self.nqubits)
        correction_batch = _normalize_batch(corrections, self.nqubits)
        if error_batch.kind != correction_batch.kind:
            error_batch = _batch_as_codes(error_batch, self.nqubits)
            correction_batch = _batch_as_codes(correction_batch, self.nqubits)
        result = _call_corrections(
            self._native_handle,
            error_batch,
            correction_batch,
            packed_syndromes,
            _qec_max_bytes(max_bytes),
        )
        syndromes, kinds, zero_indices, logical_x, logical_z = result
        return CorrectionAnalysis(
            _readonly_copy(syndromes, np.uint64 if packed_syndromes else np.uint8),
            _readonly_copy(kinds, np.uint8),
            _readonly_copy(zero_indices, np.uintp),
            _readonly_copy(logical_x, np.uint8),
            _readonly_copy(logical_z, np.uint8),
            packed_syndromes,
        )


def _validate_word(value: object, nqubits: int, name: str) -> PauliWord:
    if not isinstance(value, PauliWord):
        raise TypeError(f"{name} must be a PauliWord")
    if value.nqubits != nqubits:
        raise ValueError(f"{name} has width {value.nqubits}, expected {nqubits}")
    return value


def _qec_max_bytes(max_bytes: Optional[int]) -> int:
    """Pass the QEC estimate through without the repository's large-floor shortcut."""
    return sys.maxsize if max_bytes is None else max_bytes


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _normalize_signs(
    signs: Optional[Sequence[int]], count: int, name: str
) -> np.ndarray[Any, Any]:
    if signs is None:
        return np.ones(count, dtype=np.int8)
    try:
        values = np.asarray(signs)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a one-dimensional integer sequence") from error
    if values.ndim != 1 or values.shape[0] != count:
        raise ValueError(f"{name} must have length {count}")
    if values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise TypeError(f"{name} must contain integer values")
    if np.any((values != 1) & (values != -1)):
        raise ValueError(f"{name} must contain only +1 and -1")
    return cast(
        np.ndarray[Any, Any], np.ascontiguousarray(values, dtype=np.int8).copy()
    )


def _normalize_binary_matrix(value: object, name: str) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be a two-dimensional binary integer array"
        ) from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.dtype.kind not in "biu":
        raise TypeError(f"{name} must have boolean or integer dtype")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only zero and one")
    return cast(
        np.ndarray[Any, Any], np.ascontiguousarray(array, dtype=np.uint8).copy()
    )


def _normalize_pauli_codes(
    value: object, name: str, nqubits: int
) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a two-dimensional integer array") from error
    if array.ndim != 2 or array.shape[1] != nqubits:
        raise ValueError(f"{name} must have shape (batch, {nqubits})")
    if array.dtype.kind not in "biu" or array.dtype.kind == "b":
        raise TypeError(f"{name} must have boolean or integer dtype")
    if np.any((array < 0) | (array > 3)):
        raise ValueError(f"{name} entries must be in 0..3")
    return cast(
        np.ndarray[Any, Any], np.ascontiguousarray(array, dtype=np.uint8).copy()
    )


def _normalize_batch(value: PauliBatchInput, nqubits: int) -> _BatchInput:
    if isinstance(value, np.ndarray):
        codes = _normalize_pauli_codes(value, "errors", nqubits)
        return _BatchInput("codes", None, None, codes, int(codes.shape[0]))
    items = tuple(value)
    for index, item in enumerate(items):
        _validate_word(item, nqubits, f"error at index {index}")
    word_count = (nqubits + 63) // 64
    x_words = np.empty((len(items), word_count), dtype=np.uint64)
    z_words = np.empty((len(items), word_count), dtype=np.uint64)
    for index, item in enumerate(items):
        assert isinstance(item, PauliWord)
        x_words[index, :] = item.x_words
        z_words[index, :] = item.z_words
    return _BatchInput(
        "words",
        np.ascontiguousarray(x_words),
        np.ascontiguousarray(z_words),
        None,
        len(items),
    )


def _batch_as_codes(batch: _BatchInput, nqubits: int) -> _BatchInput:
    if batch.kind == "codes":
        return batch
    assert batch.x_words is not None and batch.z_words is not None
    codes = np.empty((batch.batch, nqubits), dtype=np.uint8)
    lookup = np.asarray([0, 1, 3, 2], dtype=np.uint8)
    for qubit in range(nqubits):
        word = qubit // 64
        shift = qubit % 64
        code_bits = ((batch.x_words[:, word] >> shift) & 1) + 2 * (
            (batch.z_words[:, word] >> shift) & 1
        )
        codes[:, qubit] = lookup[code_bits]
    return _BatchInput("codes", None, None, np.ascontiguousarray(codes), batch.batch)


def _call_syndrome_many(
    handle: _native.NativeStabilizerCodeHandle,
    batch: _BatchInput,
    packed: bool,
    max_bytes: int,
) -> np.ndarray[Any, Any]:
    if batch.kind == "codes":
        assert batch.codes is not None
        values = (
            handle.syndrome_many_codes_packed(batch.codes, max_bytes)
            if packed
            else handle.syndrome_many_codes_unpacked(batch.codes, max_bytes)
        )
    else:
        assert batch.x_words is not None and batch.z_words is not None
        values = (
            handle.syndrome_many_words_packed(batch.x_words, batch.z_words, max_bytes)
            if packed
            else handle.syndrome_many_words_unpacked(
                batch.x_words, batch.z_words, max_bytes
            )
        )
    return cast(np.ndarray[Any, Any], values)


def _call_classify_many(
    handle: _native.NativeStabilizerCodeHandle,
    batch: _BatchInput,
    packed: bool,
    max_bytes: int,
) -> Tuple[np.ndarray[Any, Any], ...]:
    if batch.kind == "codes":
        assert batch.codes is not None
        result = (
            handle.classify_many_codes_packed(batch.codes, max_bytes)
            if packed
            else handle.classify_many_codes_unpacked(batch.codes, max_bytes)
        )
    else:
        assert batch.x_words is not None and batch.z_words is not None
        result = (
            handle.classify_many_words_packed(batch.x_words, batch.z_words, max_bytes)
            if packed
            else handle.classify_many_words_unpacked(
                batch.x_words, batch.z_words, max_bytes
            )
        )
    return tuple(cast(np.ndarray[Any, Any], value) for value in result)


def _call_error_set(
    handle: _native.NativeStabilizerCodeHandle,
    batch: _BatchInput,
    max_bytes: int,
) -> Tuple[
    bool,
    Optional[Tuple[int, int]],
    Optional[Iterable[int]],
    Optional[Iterable[int]],
]:
    if batch.kind == "codes":
        assert batch.codes is not None
        result = handle.error_set_codes(batch.codes, max_bytes)
    else:
        assert batch.x_words is not None and batch.z_words is not None
        result = handle.error_set_words(batch.x_words, batch.z_words, max_bytes)
    correctable, pair, logical_x, logical_z = result
    return (
        bool(correctable),
        None if pair is None else (int(pair[0]), int(pair[1])),
        logical_x,
        logical_z,
    )


def _call_corrections(
    handle: _native.NativeStabilizerCodeHandle,
    errors: _BatchInput,
    corrections: _BatchInput,
    packed: bool,
    max_bytes: int,
) -> Tuple[np.ndarray[Any, Any], ...]:
    if errors.kind == "codes":
        assert errors.codes is not None and corrections.codes is not None
        result = (
            handle.corrections_codes_packed(errors.codes, corrections.codes, max_bytes)
            if packed
            else handle.corrections_codes_unpacked(
                errors.codes, corrections.codes, max_bytes
            )
        )
    else:
        assert errors.x_words is not None and errors.z_words is not None
        assert corrections.x_words is not None and corrections.z_words is not None
        result = (
            handle.corrections_words_packed(
                errors.x_words,
                errors.z_words,
                corrections.x_words,
                corrections.z_words,
                max_bytes,
            )
            if packed
            else handle.corrections_words_unpacked(
                errors.x_words,
                errors.z_words,
                corrections.x_words,
                corrections.z_words,
                max_bytes,
            )
        )
    return tuple(cast(np.ndarray[Any, Any], value) for value in result)


def _optional_bits(value: Optional[Iterable[int]]) -> Optional[Tuple[int, ...]]:
    return None if value is None else tuple(int(bit) for bit in value)


def _readonly_copy(value: object, dtype: Any) -> np.ndarray[Any, Any]:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.flags.writeable = False
    return cast(np.ndarray[Any, Any], array)


__all__ = [
    "CorrectionAnalysis",
    "ErrorClassification",
    "ErrorClassificationBatch",
    "ErrorSetAnalysis",
    "LogicalOperatorArrays",
    "StabilizerCheckArrays",
    "StabilizerCode",
    "StabilizerErrorKind",
]
