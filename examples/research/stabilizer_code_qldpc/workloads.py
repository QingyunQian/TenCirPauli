"""Deterministic workloads shared by the TenCirPauli/qLDPC comparison."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Workload:
    """One phase-free qubit stabilizer-code workload."""

    name: str
    family: str
    kind: str
    x_checks: np.ndarray[Any, Any]
    z_checks: np.ndarray[Any, Any]
    reference_logicals: np.ndarray[Any, Any]
    seed: int

    @property
    def nqubits(self) -> int:
        return int(self.x_checks.shape[1])

    @property
    def nchecks(self) -> int:
        if self.kind == "css":
            return int(self.x_checks.shape[0] + self.z_checks.shape[0])
        return int(self.x_checks.shape[0])

    @property
    def matrix_hash(self) -> str:
        return digest_arrays(self.x_checks, self.z_checks)


def _binary_array(value: Any) -> np.ndarray[Any, Any]:
    return np.ascontiguousarray(np.asarray(value, dtype=np.uint8))


def _pauli_rows(
    strings: Iterable[str],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rows = tuple(strings)
    if not rows:
        return np.empty((0, 0), dtype=np.uint8), np.empty((0, 0), dtype=np.uint8)
    nqubits = len(rows[0])
    x = np.zeros((len(rows), nqubits), dtype=np.uint8)
    z = np.zeros((len(rows), nqubits), dtype=np.uint8)
    code_to_xz = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
    for row, string in enumerate(rows):
        if len(string) != nqubits or any(
            character not in code_to_xz for character in string
        ):
            raise ValueError("invalid Pauli fixture")
        for column, character in enumerate(string):
            x[row, column], z[row, column] = code_to_xz[character]
    return x, z


def _codes_from_xz(
    x: np.ndarray[Any, Any], z: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    values = x + 2 * z
    return np.asarray((0, 1, 3, 2), dtype=np.uint8)[values]


def _repetition(nqubits: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    z = np.zeros((nqubits - 1, nqubits), dtype=np.uint8)
    indices = np.arange(nqubits - 1)
    z[indices, indices] = 1
    z[indices, indices + 1] = 1
    return np.empty((0, nqubits), dtype=np.uint8), z


def _hgp_repetition(length: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return the binary HGP matrix for two open-boundary repetition checks."""
    h = _repetition(length)[1]
    m, n = h.shape
    matrix_x = np.hstack(
        (np.kron(h, np.eye(n, dtype=np.uint8)), np.kron(np.eye(m, dtype=np.uint8), h.T))
    )
    matrix_z = np.hstack(
        (np.kron(np.eye(n, dtype=np.uint8), h), np.kron(h.T, np.eye(m, dtype=np.uint8)))
    )
    return matrix_x, matrix_z


def _direct_sum_five(
    blocks: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    local_x, local_z = _pauli_rows(("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"))
    nqubits = 5 * blocks
    x = np.zeros((4 * blocks, nqubits), dtype=np.uint8)
    z = np.zeros((4 * blocks, nqubits), dtype=np.uint8)
    logicals = np.zeros((2 * blocks, nqubits * 2), dtype=np.uint8)
    for block in range(blocks):
        rows = slice(4 * block, 4 * block + 4)
        columns = slice(5 * block, 5 * block + 5)
        x[rows, columns] = local_x
        z[rows, columns] = local_z
        logicals[block, 5 * block : 5 * block + 5] = 1
        logicals[blocks + block, nqubits + 5 * block : nqubits + 5 * block + 5] = 1

    permutation = (7 * np.arange(nqubits) + 3) % nqubits
    return (
        x[:, permutation],
        z[:, permutation],
        logicals[:, np.r_[permutation, nqubits + permutation]],
    )


def _steane() -> (
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
):
    h = _binary_array(
        [
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1],
        ]
    )
    logicals = np.zeros((2, 14), dtype=np.uint8)
    logicals[0, :7] = 1
    logicals[1, 7:] = 1
    return h, h.copy(), logicals


def _reference_logicals(
    x_checks: np.ndarray[Any, Any], family: str
) -> np.ndarray[Any, Any]:
    nqubits = x_checks.shape[1]
    if family == "repetition":
        logicals = np.zeros((2, 2 * nqubits), dtype=np.uint8)
        logicals[0, :nqubits] = 1
        logicals[1, nqubits] = 1
        return logicals
    if family == "five_qubit":
        _, _, logicals = _direct_sum_five(nqubits // 5)
        return logicals
    if family == "steane":
        return _steane()[2]
    if family == "hgp_repetition":
        check_rows = int(x_checks.shape[0])
        discriminant = 1 + 4 * check_rows
        length = (1 + math.isqrt(discriminant)) // 2
        if length * (length - 1) != check_rows:
            raise ValueError("invalid HGP repetition check dimensions")
        logicals = np.zeros((2, 2 * nqubits), dtype=np.uint8)
        logicals[0, :length] = 1
        logicals[1, nqubits + np.arange(length) * length] = 1
        return logicals
    return np.empty((0, 2 * nqubits), dtype=np.uint8)


def _append_redundant(
    x_checks: np.ndarray[Any, Any], z_checks: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rows = max(x_checks.shape[0], z_checks.shape[0])
    count = max(1, rows // 10)
    extra_x = np.empty((0, x_checks.shape[1]), dtype=np.uint8)
    extra_z = np.empty((0, z_checks.shape[1]), dtype=np.uint8)
    if (
        x_checks.shape[0]
        and z_checks.shape[0]
        and x_checks.shape[0] == z_checks.shape[0]
    ):
        extra_x = np.empty((count, x_checks.shape[1]), dtype=np.uint8)
        extra_z = np.empty((count, z_checks.shape[1]), dtype=np.uint8)
    elif z_checks.shape[0]:
        extra_z = np.empty((count, z_checks.shape[1]), dtype=np.uint8)
    else:
        raise ValueError("redundant workload requires at least one check row")
    for index in range(count):
        left = index % z_checks.shape[0]
        right = (index + 1) % z_checks.shape[0]
        if extra_x.shape[0]:
            extra_x[index] = (
                x_checks[index % x_checks.shape[0]]
                ^ x_checks[(index + 1) % x_checks.shape[0]]
            )
        extra_z[index] = z_checks[left] ^ z_checks[right]
    return np.vstack((x_checks, extra_x)), np.vstack((z_checks, extra_z))


def available_workloads() -> tuple[str, ...]:
    """Return the deterministic workload names."""
    return (
        "repetition_3",
        "five_qubit_1",
        "steane_7",
        "repetition_64",
        "repetition_256",
        "repetition_1024",
        "hgp_8",
        "hgp_16",
        "hgp_24",
        "hgp_32",
        "hgp_48",
        "five_qubit_16",
        "five_qubit_64",
        "five_qubit_256",
        "repetition_256_redundant",
        "five_qubit_64_redundant",
    )


def get_workload(name: str) -> Workload:
    """Build one workload without importing either compared package."""
    if name not in available_workloads():
        raise ValueError(
            f"unknown workload {name!r}; choose from {available_workloads()}"
        )

    if name.startswith("repetition_"):
        suffix = name.removeprefix("repetition_")
        if suffix.endswith("_redundant"):
            suffix = suffix.removesuffix("_redundant")
        nqubits = int(suffix)
        x_checks, z_checks = _repetition(nqubits)
        family = "repetition"
    elif name.startswith("hgp_"):
        length = int(name.removeprefix("hgp_"))
        x_checks, z_checks = _hgp_repetition(length)
        family = "hgp_repetition"
    elif name.startswith("five_qubit_"):
        suffix = name.removeprefix("five_qubit_")
        if suffix.endswith("_redundant"):
            suffix = suffix.removesuffix("_redundant")
        blocks = int(suffix)
        x_checks, z_checks, _ = _direct_sum_five(blocks)
        family = "five_qubit"
    elif name == "steane_7":
        x_checks, z_checks, _ = _steane()
        family = "steane"
    else:
        raise AssertionError(name)

    if name == "five_qubit_1":
        x_checks, z_checks, _ = _direct_sum_five(1)
        family = "five_qubit"

    if name == "five_qubit_1":
        x_checks, z_checks, reference = _direct_sum_five(1)
        reference = reference[:2]
    else:
        reference = _reference_logicals(x_checks, family)

    if name.endswith("_redundant"):
        x_checks, z_checks = _append_redundant(x_checks, z_checks)

    return Workload(
        name=name,
        family=family,
        kind=(
            "css" if family in {"repetition", "steane", "hgp_repetition"} else "general"
        ),
        x_checks=np.ascontiguousarray(x_checks, dtype=np.uint8),
        z_checks=np.ascontiguousarray(z_checks, dtype=np.uint8),
        reference_logicals=np.ascontiguousarray(reference, dtype=np.uint8),
        seed=20260822 + sum(ord(character) for character in name),
    )


def _zero_syndrome_pool(workload: Workload) -> list[np.ndarray[Any, Any]]:
    pool: list[np.ndarray[Any, Any]] = [np.zeros(workload.nqubits, dtype=np.uint8)]
    if workload.kind == "css":
        zero = np.zeros(workload.nqubits, dtype=np.uint8)
        for x_row in workload.x_checks:
            pool.append(_codes_from_xz(x_row, zero))
        for z_row in workload.z_checks:
            pool.append(_codes_from_xz(zero, z_row))
    else:
        for x_row, z_row in zip(workload.x_checks, workload.z_checks):
            pool.append(_codes_from_xz(x_row, z_row))
    return pool


def _logical_pool(workload: Workload) -> list[np.ndarray[Any, Any]]:
    if not workload.reference_logicals.size:
        return []
    nqubits = workload.nqubits
    return [
        _codes_from_xz(row[:nqubits], row[nqubits:])
        for row in workload.reference_logicals
    ]


def error_batch(workload: Workload, size: int) -> np.ndarray[Any, Any]:
    """Create a deterministic mixture of visible and zero-syndrome errors."""
    if size < 1:
        raise ValueError("batch size must be positive")
    rng = np.random.default_rng(workload.seed + size)
    values = np.zeros((size, workload.nqubits), dtype=np.uint8)
    zero_pool = _zero_syndrome_pool(workload)
    logical_pool = _logical_pool(workload)
    for index in range(size):
        if index % 3 == 1:
            values[index] = zero_pool[(index // 3) % len(zero_pool)]
            continue
        if index % 3 == 2 and logical_pool:
            values[index] = logical_pool[(index // 3) % len(logical_pool)]
            continue
        weight = 1 if index % 2 else 2
        positions = rng.choice(workload.nqubits, size=weight, replace=False)
        values[index, positions] = rng.integers(1, 4, size=weight, dtype=np.uint8)
    return values


def correction_batch(
    workload: Workload, errors: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Create deterministic exact, visible, stabilizer, and logical residuals."""
    corrections = np.zeros_like(errors)
    zero_pool = _zero_syndrome_pool(workload)
    stabilizer = zero_pool[1] if len(zero_pool) > 1 else None
    logical_pool = _logical_pool(workload)
    logical = logical_pool[0] if logical_pool else None
    for index, error in enumerate(errors):
        if index % 4 == 0:
            corrections[index] = error
        elif index % 4 == 2 and stabilizer is not None:
            corrections[index] = error ^ stabilizer
        elif index % 4 == 3 and logical is not None:
            corrections[index] = error ^ logical
    return corrections


def digest_arrays(*arrays: np.ndarray[Any, Any]) -> str:
    """Hash ordered C-contiguous array payloads with shape and dtype metadata."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def workload_hash(
    workload: Workload, errors: np.ndarray[Any, Any], corrections: np.ndarray[Any, Any]
) -> str:
    """Hash the complete input payload used by one comparison invocation."""
    return digest_arrays(workload.x_checks, workload.z_checks, errors, corrections)
