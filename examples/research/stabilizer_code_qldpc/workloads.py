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
    paper_l: int | None = None
    paper_m: int | None = None
    paper_k: int | None = None
    paper_distance: int | None = None
    paper_distance_exact: bool = False
    paper_a: tuple[str, ...] = ()
    paper_b: tuple[str, ...] = ()

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


def _cyclic_shift(size: int) -> np.ndarray[Any, Any]:
    """Return the paper's cyclic shift: row ``i`` has its one at ``i + 1``."""
    matrix = np.zeros((size, size), dtype=np.uint8)
    matrix[np.arange(size), (np.arange(size) + 1) % size] = 1
    return matrix


def _bb_monomial(ell: int, m: int, axis: str, exponent: int) -> np.ndarray[Any, Any]:
    """Return one BB polynomial monomial as a binary permutation matrix."""
    if axis == "1":
        if exponent != 0:
            raise ValueError("the BB identity monomial must be written as 1^0")
        return np.eye(ell * m, dtype=np.uint8)
    if axis == "x":
        if exponent < 0:
            raise ValueError("BB monomial exponents must be non-negative")
        return np.kron(
            np.linalg.matrix_power(_cyclic_shift(ell), exponent % ell),
            np.eye(m, dtype=np.uint8),
        )
    if axis == "y":
        if exponent < 0:
            raise ValueError("BB monomial exponents must be non-negative")
        return np.kron(
            np.eye(ell, dtype=np.uint8),
            np.linalg.matrix_power(_cyclic_shift(m), exponent % m),
        )
    raise ValueError(f"unknown BB monomial axis {axis!r}")


def _bb_polynomial(
    ell: int, m: int, terms: tuple[tuple[str, int], ...]
) -> np.ndarray[Any, Any]:
    """Add BB monomials over GF(2), preserving the paper's polynomial form."""
    if len(terms) != 3:
        raise ValueError("a BB polynomial must contain three monomials")
    matrix = np.zeros((ell * m, ell * m), dtype=np.uint8)
    monomials = []
    for axis, exponent in terms:
        monomial = _bb_monomial(ell, m, axis, exponent)
        matrix ^= monomial
        monomials.append(monomial)
    if len({bytes(monomial) for monomial in monomials}) != len(monomials):
        raise ValueError("BB polynomial terms must be distinct modulo x^l = y^m = 1")
    return matrix


def _bivariate_bicycle(
    ell: int,
    m: int,
    a_terms: tuple[tuple[str, int], ...],
    b_terms: tuple[tuple[str, int], ...],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Construct the CSS matrices ``H_X=[A|B]`` and ``H_Z=[B.T|A.T]``."""
    if ell < 1 or m < 1:
        raise ValueError("BB dimensions must be positive")
    a = _bb_polynomial(ell, m, a_terms)
    b = _bb_polynomial(ell, m, b_terms)
    return (
        np.ascontiguousarray(np.hstack((a, b)), dtype=np.uint8),
        np.ascontiguousarray(np.hstack((b.T, a.T)), dtype=np.uint8),
    )


@dataclass(frozen=True)
class _BBInstance:
    """One finite-size BB code listed in the IBM/Nature paper."""

    ell: int
    m: int
    a_terms: tuple[tuple[str, int], ...]
    b_terms: tuple[tuple[str, int], ...]
    k: int
    distance: int
    distance_exact: bool


_BB_INSTANCES: dict[str, _BBInstance] = {
    "bb_72_12_6": _BBInstance(
        6,
        6,
        (("x", 3), ("y", 1), ("y", 2)),
        (("y", 3), ("x", 1), ("x", 2)),
        12,
        6,
        True,
    ),
    "bb_90_8_10": _BBInstance(
        15,
        3,
        (("x", 9), ("y", 1), ("y", 2)),
        (("1", 0), ("x", 2), ("x", 7)),
        8,
        10,
        True,
    ),
    "bb_108_8_10": _BBInstance(
        9,
        6,
        (("x", 3), ("y", 1), ("y", 2)),
        (("y", 3), ("x", 1), ("x", 2)),
        8,
        10,
        True,
    ),
    "bb_144_12_12": _BBInstance(
        12,
        6,
        (("x", 3), ("y", 1), ("y", 2)),
        (("y", 3), ("x", 1), ("x", 2)),
        12,
        12,
        True,
    ),
    "bb_288_12_18": _BBInstance(
        12,
        12,
        (("x", 3), ("y", 2), ("y", 7)),
        (("y", 3), ("x", 1), ("x", 2)),
        12,
        18,
        True,
    ),
    "bb_360_12_le24": _BBInstance(
        30,
        6,
        (("x", 9), ("y", 1), ("y", 2)),
        (("y", 3), ("x", 25), ("x", 26)),
        12,
        24,
        False,
    ),
    "bb_756_16_le34": _BBInstance(
        21,
        18,
        (("x", 3), ("y", 10), ("y", 17)),
        (("y", 5), ("x", 3), ("x", 19)),
        16,
        34,
        False,
    ),
}


def _format_bb_term(axis: str, exponent: int) -> str:
    if axis == "1":
        return "1"
    return axis if exponent == 1 else f"{axis}^{exponent}"


def _gf2_rank(matrix: np.ndarray[Any, Any]) -> int:
    """Return the binary row rank of a small classical check matrix."""
    basis: dict[int, int] = {}
    for row in np.asarray(matrix, dtype=np.uint8):
        value = int.from_bytes(
            np.packbits(row, bitorder="little").tobytes(), byteorder="little"
        )
        while value:
            pivot = value.bit_length() - 1
            previous = basis.get(pivot)
            if previous is None:
                basis[pivot] = value
                break
            value ^= previous
    return len(basis)


def _random_regular_check_matrix(
    seed: int,
    checks: int = 72,
    bits: int = 96,
    row_weight: int = 4,
    column_weight: int = 3,
) -> np.ndarray[Any, Any]:
    """Build a reproducible simple regular binary check matrix with full row rank."""
    if checks * row_weight != bits * column_weight:
        raise ValueError("regular check degrees do not have matching edge counts")
    row_stubs = np.repeat(np.arange(checks), row_weight)
    column_stubs = np.repeat(np.arange(bits), column_weight)
    for attempt in range(10_000):
        rng = np.random.default_rng(seed + attempt)
        shuffled_rows = row_stubs.copy()
        shuffled_columns = column_stubs.copy()
        rng.shuffle(shuffled_rows)
        rng.shuffle(shuffled_columns)
        pairs = shuffled_rows * bits + shuffled_columns
        if np.unique(pairs).size != pairs.size:
            continue
        matrix = np.zeros((checks, bits), dtype=np.uint8)
        matrix[shuffled_rows, shuffled_columns] = 1
        if _gf2_rank(matrix) != checks:
            continue
        return matrix
    raise RuntimeError("failed to generate a simple full-rank regular check matrix")


def _random_hgp_14400(
    matrix_seed: int = 20260823,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return a sparse random HGP with parameters ``[[14400, 576, d]]``."""
    h = _random_regular_check_matrix(matrix_seed)
    m, n = h.shape
    matrix_x = np.hstack(
        (np.kron(h, np.eye(n, dtype=np.uint8)), np.kron(np.eye(m, dtype=np.uint8), h.T))
    )
    matrix_z = np.hstack(
        (np.kron(np.eye(n, dtype=np.uint8), h), np.kron(h.T, np.eye(m, dtype=np.uint8)))
    )
    return np.ascontiguousarray(matrix_x), np.ascontiguousarray(matrix_z)


def _row_integer(row: np.ndarray[Any, Any]) -> int:
    """Pack one little-endian binary row into a Python integer."""
    return int.from_bytes(
        np.packbits(row, bitorder="little").tobytes(), byteorder="little"
    )


def _rref_integer_rows(
    matrix: np.ndarray[Any, Any], ncolumns: int
) -> tuple[list[int], list[int]]:
    """Return reduced binary rows and pivot columns for a small fixture."""
    rows = [_row_integer(row) for row in matrix]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(ncolumns):
        bit = 1 << column
        selected = next(
            (index for index in range(pivot_row, len(rows)) if rows[index] & bit),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        for index in range(len(rows)):
            if index != pivot_row and rows[index] & bit:
                rows[index] ^= rows[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows[:pivot_row], pivots


def _binary_nullspace(matrix: np.ndarray[Any, Any]) -> list[int]:
    """Return a deterministic integer basis of a binary matrix nullspace."""
    ncolumns = int(matrix.shape[1])
    rows, pivots = _rref_integer_rows(matrix, ncolumns)
    pivot_set = set(pivots)
    result = []
    for free in range(ncolumns):
        if free in pivot_set:
            continue
        value = 1 << free
        for row, pivot in zip(rows, pivots, strict=True):
            if row & (1 << free):
                value |= 1 << pivot
        result.append(value)
    return result


def _binary_vectors_independent(values: list[int]) -> bool:
    """Return whether integer-encoded binary vectors are independent."""
    if not values:
        return True
    ncolumns = max(value.bit_length() for value in values)
    matrix = np.asarray(
        [[(value >> column) & 1 for column in range(ncolumns)] for value in values],
        dtype=np.uint8,
    )
    return len(_rref_integer_rows(matrix, ncolumns)[1]) == len(values)


def _random_css_search_checks(
    matrix_seed: int = 20260823,
    nqubits: int = 256,
    x_rank: int = 96,
    z_rank: int = 96,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Create the reproducible CSS seed for the general search-quality fixture."""
    rng = np.random.default_rng(matrix_seed)
    while True:
        x_checks = rng.integers(0, 2, size=(x_rank, nqubits), dtype=np.uint8)
        x_checks[:, 2] = x_checks[:, 0] ^ x_checks[:, 1]
        if len(_rref_integer_rows(x_checks, nqubits)[1]) == x_rank:
            break

    nullspace = _binary_nullspace(x_checks)
    target = (1 << 0) | (1 << 1) | (1 << 2)
    while True:
        selected: list[int] = []
        while len(selected) < z_rank:
            candidate = 0
            for vector in nullspace:
                if rng.integers(0, 2):
                    candidate ^= vector
            if candidate and _binary_vectors_independent([*selected, candidate]):
                selected.append(candidate)
        z_checks = np.asarray(
            [
                [(value >> column) & 1 for column in range(nqubits)]
                for value in selected
            ],
            dtype=np.uint8,
        )
        if (
            len(
                _rref_integer_rows(
                    np.vstack((z_checks, _integer_to_row(target, nqubits))), nqubits
                )[1]
            )
            == z_rank + 1
        ):
            break

    permutation = rng.permutation(nqubits)
    return x_checks[:, permutation], z_checks[:, permutation]


def _integer_to_row(value: int, nqubits: int) -> np.ndarray[Any, Any]:
    """Expand a little-endian binary integer into one fixture row."""
    return np.asarray(
        [(value >> column) & 1 for column in range(nqubits)], dtype=np.uint8
    )


def _random_general_256(
    matrix_seed: int = 20260823,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return a fixed dense ``[[256, 64, 3]]`` general stabilizer fixture."""
    css_x, css_z = _random_css_search_checks(matrix_seed)
    x_checks = np.vstack((css_x, np.zeros_like(css_x)))
    z_checks = np.vstack((np.zeros_like(css_z), css_z))
    scramble_rng = np.random.default_rng(matrix_seed ^ 0xA5A5)
    operations = scramble_rng.integers(0, 3, size=x_checks.shape[1])
    for column, operation in enumerate(operations):
        if operation == 1:
            x_checks[:, column], z_checks[:, column] = (
                z_checks[:, column].copy(),
                x_checks[:, column].copy(),
            )
        elif operation == 2:
            z_checks[:, column] ^= x_checks[:, column]
    return np.ascontiguousarray(x_checks), np.ascontiguousarray(z_checks)


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
        "bb_72_12_6",
        "bb_90_8_10",
        "bb_108_8_10",
        "bb_144_12_12",
        "bb_288_12_18",
        "bb_360_12_le24",
        "bb_756_16_le34",
        "random_hgp_14400",
        "random_general_256",
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

    paper_instance: _BBInstance | None = _BB_INSTANCES.get(name)
    if paper_instance is not None:
        x_checks, z_checks = _bivariate_bicycle(
            paper_instance.ell,
            paper_instance.m,
            paper_instance.a_terms,
            paper_instance.b_terms,
        )
        family = "bivariate_bicycle"
    elif name.startswith("repetition_"):
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
    elif name == "random_hgp_14400":
        x_checks, z_checks = _random_hgp_14400()
        family = "random_hgp"
    elif name == "random_general_256":
        x_checks, z_checks = _random_general_256()
        family = "random_general"
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
            "css"
            if family
            in {
                "repetition",
                "steane",
                "hgp_repetition",
                "random_hgp",
                "bivariate_bicycle",
            }
            else "general"
        ),
        x_checks=np.ascontiguousarray(x_checks, dtype=np.uint8),
        z_checks=np.ascontiguousarray(z_checks, dtype=np.uint8),
        reference_logicals=np.ascontiguousarray(reference, dtype=np.uint8),
        seed=(
            20260824
            if name == "random_hgp_14400"
            else (
                20260823 ^ 0x9E3779B97F4A7C15
                if name == "random_general_256"
                else 20260822 + sum(ord(character) for character in name)
            )
        ),
        paper_l=paper_instance.ell if paper_instance is not None else None,
        paper_m=paper_instance.m if paper_instance is not None else None,
        paper_k=paper_instance.k if paper_instance is not None else None,
        paper_distance=(
            paper_instance.distance if paper_instance is not None else None
        ),
        paper_distance_exact=(
            paper_instance.distance_exact if paper_instance is not None else False
        ),
        paper_a=(
            tuple(_format_bb_term(*term) for term in paper_instance.a_terms)
            if paper_instance is not None
            else ()
        ),
        paper_b=(
            tuple(_format_bb_term(*term) for term in paper_instance.b_terms)
            if paper_instance is not None
            else ()
        ),
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
        for x_row, z_row in zip(workload.x_checks, workload.z_checks, strict=True):
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
