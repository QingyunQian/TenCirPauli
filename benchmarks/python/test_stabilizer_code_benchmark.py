"""Release-mode end-to-end benchmarks for static stabilizer-code analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp


FamilyBuilder = Callable[[int], tcp.StabilizerCode]


def _repetition_code(nqubits: int) -> tcp.StabilizerCode:
    checks = np.zeros((nqubits - 1, nqubits), dtype=np.uint8)
    for row in range(nqubits - 1):
        checks[row, row : row + 2] = 1
    return tcp.StabilizerCode.from_css(np.empty((0, nqubits), dtype=np.uint8), checks)


def _finite_rate_code(nqubits: int) -> tcp.StabilizerCode:
    checks = np.zeros((nqubits // 2, nqubits), dtype=np.uint8)
    checks[np.arange(nqubits // 2), np.arange(nqubits // 2)] = 1
    return tcp.StabilizerCode.from_css(np.empty((0, nqubits), dtype=np.uint8), checks)


def _balanced_css_code(nqubits: int) -> tcp.StabilizerCode:
    half = nqubits // 4
    x_checks = np.zeros((half, nqubits), dtype=np.uint8)
    z_checks = np.zeros_like(x_checks)
    indices = np.arange(half)
    x_checks[indices, indices] = 1
    z_checks[indices, half + indices] = 1
    return tcp.StabilizerCode.from_css(x_checks, z_checks)


def _redundant_repetition_code(nqubits: int) -> tcp.StabilizerCode:
    checks = np.zeros((nqubits, nqubits), dtype=np.uint8)
    for row in range(nqubits - 1):
        checks[row, row : row + 2] = 1
    checks[-1, :2] = 1
    return tcp.StabilizerCode.from_css(np.empty((0, nqubits), dtype=np.uint8), checks)


def _general_finite_rate_code(nqubits: int) -> tcp.StabilizerCode:
    rank = nqubits // 4
    x = np.zeros((rank, nqubits), dtype=np.uint8)
    z = np.zeros_like(x)
    indices = np.arange(rank)
    x[indices, 2 * indices] = 1
    z[indices, 2 * indices] = 1
    return tcp.StabilizerCode.from_symplectic(x, z)


def _five_qubit_code(nqubits: int) -> tcp.StabilizerCode:
    if nqubits != 5:
        raise AssertionError("the five-qubit search workload has fixed width 5")
    return tcp.StabilizerCode.from_generators(
        5,
        tuple(
            tcp.PauliWord.from_string(value)
            for value in ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
        ),
    )


CONSTRUCTION_CASES: Tuple[Tuple[str, FamilyBuilder, int], ...] = (
    ("repetition", _repetition_code, 64),
    ("repetition_redundant", _redundant_repetition_code, 256),
    ("balanced_css", _balanced_css_code, 256),
    ("finite_rate_general", _general_finite_rate_code, 1024),
)

CLASSIFICATION_CASES: Tuple[Tuple[str, FamilyBuilder, int], ...] = (
    ("repetition", _repetition_code, 256),
    ("balanced_css", _balanced_css_code, 256),
    ("finite_rate_general", _general_finite_rate_code, 1024),
)

SEARCH_CASES: Tuple[Tuple[str, FamilyBuilder, int, str, int, int, int], ...] = (
    ("five_qubit_general", _five_qubit_code, 5, "general", 8, 4, 3),
    (
        "steane_css",
        lambda _: tcp.StabilizerCode.from_css(
            np.asarray(
                [
                    [1, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1],
                ],
                dtype=np.uint8,
            ),
            np.asarray(
                [
                    [1, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1],
                ],
                dtype=np.uint8,
            ),
        ),
        7,
        "auto",
        32,
        8,
        3,
    ),
    ("finite_rate_css_90", _finite_rate_code, 90, "auto", 256, 64, 1),
)


def _assert_source_tree_imports(benchmark: BenchmarkFixture) -> None:
    root = Path(__file__).resolve().parents[2]
    package_file = Path(tcp.__file__).resolve()
    native_file = Path(tcp._native.__file__).resolve()
    expected_package = root / "python" / "tencirpauli" / "__init__.py"
    expected_native_dir = expected_package.parent
    assert package_file == expected_package
    assert native_file.parent == expected_native_dir
    benchmark.extra_info.update(
        package_file=str(package_file),
        native_file=str(native_file),
    )


def _reference_classification(
    code: tcp.StabilizerCode, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    words = tuple(tcp.PauliWord.from_codes(row) for row in values)
    singular = tuple(code.classify(word) for word in words)
    syndromes = np.asarray([result.syndrome for result in singular], dtype=np.uint8)
    kinds = np.asarray([int(result.kind) for result in singular], dtype=np.uint8)
    zero_indices = np.asarray(
        [
            index
            for index, result in enumerate(singular)
            if all(value == 0 for value in result.syndrome)
        ],
        dtype=np.uintp,
    )
    logical_x = np.asarray(
        [singular[index].logical_x for index in zero_indices], dtype=np.uint8
    ).reshape((len(zero_indices), code.nlogical))
    logical_z = np.asarray(
        [singular[index].logical_z for index in zero_indices], dtype=np.uint8
    ).reshape((len(zero_indices), code.nlogical))
    return syndromes, kinds, zero_indices, logical_x, logical_z


def _pack_syndromes(syndromes: np.ndarray, nchecks: int) -> np.ndarray:
    packed = np.zeros((syndromes.shape[0], (nchecks + 63) // 64), dtype=np.uint64)
    for check in range(nchecks):
        packed[:, check // 64] |= syndromes[:, check].astype(np.uint64) << (check % 64)
    return packed


def _record_accuracy(
    benchmark: BenchmarkFixture,
    result: object,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    packed: bool,
) -> None:
    def required_array(name: str) -> np.ndarray:
        value = getattr(result, name, None)
        if not isinstance(value, np.ndarray):
            raise AssertionError(f"benchmark result is missing array {name}")
        return value

    syndromes, kinds, zero_indices, logical_x, logical_z = reference
    actual_syndromes = getattr(result, "syndromes", None)
    if actual_syndromes is None:
        actual_syndromes = required_array("residual_syndromes")
    expected_syndromes = (
        _pack_syndromes(syndromes, syndromes.shape[1]) if packed else syndromes
    )
    assert np.array_equal(actual_syndromes, expected_syndromes)
    assert np.array_equal(required_array("kinds"), kinds)
    assert np.array_equal(required_array("zero_syndrome_indices"), zero_indices)
    assert np.array_equal(required_array("logical_x"), logical_x)
    assert np.array_equal(required_array("logical_z"), logical_z)
    benchmark.extra_info.update(
        max_abs_error=0.0, accuracy_reference="singular_classify"
    )


def _batch_codes(family: str, nqubits: int, batch: int, kind: str) -> np.ndarray:
    values = np.zeros((batch, nqubits), dtype=np.uint8)
    for row in range(batch):
        qubit = row % nqubits
        if kind == "visible":
            if family.startswith("repetition"):
                values[row, qubit] = 1
            elif family == "balanced_css":
                values[row, nqubits // 4 + row % (nqubits // 4)] = 1
            elif family == "finite_rate_css":
                values[row, qubit % (nqubits // 2)] = 1
            elif family == "finite_rate_general":
                even = 2 * (row % (nqubits // 4))
                values[row, even] = 1
            else:
                values[row, qubit] = 1
        elif kind == "stabilizer":
            if family.startswith("repetition"):
                values[row, qubit] = 3
                values[row, (qubit + 1) % nqubits] = 3
            elif family == "balanced_css":
                values[row, row % (nqubits // 4)] = 1
            elif family == "finite_rate_css":
                values[row, qubit % (nqubits // 2)] = 3
            elif family == "finite_rate_general":
                even = 2 * (row % (nqubits // 4))
                values[row, even] = 3
        elif kind == "logical":
            if family.startswith("repetition"):
                values[row, :] = 1
            elif family == "balanced_css":
                values[row, nqubits // 2 + row % (nqubits // 4)] = 1
            elif family == "finite_rate_css":
                values[row, nqubits // 2 + row % (nqubits // 2)] = 1
            elif family == "finite_rate_general":
                values[row, nqubits - 1] = 1
            else:
                values[row, qubit] = 1
        else:
            raise AssertionError(kind)
    return values


def _mixed_classification_codes(family: str, nqubits: int, batch: int) -> np.ndarray:
    """Build one batch containing visible, stabilizer, and logical rows."""
    if batch % 3:
        raise AssertionError("mixed classification batch must be divisible by three")
    component = batch // 3
    return np.concatenate(
        [
            _batch_codes(family, nqubits, component, kind)
            for kind in ("visible", "stabilizer", "logical")
        ],
        axis=0,
    )


def _record_metadata(benchmark: BenchmarkFixture, result: object, batch: int) -> None:
    arrays = [
        value
        for name in (
            "syndromes",
            "residual_syndromes",
            "kinds",
            "zero_syndrome_indices",
            "logical_x",
            "logical_z",
        )
        if isinstance(value := getattr(result, name, None), np.ndarray)
    ]
    benchmark.extra_info.update(
        output_bytes=sum(int(value.nbytes) for value in arrays),
        batch=batch,
        errors_per_call=batch,
    )


@pytest.mark.parametrize(
    ("family", "builder", "nqubits"),
    CONSTRUCTION_CASES,
    ids=["repetition-64", "redundant-256", "balanced-css-256", "general-1024"],
)
def test_stabilizer_code_construction(
    benchmark: BenchmarkFixture, family: str, builder: FamilyBuilder, nqubits: int
) -> None:
    _assert_source_tree_imports(benchmark)
    result = benchmark(builder, nqubits)
    assert result.nqubits == nqubits
    benchmark.extra_info.update(family=family, retained_logical_rows=result.nlogical)


@pytest.mark.parametrize(
    ("family", "builder", "nqubits"),
    CLASSIFICATION_CASES,
    ids=["repetition-256", "balanced-css-256", "general-1024"],
)
@pytest.mark.parametrize("packed", [False, True])
def test_stabilizer_code_classification_representatives(
    benchmark: BenchmarkFixture,
    family: str,
    builder: FamilyBuilder,
    nqubits: int,
    packed: bool,
) -> None:
    code = builder(nqubits)
    errors = _mixed_classification_codes(family, nqubits, 192)
    _assert_source_tree_imports(benchmark)
    reference = _reference_classification(code, errors)
    result = benchmark(code.classify_many, errors, packed_syndromes=packed)
    assert result.kinds.shape == (192,)
    _record_accuracy(benchmark, result, reference, packed)
    _record_metadata(benchmark, result, 192)
    benchmark.extra_info.update(
        family=family,
        packed=packed,
        nchecks=code.nchecks,
        nlogical=code.nlogical,
    )


@pytest.mark.parametrize("nqubits", [64])
@pytest.mark.parametrize("packed", [False, True])
def test_stabilizer_code_syndrome_output(
    benchmark: BenchmarkFixture, nqubits: int, packed: bool
) -> None:
    code = _repetition_code(nqubits)
    errors = _batch_codes("repetition", nqubits, 512, "visible")
    _assert_source_tree_imports(benchmark)
    reference = np.asarray(
        [code.syndrome(tcp.PauliWord.from_codes(row)) for row in errors],
        dtype=np.uint8,
    )
    result = benchmark(code.syndrome_many, errors, packed=packed)
    assert result.shape[0] == 512
    expected = _pack_syndromes(reference, code.nchecks) if packed else reference
    assert np.array_equal(result, expected)
    benchmark.extra_info.update(
        output_bytes=int(result.nbytes),
        batch=512,
        packed=packed,
        nchecks=code.nchecks,
        max_abs_error=0.0,
        accuracy_reference="singular_syndrome",
    )


@pytest.mark.parametrize("nqubits", [64])
def test_stabilizer_code_word_input_boundary(
    benchmark: BenchmarkFixture, nqubits: int
) -> None:
    code = _repetition_code(nqubits)
    errors = _batch_codes("repetition", nqubits, 256, "visible")
    words = tuple(tcp.PauliWord.from_codes(row) for row in errors)
    _assert_source_tree_imports(benchmark)
    reference = _reference_classification(code, errors)
    result = benchmark(code.classify_many, words)
    assert result.kinds.shape == (256,)
    _record_accuracy(benchmark, result, reference, False)
    _record_metadata(benchmark, result, 256)
    benchmark.extra_info.update(input_form="pauli_words", nqubits=nqubits)


@pytest.mark.parametrize("residual_kind", ["visible", "stabilizer", "logical"])
@pytest.mark.parametrize("nqubits", [256])
@pytest.mark.parametrize("packed", [False, True])
def test_stabilizer_code_correction_outcomes(
    benchmark: BenchmarkFixture,
    residual_kind: str,
    nqubits: int,
    packed: bool,
) -> None:
    code = _repetition_code(nqubits)
    errors = _batch_codes("repetition", nqubits, 256, residual_kind)
    corrections = np.zeros_like(errors)
    _assert_source_tree_imports(benchmark)
    reference = _reference_classification(code, errors)
    result = benchmark(
        code.analyze_corrections,
        errors,
        corrections,
        packed_syndromes=packed,
    )
    _record_accuracy(benchmark, result, reference, packed)
    benchmark.extra_info.update(
        family="repetition",
        residual_kind=residual_kind,
        packed=packed,
        nchecks=code.nchecks,
    )


def _error_set_workload(
    nqubits: int, regime: str
) -> tuple[tcp.StabilizerCode, np.ndarray, tuple[bool, tuple[int, int] | None]]:
    code = _finite_rate_code(nqubits)
    errors = np.zeros((256, nqubits), dtype=np.uint8)
    if regime == "unique_syndromes":
        errors[np.arange(256), np.arange(256)] = 1
        return code, errors, (True, None)
    if regime == "repeated_same_coset":
        errors[:, 0] = 1
        return code, errors, (True, None)
    if regime == "multiple_conflicting_cosets":
        logical_qubit = nqubits // 2
        errors[1, logical_qubit] = 1
        errors[2, 0] = 1
        errors[3, 0] = 1
        errors[3, logical_qubit] = 1
        return code, errors, (False, (0, 1))
    raise AssertionError(regime)


@pytest.mark.parametrize(
    "regime",
    ["unique_syndromes", "repeated_same_coset", "multiple_conflicting_cosets"],
)
def test_stabilizer_code_error_set_grouping_regimes(
    benchmark: BenchmarkFixture, regime: str
) -> None:
    code, errors, expected = _error_set_workload(1024, regime)
    _assert_source_tree_imports(benchmark)
    result = benchmark(code.analyze_error_set, errors)
    assert (result.is_correctable, result.conflicting_pair) == expected
    benchmark.extra_info.update(
        batch=256,
        nchecks=code.nchecks,
        grouping_regime=regime,
        max_abs_error=0.0,
        accuracy_reference="fixed_grouping_expectation",
    )


@pytest.mark.parametrize(
    ("family", "builder", "nqubits", "sector", "max_trials", "sketch_rows", "expected"),
    SEARCH_CASES,
    ids=[case[0] for case in SEARCH_CASES],
)
def test_logical_witness_search_workloads(
    benchmark: BenchmarkFixture,
    family: str,
    builder: FamilyBuilder,
    nqubits: int,
    sector: str,
    max_trials: int,
    sketch_rows: int,
    expected: int,
) -> None:
    code = builder(nqubits)
    _assert_source_tree_imports(benchmark)
    result = benchmark(
        code.search_logical_witness,
        sector=sector,
        max_trials=max_trials,
        sketch_rows=sketch_rows,
        seed=20260822,
    )
    word = tcp.PauliWord(
        nqubits,
        tuple(int(value) for value in result.best.x_words),
        tuple(int(value) for value in result.best.z_words),
    )
    assert result.best_weight == result.best.weight == word.weight
    assert result.trials_run == max_trials
    assert code.classify(word).kind is tcp.StabilizerErrorKind.LOGICAL
    assert result.best_weight == expected
    extra_info = dict(
        family=family,
        sector=sector,
        max_trials=max_trials,
        sketch_rows=sketch_rows,
        resolved_sketch_rows=result.resolved_sketch_rows,
        initial_incumbent_weight=result.initial_incumbent_weight,
        initialization_candidates=result.initialization_candidates,
        setup_nanos=result.setup_nanos,
        initialization_nanos=result.initialization_nanos,
        trials_run=result.trials_run,
        best_weight=result.best_weight,
        witness_sector=result.best.sector,
        quality_reference="known_fixture_weight",
    )
    if benchmark.stats is not None:
        extra_info["trials_per_second"] = max_trials / benchmark.stats["mean"]
    benchmark.extra_info.update(extra_info)
