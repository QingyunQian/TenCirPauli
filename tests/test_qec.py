from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pytest

import tencirpauli as tcp
from tencirpauli.qec import (
    StabilizerCode,
    StabilizerErrorKind,
)


def _word(value: str) -> tcp.PauliWord:
    return tcp.PauliWord.from_string(value)


def _codes(values: list[str]) -> np.ndarray:
    return np.asarray([_word(value).to_codes() for value in values], dtype=np.uint8)


def _dense_pauli(word: tcp.PauliWord) -> np.ndarray[Any, Any]:
    local = (
        np.asarray([[1, 0], [0, 1]], dtype=np.complex128),
        np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
        np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
    )
    result = local[word.to_codes()[0]]
    for code in word.to_codes()[1:]:
        result = np.kron(result, local[code])
    return result


def _dense_code_projector(
    nqubits: int, checks: tuple[tcp.PauliWord, ...], signs: tuple[int, ...]
) -> np.ndarray[Any, Any]:
    dimension = 2**nqubits
    identity = np.eye(dimension, dtype=np.complex128)
    projector = identity.copy()
    for check, sign in zip(checks, signs):
        projector = projector @ (identity + sign * _dense_pauli(check)) / 2
    return projector


def _bit_flip_code() -> StabilizerCode:
    return StabilizerCode.from_generators(3, [_word("ZZI"), _word("IZZ")])


def test_repetition_construction_order_statistics_and_classification() -> None:
    code = _bit_flip_code()
    css = StabilizerCode.from_css(
        np.empty((0, 3), dtype=np.uint8),
        np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    )
    assert code.nqubits == 3
    assert code.nchecks == 2
    assert code.rank == 2
    assert code.nlogical == 1
    assert code.nrelations == 0
    assert code.qubit_order == "q0_first"
    assert code.syndrome_order == "input_checks"
    assert np.array_equal(code.check_weights(), np.asarray([2, 2], dtype=np.uintp))
    assert np.array_equal(code.qubit_degrees(), np.asarray([1, 2, 1], dtype=np.uintp))
    assert np.array_equal(
        code.syndrome(_word("XII")), np.asarray([1, 0], dtype=np.uint8)
    )
    assert np.array_equal(
        code.syndrome(_word("IXI")), np.asarray([1, 1], dtype=np.uint8)
    )
    assert np.array_equal(
        code.syndrome(_word("IIX")), np.asarray([0, 1], dtype=np.uint8)
    )
    assert np.array_equal(
        css.syndrome(_word("XII")), np.asarray([1, 0], dtype=np.uint8)
    )

    stabilizer = code.classify(_word("ZZI"))
    assert stabilizer.kind is StabilizerErrorKind.STABILIZER
    assert stabilizer.logical_x == (0,)
    assert stabilizer.logical_z == (0,)
    logical = code.classify(_word("XXX"))
    assert logical.kind is StabilizerErrorKind.LOGICAL
    assert logical.logical_x == (1,)
    assert logical.logical_z == (0,)
    for stabilizer_word in ("ZZI", "IZZ", "ZIZ"):
        equivalent = _word("XXX").multiply(_word(stabilizer_word)).word
        assert code.classify(equivalent).logical_x == logical.logical_x
        assert code.classify(equivalent).logical_z == logical.logical_z
    visible = code.classify(_word("XII"))
    assert visible.kind is StabilizerErrorKind.SYNDROME_VISIBLE
    assert visible.logical_x is None
    assert visible.logical_z is None

    checks = code.check_arrays()
    assert checks.x_words.shape == (2, 1)
    assert checks.z_words[:, 0].tolist() == [0b11, 0b110]
    assert checks.signs.tolist() == [1, 1]
    assert not checks.z_words.flags.writeable


def test_css_constructor_matches_equivalent_symplectic_constructor() -> None:
    h = np.asarray(
        [
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    zeros = np.zeros_like(h)
    css = StabilizerCode.from_css(h, h)
    symplectic = StabilizerCode.from_symplectic(
        np.vstack((h, zeros)),
        np.vstack((zeros, h)),
        signs=[1] * 6,
    )

    assert (css.rank, css.nlogical, css.nrelations) == (
        symplectic.rank,
        symplectic.nlogical,
        symplectic.nrelations,
    )
    css_checks = css.check_arrays()
    symplectic_checks = symplectic.check_arrays()
    for left, right in (
        (css_checks.x_words, symplectic_checks.x_words),
        (css_checks.z_words, symplectic_checks.z_words),
        (css_checks.signs, symplectic_checks.signs),
    ):
        assert np.array_equal(left, right)
    assert np.array_equal(css.check_weights(), symplectic.check_weights())
    assert np.array_equal(css.qubit_degrees(), symplectic.qubit_degrees())

    css_logicals = css.logical_operator_arrays()
    symplectic_logicals = symplectic.logical_operator_arrays()
    for left, right in (
        (css_logicals.logical_x_x_words, symplectic_logicals.logical_x_x_words),
        (css_logicals.logical_x_z_words, symplectic_logicals.logical_x_z_words),
        (css_logicals.logical_z_x_words, symplectic_logicals.logical_z_x_words),
        (css_logicals.logical_z_z_words, symplectic_logicals.logical_z_z_words),
    ):
        assert np.array_equal(left, right)

    errors = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert (
        css.syndrome_many(errors).tolist() == symplectic.syndrome_many(errors).tolist()
    )
    for packed in (False, True):
        left = css.classify_many(errors, packed_syndromes=packed)
        right = symplectic.classify_many(errors, packed_syndromes=packed)
        assert np.array_equal(left.syndromes, right.syndromes)
        assert np.array_equal(left.kinds, right.kinds)
        assert np.array_equal(left.zero_syndrome_indices, right.zero_syndrome_indices)
        assert np.array_equal(left.logical_x, right.logical_x)
        assert np.array_equal(left.logical_z, right.logical_z)


def test_redundant_relations_signed_dependencies_and_public_syndrome_order() -> None:
    code = StabilizerCode.from_generators(
        3,
        [_word("ZZI"), _word("IZZ"), _word("ZIZ")],
    )
    assert code.rank == 2
    assert code.nrelations == 1
    assert np.array_equal(
        code.check_relations(), np.asarray([[1, 1, 1]], dtype=np.uint8)
    )
    assert np.array_equal(
        code.check_relations(packed=True), np.asarray([[7]], dtype=np.uint64)
    )
    assert code.syndrome(_word("XII")) == (1, 0, 1)
    assert code.syndrome(_word("IXI")) == (1, 1, 0)

    reordered = StabilizerCode.from_generators(
        3,
        [_word("IZZ"), _word("ZZI"), _word("ZIZ")],
    )
    assert np.array_equal(
        code.logical_operator_arrays().logical_x_x_words,
        reordered.logical_operator_arrays().logical_x_x_words,
    )
    assert np.array_equal(
        code.logical_operator_arrays().logical_z_z_words,
        reordered.logical_operator_arrays().logical_z_z_words,
    )
    assert reordered.syndrome(_word("XII")) == (0, 1, 1)

    row_operated = StabilizerCode.from_generators(
        3,
        [_word("ZZI"), _word("ZIZ")],
    )
    assert np.array_equal(
        code.logical_operator_arrays().logical_x_x_words,
        row_operated.logical_operator_arrays().logical_x_x_words,
    )
    assert row_operated.syndrome(_word("XII")) == (1, 1)

    with pytest.raises(ValueError, match="-I"):
        StabilizerCode.from_generators(2, [_word("XX"), _word("XX")], signs=[1, -1])


def _bit_flip_error_set_oracle(
    errors: tuple[tcp.PauliWord, ...],
) -> tuple[tuple[int, int] | None, tuple[int, ...] | None, tuple[int, ...] | None]:
    checks = (_word("ZZI"), _word("IZZ"))
    stabilizers = {_word("III"), _word("ZZI"), _word("IZZ"), _word("ZIZ")}
    logical_x = _word("XXX")
    logical_z = _word("ZII")
    conflicts: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    for left in range(len(errors)):
        for right in range(left + 1, len(errors)):
            residual = errors[left].multiply(errors[right]).word
            syndrome = tuple(
                int(residual.symplectic_inner_product(check)) for check in checks
            )
            if syndrome != (0, 0) or residual in stabilizers:
                continue
            conflicts.append(
                (
                    left,
                    right,
                    (int(residual.symplectic_inner_product(logical_z)),),
                    (int(residual.symplectic_inner_product(logical_x)),),
                )
            )
    if not conflicts:
        return None, None, None
    left, right, logical_x_bits, logical_z_bits = min(conflicts)
    return (left, right), logical_x_bits, logical_z_bits


def test_error_set_witness_is_lexicographic_and_matches_pairwise_oracle() -> None:
    code = _bit_flip_code()
    regression = tuple(_word(value) for value in ("III", "IXI", "XIX", "XXX"))
    result = code.analyze_error_set(regression)
    assert result.conflicting_pair == (0, 3)
    assert result.logical_x == (1,)
    assert result.logical_z == (0,)

    all_errors = tuple(
        tcp.PauliWord.from_codes(codes) for codes in product(range(4), repeat=3)
    )
    expected_pair, expected_x, expected_z = _bit_flip_error_set_oracle(all_errors)
    actual = code.analyze_error_set(all_errors)
    assert (actual.conflicting_pair, actual.logical_x, actual.logical_z) == (
        expected_pair,
        expected_x,
        expected_z,
    )
    stabilizers = {_word(value) for value in ("III", "ZZI", "IZZ", "ZIZ")}
    for error in all_errors:
        classification = code.classify(error)
        if classification.syndrome == (0, 0):
            assert (classification.kind is StabilizerErrorKind.STABILIZER) == (
                error in stabilizers
            )
            assert classification.logical_x is not None
            assert classification.logical_z is not None
            assert (any(classification.logical_x) or any(classification.logical_z)) == (
                error not in stabilizers
            )
        else:
            assert classification.kind is StabilizerErrorKind.SYNDROME_VISIBLE


def test_phase_flip_five_qubit_and_steane_numeric_contracts() -> None:
    phase_flip = StabilizerCode.from_css(
        np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        np.empty((0, 3), dtype=np.uint8),
    )
    assert phase_flip.syndrome(_word("ZII")) == (1, 0)
    assert phase_flip.classify(_word("ZZZ")).kind is StabilizerErrorKind.LOGICAL

    five = StabilizerCode.from_generators(
        5,
        [_word("XZZXI"), _word("IXZZX"), _word("XIXZZ"), _word("ZXIXZ")],
    )
    assert (five.rank, five.nlogical, five.nrelations) == (4, 1, 0)
    for qubit in range(5):
        for code in "XYZ":
            error = _word("I" * qubit + code + "I" * (4 - qubit))
            assert five.classify(error).kind is StabilizerErrorKind.SYNDROME_VISIBLE
    logical_x, logical_z = five.logical_operators()
    five_arrays = five.logical_operator_arrays()
    assert five_arrays.logical_x_x_words.tolist() == [[31]]
    assert five_arrays.logical_x_z_words.tolist() == [[0]]
    assert five_arrays.logical_z_x_words.tolist() == [[12]]
    assert five_arrays.logical_z_z_words.tolist() == [[1]]
    assert logical_x[0].commutes_with(_word("XZZXI"))
    assert logical_z[0].commutes_with(_word("IXZZX"))
    assert logical_x[0].symplectic_inner_product(logical_z[0]) == 1

    h = np.asarray(
        [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1]],
        dtype=np.uint8,
    )
    steane = StabilizerCode.from_css(h, h)
    assert (steane.rank, steane.nlogical) == (6, 1)
    logical_x, logical_z = steane.logical_operators()
    steane_arrays = steane.logical_operator_arrays()
    assert steane_arrays.logical_x_x_words.tolist() == [[11]]
    assert steane_arrays.logical_x_z_words.tolist() == [[0]]
    assert steane_arrays.logical_z_x_words.tolist() == [[0]]
    assert steane_arrays.logical_z_z_words.tolist() == [[11]]
    for row in h:
        x_check = tcp.PauliWord.from_codes([1 if bit else 0 for bit in row])
        z_check = tcp.PauliWord.from_codes([3 if bit else 0 for bit in row])
        assert logical_x[0].commutes_with(x_check)
        assert logical_x[0].commutes_with(z_check)
        assert logical_z[0].commutes_with(x_check)
        assert logical_z[0].commutes_with(z_check)
    assert logical_x[0].symplectic_inner_product(logical_z[0]) == 1
    arrays = steane.logical_operator_arrays()
    assert arrays.logical_x_x_words.shape == (1, 1)
    assert arrays.logical_z_z_words.shape == (1, 1)


def test_dense_projector_oracle_binds_signed_code_space_and_logicals() -> None:
    checks = (_word("ZZI"), _word("IZZ"))
    projector = _dense_code_projector(3, checks, (1, 1))
    assert np.isclose(np.trace(projector), 2.0)
    assert np.isclose(np.trace(projector @ projector), 2.0)
    for check in checks:
        assert np.allclose(_dense_pauli(check) @ projector, projector)

    signed_checks = (_word("XX"), _word("ZZ"), _word("YY"))
    consistent = _dense_code_projector(2, signed_checks, (1, 1, -1))
    inconsistent = _dense_code_projector(2, signed_checks, (1, 1, 1))
    assert np.isclose(np.trace(consistent), 1.0)
    assert np.isclose(np.trace(consistent @ consistent), 1.0)
    assert np.allclose(inconsistent, np.zeros_like(inconsistent), atol=1e-12)
    for check, sign in zip(signed_checks, (1, 1, -1)):
        assert np.allclose(sign * _dense_pauli(check) @ consistent, consistent)

    code = _bit_flip_code()
    logical_x, logical_z = code.logical_operators()
    projected_x = projector @ _dense_pauli(logical_x[0]) @ projector
    projected_z = projector @ _dense_pauli(logical_z[0]) @ projector
    for projected in (projected_x, projected_z):
        assert np.linalg.norm(projected) > 0.0
        assert not np.allclose(projected, projector)
        assert np.allclose(projected @ projected, projector)
    assert np.allclose(
        projected_x @ projected_z + projected_z @ projected_x,
        np.zeros_like(projector),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="-I"):
        StabilizerCode.from_generators(
            2,
            [_word("XX"), _word("ZZ"), _word("YY")],
            signs=[1, 1, 1],
        )


def test_batch_packed_differential_and_decoder_residuals() -> None:
    code = _bit_flip_code()
    values = ["XII", "IXI", "IIX", "ZZI", "XXX", "IXX"]
    words = tuple(_word(value) for value in values)
    unpacked = code.classify_many(words)
    code_batch = code.classify_many(_codes(values))
    packed = code.classify_many(_codes(values), packed_syndromes=True)
    for array in (
        unpacked.syndromes,
        unpacked.kinds,
        unpacked.zero_syndrome_indices,
        unpacked.logical_x,
        unpacked.logical_z,
        packed.syndromes,
        packed.kinds,
        packed.zero_syndrome_indices,
        packed.logical_x,
        packed.logical_z,
    ):
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    assert np.array_equal(unpacked.syndromes, code_batch.syndromes)
    assert np.array_equal(unpacked.kinds, code_batch.kinds)
    assert np.array_equal(
        unpacked.zero_syndrome_indices, code_batch.zero_syndrome_indices
    )
    assert np.array_equal(
        packed.syndromes[:, 0] & 0b11,
        unpacked.syndromes[:, 0] + 2 * unpacked.syndromes[:, 1],
    )
    for index, word in enumerate(words):
        singular = code.classify(word)
        assert tuple(unpacked.syndromes[index]) == singular.syndrome
        assert int(unpacked.kinds[index]) == int(singular.kind)

    errors = _codes(["IXX", "XII"])
    corrections = _codes(["XII", "XII"])
    analysis = code.analyze_corrections(errors, corrections)
    assert analysis.kinds.tolist() == [
        StabilizerErrorKind.LOGICAL,
        StabilizerErrorKind.STABILIZER,
    ]
    assert analysis.logical_x.tolist() == [[1], [0]]
    assert analysis.zero_syndrome_indices.tolist() == [0, 1]
    packed_analysis = code.analyze_corrections(
        errors, corrections, packed_syndromes=True
    )
    assert np.array_equal(
        packed_analysis.residual_syndromes[:, 0] & 0b11,
        analysis.residual_syndromes[:, 0] + 2 * analysis.residual_syndromes[:, 1],
    )
    assert np.array_equal(packed_analysis.kinds, analysis.kinds)
    assert np.array_equal(
        packed_analysis.zero_syndrome_indices, analysis.zero_syndrome_indices
    )
    assert np.array_equal(packed_analysis.logical_x, analysis.logical_x)
    assert np.array_equal(packed_analysis.logical_z, analysis.logical_z)
    error_set = code.analyze_error_set(errors)
    assert error_set.is_correctable is False
    assert error_set.conflicting_pair == (0, 1)
    assert error_set.logical_x == (1,)


def test_pauliword_batch_correction_differential_against_singular_operations() -> None:
    code = _bit_flip_code()
    errors = tuple(_word(value) for value in ("XII", "ZZI", "XXX", "XII"))
    corrections = tuple(_word(value) for value in ("III", "III", "III", "XII"))
    result = code.analyze_corrections(errors, corrections)
    code_result = code.analyze_corrections(
        _codes([word.to_string() for word in errors]),
        _codes([word.to_string() for word in corrections]),
    )
    expected_syndromes = []
    expected_kinds = []
    expected_zero_indices = []
    expected_logical_x = []
    expected_logical_z = []
    for index, (error, correction) in enumerate(zip(errors, corrections)):
        residual = correction.multiply(error).word
        singular = code.classify(residual)
        expected_syndromes.append(singular.syndrome)
        expected_kinds.append(int(singular.kind))
        if singular.syndrome == (0, 0):
            assert singular.logical_x is not None
            assert singular.logical_z is not None
            expected_zero_indices.append(index)
            expected_logical_x.append(singular.logical_x)
            expected_logical_z.append(singular.logical_z)
    for actual in (result, code_result):
        assert np.array_equal(actual.residual_syndromes, expected_syndromes)
        assert np.array_equal(actual.kinds, expected_kinds)
        assert np.array_equal(actual.zero_syndrome_indices, expected_zero_indices)
        assert np.array_equal(actual.logical_x, expected_logical_x)
        assert np.array_equal(actual.logical_z, expected_logical_z)


def test_random_batch_and_correction_differential_against_singular_operations() -> None:
    rng = np.random.default_rng(20260821)
    codes = (
        _bit_flip_code(),
        StabilizerCode.from_generators(
            5,
            [
                _word("XZZXI"),
                _word("IXZZX"),
                _word("XIXZZ"),
                _word("ZXIXZ"),
            ],
        ),
    )
    for code in codes:
        values = rng.integers(0, 4, size=(48, code.nqubits), dtype=np.uint8)
        words = tuple(tcp.PauliWord.from_codes(row.tolist()) for row in values)
        batch = code.classify_many(values)
        packed = code.classify_many(values, packed_syndromes=True)
        expected_syndromes = np.asarray(
            [code.syndrome(word) for word in words], dtype=np.uint8
        )
        expected_packed = np.zeros((48, (code.nchecks + 63) // 64), dtype=np.uint64)
        for row, syndrome in enumerate(expected_syndromes):
            for check, value in enumerate(syndrome):
                if value:
                    expected_packed[row, check // 64] |= np.uint64(1 << (check % 64))
        assert np.array_equal(batch.syndromes, expected_syndromes)
        assert np.array_equal(packed.syndromes, expected_packed)
        for index, word in enumerate(words):
            singular = code.classify(word)
            assert int(batch.kinds[index]) == int(singular.kind)
            assert tuple(batch.syndromes[index]) == singular.syndrome
        corrections = rng.integers(0, 4, size=(48, code.nqubits), dtype=np.uint8)
        correction_result = code.analyze_corrections(values, corrections)
        for index, (error, correction) in enumerate(zip(words, corrections)):
            correction_word = tcp.PauliWord.from_codes(correction.tolist())
            residual = correction_word.multiply(error).word
            singular = code.classify(residual)
            assert int(correction_result.kinds[index]) == int(singular.kind)
            assert (
                tuple(correction_result.residual_syndromes[index]) == singular.syndrome
            )


def test_empty_full_rank_and_width_crossing_64_bits() -> None:
    empty_zero = StabilizerCode.from_symplectic(
        np.empty((0, 0), dtype=np.uint8), np.empty((0, 0), dtype=np.uint8)
    )
    assert (empty_zero.nqubits, empty_zero.nchecks, empty_zero.nlogical) == (0, 0, 0)
    assert empty_zero.check_arrays().x_words.shape == (0, 0)
    assert empty_zero.logical_operator_arrays().logical_x_x_words.shape == (0, 0)

    empty = StabilizerCode.from_css(
        np.empty((0, 3), dtype=np.uint8), np.empty((0, 3), dtype=np.uint8)
    )
    assert (empty.rank, empty.nlogical) == (0, 3)
    assert len(empty.logical_operators()[0]) == 3
    full = StabilizerCode.from_css(
        np.empty((0, 2), dtype=np.uint8), np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    )
    assert (full.rank, full.nlogical) == (2, 0)
    assert full.logical_operators() == ((), ())

    wide = StabilizerCode.from_css(
        np.empty((0, 65), dtype=np.uint8),
        np.asarray([[1] + [0] * 64], dtype=np.uint8),
    )
    errors = np.zeros((2, 65), dtype=np.uint8)
    errors[0, 0] = 1
    errors[1, 64] = 1
    assert wide.syndrome_many(errors).tolist() == [[1], [0]]
    assert wide.syndrome_many(errors, packed=True).tolist() == [[1], [0]]
    assert wide.check_arrays().z_words.shape == (1, 2)


def test_resource_limits_cover_logical_storage_and_packed_batch_outputs() -> None:
    zero_width = StabilizerCode.from_symplectic(
        np.empty((0, 0), dtype=np.uint8),
        np.empty((0, 0), dtype=np.uint8),
        max_bytes=0,
    )
    assert zero_width.nlogical == 0

    # The advisory construction estimate must not sum incompatible transient
    # maxima and reject a small runnable code solely for that reason.
    empty_advisory = StabilizerCode.from_css(
        np.empty((0, 64), dtype=np.uint8),
        np.empty((0, 64), dtype=np.uint8),
        max_bytes=3000,
    )
    assert (empty_advisory.rank, empty_advisory.nlogical) == (0, 64)

    with pytest.raises(MemoryError, match="memory limit"):
        StabilizerCode.from_css(
            np.empty((0, 1024), dtype=np.uint8),
            np.empty((0, 1024), dtype=np.uint8),
            max_bytes=0,
        )

    # Both packed check arrays are retained. A budget between the old
    # one-array estimate and the corrected two-array estimate must reject.
    full_rank_checks = np.eye(64, dtype=np.uint8)
    with pytest.raises(MemoryError, match="memory limit"):
        StabilizerCode.from_css(
            np.empty((0, 64), dtype=np.uint8), full_rank_checks, max_bytes=1800
        )

    z_checks = np.zeros((128, 64), dtype=np.uint8)
    z_checks[:64] = np.eye(64, dtype=np.uint8)
    z_checks[64:] = z_checks[:64]
    dependent_advisory = StabilizerCode.from_css(
        np.empty((0, 64), dtype=np.uint8), z_checks, max_bytes=5000
    )
    assert (dependent_advisory.rank, dependent_advisory.nrelations) == (64, 64)
    code = StabilizerCode.from_css(np.empty((0, 64), dtype=np.uint8), z_checks)
    errors = np.zeros((1, 64), dtype=np.uint8)
    packed_syndrome_bytes = ((code.nchecks + 63) // 64) * 8
    common_bytes = 1 + np.dtype(np.uintp).itemsize
    packed_limit = packed_syndrome_bytes + common_bytes
    assert code.classify_many(
        errors, packed_syndromes=True, max_bytes=packed_limit
    ).syndromes.shape == (1, 2)
    with pytest.raises(MemoryError, match="memory limit"):
        code.classify_many(errors, max_bytes=packed_limit)


def test_private_singular_native_abi_returns_flat_numpy_syndrome() -> None:
    code = _bit_flip_code()
    syndrome = code._native_handle.syndrome_one([1], [0])
    assert isinstance(syndrome, np.ndarray)
    assert syndrome.dtype == np.dtype(np.uint8)
    assert syndrome.ndim == 1
    assert syndrome.shape == (2,)
    assert syndrome.flags.c_contiguous
    assert syndrome.base is not None
    assert syndrome.tolist() == [1, 0]

    classified = code._native_handle.classify_one([1], [0])
    assert isinstance(classified[0], np.ndarray)
    assert classified[0].dtype == np.dtype(np.uint8)
    assert classified[0].tolist() == [1, 0]
    assert classified[1] == int(StabilizerErrorKind.SYNDROME_VISIBLE)
    assert classified[2] is None
    assert classified[3] is None

    logical = code._native_handle.classify_one([7], [0])
    for value, expected in zip(logical[2:], ([1], [0])):
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.dtype(np.uint8)
        assert value.ndim == 1
        assert value.flags.c_contiguous
        assert value.base is not None
        assert value.tolist() == expected

    error_set = code._native_handle.error_set_codes(_codes(["III", "XXX"]), 2**63 - 1)
    assert error_set[0] is False
    assert error_set[1] == (0, 1)
    for value, expected in zip(error_set[2:], ([1], [0])):
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.dtype(np.uint8)
        assert value.ndim == 1
        assert value.flags.c_contiguous
        assert value.base is not None
        assert value.tolist() == expected

    batch_codes = _codes(["XII", "ZZI", "XXX"])
    unpacked = code._native_handle.classify_many_codes_unpacked(batch_codes, 2**63 - 1)
    packed = code._native_handle.classify_many_codes_packed(batch_codes, 2**63 - 1)
    assert len(unpacked) == len(packed) == 5
    assert unpacked[0].dtype == np.dtype(np.uint8)
    assert packed[0].dtype == np.dtype(np.uint64)
    assert unpacked[0].shape == (3, 2)
    assert packed[0].shape == (3, 1)
    assert unpacked[1].tolist() == [0, 1, 2]
    assert packed[1].tolist() == [0, 1, 2]
    assert np.array_equal(unpacked[2], np.asarray([1, 2], dtype=np.uintp))
    assert np.array_equal(packed[2], np.asarray([1, 2], dtype=np.uintp))
    for output in (*unpacked, *packed):
        assert isinstance(output, np.ndarray)
        assert output.flags.c_contiguous
        assert output.flags.owndata or output.base is not None


def test_input_validation_and_css_orthogonality() -> None:
    signed_dependency = StabilizerCode.from_generators(
        2,
        [_word("XX"), _word("ZZ"), _word("YY")],
        signs=[1, 1, -1],
    )
    assert signed_dependency.check_relations().tolist() == [[1, 1, 1]]
    with pytest.raises(ValueError, match="-I"):
        StabilizerCode.from_generators(
            2,
            [_word("XX"), _word("ZZ"), _word("YY")],
            signs=[1, 1, 1],
        )
    with pytest.raises(ValueError, match="identity"):
        StabilizerCode.from_generators(1, [_word("I")])
    with pytest.raises(ValueError, match="do not commute"):
        StabilizerCode.from_generators(1, [_word("X"), _word("Z")])
    with pytest.raises(TypeError, match="boolean or integer"):
        StabilizerCode.from_symplectic(np.asarray([[0.5]]), np.asarray([[0.0]]))
    with pytest.raises(ValueError, match="zero and one"):
        StabilizerCode.from_symplectic(np.asarray([[2]]), np.asarray([[0]]))
    with pytest.raises(ValueError, match=r"orthogonality|commute"):
        StabilizerCode.from_css(
            np.asarray([[1, 0]], dtype=np.uint8), np.asarray([[1, 0]], dtype=np.uint8)
        )
    with pytest.raises(ValueError, match=r"\+1 and -1"):
        StabilizerCode.from_generators(1, [_word("X")], signs=[0])
