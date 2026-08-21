"""Runtime contracts for fixed-shape private native ABI payloads."""

from __future__ import annotations

import numpy as np

from tencirpauli import (
    BosonOperator,
    FermionOperator,
    MajoranaOperator,
    PauliOperator,
    _native,
)
from tencirpauli.structured import OperatorSpace


def _assert_array(value: object, dtype: np.dtype[object], ndim: int = 1) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    assert value.dtype == dtype
    assert value.ndim == ndim
    assert value.flags.c_contiguous
    assert value.flags.owndata or value.base is not None
    assert value.flags.writeable
    return value


def test_native_operator_materialization_tuples_have_fixed_arrays_and_values() -> None:
    pauli = PauliOperator.from_terms(1, [("Z", 2.0), ("I", 1.0)])
    term_count, nqubits, x_words, z_words, coefficients = (
        pauli._native_handle.materialize()
    )
    assert (term_count, nqubits) == (2, 1)
    _assert_array(x_words, np.dtype(np.uint64))
    _assert_array(z_words, np.dtype(np.uint64))
    values = _assert_array(coefficients, np.dtype(np.complex128))
    np.testing.assert_allclose(values, [1.0, 2.0])
    _, _, codes, code_coefficients = pauli._native_handle.materialize_arrays()
    np.testing.assert_array_equal(
        _assert_array(codes, np.dtype(np.uint8)).reshape((2, 1)), [[0], [3]]
    )
    np.testing.assert_allclose(
        _assert_array(code_coefficients, np.dtype(np.complex128)), [1.0, 2.0]
    )

    fermion = FermionOperator.from_terms(1, [(((0, "create"), (0, "annihilate")), 2.0)])
    (
        f_count,
        f_creation,
        f_creation_offsets,
        f_annihilation,
        f_annihilation_offsets,
        f_values,
    ) = fermion._native_handle.materialize()
    assert f_count == 1
    np.testing.assert_array_equal(_assert_array(f_creation, np.dtype(np.uint32)), [0])
    np.testing.assert_array_equal(
        _assert_array(f_creation_offsets, np.dtype(np.uintp)), [0, 1]
    )
    np.testing.assert_array_equal(
        _assert_array(f_annihilation, np.dtype(np.uint32)), [0]
    )
    np.testing.assert_array_equal(
        _assert_array(f_annihilation_offsets, np.dtype(np.uintp)), [0, 1]
    )
    np.testing.assert_allclose(_assert_array(f_values, np.dtype(np.complex128)), [2.0])

    boson = BosonOperator.from_terms(1, [(((0, "create"), (0, "annihilate")), 3.0)])
    b_count, b_blocks, b_offsets, b_values = boson._native_handle.materialize()
    assert b_count == 1
    np.testing.assert_array_equal(
        _assert_array(b_blocks, np.dtype(np.uint32)).reshape((-1, 3)), [[0, 1, 1]]
    )
    np.testing.assert_array_equal(_assert_array(b_offsets, np.dtype(np.uintp)), [0, 1])
    np.testing.assert_allclose(_assert_array(b_values, np.dtype(np.complex128)), [3.0])

    majorana = MajoranaOperator.from_terms(1, [((0,), 4.0)])
    m_count, m_indices, m_offsets, m_values = majorana._native_handle.materialize()
    assert m_count == 1
    np.testing.assert_array_equal(_assert_array(m_indices, np.dtype(np.uint64)), [0])
    np.testing.assert_array_equal(_assert_array(m_offsets, np.dtype(np.uintp)), [0, 1])
    np.testing.assert_allclose(_assert_array(m_values, np.dtype(np.complex128)), [4.0])


def test_native_dense_coo_csr_and_plan_apply_have_fixed_numeric_contracts() -> None:
    operator = PauliOperator.from_terms(1, [("Z", 2.0), ("I", 1.0)])
    handle = operator._native_handle
    dense_dimension, dense_values = _native.pauli_dense_handle(handle, 2**63 - 1)
    assert dense_dimension == 2
    np.testing.assert_allclose(
        _assert_array(dense_values, np.dtype(np.complex128)).reshape((2, 2)),
        [[3.0, 0.0], [0.0, -1.0]],
    )
    coo_dimension, rows, columns, coo_values = _native.pauli_coo_handle(
        handle, 2**63 - 1
    )
    assert coo_dimension == 2
    np.testing.assert_array_equal(_assert_array(rows, np.dtype(np.uint64)), [0, 1])
    np.testing.assert_array_equal(_assert_array(columns, np.dtype(np.uint64)), [0, 1])
    np.testing.assert_allclose(
        _assert_array(coo_values, np.dtype(np.complex128)), [3.0, -1.0]
    )
    csr_dimension, indptr, indices, csr_values = _native.pauli_csr_handle(
        handle, 2**63 - 1
    )
    assert csr_dimension == 2
    np.testing.assert_array_equal(_assert_array(indptr, np.dtype(np.uint64)), [0, 1, 2])
    np.testing.assert_array_equal(_assert_array(indices, np.dtype(np.uint64)), [0, 1])
    np.testing.assert_allclose(
        _assert_array(csr_values, np.dtype(np.complex128)), [3.0, -1.0]
    )

    plan = operator.compile(target="native_mvp")
    state = np.asarray([1.0 + 0.0j, 2.0 + 0.0j])
    native_output = _assert_array(
        plan._native_plan.apply(state, 2**63 - 1), np.dtype(np.complex128)
    )
    np.testing.assert_allclose(native_output, [3.0, -2.0])
    output = np.empty_like(state)
    plan._native_plan.apply_into(state, output, 2**63 - 1)
    np.testing.assert_allclose(output, native_output)

    hybrid = OperatorSpace(qubits=1).qubit.z(0)
    structured_plan = hybrid.compile(target="native_mvp")
    structured_output = _assert_array(
        structured_plan._native_plan.apply(state, 2**63 - 1),
        np.dtype(np.complex128),
    )
    np.testing.assert_allclose(structured_output, [1.0, -2.0])
