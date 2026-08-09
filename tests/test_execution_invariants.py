"""Charge-factorization execution invariants."""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp


def _balanced_fermion_sector(
    space: tcp.OperatorSpace, *, interleaved: bool
) -> tcp.ChargeSector:
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(4)})
    balance = tcp.AdditiveCharge(
        space,
        fermions={
            index: (1 if (index % 2 == 0 if interleaved else index < 2) else -1)
            for index in range(4)
        },
    )
    return tcp.ChargeSector(((total, 2), (balance, 0)))


def test_equivalent_charge_rows_select_the_same_factorized_execution() -> None:
    space = tcp.OperatorSpace(fermions=4)
    grouped = _balanced_fermion_sector(space, interleaved=False)
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(4)})
    up = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1})
    explicit = tcp.ChargeSector(((total, 2), (up, 1)))
    operator = (
        space.fermion.create(0) * space.fermion.annihilate(1)
        + space.fermion.create(1) * space.fermion.annihilate(0)
        + 0.3 * space.fermion.create(2) * space.fermion.annihilate(3)
    )
    grouped_restricted = operator.restrict_charge(grouped)
    explicit_restricted = operator.restrict_charge(explicit)
    assert grouped.dimension == explicit.dimension == 4
    assert grouped_restricted.mvp_plan().native_variant == "factorized"
    assert explicit_restricted.mvp_plan().native_variant == "factorized"
    state = np.arange(grouped.dimension, dtype=np.complex128) + 0.25j
    np.testing.assert_allclose(
        grouped_restricted.apply(state), explicit_restricted.apply(state)
    )
    np.testing.assert_allclose(grouped_restricted.dense(), explicit_restricted.dense())


def test_scaled_sign_flipped_and_permuted_rows_keep_the_same_public_sector() -> None:
    space = tcp.OperatorSpace(fermions=4)
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(4)})
    balance = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1, 2: -1, 3: -1})
    doubled_total = tcp.AdditiveCharge(space, fermions={index: 2 for index in range(4)})
    negated_balance = tcp.AdditiveCharge(space, fermions={0: -1, 1: -1, 2: 1, 3: 1})
    canonical = tcp.ChargeSector(((total, 2), (balance, 0)))
    transformed = tcp.ChargeSector(((negated_balance, 0), (doubled_total, 4)))
    assert canonical.dimension == transformed.dimension == 4
    for rank in range(canonical.dimension):
        values = canonical.unrank(rank)
        assert transformed.unrank(rank) == values
        assert canonical.rank(values) == transformed.rank(values) == rank

    hopping = (
        space.fermion.create(0) * space.fermion.annihilate(1)
        + space.fermion.create(1) * space.fermion.annihilate(0)
        + 0.3 * space.fermion.create(2) * space.fermion.annihilate(3)
    )
    canonical_restricted = hopping.restrict_charge(canonical)
    transformed_restricted = hopping.restrict_charge(transformed)
    assert canonical_restricted.mvp_plan().native_variant == "factorized"
    assert transformed_restricted.mvp_plan().native_variant == "factorized"
    state = np.arange(canonical.dimension, dtype=np.complex128) - 0.5j
    np.testing.assert_allclose(
        canonical_restricted.apply(state), transformed_restricted.apply(state)
    )


def test_inconsistent_and_zero_rows_keep_exact_empty_or_cartesian_sectors() -> None:
    space = tcp.OperatorSpace(fermions=2)
    total = tcp.AdditiveCharge(space, fermions={0: 1, 1: 1})
    doubled_total = tcp.AdditiveCharge(space, fermions={0: 2, 1: 2})
    inconsistent = tcp.ChargeSector(((total, 1), (doubled_total, 3)))
    assert inconsistent.dimension == 0

    zero = tcp.AdditiveCharge(space)
    full = zero.sector(0)
    empty = zero.sector(1)
    assert full.dimension == 4
    assert [full.unrank(rank) for rank in range(full.dimension)] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert empty.dimension == 0


def test_factorized_charge_execution_matches_dense_without_python_hints() -> None:
    space = tcp.OperatorSpace(qubits=4, bosons=1)
    left = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    right = tcp.AdditiveCharge(space, qubits={2: (0, 1), 3: (0, 1)})
    sector = tcp.ChargeSector(((left, 1), (right, 1)), boson_cutoffs={0: 2})
    operator = (
        space.qubit.x(0) * space.qubit.x(1)
        + space.qubit.y(0) * space.qubit.y(1)
        + 0.7 * space.qubit.z(2)
        + 0.2 * space.boson.create(0) * space.boson.annihilate(0)
    )
    restricted = operator.restrict_charge(sector)
    plan = restricted.mvp_plan()
    assert plan.execution_family == "factorized"
    state = np.arange(plan.dimension, dtype=np.complex128) - 0.4j
    output = np.empty_like(state)
    plan.apply_into(state, output, max_bytes=None)
    np.testing.assert_allclose(output, restricted.dense() @ state)


def test_coupled_finite_additive_sector_remains_exact() -> None:
    space = tcp.OperatorSpace(qubits=1, bosons=1)
    coupled = tcp.AdditiveCharge(space, qubits={0: (0, 1)}, bosons={0: 1})
    sector = coupled.sector(1, boson_cutoffs={0: 1})
    operator = space.qubit.z(0) + 0.5 * space.boson.create(0) * space.boson.annihilate(
        0
    )
    restricted = operator.restrict_charge(sector)
    # A one-row finite additive constraint is a compact component.  The
    # genuinely coupled case is the irreducible multi-row sector below.
    assert restricted.mvp_plan().execution_family == "factorized"
    state = np.asarray([0.5 - 0.2j, -0.3 + 0.7j])
    np.testing.assert_allclose(restricted.apply(state), restricted.dense() @ state)


def test_irreducible_multirow_sector_uses_general_execution() -> None:
    space = tcp.OperatorSpace(qubits=2, bosons=1)
    first = tcp.AdditiveCharge(space, qubits={0: (0, 1)}, bosons={0: 1})
    second = tcp.AdditiveCharge(space, qubits={1: (0, 1)}, bosons={0: 1})
    sector = tcp.ChargeSector(((first, 1), (second, 1)), boson_cutoffs={0: 2})
    operator = (
        0.2 * space.qubit.z(0)
        + 0.3 * space.qubit.z(1)
        + 0.4 * space.boson.create(0) * space.boson.annihilate(0)
    )
    restricted = operator.restrict_charge(sector)
    plan = restricted.mvp_plan()
    assert plan.execution_family == "generic"
    state = np.asarray([0.5 - 0.2j, -0.3 + 0.7j])
    output = np.empty_like(state)
    plan.apply_into(state, output, max_bytes=None)
    np.testing.assert_allclose(output, restricted.dense() @ state)
