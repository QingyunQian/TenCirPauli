"""End-to-end execution-family and diagnostic-counter invariants."""

from __future__ import annotations

from typing import Any

import numpy as np

import tencirpauli as tcp


def _charge_restrict(operator: Any, sector: Any) -> Any:
    charge_module = __import__("tencirpauli.charge", fromlist=["dummy"])
    return charge_module.restrict_charge(operator, sector)


def _two_factor_product(cutoff: int) -> tuple[Any, Any]:
    space = tcp.OperatorSpace(qubits=4, bosons=1)
    first = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    second = tcp.AdditiveCharge(space, qubits={2: (0, 1), 3: (0, 1)})
    sector = tcp.ChargeSector(((first, 1), (second, 1)), boson_cutoffs={0: cutoff})
    operator = (
        space.qubit.x(0) * space.qubit.x(1)
        + space.qubit.y(0) * space.qubit.y(1)
        + space.qubit.x(2) * space.qubit.x(3)
        + space.qubit.y(2) * space.qubit.y(3)
        + 0.25 * space.boson.create(0) * space.boson.annihilate(0)
    )
    return operator.restrict_charge(sector), sector


def test_specialized_capability_winners_are_selected_before_general_product() -> None:
    qubit_space = tcp.OperatorSpace(qubits=4)
    qubit_charge = tcp.AdditiveCharge(
        qubit_space, qubits={index: (0, 1) for index in range(4)}
    )
    qubit_operator = qubit_space.qubit.x(0) * qubit_space.qubit.x(
        1
    ) + qubit_space.qubit.y(0) * qubit_space.qubit.y(1)
    qubit = _charge_restrict(qubit_operator, qubit_charge.sector(2))
    assert qubit.mvp_plan().native_variant == "qubit"

    qudit_space = tcp.OperatorSpace(qudits=(3, 3))
    qudit_charge = tcp.AdditiveCharge(qudit_space, qudits={0: 1, 1: 1})
    qudit_operator = qudit_space.qudit.weyl(0, 0, 1) + qudit_space.qudit.weyl(1, 0, 1)
    qudit = _charge_restrict(qudit_operator, qudit_charge.sector(0))
    assert qudit.mvp_plan().native_variant == "qudit"

    boson_space = tcp.OperatorSpace(bosons=2)
    boson_charge = tcp.AdditiveCharge(boson_space, bosons={0: 1, 1: 1})
    boson_operator = boson_space.boson.create(0) * boson_space.boson.annihilate(
        1
    ) + boson_space.boson.create(1) * boson_space.boson.annihilate(0)
    boson = _charge_restrict(
        boson_operator, boson_charge.sector(1, boson_cutoffs={0: 2, 1: 2})
    )
    assert boson.mvp_plan().native_variant == "boson"

    hybrid_space = tcp.OperatorSpace(fermions=4, bosons=1)
    total = tcp.AdditiveCharge(hybrid_space, fermions={index: 1 for index in range(4)})
    balance = tcp.AdditiveCharge(hybrid_space, fermions={0: 1, 1: 1, 2: -1, 3: -1})
    hybrid_operator = (
        hybrid_space.fermion.create(0) * hybrid_space.fermion.annihilate(1)
        + hybrid_space.fermion.create(1) * hybrid_space.fermion.annihilate(0)
        + hybrid_space.boson.create(0) * hybrid_space.boson.annihilate(0)
    )
    hybrid = _charge_restrict(
        hybrid_operator,
        tcp.ChargeSector(((total, 2), (balance, 0)), boson_cutoffs={0: 2}),
    )
    assert hybrid.mvp_plan().native_variant == "fermion"


def test_product_counters_prove_two_factor_path_without_fallback_or_ranker() -> None:
    restricted, sector = _two_factor_product(2)
    plan = restricted.mvp_plan()
    assert plan.native_variant == "product"
    state = np.arange(sector.dimension, dtype=np.complex128) + 0.2j
    output = np.empty_like(state)
    plan.apply_into(state, output, max_bytes=0)
    np.testing.assert_allclose(output, restricted.dense() @ state)

    counters = tuple(plan._native_execution.debug_counters())
    assert len(counters) == 7
    (
        factor_evaluations,
        segments,
        destination_allocations,
        fallbacks,
        rankers,
        groups,
        inserts,
    ) = counters
    assert factor_evaluations > 0
    assert groups > 0
    assert segments > 0
    assert inserts >= 0
    assert destination_allocations == 0
    assert fallbacks == 0
    assert rankers == 0


def test_product_identity_spectator_scales_segments_not_factor_work() -> None:
    small, small_sector = _two_factor_product(1)
    large, large_sector = _two_factor_product(3)
    small_plan = small.mvp_plan()
    large_plan = large.mvp_plan()
    small_state = np.arange(small_sector.dimension, dtype=np.complex128)
    large_state = np.arange(large_sector.dimension, dtype=np.complex128)
    small_output = np.empty_like(small_state)
    large_output = np.empty_like(large_state)
    small_plan.apply_into(small_state, small_output, max_bytes=0)
    large_plan.apply_into(large_state, large_output, max_bytes=0)
    small_counters = tuple(small_plan._native_execution.debug_counters())
    large_counters = tuple(large_plan._native_execution.debug_counters())
    assert large_counters[0] == small_counters[0]
    assert large_counters[1] > small_counters[1]
    assert large_counters[2] == 0
    assert large_counters[3] == 0
    assert large_counters[4] == 0
    assert large_counters[5] > 0
    assert large_counters[6] >= 0


def test_product_inactive_compact_factor_scales_segments_not_active_work(
    monkeypatch: Any,
) -> None:
    charge_module = __import__("tencirpauli.charge", fromlist=["dummy"])
    monkeypatch.setattr(charge_module, "_fast_qubit_particles", lambda *_: None)

    def make(inactive_qubits: int) -> Any:
        space = tcp.OperatorSpace(qubits=inactive_qubits + 2)
        active = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
        inactive = tcp.AdditiveCharge(
            space,
            qubits={index: (0, 1) for index in range(2, inactive_qubits + 2)},
        )
        operator = space.qubit.x(0) * space.qubit.x(1) + space.qubit.y(
            0
        ) * space.qubit.y(1)
        return operator.restrict_charge(
            tcp.ChargeSector(((active, 1), (inactive, inactive_qubits // 2)))
        )

    counters: list[tuple[int, ...]] = []
    dimensions: list[int] = []
    for inactive_qubits in (2, 4, 6):
        restricted = make(inactive_qubits)
        plan = restricted.mvp_plan()
        assert plan.native_variant == "product"
        state = np.arange(plan.dimension, dtype=np.complex128) + 0.3j
        output = np.empty_like(state)
        plan.apply_into(state, output, max_bytes=0)
        np.testing.assert_allclose(output, restricted.dense() @ state)
        dimensions.append(plan.dimension)
        counters.append(tuple(plan._native_execution.debug_counters()))

    assert dimensions == [4, 12, 40]
    assert [counter[0] for counter in counters] == [counters[0][0]] * 3
    assert [counter[5] for counter in counters] == [counters[0][5]] * 3
    assert [counter[6] for counter in counters] == [counters[0][6]] * 3
    assert counters[0][1] < counters[1][1] < counters[2][1]
    assert all(counter[2:5] == (0, 0, 0) for counter in counters)


def test_product_public_index_storage_is_independent_of_spectator_volume() -> None:
    def make(cutoff: int) -> Any:
        space = tcp.OperatorSpace(qubits=4, bosons=1)
        first = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
        second = tcp.AdditiveCharge(space, qubits={2: (0, 1), 3: (0, 1)})
        sector = tcp.ChargeSector(((first, 1), (second, 1)), boson_cutoffs={0: cutoff})
        operator = (
            space.qubit.x(0) * space.qubit.x(1)
            + space.qubit.y(0) * space.qubit.y(1)
            + space.qubit.x(2) * space.qubit.x(3)
            + space.qubit.y(2) * space.qubit.y(3)
        )
        return operator.restrict_charge(sector)

    small = make(1).mvp_plan()
    large = make(511).mvp_plan()
    assert small.native_variant == large.native_variant == "product"
    # The sector DP naturally grows with the unrestricted Boson dimension,
    # but Product indexing must not add a second selected-space-sized public
    # permutation. A 512x spectator change therefore stays far below the
    # linear-in-D footprint of two u64 D-entry maps.
    assert large.estimated_retained_bytes < 4 * small.estimated_retained_bytes


def test_product_public_index_handles_interleaved_factor_without_ranker(
    monkeypatch: Any,
) -> None:
    charge_module = __import__("tencirpauli.charge", fromlist=["dummy"])
    monkeypatch.setattr(charge_module, "_fast_qubit_particles", lambda *_: None)
    space = tcp.OperatorSpace(qubits=3)
    charge = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 0), 2: (0, 1)})
    operator = (
        space.qubit.x(0) * space.qubit.x(2)
        + space.qubit.y(0) * space.qubit.y(2)
        + 0.25 * space.qubit.z(1)
    )
    restricted = operator.restrict_charge(charge.sector(1))
    plan = restricted.mvp_plan()
    assert plan.native_variant == "product"
    state = np.arange(plan.dimension, dtype=np.complex128) + 0.4j
    output = np.empty_like(state)
    plan.apply_into(state, output, max_bytes=0)
    np.testing.assert_allclose(output, restricted.dense() @ state)
    counters = tuple(plan._native_execution.debug_counters())
    assert counters[0] > 0
    assert counters[1] > 0
    assert counters[3] == 0
    assert counters[4] == 0
    assert counters[5] > 0


def test_csr_chunk_capacities_have_identical_arrays_and_mvp() -> None:
    space = tcp.OperatorSpace(fermions=4)
    charge = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(4)})
    operator = space.fermion.create(0) * space.fermion.annihilate(
        1
    ) + space.fermion.create(1) * space.fermion.annihilate(0)
    restricted = operator.restrict_charge(charge.sector(2))
    native = restricted.mvp_plan()._native_execution
    reference = native.compile_eager_with_chunk_capacity(2**63 - 1, 1 << 20)
    reference_arrays = tuple(np.asarray(value).copy() for value in reference.csr())
    state = np.arange(restricted.dimension, dtype=np.complex128) + 0.2j
    reference_output = np.empty_like(state)
    reference.apply_into(state, reference_output, max_bytes=2**63 - 1)
    for capacity in (1, 2, 7):
        eager = native.compile_eager_with_chunk_capacity(2**63 - 1, capacity)
        actual_arrays = tuple(np.asarray(value) for value in eager.csr())
        for actual, expected in zip(actual_arrays, reference_arrays):
            np.testing.assert_array_equal(actual, expected)
        output = np.empty_like(state)
        eager.apply_into(state, output, max_bytes=2**63 - 1)
        np.testing.assert_array_equal(output, reference_output)


def test_scalar_and_wide_generic_counters_have_positive_path_activity(
    monkeypatch: Any,
) -> None:
    charge_module = __import__("tencirpauli.charge", fromlist=["dummy"])
    monkeypatch.setattr(
        charge_module, "_supports_independent_factor_product", lambda _: None
    )
    monkeypatch.setattr(charge_module, "_fast_qubit_particles", lambda *_: None)
    monkeypatch.setattr(charge_module, "_fast_qudit_factor", lambda _: None)
    monkeypatch.setattr(charge_module, "_fast_boson_factor", lambda _: None)
    monkeypatch.setattr(charge_module, "_fast_fermion_particles", lambda *_: None)

    scalar_space = tcp.OperatorSpace(qubits=4)
    scalar_number = tcp.AdditiveCharge(
        scalar_space, qubits={index: (0, 1) for index in range(4)}
    )
    scalar_marked = tcp.AdditiveCharge(scalar_space, qubits={0: (0, 1)})
    scalar_operator = (
        scalar_space.qubit.x(1) * scalar_space.qubit.x(2)
        + scalar_space.qubit.y(1) * scalar_space.qubit.y(2)
        + 0.5 * scalar_space.qubit.z(1)
    )
    scalar = scalar_operator.restrict_charge(
        tcp.ChargeSector(((scalar_number, 1), (scalar_marked, 0)))
    )
    scalar_plan = scalar.mvp_plan()
    assert scalar_plan.native_variant == "generic"
    scalar_state = np.arange(scalar.dimension, dtype=np.complex128)
    scalar_output = np.empty_like(scalar_state)
    scalar_plan.apply_into(scalar_state, scalar_output, max_bytes=None)
    scalar_plan.apply_into(scalar_state, scalar_output, max_bytes=None)
    scalar_eager = scalar.mvp_plan(storage="eager")
    np.testing.assert_allclose(scalar_eager.apply(scalar_state), scalar_output)
    scalar_counters = tuple(scalar_plan._native_execution.debug_counters())
    assert len(scalar_counters) == 8
    assert scalar_counters[0] > 0
    assert scalar_counters[1] == 0
    assert scalar_counters[2:4] == (0, 0)
    assert scalar_counters[4] > 0
    assert scalar_counters[5] > 0
    assert scalar_counters[6] > 0
    assert scalar_counters[7] == 0

    wide_space = tcp.OperatorSpace(qubits=65)
    wide_number = tcp.AdditiveCharge(
        wide_space, qubits={index: (0, 1) for index in range(65)}
    )
    wide_marked = tcp.AdditiveCharge(wide_space, qubits={0: (0, 1)})
    wide_operator = (
        wide_space.qubit.x(1) * wide_space.qubit.x(2)
        + wide_space.qubit.y(1) * wide_space.qubit.y(2)
        + 0.5 * wide_space.qubit.z(1)
    )
    wide = wide_operator.restrict_charge(
        tcp.ChargeSector(((wide_number, 1), (wide_marked, 0)))
    )
    wide_plan = wide.mvp_plan()
    assert wide_plan.native_variant == "generic"
    wide_state = np.arange(wide.dimension, dtype=np.complex128)
    wide_output = np.empty_like(wide_state)
    wide_plan.apply_into(wide_state, wide_output, max_bytes=None)
    wide_plan.apply_into(wide_state, wide_output, max_bytes=None)
    wide_eager = wide.mvp_plan(storage="eager")
    np.testing.assert_allclose(wide_eager.apply(wide_state), wide_output)
    wide_counters = tuple(wide_plan._native_execution.debug_counters())
    assert wide_counters[0] == 0
    assert wide_counters[1] > 0
    assert wide_counters[2:4] == (0, 0)
    assert wide_counters[4] > 0
    assert wide_counters[5] > 0
    assert wide_counters[6] > 0
    assert wide_counters[7] == 0


def test_structured_fermion_metadata_and_counters_are_wired() -> None:
    space = tcp.OperatorSpace(fermions=3, bosons=1)
    operator = space.fermion.create(0) * space.fermion.annihilate(1)
    plan = operator.compile("native_mvp", boson_cutoffs={0: 1})
    state = np.arange(plan.dimension, dtype=np.complex128)
    output = np.empty_like(state)
    plan.apply_into(state, output, max_bytes=0)
    counters = tuple(plan._native_plan.debug_counters())
    metadata = tuple(plan._native_plan.debug_metadata())
    assert counters[0] > 0
    assert counters[1] > 0
    assert counters[2] > 0
    assert counters[3] == 0
    assert metadata[0] == metadata[1] > 0
    assert metadata[2] == 0


def test_raw_fermion_source_packing_is_shared_across_schedule_members() -> None:
    operator = tcp.FermionOperator.from_terms(
        4,
        [
            (((0, "create"), (0, "annihilate")), 0.25),
            (((1, "create"), (1, "annihilate")), -0.5),
            (((0, "create"), (1, "annihilate")), 0.7 + 0.2j),
            (((1, "create"), (0, "annihilate")), 0.7 - 0.2j),
            (((2, "create"), (3, "annihilate")), -0.4 + 0.1j),
            (((3, "create"), (2, "annihilate")), -0.4 - 0.1j),
        ],
    )
    raw = operator.compile("native_mvp")
    mapped = operator.map_fermions().compile("native_mvp")
    state = np.arange(raw.dimension, dtype=np.complex128) - 0.25j
    output = np.empty_like(state)
    raw.apply_into(state, output, max_bytes=0)
    np.testing.assert_allclose(output, mapped.apply(state), atol=1e-12)

    profile = raw._native_plan.debug_fermion_mask_phase_profile(64)
    assert profile is not None
    descriptor_count, sample_count, packed_units, member_evaluations, *_ = (
        int(value) for value in profile
    )
    assert descriptor_count > 1
    assert packed_units == sample_count
    assert member_evaluations == descriptor_count * sample_count
    assert packed_units < member_evaluations
