"""Stable public workloads for additive-charge factorization benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import tencirpauli as tcp


SectorFactory = Callable[[], tcp.ChargeSector]
WorkloadBuilder = Callable[[], tuple[Any, SectorFactory]]


@dataclass(frozen=True)
class ChargeFactorizationWorkload:
    """One reproducible public charge-sector benchmark workload."""

    name: str
    component_dimensions: tuple[int, ...]
    build: WorkloadBuilder
    expected_execution_family: str = "factorized"


def _fermion_total() -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(fermions=20)
    number = tcp.AdditiveCharge(space, fermions={mode: 1 for mode in range(20)})
    operator: Any = 0.0 * space.fermion.create(0) * space.fermion.annihilate(0)
    for mode in range(19):
        operator = operator + (
            space.fermion.create(mode) * space.fermion.annihilate(mode + 1)
            + space.fermion.create(mode + 1) * space.fermion.annihilate(mode)
        )
    return operator, lambda: number.sector(10)


def _spin_components(
    up_population: int, *, spin_row: bool
) -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(fermions=20)
    total = tcp.AdditiveCharge(space, fermions={mode: 1 for mode in range(20)})
    up = tcp.AdditiveCharge(space, fermions={mode: 1 for mode in range(10)})
    spin = tcp.AdditiveCharge(
        space, fermions={mode: (1 if mode < 10 else -1) for mode in range(20)}
    )
    operator: Any = 0.0 * space.fermion.create(0) * space.fermion.annihilate(0)
    for offset in (0, 10):
        for mode in range(offset, offset + 9):
            operator = operator + (
                space.fermion.create(mode) * space.fermion.annihilate(mode + 1)
                + space.fermion.create(mode + 1) * space.fermion.annihilate(mode)
            )
    if spin_row:
        return operator, lambda: tcp.ChargeSector(
            ((total, 10), (spin, 2 * up_population - 10))
        )
    return operator, lambda: tcp.ChargeSector(((total, 10), (up, up_population)))


def _qubit_fixed(population: int) -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(qubits=10, bosons=1)
    number = tcp.AdditiveCharge(space, qubits={mode: (0, 1) for mode in range(10)})
    operator: Any = 0.0 * space.qubit.z(0)
    for mode in range(9):
        operator = operator + (
            space.qubit.x(mode) * space.qubit.x(mode + 1)
            + space.qubit.y(mode) * space.qubit.y(mode + 1)
        )
    operator = operator + 0.1 * space.boson.create(0) * space.boson.annihilate(0)
    return operator, lambda: number.sector(population, boson_cutoffs={0: 1})


def _qubit_interleaved_spectators() -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(qubits=6, bosons=1)
    fixed = tcp.AdditiveCharge(space, qubits={0: (0, 1), 2: (0, 1), 4: (0, 1)})
    operator = (
        space.qubit.x(0) * space.qubit.x(2)
        + space.qubit.y(0) * space.qubit.y(2)
        + 0.3 * space.qubit.z(1) * space.qubit.z(5)
        + 0.2 * space.boson.create(0) * space.boson.annihilate(0)
    )
    return operator, lambda: fixed.sector(1, boson_cutoffs={0: 1})


def _qudit_regions(two_regions: bool) -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(qudits=(3, 3, 3, 3))
    left = tcp.AdditiveCharge(space, qudits={0: 1, 1: 1})
    right = tcp.AdditiveCharge(space, qudits={2: 1, 3: 1})
    operator = (
        0.2 * space.qudit.weyl(0, 0, 1)
        + 0.3 * space.qudit.weyl(1, 0, 1)
        + 0.5 * space.qudit.weyl(2, 0, 1)
        + 0.7 * space.qudit.weyl(3, 0, 1)
    )
    if two_regions:
        return operator, lambda: tcp.ChargeSector(((left, 0), (right, 0)))
    return operator, lambda: left.sector(0)


def _boson_weighted(two_regions: bool) -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(bosons=4)
    left = tcp.AdditiveCharge(space, bosons={0: 1, 1: 2})
    right = tcp.AdditiveCharge(space, bosons={2: 1, 3: 1})
    operator = (
        0.2 * space.boson.create(0) * space.boson.annihilate(0)
        + 0.3 * space.boson.create(1) * space.boson.annihilate(1)
        + space.boson.create(2) * space.boson.annihilate(3)
        + space.boson.create(3) * space.boson.annihilate(2)
    )
    cutoffs = {mode: 3 for mode in range(4)}
    if two_regions:
        return operator, lambda: tcp.ChargeSector(
            ((left, 2), (right, 2)), boson_cutoffs=cutoffs
        )
    return operator, lambda: left.sector(2, boson_cutoffs=cutoffs)


def _hybrid_factorized() -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(fermions=4, bosons=2)
    number = tcp.AdditiveCharge(space, fermions={mode: 1 for mode in range(4)})
    operator = (
        space.fermion.create(0) * space.fermion.annihilate(1)
        + space.fermion.create(1) * space.fermion.annihilate(0)
        + 0.2 * space.boson.create(0) * space.boson.annihilate(0)
        + 0.3 * space.boson.create(1) * space.boson.annihilate(1)
    )
    return operator, lambda: number.sector(2, boson_cutoffs={0: 3, 1: 3})


def _coupled_general(two_rows: bool) -> tuple[Any, SectorFactory]:
    space = tcp.OperatorSpace(qubits=2, bosons=1)
    first = tcp.AdditiveCharge(space, qubits={0: (0, 1)}, bosons={0: 1})
    second = tcp.AdditiveCharge(space, qubits={1: (0, 1)}, bosons={0: 1})
    operator = (
        0.2 * space.qubit.z(0)
        + 0.3 * space.qubit.z(1)
        + 0.4 * space.boson.create(0) * space.boson.annihilate(0)
    )
    if two_rows:
        return operator, lambda: tcp.ChargeSector(
            ((first, 1), (second, 1)), boson_cutoffs={0: 2}
        )
    return operator, lambda: first.sector(1, boson_cutoffs={0: 2})


def workloads() -> tuple[ChargeFactorizationWorkload, ...]:
    """Return the committed capability matrix in stable display order."""
    return (
        ChargeFactorizationWorkload("fermion-total-20-n10", (184_756,), _fermion_total),
        ChargeFactorizationWorkload(
            "fermion-balanced-10-10",
            (252, 252),
            lambda: _spin_components(5, spin_row=False),
        ),
        ChargeFactorizationWorkload(
            "fermion-unbalanced-10-10",
            (210, 210),
            lambda: _spin_components(6, spin_row=False),
        ),
        ChargeFactorizationWorkload(
            "fermion-equivalent-n-sz",
            (252, 252),
            lambda: _spin_components(5, spin_row=True),
        ),
        ChargeFactorizationWorkload(
            "qubit-low-population", (10, 2), lambda: _qubit_fixed(1)
        ),
        ChargeFactorizationWorkload(
            "qubit-central-population", (252, 2), lambda: _qubit_fixed(5)
        ),
        ChargeFactorizationWorkload(
            "qubit-interleaved-spectators", (3, 16), _qubit_interleaved_spectators
        ),
        ChargeFactorizationWorkload(
            "qudit-one-region", (3, 9), lambda: _qudit_regions(False)
        ),
        ChargeFactorizationWorkload(
            "qudit-two-regions", (3, 3), lambda: _qudit_regions(True)
        ),
        ChargeFactorizationWorkload(
            "boson-weighted", (2, 16), lambda: _boson_weighted(False)
        ),
        ChargeFactorizationWorkload(
            "boson-independent", (2, 3), lambda: _boson_weighted(True)
        ),
        ChargeFactorizationWorkload("hybrid-factorized", (6, 16), _hybrid_factorized),
        ChargeFactorizationWorkload(
            "coupled-scalar", (4,), lambda: _coupled_general(False)
        ),
        ChargeFactorizationWorkload(
            "coupled-multirow",
            (2,),
            lambda: _coupled_general(True),
            expected_execution_family="generic",
        ),
    )
