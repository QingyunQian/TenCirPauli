"""Stable public-path benchmarks for additive-charge factorization."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

factorized_charge = import_module("benchmarks.workloads.factorized_charge")
ChargeFactorizationWorkload = factorized_charge.ChargeFactorizationWorkload
workloads = factorized_charge.workloads


CASES = workloads()


def _assert_execution_family(case: ChargeFactorizationWorkload, plan: Any) -> None:
    assert plan.execution_family == case.expected_execution_family


def _metadata(
    benchmark: BenchmarkFixture,
    case: ChargeFactorizationWorkload,
    plan: Any,
    checksum: complex,
) -> None:
    benchmark.extra_info.update(
        {
            "component_dimensions": list(case.component_dimensions),
            "dimension": int(plan.dimension),
            "execution_family": str(plan.execution_family),
            "native_variant": str(plan.native_variant),
            "retained_bytes": int(plan.estimated_bytes),
            "schedule_count": int(plan.schedule_count),
            "checksum": [float(checksum.real), float(checksum.imag)],
        }
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_charge_factorization_sector_setup(
    benchmark: BenchmarkFixture, case: ChargeFactorizationWorkload
) -> None:
    _, make_sector = case.build()
    sector = benchmark(make_sector)
    benchmark.extra_info.update(
        {
            "component_dimensions": list(case.component_dimensions),
            "dimension": int(sector.dimension),
            "retained_bytes": int(sector.estimated_bytes),
        }
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_charge_factorization_lazy_setup(
    benchmark: BenchmarkFixture, case: ChargeFactorizationWorkload
) -> None:
    operator, make_sector = case.build()
    sector = make_sector()
    restricted = benchmark(operator.restrict_charge, sector)
    plan = restricted.mvp_plan()
    _assert_execution_family(case, plan)
    state = np.linspace(-0.5, 0.5, plan.dimension).astype(np.complex128)
    _metadata(benchmark, case, plan, np.sum(plan.apply(state)))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_charge_factorization_apply_into(
    benchmark: BenchmarkFixture, case: ChargeFactorizationWorkload
) -> None:
    operator, make_sector = case.build()
    plan = operator.restrict_charge(make_sector()).mvp_plan()
    _assert_execution_family(case, plan)
    state = np.linspace(-0.5, 0.5, plan.dimension).astype(np.complex128)
    output = np.empty_like(state)
    benchmark(plan.apply_into, state, output)
    _metadata(benchmark, case, plan, np.sum(output))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_charge_factorization_allocating_apply(
    benchmark: BenchmarkFixture, case: ChargeFactorizationWorkload
) -> None:
    operator, make_sector = case.build()
    plan = operator.restrict_charge(make_sector()).mvp_plan()
    _assert_execution_family(case, plan)
    state = np.linspace(-0.5, 0.5, plan.dimension).astype(np.complex128)
    output = benchmark(plan.apply, state)
    _metadata(benchmark, case, plan, np.sum(output))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_charge_factorization_direct_eager_and_csr(
    benchmark: BenchmarkFixture, case: ChargeFactorizationWorkload
) -> None:
    operator, make_sector = case.build()
    sector = make_sector()

    def build_and_read_csr() -> int:
        restricted = operator.restrict_charge(sector, storage="eager")
        assert restricted._lazy_plan is None
        return int(restricted.csr().data.size)

    nonzeros = benchmark(build_and_read_csr)
    eager = operator.restrict_charge(sector, storage="eager")
    plan = eager.mvp_plan(storage="eager")
    _assert_execution_family(case, plan)
    state = np.linspace(-0.5, 0.5, plan.dimension).astype(np.complex128)
    _metadata(benchmark, case, plan, np.sum(plan.apply(state)))
    benchmark.extra_info["csr_nonzeros"] = nonzeros
