"""Record one coarse, reproducible additive-charge capability matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

factorized_charge = import_module("benchmarks.workloads.factorized_charge")
ChargeFactorizationWorkload = factorized_charge.ChargeFactorizationWorkload
workloads = factorized_charge.workloads


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    """Run one public operation and return its elapsed wall time in milliseconds."""
    start = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - start) * 1_000.0


def checksum(values: np.ndarray[Any, Any]) -> tuple[float, float]:
    """Return a compact numerical witness without serializing a state vector."""
    total = np.sum(values, dtype=np.complex128)
    return float(total.real), float(total.imag)


def measure_case(case: ChargeFactorizationWorkload) -> dict[str, object]:
    """Measure one stable workload through only public charge APIs."""
    operator, sector_factory = case.build()
    sector, sector_ms = timed(sector_factory)
    restricted, lazy_setup_ms = timed(lambda: operator.restrict_charge(sector))
    plan = restricted.mvp_plan()
    if plan.execution_family != case.expected_execution_family:
        raise AssertionError(
            f"{case.name} selected {plan.execution_family!r}, expected "
            f"{case.expected_execution_family!r}"
        )
    state = np.linspace(-0.5, 0.5, plan.dimension, dtype=np.float64).astype(
        np.complex128
    )
    output = np.empty_like(state)
    _, apply_into_ms = timed(lambda: plan.apply_into(state, output))
    allocated, apply_ms = timed(lambda: plan.apply(state))
    eager, eager_setup_ms = timed(
        lambda: operator.restrict_charge(sector, storage="eager")
    )
    csr, csr_ms = timed(eager.csr)
    return {
        "name": case.name,
        "component_dimensions": list(case.component_dimensions),
        "dimension": plan.dimension,
        "execution_family": plan.execution_family,
        "expected_execution_family": case.expected_execution_family,
        "native_variant": plan.native_variant,
        "schedule_count": plan.schedule_count,
        "retained_bytes": plan.estimated_bytes,
        "eager_retained_bytes": eager.estimated_bytes,
        "sector_setup_ms": sector_ms,
        "lazy_setup_ms": lazy_setup_ms,
        "apply_into_ms": apply_into_ms,
        "apply_ms": apply_ms,
        "direct_eager_setup_ms": eager_setup_ms,
        "csr_readback_ms": csr_ms,
        "csr_nonzeros": int(csr.data.size),
        "checksum": list(checksum(allocated)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".benchmarks" / "factorized-charge-matrix.json",
        help="local JSON result path (default: .benchmarks/factorized-charge-matrix.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [measure_case(case) for case in workloads()]
    payload = {"schema_version": 1, "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(
            f"{row['name']}: apply_into={row['apply_into_ms']:.2f} ms, "
            f"eager={row['direct_eager_setup_ms']:.2f} ms, "
            f"csr={row['csr_readback_ms']:.2f} ms"
        )
    print(f"Saved factorized charge matrix: {args.output}")


if __name__ == "__main__":
    main()
