"""Release A/B for a 16-mode U1 Product plan and its fallbacks.

The three records use the identical two-term operator and four-particle
sector.  Only the Python selector hooks differ: the shared Product executor,
the legacy specialized qubit executor, and the generic compiled schedule.
Each record reports the native variant actually selected, numerical agreement
with the generic reference, and steady caller-owned execution time.
"""

from __future__ import annotations

import importlib
import json
import statistics
import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

import tencirpauli as tcp


charge_module = importlib.import_module("tencirpauli.charge")


@contextmanager
def _selector_mode(mode: str) -> Iterator[None]:
    original_product = charge_module._supports_independent_factor_product
    original_qubit = charge_module._fast_qubit_particles
    if mode == "product":
        replacements = (original_product, lambda *_: None)
    elif mode == "specialized_qubit":
        replacements = (lambda _: None, original_qubit)
    elif mode == "generic":
        replacements = (lambda _: None, lambda *_: None)
    else:
        raise ValueError(f"unknown selector mode: {mode!r}")
    charge_module._supports_independent_factor_product = replacements[0]
    charge_module._fast_qubit_particles = replacements[1]
    try:
        yield
    finally:
        charge_module._supports_independent_factor_product = original_product
        charge_module._fast_qubit_particles = original_qubit


def _operator_and_sector() -> tuple[Any, Any]:
    operator = tcp.PauliOperator.from_terms(
        16,
        [
            ("XX" + "I" * 14, 1.0),
            ("YY" + "I" * 14, 1.0),
        ],
    )
    space = tcp.OperatorSpace(qubits=16)
    number = tcp.AdditiveCharge(
        space,
        qubits={index: (0, 1) for index in range(16)},
    )
    return operator, tcp.ChargeSector(((number, 4),))


def _measure(mode: str, repeats: int) -> tuple[dict[str, Any], np.ndarray[Any, Any]]:
    construction: list[float] = []
    restricted: Any = None
    plan: Any = None
    with _selector_mode(mode):
        for _ in range(3):
            start = time.perf_counter()
            operator, sector = _operator_and_sector()
            # Call the general charge boundary directly so the benchmark can
            # compare Product and the legacy Qubit executor on one API path;
            # PauliOperator.restrict_charge(U1Sector(...)) intentionally keeps
            # the separate compatibility U1 facade.
            restricted = charge_module.restrict_charge(operator, sector)
            plan = restricted.mvp_plan()
            construction.append(time.perf_counter() - start)
        state = np.arange(plan.dimension, dtype=np.complex128) + 0.125j
        output = np.empty_like(state)
        for _ in range(2):
            plan.apply_into(state, output, max_bytes=None)
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            plan.apply_into(state, output, max_bytes=None)
            samples.append(time.perf_counter() - start)
        counters = plan._native_execution.debug_counters()
        record = {
            "case": mode,
            "dimension": int(plan.dimension),
            "term_count": int(plan.term_count),
            "execution_family": str(plan.execution_family),
            "native_variant": str(plan.native_variant),
            "retained_format": str(plan.retained_format),
            "construction_seconds_median": statistics.median(construction),
            "steady_apply_into_seconds_median": statistics.median(samples),
            "steady_apply_into_seconds": samples,
            "estimated_retained_bytes": int(plan.estimated_retained_bytes),
            "debug_counters": counters,
        }
        return record, output.copy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray[Any, Any]] = {}
    for mode in ("generic", "product", "specialized_qubit"):
        record, output = _measure(mode, args.repeats)
        records.append(record)
        outputs[mode] = output
    reference = outputs["generic"]
    expected_variants = {
        "generic": "generic",
        "product": "product",
        "specialized_qubit": "qubit",
    }
    for record in records:
        expected_variant = expected_variants[record["case"]]
        if record["native_variant"] != expected_variant:
            raise RuntimeError(
                f"{record['case']} selected native variant "
                f"{record['native_variant']!r}, expected {expected_variant!r}"
            )
        record["max_abs_error_vs_generic"] = float(
            np.max(np.abs(outputs[record["case"]] - reference))
        )
    print(
        json.dumps(
            {"workload": "u1_16_mode_product_ab", "records": records}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
