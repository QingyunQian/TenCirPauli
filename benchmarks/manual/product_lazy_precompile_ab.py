"""Release-mode Product lazy descriptor benchmark.

The benchmark deliberately measures the public construction boundary together
with the native plan, then separates first, steady ``apply_into`` and repeated
allocating ``apply``.  Run it with the release-built project environment; the
JSON record is evidence, not a CI wall-time gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from typing import Any, Callable

import numpy as np

import tencirpauli as tcp


charge_module = importlib.import_module("tencirpauli.charge")


def _spinful_product(sites: int, cutoff: int, particles: int) -> tuple[Any, Any]:
    space = tcp.OperatorSpace(fermions=2 * sites, bosons=sites)
    total = tcp.AdditiveCharge(space, fermions={mode: 1 for mode in range(2 * sites)})
    balance = tcp.AdditiveCharge(
        space,
        fermions={
            2 * site + spin: (1 if spin == 0 else -1)
            for site in range(sites)
            for spin in range(2)
        },
    )
    terms: Any = 0.2 * space.boson.create(0) * space.boson.annihilate(0)
    for site in range(sites - 1):
        for spin in range(2):
            left = 2 * site + spin
            right = 2 * (site + 1) + spin
            terms = terms + space.fermion.create(left) * space.fermion.annihilate(right)
            terms = terms + space.fermion.create(right) * space.fermion.annihilate(left)
    sector = tcp.ChargeSector(
        ((total, 2 * particles), (balance, 0)),
        boson_cutoffs={site: cutoff for site in range(sites)},
    )
    return terms.restrict_charge(sector), sector


def _measure(
    name: str,
    factory: Callable[[], tuple[Any, Any]],
    repeats: int,
    *,
    force_generic: bool = False,
) -> dict[str, Any]:
    construction_samples: list[float] = []
    restricted: Any = None
    sector: Any = None
    for _ in range(3):
        start = time.perf_counter()
        if force_generic:
            original = charge_module._supports_independent_factor_product
            original_fermion = charge_module._fast_fermion_particles
            charge_module._supports_independent_factor_product = lambda _: None
            charge_module._fast_fermion_particles = lambda *_: None
            try:
                restricted, sector = factory()
                restricted.mvp_plan()
            finally:
                charge_module._supports_independent_factor_product = original
                charge_module._fast_fermion_particles = original_fermion
        else:
            restricted, sector = factory()
            restricted.mvp_plan()
        construction_samples.append(time.perf_counter() - start)
    plan = restricted.mvp_plan()
    state = np.arange(sector.dimension, dtype=np.complex128) + 0.17j
    output = np.empty_like(state)
    apply_budget = None if force_generic else 0
    for _ in range(2):
        plan.apply_into(state, output, max_bytes=apply_budget)
    first_start = time.perf_counter()
    plan.apply_into(state, output, max_bytes=apply_budget)
    first_seconds = time.perf_counter() - first_start
    steady: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        plan.apply_into(state, output, max_bytes=apply_budget)
        steady.append(time.perf_counter() - start)
    repeated: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        plan.apply(state)
        repeated.append(time.perf_counter() - start)
    return {
        "case": name,
        "implementation": (
            "generic_term_interpreter_baseline"
            if force_generic
            else "product_factor_local_action_descriptors"
        ),
        "dimension": int(sector.dimension),
        "term_count": int(plan.term_count),
        "execution_family": str(plan.execution_family),
        "native_variant": str(getattr(plan, "native_variant", "unknown")),
        "retained_format": str(plan.retained_format),
        "estimated_retained_bytes": int(plan.estimated_retained_bytes),
        "construction_seconds_median": statistics.median(construction_samples),
        "first_apply_into_seconds": first_seconds,
        "steady_apply_into_seconds": steady,
        "steady_apply_into_median_seconds": statistics.median(steady),
        "repeated_apply_seconds": repeated,
        "repeated_apply_median_seconds": statistics.median(repeated),
        "debug_counters": plan._native_execution.debug_counters(),
        "max_abs_output": float(np.max(np.abs(output))) if output.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--particles", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    records = [
        _measure(
            "spinful_generic_term_interpreter_baseline",
            lambda: _spinful_product(args.sites, args.cutoff, args.particles),
            args.repeats,
            force_generic=True,
        ),
        _measure(
            "spinful_fermion_factor_cartesian_bosons",
            lambda: _spinful_product(args.sites, args.cutoff, args.particles),
            args.repeats,
        ),
    ]
    print(
        json.dumps(
            {"release_benchmark": "product_lazy_precompile", "records": records},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
