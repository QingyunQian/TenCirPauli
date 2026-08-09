"""Profile the raw FermionMask path against explicit Jordan--Wigner mapping.

The timing records include public construction and caller-owned native MVP
application. The native phase values are batched descriptor microbenchmarks.
They also report source-packing units and schedule-member evaluations, so
source reuse is visible without per-contribution clock reads.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from _provenance import worktree_fingerprint

import tencirpauli as tcp


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "fermion-mask-phase-v3"


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _operator(modes: int) -> tcp.FermionOperator:
    """Build a local hopping-and-density model on the requested mode count."""
    if modes < 2:
        raise ValueError("modes must be at least two")
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for mode in range(modes):
        terms.append((((mode, "create"), (mode, "annihilate")), 0.15 * (mode + 1)))
    for mode in range(modes - 1):
        coefficient = 0.7 + 0.05j * (mode + 1)
        terms.append((((mode, "create"), (mode + 1, "annihilate")), coefficient))
        terms.append(
            (((mode + 1, "create"), (mode, "annihilate")), coefficient.conjugate())
        )
    return tcp.FermionOperator.from_terms(modes, terms)


def _phase_record(plan: Any, samples: int) -> dict[str, Any] | None:
    native = getattr(plan, "_native_plan", None)
    profile = getattr(native, "debug_fermion_mask_phase_profile", lambda _: None)(
        samples
    )
    if profile is None:
        return None
    (
        descriptor_count,
        sample_count,
        packed_source_units,
        schedule_member_evaluations,
        extraction,
        parity,
        index_update,
        input_output,
    ) = (int(value) for value in profile)
    phases = {
        "packed_source_extraction_nanos": extraction,
        "parity_nanos": parity,
        "index_update_nanos": index_update,
        "input_output_nanos": input_output,
    }
    total = sum(phases.values())
    return {
        "descriptor_count": descriptor_count,
        "sample_count": sample_count,
        "packed_source_unit_count": packed_source_units,
        "schedule_member_evaluations": schedule_member_evaluations,
        **phases,
        "phase_total_nanos": total,
        "phase_shares": {
            name.removesuffix("_nanos"): (value / total if total else 0.0)
            for name, value in phases.items()
        },
    }


def _measure(
    name: str,
    factory: Callable[[tcp.FermionOperator], Any],
    repeats: int,
    phase_samples: int,
    modes: int,
) -> tuple[dict[str, Any], np.ndarray[Any, Any]]:
    construction: list[float] = []
    operator = _operator(modes)
    plan: Any = None
    for _ in range(3):
        start = time.perf_counter()
        plan = factory(operator)
        construction.append(time.perf_counter() - start)
    assert plan is not None
    rng = np.random.default_rng(20260809)
    state = np.asarray(
        rng.normal(size=plan.dimension) + 1j * rng.normal(size=plan.dimension),
        dtype=np.complex128,
    )
    output = np.empty_like(state)
    for _ in range(2):
        plan.apply_into(state, output, max_bytes=None)
    start = time.perf_counter()
    plan.apply_into(state, output, max_bytes=None)
    first_seconds = time.perf_counter() - start
    steady: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        plan.apply_into(state, output, max_bytes=None)
        steady.append(time.perf_counter() - start)
    native = getattr(plan, "_native_plan", None)
    return (
        {
            "variant": name,
            "native_variant": str(
                getattr(
                    plan, "native_variant", getattr(native, "native_variant", "unknown")
                )
            ),
            "execution_family": str(plan.execution_family),
            "dimension": int(plan.dimension),
            "source_term_count": int(
                getattr(plan, "source_term_count", plan.term_count)
            ),
            "schedule_count": int(getattr(plan, "schedule_count", plan.term_count)),
            "construction_seconds": construction,
            "construction_median_seconds": statistics.median(construction),
            "first_apply_into_seconds": first_seconds,
            "steady_apply_into_seconds": steady,
            "steady_apply_into_median_seconds": statistics.median(steady),
            "estimated_retained_bytes": int(plan.estimated_retained_bytes),
            "peak_rss_bytes": _rss_bytes(),
            "fermion_mask_phase_profile": _phase_record(plan, phase_samples),
            "thread_environment": {
                key: os.environ.get(key, "unset")
                for key in (
                    "RAYON_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        output.copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--phase-samples", type=int, default=4096)
    parser.add_argument(
        "--modes",
        type=int,
        default=None,
        help="fermion mode count; defaults to 12 or 18 with --throughput",
    )
    parser.add_argument(
        "--throughput",
        action="store_true",
        help="use the 18-mode, D=262144 throughput workload",
    )
    args = parser.parse_args()
    if args.repeats < 7 or args.phase_samples < 1:
        raise SystemExit("repeats must be at least seven and phase samples positive")
    modes = 18 if args.throughput and args.modes is None else (args.modes or 12)
    if modes < 2 or modes > 63:
        raise SystemExit("modes must be between 2 and 63")
    raw, raw_output = _measure(
        "raw_fermion_mask",
        lambda operator: operator.compile("native_mvp"),
        args.repeats,
        args.phase_samples,
        modes,
    )
    mapped, mapped_output = _measure(
        "explicit_jordan_wigner_pauli",
        lambda operator: operator.map_fermions("jordan_wigner").compile("native_mvp"),
        args.repeats,
        args.phase_samples,
        modes,
    )
    error = float(np.max(np.abs(raw_output - mapped_output)))
    if error > 5e-12:
        raise AssertionError(f"raw/mapped numerical disagreement: {error}")
    print(
        json.dumps(
            {
                "benchmark": "fermion_mask_phase_profile",
                "workload_manifest": MANIFEST_VERSION,
                "modes": modes,
                "records": [raw, mapped],
                "max_abs_error": error,
                "exact_command": [sys.executable, str(Path(__file__).resolve())],
                **worktree_fingerprint(ROOT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
