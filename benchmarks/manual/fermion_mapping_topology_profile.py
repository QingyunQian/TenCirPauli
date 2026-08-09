"""Measure raw FermionMask versus Jordan--Wigner across operator locality.

This release-mode diagnostic uses a common 18-mode full-space statevector for
a local chain, long-range one-body hopping, and nonlocal quartic pair transfer.
It records both source and mapped term counts, construction, caller-owned MVP
application, and one raw descriptor-phase profile. Run with
``python benchmarks/manual/fermion_mapping_topology_profile.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from _provenance import worktree_fingerprint

import tencirpauli as tcp


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "fermion-mapping-topology-v1"
ModelName = Literal["local_chain", "long_range_hopping", "nonlocal_pair_transfer"]
MODELS: tuple[ModelName, ...] = (
    "local_chain",
    "long_range_hopping",
    "nonlocal_pair_transfer",
)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _local_chain_terms(modes: int) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for mode in range(modes):
        terms.append((((mode, "create"), (mode, "annihilate")), 0.15 * (mode + 1)))
    for mode in range(modes - 1):
        coefficient = 0.7 + 0.05j * (mode + 1)
        terms.append((((mode, "create"), (mode + 1, "annihilate")), coefficient))
        terms.append(
            (((mode + 1, "create"), (mode, "annihilate")), coefficient.conjugate())
        )
    return terms


def _long_range_hopping_terms(
    modes: int,
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    separation = modes // 3
    for mode in range(modes):
        terms.append((((mode, "create"), (mode, "annihilate")), 0.11 * (mode + 1)))
    for left in range(modes):
        for right in range(left + separation, modes):
            coefficient = (0.37 + 0.02j * (left + right + 1)) / (right - left)
            terms.append((((left, "create"), (right, "annihilate")), coefficient))
            terms.append(
                (((right, "create"), (left, "annihilate")), coefficient.conjugate())
            )
    return terms


def _nonlocal_pair_transfer_terms(
    modes: int,
) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    if modes % 3:
        raise ValueError(
            "nonlocal pair-transfer workload requires a mode count divisible by three"
        )
    block = modes // 3
    left_pairs = tuple(itertools.combinations(range(block), 2))
    right_pairs = tuple(itertools.combinations(range(2 * block, 3 * block), 2))
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for left_index, (first, second) in enumerate(left_pairs):
        for offset in (0, block - 1):
            third, fourth = right_pairs[(5 * left_index + offset) % len(right_pairs)]
            coefficient = 0.09 + 0.01j * (left_index + offset + 1)
            terms.append(
                (
                    (
                        (first, "create"),
                        (second, "create"),
                        (third, "annihilate"),
                        (fourth, "annihilate"),
                    ),
                    coefficient,
                )
            )
            terms.append(
                (
                    (
                        (fourth, "create"),
                        (third, "create"),
                        (second, "annihilate"),
                        (first, "annihilate"),
                    ),
                    coefficient.conjugate(),
                )
            )
    return terms


def _operator(modes: int, model: ModelName) -> tcp.FermionOperator:
    if modes < 6:
        raise ValueError("modes must be at least six")
    if model == "local_chain":
        terms = _local_chain_terms(modes)
    elif model == "long_range_hopping":
        terms = _long_range_hopping_terms(modes)
    else:
        terms = _nonlocal_pair_transfer_terms(modes)
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
    operator: tcp.FermionOperator,
    repeats: int,
    phase_samples: int,
    state: np.ndarray[Any, Any],
) -> tuple[dict[str, Any], np.ndarray[Any, Any]]:
    construction: list[float] = []
    plan: Any = None
    for _ in range(3):
        start = time.perf_counter()
        plan = factory(operator)
        construction.append(time.perf_counter() - start)
    assert plan is not None
    if state.shape != (plan.dimension,):
        raise RuntimeError("shared state does not match the compiled plan dimension")
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


def _profile_model(
    model: ModelName, modes: int, repeats: int, phase_samples: int
) -> dict[str, Any]:
    operator = _operator(modes, model)
    dimension = 1 << modes
    generator = np.random.default_rng(20260809)
    state = np.asarray(
        generator.normal(size=dimension) + 1j * generator.normal(size=dimension),
        dtype=np.complex128,
    )
    raw, raw_output = _measure(
        "raw_fermion_mask",
        lambda value: value.compile("native_mvp"),
        operator,
        repeats,
        phase_samples,
        state,
    )
    mapped, mapped_output = _measure(
        "explicit_jordan_wigner_pauli",
        lambda value: value.map_fermions("jordan_wigner").compile("native_mvp"),
        operator,
        repeats,
        phase_samples,
        state,
    )
    error = float(np.max(np.abs(raw_output - mapped_output)))
    if error > 5e-12:
        raise AssertionError(f"{model} raw/mapped numerical disagreement: {error}")
    return {
        "model": model,
        "input_term_count": len(_operator(modes, model).terms),
        "records": [raw, mapped],
        "max_abs_error": error,
        "raw_over_jordan_wigner_apply_ratio": (
            raw["steady_apply_into_median_seconds"]
            / mapped["steady_apply_into_median_seconds"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=18)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--phase-samples", type=int, default=4096)
    parser.add_argument("--model", choices=MODELS, action="append")
    args = parser.parse_args()
    if args.modes < 6 or args.modes > 24:
        raise SystemExit("modes must be between 6 and 24")
    if args.repeats < 7 or args.phase_samples < 1:
        raise SystemExit("repeats must be at least seven and phase samples positive")
    selected = tuple(args.model) if args.model else MODELS
    print(
        json.dumps(
            {
                "benchmark": "fermion_mapping_topology_profile",
                "workload_manifest": MANIFEST_VERSION,
                "modes": args.modes,
                "profiles": [
                    _profile_model(model, args.modes, args.repeats, args.phase_samples)
                    for model in selected
                ],
                "exact_command": [sys.executable, str(Path(__file__).resolve())],
                **worktree_fingerprint(ROOT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
