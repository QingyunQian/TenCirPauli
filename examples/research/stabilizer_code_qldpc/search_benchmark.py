"""Profile logical-witness quality and end-to-end time on larger research workloads.

This script is intentionally outside the repository's bounded regression
benchmark suite. It is for choosing search profiles on HGP and other large
static-code workloads; the default run has a three-minute wall-clock budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from workloads import Workload, get_workload

import tencirpauli as tcp


DEFAULT_WORKLOADS = ("hgp_8", "hgp_16", "hgp_24", "hgp_32")
DEFAULT_SKETCH_ROWS = (32, 64)
DEFAULT_TRIALS = (16, 64)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _build(workload: Workload) -> tcp.StabilizerCode:
    if workload.kind == "css":
        return tcp.StabilizerCode.from_css(
            workload.x_checks, workload.z_checks, max_bytes=None
        )
    return tcp.StabilizerCode.from_symplectic(
        workload.x_checks, workload.z_checks, max_bytes=None
    )


def _parse_csv(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated integers"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def _digest_witness(result: tcp.LogicalWitnessSearchResult) -> str:
    digest = hashlib.sha256()
    for array in (
        result.best.x_words,
        result.best.z_words,
        result.best.logical_x_words,
        result.best.logical_z_words,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_result(
    code: tcp.StabilizerCode, result: tcp.LogicalWitnessSearchResult
) -> None:
    word = tcp.PauliWord(
        result.best.nqubits,
        tuple(int(value) for value in result.best.x_words),
        tuple(int(value) for value in result.best.z_words),
    )
    classification = code.classify(word)
    if classification.kind is not tcp.StabilizerErrorKind.LOGICAL:
        raise RuntimeError("search returned a non-logical witness")
    if any(classification.syndrome):
        raise RuntimeError("search returned a witness with a nonzero syndrome")
    if result.best_weight != result.best.weight or result.best.weight != word.weight:
        raise RuntimeError("search returned an inconsistent witness weight")


def _measure(
    code: tcp.StabilizerCode,
    workload: Workload,
    sector: str,
    sketch_rows: int,
    max_trials: int,
    repetitions: int,
) -> dict[str, Any]:
    def run() -> tcp.LogicalWitnessSearchResult:
        return code.search_logical_witness(
            sector=sector,
            sketch_rows=sketch_rows,
            max_trials=max_trials,
            seed=workload.seed,
            max_bytes=None,
        )

    first_start = time.perf_counter()
    first = run()
    first_seconds = time.perf_counter() - first_start
    _validate_result(code, first)
    steady_seconds = []
    steady_metadata: list[dict[str, int | bool]] = []
    result = first
    for _ in range(repetitions):
        start = time.perf_counter()
        result = run()
        steady_seconds.append(time.perf_counter() - start)
        steady_metadata.append(
            {
                "setup_nanos": result.setup_nanos,
                "initialization_nanos": result.initialization_nanos,
                "initial_incumbent_weight": result.initial_incumbent_weight,
                "initialization_candidates": result.initialization_candidates,
                "resolved_sketch_rows": result.resolved_sketch_rows,
            }
        )
        _validate_result(code, result)
        if result.best_weight != first.best_weight or _digest_witness(
            result
        ) != _digest_witness(first):
            raise RuntimeError("seeded search replay changed during the benchmark")
    median_seconds = statistics.median(steady_seconds)
    return {
        "workload": workload.name,
        "family": workload.family,
        "nqubits": code.nqubits,
        "nchecks": code.nchecks,
        "rank": code.rank,
        "nlogical": code.nlogical,
        "sector": sector,
        "algorithm": result.algorithm,
        "sketch_rows_requested": sketch_rows,
        "resolved_sketch_rows": first.resolved_sketch_rows,
        "max_trials": max_trials,
        "trials_run": result.trials_run,
        "first_seconds": first_seconds,
        "steady_seconds": [float(value) for value in steady_seconds],
        "steady_seconds_median": median_seconds,
        "trials_per_second": (
            result.trials_run / median_seconds if median_seconds else 0.0
        ),
        "best_weight": result.best_weight,
        "initial_incumbent_weight": first.initial_incumbent_weight,
        "improvement_weight": (first.initial_incumbent_weight - result.best_weight),
        "improved": result.best_weight < first.initial_incumbent_weight,
        "initialization_candidates": first.initialization_candidates,
        "setup_nanos": first.setup_nanos,
        "initialization_nanos": first.initialization_nanos,
        "steady_metadata": steady_metadata,
        "witness_sector": result.best.sector,
        "witness_digest": _digest_witness(result),
        "seed": result.seed,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        default=",".join(DEFAULT_WORKLOADS),
        help="comma-separated names from workloads.available_workloads()",
    )
    parser.add_argument(
        "--sketch-rows",
        default=",".join(str(value) for value in DEFAULT_SKETCH_ROWS),
        help="comma-separated sketch caps; tune these outside the regression suite",
    )
    parser.add_argument(
        "--trials",
        default=",".join(str(value) for value in DEFAULT_TRIALS),
        help="comma-separated randomized trial budgets",
    )
    parser.add_argument(
        "--sectors",
        help="optional comma-separated sectors; defaults to all applicable sectors",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--time-budget", type=float, default=180.0)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="fail unless at least one record beats its deterministic incumbent",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if arguments.time_budget <= 0:
        raise SystemExit("--time-budget must be positive")
    names = tuple(
        item.strip() for item in arguments.workloads.split(",") if item.strip()
    )
    if not names:
        raise SystemExit("--workloads must not be empty")
    sketch_rows = _parse_csv(arguments.sketch_rows, "--sketch-rows")
    trials = _parse_csv(arguments.trials, "--trials")
    requested_sectors = (
        tuple(item.strip() for item in arguments.sectors.split(",") if item.strip())
        if arguments.sectors is not None
        else None
    )
    if requested_sectors is not None:
        allowed_sectors = {"auto", "x", "z", "general"}
        if not requested_sectors or any(
            sector not in allowed_sectors for sector in requested_sectors
        ):
            raise SystemExit("--sectors must contain auto, x, z, or general")
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    stopped_early = False
    saw_improvement = False
    for name in names:
        workload = get_workload(name)
        matrix_hash = workload.matrix_hash
        build_start = time.perf_counter()
        code = _build(workload)
        build_seconds = time.perf_counter() - build_start
        available_sectors = (
            ("auto", "x", "z", "general") if workload.kind == "css" else ("general",)
        )
        sectors = available_sectors if requested_sectors is None else requested_sectors
        if any(sector not in available_sectors for sector in sectors):
            raise SystemExit(
                f"workload {name!r} does not support requested sectors {sectors}"
            )
        for sector in sectors:
            for requested_sketch_rows in sketch_rows:
                for max_trials in trials:
                    if time.perf_counter() - started >= arguments.time_budget:
                        stopped_early = True
                        break
                    record = _measure(
                        code,
                        workload,
                        sector,
                        requested_sketch_rows,
                        max_trials,
                        arguments.repetitions,
                    )
                    record["construction_seconds"] = build_seconds
                    record["matrix_hash"] = matrix_hash
                    records.append(record)
                    saw_improvement = saw_improvement or bool(record["improved"])
                if stopped_early:
                    break
            if stopped_early:
                break
        if stopped_early:
            break
    if arguments.require_improvement and not saw_improvement:
        raise SystemExit(
            "--require-improvement requested, but no randomized trial improved "
            "the deterministic incumbent"
        )
    report = {
        "schema_version": 1,
        "purpose": "logical_witness_quality_vs_end_to_end_time",
        "time_budget_seconds": arguments.time_budget,
        "elapsed_seconds": time.perf_counter() - started,
        "stopped_early": stopped_early,
        "saw_improvement": saw_improvement,
        "package_file": str(Path(tcp.__file__).resolve()),
        "native_file": str(Path(tcp._native.__file__).resolve()),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "unset"),
        "records": records,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload)
        print(arguments.output)


if __name__ == "__main__":
    main()
