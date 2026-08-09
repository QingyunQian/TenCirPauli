"""Measure coupled scalar and wide Generic charge execution in fresh processes."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from _provenance import worktree_fingerprint

import tencirpauli as tcp


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "generic-wide-profile-v1"
charge_module = importlib.import_module("tencirpauli.charge")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@contextmanager
def _force_generic() -> Iterator[None]:
    names = (
        "_supports_independent_factor_product",
        "_fast_qubit_particles",
        "_fast_qudit_factor",
        "_fast_boson_factor",
        "_fast_fermion_particles",
    )
    originals = {name: getattr(charge_module, name) for name in names}
    for name in names:
        setattr(charge_module, name, lambda *args: None)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(charge_module, name, function)


def _case(n_qubits: int) -> Any:
    space = tcp.OperatorSpace(qubits=n_qubits)
    number = tcp.AdditiveCharge(
        space, qubits={index: (0, 1) for index in range(n_qubits)}
    )
    marked = tcp.AdditiveCharge(space, qubits={0: (0, 1)})
    operator = (
        space.qubit.x(1) * space.qubit.x(2)
        + space.qubit.y(1) * space.qubit.y(2)
        + 0.5 * space.qubit.z(1)
    )
    sector = tcp.ChargeSector(((number, 1), (marked, 0)))
    return operator.restrict_charge(sector)


def _measure_child(n_qubits: int, repeats: int) -> dict[str, Any]:
    with _force_generic():
        construction: list[float] = []
        restricted: Any = None
        plan: Any = None
        for _ in range(3):
            start = time.perf_counter()
            restricted = _case(n_qubits)
            plan = restricted.mvp_plan()
            construction.append(time.perf_counter() - start)
        assert restricted is not None and plan is not None
        if plan.native_variant != "generic":
            raise RuntimeError(f"expected generic variant, got {plan.native_variant!r}")
        state = np.arange(plan.dimension, dtype=np.complex128) + 0.125j
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
        eager_start = time.perf_counter()
        eager = restricted.mvp_plan(storage="eager")
        eager_seconds = time.perf_counter() - eager_start
        eager_output = eager.apply(state)
        error = float(np.max(np.abs(output - eager_output))) if output.size else 0.0
        if error > 5e-12:
            raise AssertionError(f"lazy/eager numerical disagreement: {error}")
        counters = tuple(
            int(value) for value in plan._native_execution.debug_counters()
        )
        return {
            "n_qubits": n_qubits,
            "key_width": n_qubits,
            "dimension": int(plan.dimension),
            "native_variant": str(plan.native_variant),
            "execution_family": str(plan.execution_family),
            "schedule_count": int(plan.schedule_count),
            "construction_seconds": construction,
            "construction_median_seconds": statistics.median(construction),
            "first_apply_into_seconds": first_seconds,
            "steady_apply_into_seconds": steady,
            "steady_apply_into_median_seconds": statistics.median(steady),
            "eager_construction_seconds": eager_seconds,
            "estimated_retained_bytes": int(plan.estimated_retained_bytes),
            "estimated_eager_retained_bytes": int(eager.estimated_retained_bytes),
            "peak_rss_bytes": _rss_bytes(),
            "debug_counters": counters,
            "max_abs_error_lazy_vs_eager": error,
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
            **worktree_fingerprint(ROOT),
        }


def _parse_record(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise RuntimeError("wide Generic child did not emit JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.repeats < 7:
        raise SystemExit("repeats must be at least seven")
    if args.child:
        if args.n_qubits not in {4, 65}:
            raise SystemExit("--child requires --n-qubits 4 or 65")
        print(json.dumps(_measure_child(args.n_qubits, args.repeats), sort_keys=True))
        return
    records: list[dict[str, Any]] = []
    for n_qubits in (4, 65):
        command = [
            args.python,
            str(Path(__file__).resolve()),
            "--child",
            "--n-qubits",
            str(n_qubits),
            "--repeats",
            str(args.repeats),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ),
            check=True,
            capture_output=True,
            text=True,
        )
        record = _parse_record(completed.stdout)
        record["exact_command"] = command
        records.append(record)
    scalar, wide = records
    if scalar["debug_counters"][0] <= 0 or wide["debug_counters"][1] <= 0:
        raise RuntimeError("Generic intended-path counter was not active")
    for record in records:
        if tuple(record["debug_counters"])[2:4] != (0, 0):
            raise RuntimeError("Generic accumulator unexpectedly grew or owned a key")
    print(
        json.dumps(
            {
                "benchmark": "generic_wide_profile",
                "workload_manifest": MANIFEST_VERSION,
                "records": records,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
