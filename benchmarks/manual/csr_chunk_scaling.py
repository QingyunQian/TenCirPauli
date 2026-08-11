"""Measure deterministic CSR construction across independent chunk processes.

The parent launches one fresh child per target chunk count.  Every child
compares its materialized CSR arrays and MVP output with an exact one-chunk
reference, so the reported RSS is a per-capacity process high-water mark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from _provenance import worktree_fingerprint

import tencirpauli as tcp


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()


def _operator() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(fermions=8)
    charge = tcp.AdditiveCharge(space, fermions={mode: 1 for mode in range(8)})
    terms: list[Any] = []
    for left in range(8):
        for right in range(left + 1, 8):
            terms.append(space.fermion.create(left) * space.fermion.annihilate(right))
            terms.append(space.fermion.create(right) * space.fermion.annihilate(left))
    operator = terms[0]
    for term in terms[1:]:
        operator = operator + term
    return operator, charge.sector(4)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _arrays(eager: Any) -> tuple[np.ndarray[Any, Any], ...]:
    return tuple(np.asarray(value).copy() for value in eager.csr(2**63 - 1))


def _measure_child(target_chunks: int, repeats: int) -> dict[str, Any]:
    operator, sector = _operator()
    lazy = operator.restrict_charge(sector).mvp_plan()
    native = lazy._native_execution
    probe = native.compile_eager_with_chunk_capacity(2**63 - 1, 1 << 20)
    transition_count = int(probe.transition_count)
    del probe
    reference_capacity = max(1, transition_count)
    reference = native.compile_eager_with_chunk_capacity(2**63 - 1, reference_capacity)
    reference_arrays = _arrays(reference)
    state = np.arange(sector.dimension, dtype=np.complex128) + 0.125j
    reference_output = np.empty_like(state)
    reference.apply_into(state, reference_output, max_bytes=2**63 - 1)
    del reference

    capacity = max(1, math.ceil(transition_count / target_chunks))
    samples: list[float] = []
    profiles: list[tuple[int, ...]] = []
    csr_equal = True
    mvp_equal = True
    for _ in range(repeats):
        start = time.perf_counter()
        eager = native.compile_eager_with_chunk_capacity(2**63 - 1, capacity)
        samples.append(time.perf_counter() - start)
        profile = tuple(int(value) for value in eager.construction_profile())
        profiles.append(profile)
        actual_arrays = _arrays(eager)
        csr_equal = csr_equal and all(
            np.array_equal(actual, expected)
            for actual, expected in zip(actual_arrays, reference_arrays)
        )
        output = np.empty_like(state)
        eager.apply_into(state, output, max_bytes=2**63 - 1)
        mvp_equal = mvp_equal and bool(np.array_equal(output, reference_output))
        del output, actual_arrays, eager
    if not csr_equal or not mvp_equal:
        raise AssertionError(
            f"chunk capacity {capacity} disagrees with the one-chunk reference"
        )
    profile = profiles[-1]
    return {
        "target_chunks": target_chunks,
        "chunk_capacity": capacity,
        "transition_count": transition_count,
        "construction_seconds": samples,
        "construction_median_seconds": statistics.median(samples),
        "peak_rss_bytes": _rss_bytes(),
        "generation_nanos": profile[0],
        "chunk_count": profile[1],
        "sorted_entries": profile[2],
        "merged_entries": profile[3],
        "heap_operations": profile[4],
        "final_array_allocation_entries": profile[5],
        "chunk_metadata_bytes": profile[6],
        "sort_nanos": profile[7],
        "merge_nanos": profile[8],
        "final_array_allocation_nanos": profile[9],
        "csr_equal_to_reference": csr_equal,
        "mvp_equal_to_reference": mvp_equal,
        "execution_family": str(lazy.execution_family),
        "native_variant": str(lazy.native_variant),
        "thread_environment": {
            name: os.environ.get(name, "unset")
            for name in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def _parse_record(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise RuntimeError("CSR child did not emit JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--target-chunks", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--targets", type=int, nargs="+", default=[1, 8, 64, 512])
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.targets):
        raise SystemExit("repeats and target chunk counts must be positive")
    if args.child:
        if args.target_chunks is None or args.target_chunks < 1:
            raise SystemExit("--child requires a positive --target-chunks")
        print(
            json.dumps(_measure_child(args.target_chunks, args.repeats), sort_keys=True)
        )
        return

    records: list[dict[str, Any]] = []
    for target in args.targets:
        command = [
            args.python,
            str(SCRIPT),
            "--child",
            "--target-chunks",
            str(target),
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
    if not all(
        record["csr_equal_to_reference"] and record["mvp_equal_to_reference"]
        for record in records
    ):
        raise RuntimeError("CSR reference equality failed")
    print(
        json.dumps(
            {
                "workload": "eight_mode_fixed_number_hopping",
                "workload_manifest": "csr-chunk-scaling-v1",
                "basis_dimension": 70,
                "independent_process_per_capacity": True,
                "records": records,
                **worktree_fingerprint(ROOT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
