"""Measure Product lazy execution across explicit Rayon thread settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples" / "research" / "holstein" / "run_tencirpauli.py"


def parse_record(stdout: str) -> dict[str, Any]:
    """Read the flat JSON record emitted by the workload runner."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise RuntimeError("workload runner did not emit JSON")


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", default="1,2,4,physical")
    parser.add_argument(
        "--expected-variant",
        default="fermion",
        help="require this native variant in every fresh child record",
    )
    return parser.parse_args()


def main() -> None:
    args = cli()
    physical = os.cpu_count() or 1
    requested = [item.strip() for item in args.threads.split(",") if item.strip()]
    thread_values = [
        physical if item == "physical" else int(item) for item in requested
    ]
    records: list[dict[str, Any]] = []
    for threads in thread_values:
        environment = {
            **os.environ,
            "RAYON_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
        }
        command = [
            args.python,
            str(RUNNER),
            "--sites",
            str(args.sites),
            "--cutoff",
            str(args.cutoff),
            "--particles",
            str(args.particles),
            "--repeats",
            str(args.repeats),
            "--storage",
            "lazy",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        record = parse_record(completed.stdout)
        actual_variant = str(record.get("native_variant", "unknown"))
        if actual_variant != args.expected_variant:
            raise RuntimeError(
                "unexpected native variant for the measured workload: "
                f"expected {args.expected_variant!r}, got {actual_variant!r}"
            )
        record["requested_threads"] = threads
        records.append(record)
    result = {
        "workload": "one_dimensional_spinful_holstein",
        "sites": args.sites,
        "cutoff": args.cutoff,
        "particles_per_spin": args.particles,
        "records": records,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
