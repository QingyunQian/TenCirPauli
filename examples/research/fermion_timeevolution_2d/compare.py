"""Compare raw Fermion native, JAX callback, and mapped JAX paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


_HERE = Path(__file__).resolve().parent


def _run(script: Path, args: Iterable[str]) -> Dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script} failed with exit code {result.returncode}:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{script} produced no JSON output")
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--interaction", type=float, default=4.0)
    parser.add_argument("--mapping", default="jordan_wigner")
    parser.add_argument("--times", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--subspace-dimension", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solvers", nargs="+", default=["krylov", "chebyshev"])
    args = parser.parse_args()
    common_args = [
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--hopping",
        str(args.hopping),
        "--interaction",
        str(args.interaction),
        "--times",
        *[str(value) for value in args.times],
        "--subspace-dimension",
        str(args.subspace_dimension),
        "--repeats",
        str(args.repeats),
        "--solvers",
        *args.solvers,
    ]
    native = _run(_HERE / "run_native.py", common_args)
    native_callback = _run(_HERE / "run_native_callback.py", common_args)
    backend = _run(_HERE / "run_backend.py", [*common_args, "--mapping", args.mapping])
    summary: Dict[str, Any] = {
        "rows": args.rows,
        "cols": args.cols,
        "dimension": native["dimension"],
        "native_plan_seconds": native["native_plan_seconds"],
        "native_callback_plan_seconds": native_callback["native_plan_seconds"],
        "mapping_plan_seconds": backend["mapping_plan_seconds"],
        "native_peak_rss_bytes": native["peak_rss_bytes"],
        "native_callback_peak_rss_bytes": native_callback["peak_rss_bytes"],
        "backend_peak_rss_bytes": backend["peak_rss_bytes"],
        "solvers": {},
    }
    for solver in args.solvers:
        native_result = native["results"][solver]
        callback_result = native_callback["results"][solver]
        backend_result = backend["results"][solver]
        summary["solvers"][solver] = {
            "native_first_seconds": native_result["first_seconds"],
            "native_callback_first_jit_seconds": callback_result["first_jit_seconds"],
            "backend_first_jit_seconds": backend_result["first_jit_seconds"],
            "native_warm_median_seconds": native_result["warm_median_seconds"],
            "native_callback_warm_median_seconds": callback_result[
                "warm_median_seconds"
            ],
            "backend_warm_median_seconds": backend_result["warm_median_seconds"],
            "native_callback_over_native_first_ratio": callback_result[
                "first_jit_seconds"
            ]
            / native_result["first_seconds"],
            "native_callback_over_native_warm_ratio": callback_result[
                "warm_median_seconds"
            ]
            / native_result["warm_median_seconds"],
            "backend_over_native_first_ratio": backend_result["first_jit_seconds"]
            / native_result["first_seconds"],
            "backend_over_native_warm_ratio": backend_result["warm_median_seconds"]
            / native_result["warm_median_seconds"],
            "native_callback_over_backend_first_ratio": callback_result[
                "first_jit_seconds"
            ]
            / backend_result["first_jit_seconds"],
            "native_callback_over_backend_warm_ratio": callback_result[
                "warm_median_seconds"
            ]
            / backend_result["warm_median_seconds"],
        }
    print(
        json.dumps(
            {
                "native": native,
                "native_callback": native_callback,
                "backend": backend,
                "summary": summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
