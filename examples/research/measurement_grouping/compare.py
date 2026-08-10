"""Run and summarize the measurement-grouping cross-environment comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


_HERE = Path(__file__).resolve().parent


def _run_script(python: str, script: Path, args: Iterable[str]) -> Dict[str, Any]:
    result = subprocess.run(
        [python, str(script), *args],
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


def _args(args: argparse.Namespace) -> list[str]:
    return [
        "--nqubits",
        str(args.nqubits),
        "--term-count",
        str(args.term_count),
        "--support-width",
        str(args.support_width),
        "--shots",
        str(args.shots),
        "--seed",
        str(args.seed),
        "--repetitions",
        str(args.repetitions),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=64)
    parser.add_argument("--term-count", type=int, default=1024)
    parser.add_argument("--support-width", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tencirpauli-python", default=sys.executable)
    parser.add_argument(
        "--qiskit-python",
        default=os.environ.get("QISKIT_PYTHON"),
        help="optional interpreter with Qiskit installed",
    )
    args = parser.parse_args()
    script_args = _args(args)
    native = _run_script(
        args.tencirpauli_python, _HERE / "run_tencirpauli.py", script_args
    )
    result: Dict[str, Any] = {"tencirpauli": native}
    if args.qiskit_python:
        result["qiskit"] = _run_script(
            args.qiskit_python, _HERE / "run_qiskit.py", script_args
        )
    native_grouping = native["timings_seconds"]["native_grouping"]
    python_grouping = native["timings_seconds"]["python_grouping"]
    summary: Dict[str, Any] = {
        "native_over_python_grouping_speedup": python_grouping / native_grouping,
        "native_group_count": native["native_group_count"],
        "python_group_count": native["python_group_count"],
        "correct": bool(
            native["qwc_correct"]
            and native["reconstruction_correct"]
            and native["python_grouping_matches_native"]
        ),
    }
    if "qiskit" in result:
        qiskit = result["qiskit"]
        summary.update(
            {
                "qiskit_group_count": qiskit["group_count"],
                "native_over_qiskit_grouping_speedup": qiskit["timings_seconds"][
                    "grouping"
                ]
                / native_grouping,
                "qiskit_qwc_correct": qiskit["qwc_correct"],
            }
        )
        summary["correct"] = bool(summary["correct"] and qiskit["qwc_correct"])
    result["summary"] = summary
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
