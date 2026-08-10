"""Run and numerically compare the two 60-qubit U(1) VQE runners."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _record(stdout: str) -> dict[str, Any]:
    """Parse the final JSON object emitted by one child runner."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("{") and line.endswith("}"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise RuntimeError("runner did not emit a JSON object")


def _run(
    executable: str,
    script: Path,
    seed: int,
    repeats: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Run one implementation in a fresh process."""
    completed = subprocess.run(
        [executable, str(script), "--seed", str(seed), "--repeats", str(repeats)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{script.name} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return _record(completed.stdout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--tencirpauli-python", default=sys.executable)
    parser.add_argument("--tensorcircuit-python", default=sys.executable)
    parser.add_argument(
        "--tensorcircuit-source",
        type=Path,
        default=Path(
            os.environ.get("TENSORCIRCUIT_SOURCE", str(ROOT.parent / "tensorcircuit"))
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run both children, verify energy/gradient equality, and print a summary."""
    args = _parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    base_environment = dict(os.environ)
    tencirpauli = _run(
        args.tencirpauli_python,
        HERE / "run_tencirpauli.py",
        args.seed,
        args.repeats,
        base_environment,
    )
    tensorcircuit_environment = {
        **base_environment,
        "PYTHONPATH": os.pathsep.join(
            [
                str(args.tensorcircuit_source),
                str(ROOT / "python"),
                base_environment.get("PYTHONPATH", ""),
            ]
        ),
    }
    tensorcircuit = _run(
        args.tensorcircuit_python,
        HERE / "run_tensorcircuit.py",
        args.seed,
        args.repeats,
        tensorcircuit_environment,
    )

    native_gradient = np.asarray(tencirpauli["gradient_values"], dtype=np.float64)
    tensorcircuit_gradient = np.asarray(
        tensorcircuit["gradient_values"], dtype=np.float64
    )
    if native_gradient.shape != tensorcircuit_gradient.shape:
        raise RuntimeError(
            f"gradient shape mismatch: {native_gradient.shape} vs "
            f"{tensorcircuit_gradient.shape}"
        )
    energy_abs_error = abs(
        float(tencirpauli["energy"]) - float(tensorcircuit["energy"])
    )
    gradient_delta = native_gradient - tensorcircuit_gradient
    gradient_max_abs_error = float(np.max(np.abs(gradient_delta)))
    gradient_l2_error = float(np.linalg.norm(gradient_delta))
    if energy_abs_error > 1e-10 or gradient_max_abs_error > 1e-10:
        raise RuntimeError(
            f"numerical mismatch: energy error={energy_abs_error}, "
            f"gradient max error={gradient_max_abs_error}"
        )

    output = {
        "comparison": "tencirpauli_native_vs_tensorcircuit_jax_scan",
        "seed": args.seed,
        "repeats": args.repeats,
        "dimension": tencirpauli["dimension"],
        "parameter_count": tencirpauli["parameter_count"],
        "energy_abs_error": energy_abs_error,
        "gradient_max_abs_error": gradient_max_abs_error,
        "gradient_l2_error": gradient_l2_error,
        "first_seconds": {
            "tencirpauli": tencirpauli["first_value_and_grad_seconds"],
            "tensorcircuit": tensorcircuit["first_value_and_grad_seconds"],
        },
        "steady_seconds_median": {
            "tencirpauli": tencirpauli["steady_value_and_grad_seconds_median"],
            "tensorcircuit": tensorcircuit["steady_value_and_grad_seconds_median"],
        },
        "first_end_to_end_seconds": {
            "tencirpauli": tencirpauli["end_to_end_first_seconds"],
            "tensorcircuit": tensorcircuit["end_to_end_first_seconds"],
        },
        "tencirpauli_speedup_over_tensorcircuit_first": float(
            tensorcircuit["first_value_and_grad_seconds"]
            / tencirpauli["first_value_and_grad_seconds"]
        ),
        "tencirpauli_speedup_over_tensorcircuit_first_end_to_end": float(
            tensorcircuit["end_to_end_first_seconds"]
            / tencirpauli["end_to_end_first_seconds"]
        ),
        "tencirpauli_speedup_over_tensorcircuit_steady": float(
            tensorcircuit["steady_value_and_grad_seconds_median"]
            / tencirpauli["steady_value_and_grad_seconds_median"]
        ),
        "tencirpauli": {
            key: value for key, value in tencirpauli.items() if key != "gradient_values"
        },
        "tensorcircuit": {
            key: value
            for key, value in tensorcircuit.items()
            if key != "gradient_values"
        },
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
