"""Run the TenCirPauli and QuSpin Holstein runners in separate processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "workloads" / "holstein.json"
TENCIRPAULI_RUNNER = ROOT / "examples" / "research" / "holstein" / "run_tencirpauli.py"
QUSPIN_RUNNER = ROOT / "examples" / "research" / "holstein" / "run_quspin.py"


def parse_json(stdout: str) -> dict[str, Any]:
    """Parse the final JSON record emitted by a manual runner."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise RuntimeError("runner did not emit a JSON record")


def run_runner(
    executable: str,
    runner: Path,
    arguments: list[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    """Run one library in its own Python process and return its flat record."""
    command = [executable, str(runner), *arguments]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_json(completed.stdout)


def run_environment(threads: int) -> dict[str, str]:
    """Return matched explicit thread settings for both subprocesses."""
    value = str(threads)
    return {
        **os.environ,
        "RAYON_NUM_THREADS": value,
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
    }


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--tencirpauli-python",
        default=sys.executable,
        help="Python executable for the TenCirPauli subprocess",
    )
    parser.add_argument(
        "--quspin-python",
        default=None,
        help=(
            "Python executable for the isolated QuSpin subprocess; may also be "
            "provided through QUSPIN_PYTHON"
        ),
    )
    parser.add_argument(
        "--mode", choices=("matrix_free", "explicit"), default="matrix_free"
    )
    parser.add_argument("--storage", choices=("lazy", "eager"), default="lazy")
    parser.add_argument("--eigsh", action="store_true")
    parser.add_argument("--explicit-csr", action="store_true")
    parser.add_argument("--sites", type=int, default=None)
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = cli()
    quspin_python = args.quspin_python or os.environ.get("QUSPIN_PYTHON")
    if not quspin_python:
        raise SystemExit(
            "provide the isolated QuSpin interpreter with --quspin-python "
            "or QUSPIN_PYTHON"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            key: value
            for key, value in {
                "sites": args.sites,
                "cutoff": args.cutoff,
                "particles_per_spin": args.particles,
                "repeats": args.repeats,
            }.items()
            if value is not None
        }
    )
    if manifest.get("dtype") != "complex128":
        raise SystemExit("the comparison runner currently requires complex128")
    common = [
        "--sites",
        str(manifest["sites"]),
        "--cutoff",
        str(manifest["cutoff"]),
        "--particles",
        str(manifest["particles_per_spin"]),
        "--t",
        str(manifest["hopping_strength"]),
        "--omega",
        str(manifest["phonon_frequency"]),
        "--g",
        str(manifest["coupling_strength"]),
        "--u",
        str(manifest["interaction_strength"]),
        "--seed",
        str(manifest["seed"]),
        "--repeats",
        str(manifest["repeats"]),
    ]
    quspin_arguments = [*common, "--mode", args.mode]
    tencirpauli_arguments = [
        *common,
        "--storage",
        args.storage,
        "--max-bytes",
        str(args.max_bytes),
    ]
    if args.explicit_csr:
        if args.storage != "eager":
            raise SystemExit("--explicit-csr requires --storage eager")
        tencirpauli_arguments.append(
            "--csr-eigsh" if args.eigsh else "--materialize-csr"
        )
    if args.eigsh:
        quspin_arguments += [
            "--eigsh",
            "--tol",
            str(manifest["tol"]),
            "--maxiter",
            str(manifest["maxiter"]),
            "--ncv",
            str(manifest["ncv"]),
        ]
        tencirpauli_arguments += [
            "--eigsh",
            "--tol",
            str(manifest["tol"]),
            "--maxiter",
            str(manifest["maxiter"]),
            "--ncv",
            str(manifest["ncv"]),
        ]
    environment = run_environment(args.threads)
    tencirpauli = run_runner(
        args.tencirpauli_python, TENCIRPAULI_RUNNER, tencirpauli_arguments, environment
    )
    quspin = run_runner(quspin_python, QUSPIN_RUNNER, quspin_arguments, environment)
    if tencirpauli["dimension"] != quspin["dimension"]:
        raise RuntimeError(
            f"dimension mismatch: TenCirPauli={tencirpauli['dimension']} "
            f"QuSpin={quspin['dimension']}"
        )
    record: dict[str, Any] = {
        "manifest": str(args.manifest.relative_to(ROOT)),
        "mode": args.mode,
        "storage": args.storage,
        "dimension": tencirpauli["dimension"],
        "tencirpauli": tencirpauli,
        "quspin": quspin,
        "dimension_match": True,
        "threads": args.threads,
    }
    if args.eigsh:
        energy_left = float(tencirpauli["native_ground_energy"])
        energy_right = float(quspin["ground_energy"])
        error = abs(energy_left - energy_right)
        if error > 1.0e-8:
            raise RuntimeError(f"ground-energy mismatch: {error}")
        record["ground_energy_error"] = error
    payload = json.dumps(record, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
