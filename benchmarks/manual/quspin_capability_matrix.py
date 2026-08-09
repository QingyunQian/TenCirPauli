"""Run the frozen QuSpin capability matrix in isolated child processes.

The parent never imports TenCirPauli or QuSpin.  Direct-comparison rows use an
ABBA subprocess order, verify the bounded action matrices after conversion to
the same occupation ordering, and preserve both caller-owned and
allocation-inclusive measurement boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from quspin_capabilities import (
    DIRECT_COMPARISON,
    NO_FAIR_BASELINE,
    ROOT,
    capability_manifest_paths,
    load_manifest,
)


TENCIRPAULI_RUNNER = ROOT / "benchmarks" / "manual" / "run_quspin_tencirpauli.py"
QUSPIN_RUNNER = ROOT / "benchmarks" / "manual" / "run_quspin_quspin.py"


def _parse_record(stdout: str) -> dict[str, Any]:
    """Parse the final flat JSON record emitted by a child runner."""
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
    raise RuntimeError("comparison child did not emit a JSON object")


def _environment(threads: int) -> dict[str, str]:
    """Set the same explicit thread policy for each isolated process."""
    if threads < 1:
        raise ValueError("threads must be positive")
    value = str(threads)
    return {
        **os.environ,
        "RAYON_NUM_THREADS": value,
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
    }


def _run_child(
    executable: str,
    runner: Path,
    manifest: Path,
    artifact: Path,
    environment: dict[str, str],
    *,
    preflight: bool = False,
) -> dict[str, Any]:
    """Run one library with no in-process dependency overlap."""
    command = [
        executable,
        str(runner),
        "--manifest",
        str(manifest),
        "--artifact",
        str(artifact),
    ]
    if preflight:
        command.append("--preflight")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{manifest.name} via {runner.name} failed with exit code "
            f"{completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return _parse_record(completed.stdout)


def _load_artifact(
    path: Path,
) -> tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load one canonical action artifact without trusting object payloads."""
    with np.load(path, allow_pickle=False) as artifact:
        kind = str(artifact["kind"].item()) if "kind" in artifact else "matrix"
        values = np.asarray(
            artifact["matrix"] if kind == "matrix" else artifact["values"],
            dtype=np.complex128,
        )
        states = np.asarray(artifact["states"], dtype=np.uint64)
    if kind not in {"matrix", "vector"}:
        raise RuntimeError(f"unknown action artifact kind {kind!r}")
    return kind, values, states


def _verify_actions(paths: list[Path], tolerance: float) -> dict[str, float]:
    """Verify all child artifacts equal after their documented basis permutation."""
    reference_kind, reference_values, reference_states = _load_artifact(paths[0])
    max_error = 0.0
    for path in paths[1:]:
        kind, values, states = _load_artifact(path)
        if kind != reference_kind:
            raise RuntimeError(f"action kinds differ in artifact {path.name}")
        if not np.array_equal(states, reference_states):
            raise RuntimeError(
                f"basis occupations differ in action artifact {path.name}"
            )
        error = float(np.max(np.abs(values - reference_values)))
        max_error = max(max_error, error)
    if max_error > tolerance:
        raise RuntimeError(
            f"mapped action mismatch {max_error} exceeds manifest tolerance {tolerance}"
        )
    return {
        "action_kind": reference_kind,
        "action_max_error": max_error,
        "action_tolerance": tolerance,
    }


def _aggregate(records: list[dict[str, Any]], key: str) -> float:
    """Take the median across independent fresh-process child medians."""
    return float(statistics.median(float(record[key]) for record in records))


def _direct_row(
    manifest_path: Path,
    manifest: dict[str, Any],
    threads: int,
    tencirpauli_python: str,
    quspin_python: str,
) -> dict[str, Any]:
    """Preflight correctness, then time a direct comparison in ABBA order."""
    order = ("tencirpauli", "quspin", "quspin", "tencirpauli")
    environment = _environment(threads)
    records: dict[str, list[dict[str, Any]]] = {"tencirpauli": [], "quspin": []}
    with tempfile.TemporaryDirectory(prefix="tencirpauli-quspin-") as directory:
        temporary = Path(directory)
        preflight_records: dict[str, dict[str, Any]] = {}
        preflight_artifacts: list[Path] = []
        for library in ("tencirpauli", "quspin"):
            artifact = temporary / f"preflight-{library}.npz"
            record = _run_child(
                tencirpauli_python if library == "tencirpauli" else quspin_python,
                TENCIRPAULI_RUNNER if library == "tencirpauli" else QUSPIN_RUNNER,
                manifest_path,
                artifact,
                environment,
                preflight=True,
            )
            if record.get("stage") != "preflight":
                raise RuntimeError(
                    "comparison child did not report numerical preflight"
                )
            if int(record["dimension"]) < 1:
                raise RuntimeError("preflight child reported a non-positive dimension")
            if record["capability"] != manifest["capability"]:
                raise RuntimeError("preflight child reported a different capability")
            preflight_records[library] = record
            preflight_artifacts.append(artifact)
        action = _verify_actions(
            preflight_artifacts, float(manifest["action_tolerance"])
        )
        if int(preflight_records["tencirpauli"]["dimension"]) != int(
            preflight_records["quspin"]["dimension"]
        ):
            raise RuntimeError("preflight comparison dimensions do not agree")
        for index, library in enumerate(order):
            artifact = temporary / f"timing-{index:02d}-{library}.npz"
            record = _run_child(
                tencirpauli_python if library == "tencirpauli" else quspin_python,
                TENCIRPAULI_RUNNER if library == "tencirpauli" else QUSPIN_RUNNER,
                manifest_path,
                artifact,
                environment,
            )
            if int(record["dimension"]) < 1:
                raise RuntimeError("comparison child reported a non-positive dimension")
            if record["capability"] != manifest["capability"]:
                raise RuntimeError("comparison child reported a different capability")
            records[library].append(record)

    dimension = int(preflight_records["tencirpauli"]["dimension"])
    if any(
        int(record["dimension"]) != dimension
        for rows in records.values()
        for record in rows
    ):
        raise RuntimeError("comparison dimensions do not agree")
    tencirpauli_public = _aggregate(
        records["tencirpauli"], "public_apply_seconds_median"
    )
    quspin_public = _aggregate(records["quspin"], "public_apply_seconds_median")
    ratio = tencirpauli_public / quspin_public
    if ratio < 1.0:
        verdict = "win"
    elif ratio <= 1.10:
        verdict = "parity"
    else:
        verdict = "above_parity_ceiling"
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "capability": manifest["capability"],
        "comparison": DIRECT_COMPARISON,
        "operator_family": manifest["operator_family"],
        "threads": threads,
        "numerical_preflight_before_timing": True,
        "subprocess_order": list(order),
        "dimension": dimension,
        "action_validation": action,
        "measurement_boundary": {
            "tencirpauli_apply_into": "caller-owned output buffer",
            "tencirpauli_public_apply": "allocating public plan.apply",
            "quspin_public_apply": "allocating operator.dot",
            "quspin_caller_owned_apply": "unavailable",
        },
        "tencirpauli": {
            "records": records["tencirpauli"],
            "apply_into_seconds_median": _aggregate(
                records["tencirpauli"], "apply_into_seconds_median"
            ),
            "public_apply_seconds_median": tencirpauli_public,
        },
        "quspin": {
            "records": records["quspin"],
            "public_apply_seconds_median": quspin_public,
        },
        "public_apply_ratio_tencirpauli_over_quspin": ratio,
        "performance_verdict": verdict,
    }


def _no_baseline_row(
    manifest_path: Path,
    manifest: dict[str, Any],
    threads: int,
    tencirpauli_python: str,
) -> dict[str, Any]:
    """Record internal evidence for a capability with no fair optimized baseline."""
    environment = _environment(threads)
    with tempfile.TemporaryDirectory(prefix="tencirpauli-internal-") as directory:
        artifact = Path(directory) / "tencirpauli.npz"
        record = _run_child(
            tencirpauli_python,
            TENCIRPAULI_RUNNER,
            manifest_path,
            artifact,
            environment,
        )
        kind, values, states = _load_artifact(artifact)
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "capability": manifest["capability"],
        "comparison": NO_FAIR_BASELINE,
        "no_fair_baseline_reason": manifest["no_fair_baseline_reason"],
        "operator_family": manifest["operator_family"],
        "threads": threads,
        "dimension": int(record["dimension"]),
        "internal_action_kind": kind,
        "internal_action_shape": list(values.shape),
        "internal_basis_rows": int(states.shape[0]),
        "tencirpauli": record,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--manifest", type=Path, action="append")
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--tencirpauli-python", default=sys.executable)
    parser.add_argument("--quspin-python", default=os.environ.get("QUSPIN_PYTHON"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-parity",
        action="store_true",
        help="exit nonzero when a direct public-call ratio exceeds 1.10",
    )
    return parser.parse_args()


def main() -> None:
    """Execute selected manifest rows and emit a flat reproducible JSON record."""
    args = _parse_args()
    paths = (
        capability_manifest_paths()
        if args.all
        else tuple(path.resolve() for path in args.manifest)
    )
    manifests = [(path, load_manifest(path)) for path in paths]
    if any(manifest["comparison"] == DIRECT_COMPARISON for _, manifest in manifests):
        if not args.quspin_python:
            raise SystemExit(
                "provide --quspin-python or QUSPIN_PYTHON for direct QuSpin rows"
            )
    rows: list[dict[str, Any]] = []
    for path, manifest in manifests:
        for threads in args.threads:
            if manifest["comparison"] == DIRECT_COMPARISON:
                rows.append(
                    _direct_row(
                        path,
                        manifest,
                        threads,
                        args.tencirpauli_python,
                        str(args.quspin_python),
                    )
                )
            else:
                rows.append(
                    _no_baseline_row(path, manifest, threads, args.tencirpauli_python)
                )
    output = {
        "matrix_manifest_version": "quspin-capability-matrix-v1",
        "rows": rows,
        "all_direct_rows_within_parity_ceiling": all(
            row.get("performance_verdict") != "above_parity_ceiling" for row in rows
        ),
        "exact_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    }
    payload = json.dumps(output, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.enforce_parity and not output["all_direct_rows_within_parity_ceiling"]:
        raise SystemExit("one or more direct QuSpin rows exceed the 10% parity ceiling")


if __name__ == "__main__":
    main()
