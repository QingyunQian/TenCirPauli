"""Record and compare local Rust and Python benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / ".benchmarks"
RUST_TARGET = RESULTS_ROOT / "rust-target"
PYTHON_STORAGE = RESULTS_ROOT / "python"
RUNS_ROOT = RESULTS_ROOT / "runs"
SUITES = (
    "all",
    "rust",
    "python",
    "structured",
    "majorana_charge",
    "charge_factorization",
)

# Benchmark names are normally immutable. These three entries were renamed
# solely to remove an obsolete executor name after the tested workload kept
# its exact input, timing boundary, and numerical assertion. Keeping this
# migration here preserves the historical measurement instead of silently
# treating it as a new, incomparable benchmark.
BENCHMARK_NAME_MIGRATIONS = {
    "benchmarks/python/test_native_mvp_resources_benchmark.py::"
    "test_product_fermion_boson_apply_into": "benchmarks/python/"
    "test_native_mvp_resources_benchmark.py::test_factorized_fermion_boson_apply_into",
    "benchmarks/python/test_native_mvp_resources_benchmark.py::"
    "test_product_charge_aggregation_steady_apply": "benchmarks/python/"
    "test_native_mvp_resources_benchmark.py::test_factorized_charge_aggregation_steady_apply",
    "benchmarks/python/test_native_mvp_resources_benchmark.py::"
    "test_product_charge_prepared_layout_apply_into": "benchmarks/python/"
    "test_native_mvp_resources_benchmark.py::"
    "test_factorized_charge_prepared_layout_apply_into",
}


def command_environment() -> Dict[str, str]:
    """Bind subprocesses to the environment that owns the active Python."""
    environment = os.environ.copy()
    python_bin = str(Path(sys.executable).parent)
    environment["PATH"] = python_bin + os.pathsep + environment.get("PATH", "")
    environment["CONDA_PREFIX"] = sys.prefix
    environment.pop("VIRTUAL_ENV", None)
    return environment


def run(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run a command from the repository root and show it first."""
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=env or command_environment(),
        check=True,
    )


def capture(command: List[str]) -> str:
    """Capture a best-effort single-line command result."""
    try:
        return subprocess.check_output(
            command, cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_commit() -> str:
    """Return the current commit or a marker for a new repository."""
    value = capture(["git", "rev-parse", "HEAD"])
    return value if value != "unavailable" else "no-commit"


def git_dirty() -> bool:
    """Return whether tracked or untracked project files differ from HEAD."""
    return bool(capture(["git", "status", "--porcelain", "--untracked-files=normal"]))


def default_label() -> str:
    """Build a unique, human-readable run label."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit()
    short_commit = commit[:12] if commit != "no-commit" else commit
    suffix = "-dirty" if git_dirty() else ""
    return f"{timestamp}_{short_commit}{suffix}"


def validate_label(label: str) -> str:
    """Reject labels that are unsafe as file and Criterion baseline names."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("labels may contain only letters, numbers, '.', '_', and '-'")
    return label


def tool_version(command: List[str]) -> str:
    """Return the first line of a tool version."""
    return capture(command).splitlines()[0]


def metadata(label: str, suite: str) -> Dict[str, object]:
    """Collect reproducibility metadata without usernames, paths, or hostnames."""
    return {
        "schema_version": 1,
        "label": label,
        "suite": suite,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "tools": {
            "python": platform.python_version(),
            "rustc": tool_version(["rustc", "--version"]),
            "cargo": tool_version(["cargo", "--version"]),
            "maturin": tool_version(["maturin", "--version"]),
        },
        "thread_environment": {
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "unset"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        },
        "status": "running",
    }


def write_manifest(data: Dict[str, object]) -> None:
    """Persist one local benchmark-run manifest."""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    path = RUNS_ROOT / f"{data['label']}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def rust_environment() -> Dict[str, str]:
    """Keep Criterion build products and baselines outside Cargo's normal target."""
    environment = command_environment()
    environment["CARGO_TARGET_DIR"] = str(RUST_TARGET)
    return environment


def record_rust(label: str) -> None:
    """Record a named Criterion baseline."""
    for bench in ("pauli_word", "symmetry", "propagation"):
        run(
            [
                "cargo",
                "bench",
                "--locked",
                "-p",
                "tencir-pauli-core",
                "--bench",
                bench,
                "--",
                "--save-baseline",
                label,
            ],
            env=rust_environment(),
        )


def python_storage_uri() -> str:
    """Return pytest-benchmark's local file-storage URI."""
    PYTHON_STORAGE.mkdir(parents=True, exist_ok=True)
    return "file://.benchmarks/python"


def build_python_extension() -> None:
    """Install the current native extension in release mode before timing it."""
    run(["maturin", "develop", "--release", "--locked"])


def record_python(label: str, paths: Optional[List[str]] = None) -> None:
    """Record a named pytest-benchmark run for the selected paths."""
    build_python_extension()
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            *(paths or ["benchmarks/python"]),
            "--benchmark-only",
            "--benchmark-quiet",
            f"--benchmark-storage={python_storage_uri()}",
            f"--benchmark-save={label}",
            "--benchmark-sort=name",
        ]
    )


def record(label: str, suite: str) -> None:
    """Record selected suites and their reproducibility metadata."""
    manifest_path = RUNS_ROOT / f"{label}.json"
    if manifest_path.exists():
        raise ValueError(f"benchmark label already exists: {label}")

    data = metadata(label, suite)
    write_manifest(data)
    try:
        if suite in ("all", "rust"):
            record_rust(label)
        if suite in ("all", "python"):
            record_python(label)
        if suite == "structured":
            record_python(
                label, ["benchmarks/python/test_structured_algebra_benchmark.py"]
            )
        if suite == "majorana_charge":
            record_python(
                label, ["benchmarks/python/test_majorana_mapping_charge_benchmark.py"]
            )
        if suite == "charge_factorization":
            record_python(
                label, ["benchmarks/python/test_charge_factorization_benchmark.py"]
            )
    except (OSError, subprocess.CalledProcessError):
        data["status"] = "failed"
        write_manifest(data)
        raise
    data["status"] = "complete"
    write_manifest(data)
    print(f"Recorded benchmark run: {label}")


def find_python_result(label: str) -> str:
    """Resolve a saved label to pytest-benchmark's generated result ID."""
    return find_python_result_path(label).stem


def find_python_result_path(label: str) -> Path:
    """Resolve a saved label to its pytest-benchmark JSON result."""
    matches = sorted(
        PYTHON_STORAGE.glob(f"**/*_{label}.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise ValueError(f"no Python benchmark result found for label: {label}")
    return matches[-1]


def benchmark_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load benchmark rows keyed by their stable pytest node identifier."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid benchmark JSON: {path}") from error
    rows = data.get("benchmarks")
    if not isinstance(rows, list):
        raise ValueError(f"benchmark JSON has no benchmark rows: {path}")

    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid benchmark row in {path}")
        fullname = row.get("fullname")
        name = row.get("name")
        if isinstance(fullname, str) and fullname:
            key = f"fullname:{BENCHMARK_NAME_MIGRATIONS.get(fullname, fullname)}"
        elif isinstance(name, str) and name:
            # pytest-benchmark's parameter dictionary can contain memory-address
            # representations, whereas its generated name is stable for our cases.
            key = f"name:{name}"
        else:
            raise ValueError(f"benchmark row has no stable name in {path}")
        if key in indexed:
            raise ValueError(f"duplicate benchmark identifier {key!r} in {path}")
        indexed[key] = row
    return indexed


def benchmark_median(row: Dict[str, Any], path: Path) -> float:
    """Return one finite, positive median measurement in seconds."""
    stats = row.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"benchmark row has no statistics in {path}")
    median = stats.get("median")
    if not isinstance(median, (int, float)) or median <= 0:
        raise ValueError(f"benchmark row has invalid median in {path}")
    return float(median)


def comparison_row(
    key: str,
    baseline: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
    baseline_path: Path,
    current_path: Path,
    threshold: float,
    small_regression_threshold: float,
    minimum_delta_seconds: float,
    enforce_threshold: bool,
) -> Dict[str, object]:
    """Create one deterministic comparison record for a benchmark identifier."""
    display_name = (current or baseline or {}).get("fullname") or (
        current or baseline or {}
    ).get("name", key)
    result: Dict[str, object] = {"benchmark": display_name, "key": key}
    if baseline is None:
        result.update(
            status="NEW",
            baseline_median_seconds=None,
            current_median_seconds=benchmark_median(current or {}, current_path),
            ratio=None,
            delta_seconds=None,
        )
        return result
    if current is None:
        result.update(
            status="MISSING_CURRENT",
            baseline_median_seconds=benchmark_median(baseline, baseline_path),
            current_median_seconds=None,
            ratio=None,
            delta_seconds=None,
        )
        return result

    baseline_median = benchmark_median(baseline, baseline_path)
    current_median = benchmark_median(current, current_path)
    ratio = current_median / baseline_median
    material_regression = current_median - baseline_median >= minimum_delta_seconds
    applicable_threshold = (
        threshold if material_regression else small_regression_threshold
    )
    status = (
        "PASS"
        if ratio <= threshold
        else (
            "BLOCK"
            if enforce_threshold and ratio > applicable_threshold
            else "INVESTIGATE"
        )
    )
    result.update(
        status=status,
        baseline_median_seconds=baseline_median,
        current_median_seconds=current_median,
        ratio=ratio,
        delta_seconds=current_median - baseline_median,
    )
    return result


def formatted_measurement(value: object, scale: float, suffix: str) -> str:
    """Format a deliberately coarse human summary of a numeric measurement."""
    if value is None:
        return "-"
    if not isinstance(value, (int, float)):
        raise ValueError("comparison report contains a non-numeric measurement")
    measurement = float(value) * scale
    if suffix == "x":
        return f"{measurement:.1f}{suffix}"
    if abs(measurement) < 1.0:
        return f"~0{suffix}"
    return f"{measurement:.0f}{suffix}"


def write_python_comparison(
    baseline_label: str,
    suite: str,
    threshold: float,
    small_regression_threshold: float,
    minimum_delta_seconds: float,
    baseline_path: Path,
    current_path: Path,
    enforce_threshold: bool,
) -> Tuple[Path, bool]:
    """Persist and print a median-based benchmark comparison report."""
    baseline_rows = benchmark_rows(baseline_path)
    current_rows = benchmark_rows(current_path)
    rows = [
        comparison_row(
            key,
            baseline_rows.get(key),
            current_rows.get(key),
            baseline_path,
            current_path,
            threshold,
            small_regression_threshold,
            minimum_delta_seconds,
            enforce_threshold,
        )
        for key in sorted(set(baseline_rows) | set(current_rows))
    ]
    blocked = any(row["status"] in ("BLOCK", "MISSING_CURRENT") for row in rows)
    report = {
        "schema_version": 1,
        "baseline_label": baseline_label,
        "suite": suite,
        "threshold_ratio": threshold,
        "small_regression_threshold_ratio": small_regression_threshold,
        "minimum_delta_seconds": minimum_delta_seconds,
        "enforces_threshold": enforce_threshold,
        "baseline_result": str(baseline_path.relative_to(ROOT)),
        "current_result": str(current_path.relative_to(ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blocked else "complete",
        "benchmarks": rows,
    }
    report_label = f"comparison-{baseline_label}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report_path = RUNS_ROOT / f"{report_label}.json"
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in sorted({str(row["status"]) for row in rows})
    }
    print(f"\nPython benchmark comparison: {status_counts}")
    attention_rows = [
        row
        for row in rows
        if row["status"] in ("BLOCK", "MISSING_CURRENT", "INVESTIGATE")
    ]
    for row in attention_rows:
        baseline_median = row["baseline_median_seconds"]
        current_median = row["current_median_seconds"]
        ratio = row["ratio"]
        delta = row["delta_seconds"]
        baseline_text = formatted_measurement(baseline_median, 1e3, " ms")
        current_text = formatted_measurement(current_median, 1e3, " ms")
        ratio_text = formatted_measurement(ratio, 1.0, "x")
        delta_text = formatted_measurement(delta, 1e3, " ms")
        print(
            f"{row['status']:>15}  {ratio_text:>8}  {delta_text:>12}  "
            f"baseline={baseline_text:>12}  current={current_text:>12}  {row['benchmark']}"
        )
    new_count = status_counts.get("NEW", 0)
    if new_count:
        print(f"{new_count} new benchmark(s) have no matching saved baseline.")
    print(f"Comparison report: {report_path.relative_to(ROOT)}")
    return report_path, blocked


def compare_rust(label: str) -> None:
    """Run Criterion and compare current measurements with a named baseline."""
    for bench in ("pauli_word", "symmetry", "propagation"):
        run(
            [
                "cargo",
                "bench",
                "--locked",
                "-p",
                "tencir-pauli-core",
                "--bench",
                bench,
                "--",
                "--baseline",
                label,
            ],
            env=rust_environment(),
        )


def rust_baseline_exists(label: str) -> bool:
    """Return whether the local Criterion storage contains a named baseline."""
    criterion_root = RUST_TARGET / "criterion"
    return criterion_root.exists() and any(criterion_root.glob(f"**/{label}"))


def compare_python(
    label: str,
    suite: str,
    threshold: float,
    small_regression_threshold: float,
    minimum_delta_seconds: float,
    paths: Optional[List[str]] = None,
    current_label: Optional[str] = None,
) -> bool:
    """Compare a saved baseline with either fresh or saved current medians."""
    baseline_path = find_python_result_path(label)
    if current_label is None:
        build_python_extension()
        current_label = validate_label(
            f"compare-{label}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                *(paths or ["benchmarks/python"]),
                "--benchmark-only",
                "--benchmark-quiet",
                f"--benchmark-storage={python_storage_uri()}",
                f"--benchmark-save={current_label}",
                "--benchmark-sort=name",
            ]
        )
    current_path = find_python_result_path(current_label)
    _, blocked = write_python_comparison(
        label,
        suite,
        threshold,
        small_regression_threshold,
        minimum_delta_seconds,
        baseline_path,
        current_path,
        enforce_threshold=suite in ("charge_factorization", "python", "all"),
    )
    return blocked


def compare(
    label: str,
    suite: Optional[str],
    threshold: float,
    small_regression_threshold: float,
    minimum_delta_seconds: float,
    current_label: Optional[str],
) -> None:
    """Compare current code with a previously recorded local run."""
    manifest_path = RUNS_ROOT / f"{label}.json"
    if manifest_path.exists():
        saved_data = json.loads(manifest_path.read_text())
        selected_suite = suite or str(saved_data["suite"])
    elif suite is not None:
        # pytest-benchmark files are themselves persisted baselines.  A manifest is
        # useful metadata, but old records must remain comparable after this runner
        # gained manifests.
        selected_suite = suite
    else:
        raise ValueError(
            f"unknown benchmark label: {label}; pass --suite to compare a stored "
            "pytest-benchmark result without a manifest"
        )
    blocked = False
    if selected_suite in ("all", "rust"):
        if current_label is not None:
            raise ValueError(
                "--current may only be used with a Python-only benchmark suite"
            )
        if not rust_baseline_exists(label):
            raise ValueError(
                f"Rust Criterion baseline {label!r} is not present locally; "
                "record a matching Rust baseline or compare the persisted "
                "Python/FFI result with --suite python"
            )
        compare_rust(label)
    if selected_suite in ("all", "python"):
        blocked = (
            compare_python(
                label,
                selected_suite,
                threshold,
                small_regression_threshold,
                minimum_delta_seconds,
                current_label=current_label,
            )
            or blocked
        )
    if selected_suite == "structured":
        blocked = compare_python(
            label,
            selected_suite,
            threshold,
            small_regression_threshold,
            minimum_delta_seconds,
            ["benchmarks/python/test_structured_algebra_benchmark.py"],
            current_label,
        )
    if selected_suite == "majorana_charge":
        blocked = compare_python(
            label,
            selected_suite,
            threshold,
            small_regression_threshold,
            minimum_delta_seconds,
            ["benchmarks/python/test_majorana_mapping_charge_benchmark.py"],
            current_label,
        )
    if selected_suite == "charge_factorization":
        blocked = compare_python(
            label,
            selected_suite,
            threshold,
            small_regression_threshold,
            minimum_delta_seconds,
            ["benchmarks/python/test_charge_factorization_benchmark.py"],
            current_label,
        )
    if blocked:
        raise ValueError(
            f"benchmark regression exceeds the {threshold:.3f}x threshold; "
            "see the persisted comparison report"
        )


def import_baseline(label: str, suite: str) -> None:
    """Register an existing pytest-benchmark JSON as a named local baseline."""
    manifest_path = RUNS_ROOT / f"{label}.json"
    if manifest_path.exists():
        raise ValueError(f"benchmark label already exists: {label}")
    result_path = find_python_result_path(label)
    data = metadata(label, suite)
    data.update(
        status="complete",
        source="imported_pytest_benchmark_result",
        python_result=str(result_path.relative_to(ROOT)),
    )
    write_manifest(data)
    print(f"Imported benchmark baseline: {label}")


def list_runs() -> None:
    """Print locally recorded benchmark manifests."""
    if not RUNS_ROOT.exists():
        print("No local benchmark runs recorded.")
        return
    for path in sorted(RUNS_ROOT.glob("*.json")):
        data = json.loads(path.read_text())
        if "label" not in data:
            if "baseline_label" in data:
                print(
                    f"{path.stem}  comparison={data.get('status', 'unknown')}  "
                    f"baseline={data['baseline_label']}"
                )
            continue
        print(
            f"{data['label']}  {data['status']}  "
            f"commit={str(data['git_commit'])[:12]}  dirty={data['git_dirty']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse the local benchmark command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="record a new local run")
    record_parser.add_argument("--label", help="optional unique result label")
    record_parser.add_argument("--suite", choices=SUITES, default="all")

    compare_parser = subparsers.add_parser(
        "compare", help="compare current code with a saved run"
    )
    compare_parser.add_argument("baseline", help="saved run label")
    compare_parser.add_argument(
        "--suite", choices=SUITES, help="defaults to the suite saved in the baseline"
    )
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=1.15,
        help="maximum current/baseline median ratio for >=1 ms regressions (default: 1.15)",
    )
    compare_parser.add_argument(
        "--minimum-delta-ms",
        type=float,
        default=1.0,
        help="absolute regression boundary for the standard threshold (default: 1.0 ms)",
    )
    compare_parser.add_argument(
        "--small-regression-threshold",
        type=float,
        default=1.30,
        help="maximum ratio below --minimum-delta-ms before a comparison fails (default: 1.30)",
    )
    compare_parser.add_argument(
        "--current",
        help="saved current pytest-benchmark label; skips fresh measurement for Python-only comparison",
    )

    import_parser = subparsers.add_parser(
        "import-baseline", help="register an existing pytest-benchmark result"
    )
    import_parser.add_argument("label", help="existing pytest-benchmark result label")
    import_parser.add_argument("--suite", choices=SUITES, required=True)

    subparsers.add_parser("list", help="list saved local runs")
    return parser.parse_args()


def main() -> int:
    """Run the selected benchmark operation."""
    arguments = parse_args()
    try:
        if arguments.command == "record":
            label = validate_label(arguments.label or default_label())
            record(label, arguments.suite)
        elif arguments.command == "compare":
            if (
                arguments.threshold <= 0
                or arguments.small_regression_threshold < arguments.threshold
                or arguments.minimum_delta_ms < 0
            ):
                raise ValueError(
                    "--threshold must be positive, --small-regression-threshold must be at least --threshold, and --minimum-delta-ms must be non-negative"
                )
            compare(
                validate_label(arguments.baseline),
                arguments.suite,
                arguments.threshold,
                arguments.small_regression_threshold,
                arguments.minimum_delta_ms / 1e3,
                (
                    None
                    if arguments.current is None
                    else validate_label(arguments.current)
                ),
            )
        elif arguments.command == "import-baseline":
            import_baseline(validate_label(arguments.label), arguments.suite)
        else:
            list_runs()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"benchmark command failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
