"""Regression tests for the local benchmark comparison policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_runner", ROOT / "benchmarks/run.py"
)
assert SPEC is not None
assert SPEC.loader is not None
benchmark_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_runner)


def _row(median: float) -> dict[str, object]:
    return {"fullname": "stable-case", "stats": {"median": median}}


def test_sub_millisecond_regression_above_relaxed_threshold_blocks(
    tmp_path: Path,
) -> None:
    result = benchmark_runner.comparison_row(
        "stable-case",
        _row(10e-6),
        _row(1.21 * 10e-6),
        tmp_path / "baseline.json",
        tmp_path / "current.json",
        1.10,
        1.20,
        1e-3,
        True,
    )
    assert result["status"] == "BLOCK"


def test_sub_millisecond_small_regression_remains_investigation(tmp_path: Path) -> None:
    result = benchmark_runner.comparison_row(
        "stable-case",
        _row(10e-6),
        _row(11.5e-6),
        tmp_path / "baseline.json",
        tmp_path / "current.json",
        1.10,
        1.20,
        1e-3,
        True,
    )
    assert result["status"] == "INVESTIGATE"


def test_millisecond_regression_keeps_the_stricter_threshold(tmp_path: Path) -> None:
    result = benchmark_runner.comparison_row(
        "stable-case",
        _row(10e-3),
        _row(11.5e-3),
        tmp_path / "baseline.json",
        tmp_path / "current.json",
        1.10,
        1.20,
        1e-3,
        True,
    )
    assert result["status"] == "BLOCK"
