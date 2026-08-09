"""Shared manifest and measurement helpers for isolated QuSpin comparisons.

This module deliberately imports neither TenCirPauli nor QuSpin.  The two
environment-specific child runners import it only for frozen-workload parsing,
deterministic state construction, and flat artifact handling.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


try:
    _resource = importlib.import_module("resource")
except ImportError:  # pragma: no cover - Windows does not provide resource
    _resource = None


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "quspin-capability-v1"
DIRECT_COMPARISON = "directly_comparable"
NO_FAIR_BASELINE = "no_fair_optimized_baseline"

DIRECT_CAPABILITIES = frozenset(
    {
        "spin_full",
        "spin_full_throughput",
        "spin_fixed_hamming",
        "fermion_full",
        "boson_full",
        "fermion_boson_full",
        "fermion_boson_full_throughput",
        "fermion_fixed_boson_spectators",
        "boson_fixed_fermion_spectators",
        "independent_spin_factors",
        "independent_spin_factors_throughput",
        "reversed_boson_spin_factors",
    }
)
NO_BASELINE_CAPABILITIES = frozenset({"generic_coupled_charge", "qudit_weyl"})
ALL_CAPABILITIES = DIRECT_CAPABILITIES | NO_BASELINE_CAPABILITIES


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate one committed, capability-named workload manifest."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    if value.get("workload_manifest") != MANIFEST_VERSION:
        raise ValueError(f"{path}: unsupported workload manifest version")
    capability = value.get("capability")
    if capability not in ALL_CAPABILITIES:
        raise ValueError(f"{path}: unknown capability {capability!r}")
    comparison = value.get("comparison")
    expected = (
        DIRECT_COMPARISON if capability in DIRECT_CAPABILITIES else NO_FAIR_BASELINE
    )
    if comparison != expected:
        raise ValueError(
            f"{path}: capability {capability!r} requires comparison={expected!r}"
        )
    if value.get("dtype") != "complex128":
        raise ValueError(f"{path}: only complex128 comparisons are supported")
    repeats = value.get("repeats")
    if not isinstance(repeats, int) or repeats < 7:
        raise ValueError(f"{path}: repeats must be an integer of at least seven")
    warmups = value.get("warmups")
    if warmups != 2:
        raise ValueError(f"{path}: exactly two untimed warmups are required")
    seed = value.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"{path}: seed must be an integer")
    tolerance = value.get("action_tolerance")
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError(f"{path}: action_tolerance must be positive")
    validation = value.get("action_validation", "dense")
    if validation not in {"dense", "vector"}:
        raise ValueError(f"{path}: action_validation must be 'dense' or 'vector'")
    return value


def capability_manifest_paths() -> tuple[Path, ...]:
    """Return all committed capability manifests in deterministic order."""
    directory = ROOT / "benchmarks" / "workloads" / "quspin"
    return tuple(sorted(directory.glob("*.json")))


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes when the platform exposes it."""
    if _resource is None:
        return 0
    value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def thread_environment() -> dict[str, str]:
    """Return the complete thread configuration carried into a child process."""
    return {
        name: os.environ.get(name, "unset")
        for name in (
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }


def canonical_state(states: Sequence[Sequence[int]], seed: int) -> np.ndarray[Any, Any]:
    """Build the same normalized vector for any permutation of occupation rows."""
    keys = [tuple(int(value) for value in row) for row in states]
    if len(set(keys)) != len(keys):
        raise ValueError("canonical-state occupations must be unique")
    generator = np.random.default_rng(seed)
    ordered = sorted(keys)
    values = generator.normal(size=len(keys)) + 1j * generator.normal(size=len(keys))
    values = np.asarray(values / np.linalg.norm(values), dtype=np.complex128)
    by_key = {key: values[index] for index, key in enumerate(ordered)}
    return np.asarray([by_key[key] for key in keys], dtype=np.complex128)


def canonicalize_action(
    matrix: np.ndarray[Any, Any], states: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return an action and state rows sorted by physical occupations."""
    values = np.asarray(matrix, dtype=np.complex128)
    occupations = np.asarray(states, dtype=np.uint64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("action artifact must be a square matrix")
    if occupations.ndim != 2 or occupations.shape[0] != values.shape[0]:
        raise ValueError("action artifact occupations do not match matrix dimension")
    order = np.asarray(
        sorted(
            range(occupations.shape[0]), key=lambda index: tuple(occupations[index])
        ),
        dtype=np.intp,
    )
    return values[np.ix_(order, order)], occupations[order]


def save_action_artifact(
    path: Path, matrix: np.ndarray[Any, Any], states: np.ndarray[Any, Any]
) -> None:
    """Save a bounded canonical action artifact for cross-process validation."""
    canonical_matrix, canonical_states = canonicalize_action(matrix, states)
    np.savez_compressed(
        path,
        kind=np.asarray("matrix"),
        matrix=np.asarray(canonical_matrix, dtype=np.complex128),
        states=np.asarray(canonical_states, dtype=np.uint64),
    )


def save_action_vector_artifact(
    path: Path, values: np.ndarray[Any, Any], states: np.ndarray[Any, Any]
) -> None:
    """Save one deterministic action vector in canonical occupation order."""
    action = np.asarray(values, dtype=np.complex128)
    occupations = np.asarray(states, dtype=np.uint64)
    if (
        action.ndim != 1
        or occupations.ndim != 2
        or action.shape[0] != occupations.shape[0]
    ):
        raise ValueError("action vector and occupation rows must have equal length")
    order = np.asarray(
        sorted(
            range(occupations.shape[0]), key=lambda index: tuple(occupations[index])
        ),
        dtype=np.intp,
    )
    np.savez_compressed(
        path,
        kind=np.asarray("vector"),
        values=action[order],
        states=occupations[order],
    )


def write_preflight_artifact(
    validation_kind: str,
    states: np.ndarray[Any, Any],
    action: np.ndarray[Any, Any],
    path: Path,
) -> None:
    """Write the shared flat preflight artifact for a matrix or action vector."""
    if validation_kind == "dense":
        save_action_artifact(path, action, states)
    elif validation_kind == "vector":
        save_action_vector_artifact(path, action, states)
    else:
        raise ValueError("validation_kind must be 'dense' or 'vector'")


def measure_apply(
    apply: Any, warmups: int, repeats: int
) -> dict[str, float | list[float]]:
    """Measure one unchanged public apply expression after fixed warmups."""
    for _ in range(warmups):
        apply()
    start = time.perf_counter()
    apply()
    first = time.perf_counter() - start
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        apply()
        samples.append(time.perf_counter() - start)
    return {"first": first, "samples": samples, "median": float(np.median(samples))}


_TIMING_REQUIRED_KEYS = frozenset(
    {
        "library",
        "capability",
        "comparison",
        "operator_family",
        "dimension",
        "dtype",
        "warmups",
        "sample_count",
        "peak_rss_bytes",
        "thread_environment",
        "workload_manifest",
        "exact_command",
        "commit",
        "dirty",
        "worktree_fingerprint",
    }
)


def build_timing_record(
    manifest: Mapping[str, Any],
    *,
    library: str,
    dimension: int,
    warmups: int,
    repeats: int,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the common isolated-child timing record schema."""
    record: dict[str, Any] = {
        "library": library,
        "capability": manifest["capability"],
        "comparison": manifest["comparison"],
        "operator_family": manifest["operator_family"],
        "dtype": manifest["dtype"],
        "dimension": dimension,
        "warmups": warmups,
        "sample_count": repeats,
        "peak_rss_bytes": peak_rss_bytes(),
        "thread_environment": thread_environment(),
        "workload_manifest": manifest["workload_manifest"],
        "exact_command": exact_command(),
    }
    record.update(fields)
    validate_timing_record(record)
    return record


def validate_timing_record(record: Mapping[str, Any]) -> None:
    """Reject incomplete or nondeterministic child timing records before JSON output."""
    missing = _TIMING_REQUIRED_KEYS.difference(record)
    if missing:
        raise ValueError(f"timing record is missing required keys: {sorted(missing)}")
    if record["dimension"] <= 0 or record["sample_count"] <= 0:
        raise ValueError("timing record dimension and sample_count must be positive")
    if not isinstance(record["thread_environment"], dict):
        raise ValueError("timing record thread_environment must be a mapping")
    command = record["exact_command"]
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise ValueError("timing record exact_command must be a string list")
    if not isinstance(record["commit"], str) or not record["commit"]:
        raise ValueError("timing record commit must be a non-empty string")
    if not isinstance(record["dirty"], bool):
        raise ValueError("timing record dirty must be a boolean")
    if (
        not isinstance(record["worktree_fingerprint"], str)
        or not record["worktree_fingerprint"]
    ):
        raise ValueError(
            "timing record worktree_fingerprint must be a non-empty string"
        )


def exact_command() -> list[str]:
    """Return the child command exactly as the child received it."""
    return [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]


def require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Narrow an untrusted JSON value to a string-keyed mapping."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value
