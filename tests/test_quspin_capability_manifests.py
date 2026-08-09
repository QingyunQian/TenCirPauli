"""Static completeness checks for the isolated QuSpin capability matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MANUAL = Path(__file__).resolve().parents[1] / "benchmarks" / "manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))

from quspin_capabilities import (  # noqa: E402
    DIRECT_CAPABILITIES,
    DIRECT_COMPARISON,
    NO_BASELINE_CAPABILITIES,
    NO_FAIR_BASELINE,
    build_timing_record,
    capability_manifest_paths,
    exact_command,
    load_manifest,
    measure_apply,
    validate_timing_record,
    write_preflight_artifact,
)


def test_quspin_capability_manifests_cover_every_declared_comparison_class() -> None:
    manifests = [load_manifest(path) for path in capability_manifest_paths()]
    direct = {
        str(manifest["capability"])
        for manifest in manifests
        if manifest["comparison"] == DIRECT_COMPARISON
    }
    no_baseline = {
        str(manifest["capability"])
        for manifest in manifests
        if manifest["comparison"] == NO_FAIR_BASELINE
    }
    assert direct == DIRECT_CAPABILITIES
    assert no_baseline == NO_BASELINE_CAPABILITIES


def test_shared_quspin_protocol_builds_valid_record_and_canonical_artifacts(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(capability_manifest_paths()[0])
    calls = 0

    def apply() -> None:
        nonlocal calls
        calls += 1

    timing = measure_apply(apply, warmups=2, repeats=3)
    assert calls == 6
    assert len(timing["samples"]) == 3
    record = build_timing_record(
        manifest,
        library="test",
        dimension=2,
        warmups=2,
        repeats=3,
        fields={
            "commit": "0123456789abcdef",
            "dirty": False,
            "worktree_fingerprint": "0123456789ab-clean-test",
        },
    )
    assert record["sample_count"] == 3
    states = np.asarray([(1, 0), (0, 1)], dtype=np.uint64)
    write_preflight_artifact(
        "dense",
        states,
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.complex128),
        tmp_path / "matrix.npz",
    )
    write_preflight_artifact(
        "vector",
        states,
        np.asarray((1.0, 0.0), dtype=np.complex128),
        tmp_path / "vector.npz",
    )
    assert np.load(tmp_path / "matrix.npz")["kind"].item() == "matrix"
    assert np.load(tmp_path / "vector.npz")["kind"].item() == "vector"


def test_shared_quspin_protocol_rejects_missing_provenance_fields() -> None:
    manifest = load_manifest(capability_manifest_paths()[0])
    with pytest.raises(ValueError, match="worktree_fingerprint"):
        build_timing_record(
            manifest,
            library="test",
            dimension=2,
            warmups=2,
            repeats=3,
            fields={"commit": "0123456789abcdef", "dirty": False},
        )
    with pytest.raises(ValueError, match="exact_command"):
        validate_timing_record(
            {
                "library": "test",
                "capability": manifest["capability"],
                "comparison": manifest["comparison"],
                "operator_family": manifest["operator_family"],
                "dimension": 2,
                "dtype": "complex128",
                "warmups": 2,
                "sample_count": 3,
                "peak_rss_bytes": 1,
                "thread_environment": {},
                "workload_manifest": manifest["workload_manifest"],
                "commit": "0123456789abcdef",
                "dirty": False,
                "worktree_fingerprint": "0123456789ab-clean-test",
            }
        )


def test_peak_rss_bytes_is_zero_when_platform_has_no_resource_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quspin_capabilities as protocol

    monkeypatch.setattr(protocol, "_resource", None)
    assert protocol.peak_rss_bytes() == 0


def test_exact_command_is_deterministic_from_child_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quspin_capabilities as protocol

    monkeypatch.setattr(protocol.sys, "executable", "/test/python")
    monkeypatch.setattr(protocol.sys, "argv", ["child.py", "--manifest", "row.json"])
    expected = [
        "/test/python",
        str(Path("child.py").resolve()),
        "--manifest",
        "row.json",
    ]
    assert exact_command() == expected
    assert exact_command() == expected
