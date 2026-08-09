"""Static completeness checks for the isolated QuSpin capability matrix."""

from __future__ import annotations

import sys
from pathlib import Path


MANUAL = Path(__file__).resolve().parents[1] / "benchmarks" / "manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))

from quspin_capabilities import (  # noqa: E402
    DIRECT_CAPABILITIES,
    DIRECT_COMPARISON,
    NO_BASELINE_CAPABILITIES,
    NO_FAIR_BASELINE,
    capability_manifest_paths,
    load_manifest,
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
