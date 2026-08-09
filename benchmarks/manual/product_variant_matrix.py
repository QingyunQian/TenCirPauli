"""Serial release-mode A/B matrix for the supported charge executors.

The parent process launches one fresh child for each case/variant pair.  A
child changes only the private selector hooks, records the native variant
actually constructed, and checks its output against a dense reference.  The
runner is intentionally evidence-oriented: performance enforcement is opt-in
and never runs cases concurrently.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from _provenance import worktree_fingerprint

import tencirpauli as tcp


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "product-variant-matrix-v1"
charge_module = importlib.import_module("tencirpauli.charge")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _qubit_case() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(qubits=8)
    charge = tcp.AdditiveCharge(space, qubits={index: (0, 1) for index in range(8)})
    operator = (
        space.qubit.x(0) * space.qubit.x(1)
        + space.qubit.y(0) * space.qubit.y(1)
        + 0.17 * space.qubit.z(4)
    )
    return operator, charge.sector(4)


def _qudit_case() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(qudits=(3, 3))
    charge = tcp.AdditiveCharge(space, qudits={0: 1, 1: 1})
    operator = space.qudit.weyl(0, 0, 1) + 0.3 * space.qudit.weyl(1, 0, 2)
    return operator, charge.sector(0)


def _boson_case() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(bosons=3)
    charge = tcp.AdditiveCharge(space, bosons={index: 1 for index in range(3)})
    operator = (
        space.boson.create(0) * space.boson.annihilate(1)
        + space.boson.create(1) * space.boson.annihilate(0)
        + 0.2 * space.boson.create(2) * space.boson.annihilate(2)
    )
    return operator, charge.sector(1, boson_cutoffs={index: 2 for index in range(3)})


def _pure_fermion_case() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(fermions=6)
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(6)})
    balance = tcp.AdditiveCharge(
        space, fermions={index: (1 if index < 3 else -1) for index in range(6)}
    )
    operator = (
        space.fermion.create(0) * space.fermion.annihilate(1)
        + space.fermion.create(1) * space.fermion.annihilate(0)
        + space.fermion.create(3) * space.fermion.annihilate(4)
        + space.fermion.create(4) * space.fermion.annihilate(3)
    )
    return operator, tcp.ChargeSector(((total, 2), (balance, 0)))


def _mixed_fermion_boson_case() -> tuple[Any, Any]:
    sites = 3
    space = tcp.OperatorSpace(fermions=2 * sites, bosons=sites)
    total = tcp.AdditiveCharge(space, fermions={index: 1 for index in range(2 * sites)})
    balance = tcp.AdditiveCharge(
        space,
        fermions={
            2 * site + spin: (1 if spin == 0 else -1)
            for site in range(sites)
            for spin in range(2)
        },
    )
    operator: Any = 0.2 * space.boson.create(0) * space.boson.annihilate(0)
    for site in range(sites - 1):
        for spin in range(2):
            left = 2 * site + spin
            right = 2 * (site + 1) + spin
            operator = operator + space.fermion.create(left) * space.fermion.annihilate(
                right
            )
            operator = operator + space.fermion.create(
                right
            ) * space.fermion.annihilate(left)
    operator = operator + 0.1 * (
        space.fermion.create(0)
        * space.fermion.annihilate(0)
        * (space.boson.create(1) + space.boson.annihilate(1))
    )
    sector = tcp.ChargeSector(
        ((total, 2), (balance, 0)),
        boson_cutoffs={index: 2 for index in range(sites)},
    )
    return operator, sector


def _two_factor_case() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(qubits=4, bosons=1)
    first = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 1)})
    second = tcp.AdditiveCharge(space, qubits={2: (0, 1), 3: (0, 1)})
    sector = tcp.ChargeSector(((first, 1), (second, 1)), boson_cutoffs={0: 2})
    operator = (
        space.qubit.x(0) * space.qubit.x(1)
        + space.qubit.y(0) * space.qubit.y(1)
        + space.qubit.x(2) * space.qubit.x(3)
        + space.qubit.y(2) * space.qubit.y(3)
        + 0.25 * space.boson.create(0) * space.boson.annihilate(0)
    )
    return operator, sector


def _reversed_factor_sizes_case() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(bosons=3, qubits=4)
    boson_charge = tcp.AdditiveCharge(space, bosons={index: 1 for index in range(3)})
    qubit_charge = tcp.AdditiveCharge(
        space, qubits={index: (0, 1) for index in range(4)}
    )
    operator = (
        space.boson.create(0) * space.boson.annihilate(1)
        + space.boson.create(1) * space.boson.annihilate(0)
        + space.qubit.x(0) * space.qubit.x(1)
        + space.qubit.y(0) * space.qubit.y(1)
    )
    sector = tcp.ChargeSector(
        ((boson_charge, 1), (qubit_charge, 2)),
        boson_cutoffs={index: 2 for index in range(3)},
    )
    return operator, sector


def _interleaved_factor_case() -> tuple[Any, Any]:
    space = tcp.OperatorSpace(qubits=3)
    charge = tcp.AdditiveCharge(space, qubits={0: (0, 1), 1: (0, 0), 2: (0, 1)})
    operator = (
        space.qubit.x(0) * space.qubit.x(2)
        + space.qubit.y(0) * space.qubit.y(2)
        + 0.2 * space.qubit.z(1)
    )
    return operator, charge.sector(1)


CASES: dict[str, tuple[Callable[[], tuple[Any, Any]], tuple[str, ...], str]] = {
    "qubit": (_qubit_case, ("generic", "product", "qubit"), "qubit"),
    "qudit": (_qudit_case, ("generic", "product", "qudit"), "qudit"),
    "boson": (_boson_case, ("generic", "product", "boson"), "boson"),
    "pure_fermion": (
        _pure_fermion_case,
        ("generic", "product", "fermion"),
        "fermion",
    ),
    "mixed_fermion_boson": (
        _mixed_fermion_boson_case,
        ("generic", "product", "fermion"),
        "fermion",
    ),
    "two_factor": (_two_factor_case, ("generic", "product"), "product"),
    "reversed_factor_sizes": (
        _reversed_factor_sizes_case,
        ("generic", "product"),
        "product",
    ),
    "interleaved_factor": (
        _interleaved_factor_case,
        ("generic", "product"),
        "product",
    ),
}


@contextmanager
def _selector_mode(mode: str) -> Iterator[None]:
    names = (
        "_supports_independent_factor_product",
        "_fast_qubit_particles",
        "_fast_qudit_factor",
        "_fast_boson_factor",
        "_fast_fermion_particles",
    )
    originals = {name: getattr(charge_module, name) for name in names}
    for name in names:
        setattr(charge_module, name, lambda *args: None)
    if mode == "product":
        charge_module._supports_independent_factor_product = originals[
            "_supports_independent_factor_product"
        ]
    elif mode in {"qubit", "fermion"}:
        setattr(
            charge_module,
            f"_fast_{mode}_particles",
            originals[f"_fast_{mode}_particles"],
        )
    elif mode in {"qudit", "boson"}:
        setattr(
            charge_module, f"_fast_{mode}_factor", originals[f"_fast_{mode}_factor"]
        )
    elif mode != "generic":
        raise ValueError(f"unknown selector mode: {mode!r}")
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(charge_module, name, function)


def _measure_child(case_name: str, variant: str, repeats: int) -> dict[str, Any]:
    factory, allowed_variants, expected_public_variant = CASES[case_name]
    if variant not in allowed_variants:
        raise ValueError(f"variant {variant!r} is not valid for case {case_name!r}")
    with _selector_mode(variant):
        construction: list[float] = []
        restricted: Any = None
        sector: Any = None
        for _ in range(2):
            start = time.perf_counter()
            operator, sector = factory()
            restricted = charge_module.restrict_charge(operator, sector)
            restricted.mvp_plan()
            construction.append(time.perf_counter() - start)
        assert restricted is not None and sector is not None
        plan = restricted.mvp_plan()
        actual_variant = str(plan.native_variant)
        expected_variant = "generic" if variant == "generic" else variant
        if actual_variant != expected_variant:
            raise RuntimeError(
                f"{case_name}/{variant} selected {actual_variant!r}, "
                f"expected {expected_variant!r}"
            )
        rng = np.random.default_rng(20260809)
        state = rng.normal(size=plan.dimension) + 1j * rng.normal(size=plan.dimension)
        state = np.asarray(state, dtype=np.complex128)
        output = np.empty_like(state)
        plan.apply_into(state, output, max_bytes=None)
        first_start = time.perf_counter()
        plan.apply_into(state, output, max_bytes=None)
        first_seconds = time.perf_counter() - first_start
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            plan.apply_into(state, output, max_bytes=None)
            samples.append(time.perf_counter() - start)
        reference = restricted.dense() @ state
        max_error = float(np.max(np.abs(output - reference))) if output.size else 0.0
        if max_error > 5e-12:
            raise AssertionError(f"{case_name}/{variant} dense error {max_error}")
        counters = plan._native_execution.debug_counters()
        return {
            "case": case_name,
            "requested_variant": variant,
            "expected_public_variant": expected_public_variant,
            "native_variant": actual_variant,
            "execution_family": str(plan.execution_family),
            "dimension": int(plan.dimension),
            "term_count": int(plan.term_count),
            "construction_seconds_median": statistics.median(construction),
            "first_apply_into_seconds": first_seconds,
            "steady_apply_into_seconds": samples,
            "steady_apply_into_seconds_median": statistics.median(samples),
            "estimated_retained_bytes": int(plan.estimated_retained_bytes),
            "estimated_construction_peak_bytes": int(
                plan.estimated_construction_peak_bytes
            ),
            "debug_counters": counters,
            "max_abs_error_vs_dense": max_error,
            "peak_rss_bytes": _rss_bytes(),
            "thread_environment": {
                name: os.environ.get(name, "unset")
                for name in (
                    "RAYON_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "workload_manifest": MANIFEST_VERSION,
            **worktree_fingerprint(ROOT),
        }


def _parse_record(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise RuntimeError("variant child did not emit a JSON record")


def _child_main(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            _measure_child(args.case, args.variant, args.repeats), sort_keys=True
        )
    )


def _parent_main(args: argparse.Namespace) -> None:
    selected_cases = args.cases or list(CASES)
    records: list[dict[str, Any]] = []
    for case_name in selected_cases:
        if case_name not in CASES:
            raise SystemExit(f"unknown case: {case_name}")
        _, variants, _ = CASES[case_name]
        for variant in variants:
            command = [
                args.python,
                str(Path(__file__).resolve()),
                "--child",
                "--case",
                case_name,
                "--variant",
                variant,
                "--repeats",
                str(args.repeats),
            ]
            environment = {
                **os.environ,
                "RAYON_NUM_THREADS": str(args.threads),
                "OMP_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads),
                "VECLIB_MAXIMUM_THREADS": str(args.threads),
            }
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            record = _parse_record(completed.stdout)
            record["exact_command"] = command
            records.append(record)
    if args.enforce_performance:
        by_key = {
            (record["case"], record["requested_variant"]): record for record in records
        }
        for case_name in selected_cases:
            _, variants, specialized = CASES[case_name]
            if "product" not in variants or specialized not in variants:
                continue
            product = by_key[(case_name, "product")]["steady_apply_into_seconds_median"]
            winner = by_key[(case_name, specialized)][
                "steady_apply_into_seconds_median"
            ]
            if product > args.max_product_ratio * winner:
                raise RuntimeError(
                    f"{case_name} Product median {product:.6g}s exceeds "
                    f"{args.max_product_ratio:.3g}x {specialized} median {winner:.6g}s"
                )
    print(
        json.dumps(
            {
                "benchmark": "product_variant_matrix",
                "workload_manifest": MANIFEST_VERSION,
                "threads": args.threads,
                "cases": selected_cases,
                "records": records,
                "performance_policy": {
                    "enforced": bool(args.enforce_performance),
                    "max_product_ratio": args.max_product_ratio,
                    "expected_goal": "Product is not a default winner until its matched median is within the recorded gate.",
                },
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--variant")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--enforce-performance", action="store_true")
    parser.add_argument("--max-product-ratio", type=float, default=2.0)
    args = parser.parse_args()
    if args.repeats < 7 or args.threads < 1:
        raise SystemExit("repeats must be at least seven and threads must be positive")
    if args.child:
        if args.case is None or args.variant is None:
            raise SystemExit("--child requires --case and --variant")
        _child_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
