"""Run one frozen QuSpin-capability workload with TenCirPauli only."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from _provenance import worktree_fingerprint
from quspin_capabilities import (
    ROOT,
    canonical_state,
    exact_command,
    load_manifest,
    peak_rss_bytes,
    save_action_artifact,
    save_action_vector_artifact,
    thread_environment,
)

import tencirpauli as tcp


SectorFactory = Callable[[], tcp.ChargeSector]
CaseBuild = tuple[Any, SectorFactory | None, dict[int, int] | None]


def _exchange(space: tcp.OperatorSpace, left: int, right: int) -> Any:
    """Return the number-conserving spin exchange on one qubit bond."""
    qubit = space.qubit
    return 0.5 * (qubit.x(left) * qubit.x(right) + qubit.y(left) * qubit.y(right))


def _cartesian_states(
    space: tcp.OperatorSpace, boson_cutoffs: dict[int, int] | None
) -> np.ndarray[Any, Any]:
    """Enumerate a bounded Cartesian basis in public operator-space order."""
    dimensions = tuple(
        (
            (boson_cutoffs or {})[axis.index] + 1
            if axis.domain == "boson"
            else axis.dimension
        )
        for axis in space._axes
    )
    return np.asarray(
        list(itertools.product(*(range(size) for size in dimensions))), dtype=np.uint64
    )


def _native_diagnostics(plan: Any) -> dict[str, list[int] | None]:
    """Read optional native counters after all timed work is complete."""
    native = getattr(plan, "_native_plan", None)
    counters = getattr(native, "debug_counters", lambda: None)()
    metadata = getattr(native, "debug_metadata", lambda: None)()
    return {
        "debug_counters": (
            None if counters is None else [int(value) for value in counters]
        ),
        "debug_metadata": (
            None if metadata is None else [int(value) for value in metadata]
        ),
    }


def _operator_and_sector(capability: str) -> CaseBuild:
    """Build one native operator and its finite restricted basis factory."""
    if capability == "spin_full":
        space = tcp.OperatorSpace(qubits=6)
        qubit = space.qubit
        operator = (
            0.41 * qubit.x(0)
            - 0.23 * qubit.z(2)
            + 0.37 * qubit.z(0) * qubit.z(1)
            + 0.29 * _exchange(space, 1, 4)
        )
        return operator, None, None
    if capability == "spin_full_throughput":
        qubits = 18
        space = tcp.OperatorSpace(qubits=qubits)
        operator: Any = 0.13 * space.qubit.x(0) - 0.19 * space.qubit.z(0)
        for index in range(qubits - 1):
            operator = operator + 0.07 * space.qubit.z(index) * space.qubit.z(index + 1)
            operator = operator + 0.23 * _exchange(space, index, index + 1)
        return operator, None, None
    if capability == "spin_fixed_hamming":
        space = tcp.OperatorSpace(qubits=6)
        qubit = space.qubit
        number = tcp.AdditiveCharge(
            space, qubits={index: (0, 1) for index in range(space.qubits)}
        )
        operator = (
            0.67 * _exchange(space, 0, 1)
            - 0.29 * _exchange(space, 2, 5)
            + 0.22 * qubit.z(0) * qubit.z(4)
            + 0.11 * qubit.z(3)
        )
        return operator, lambda: number.sector(2), None
    if capability == "fermion_full":
        space = tcp.OperatorSpace(fermions=4)
        fermion = space.fermion
        operator = (
            -0.73
            * (
                fermion.create(0) * fermion.annihilate(1)
                + fermion.create(1) * fermion.annihilate(0)
            )
            + 0.19j
            * (
                fermion.create(2) * fermion.annihilate(3)
                - fermion.create(3) * fermion.annihilate(2)
            )
            + 0.35 * fermion.create(2) * fermion.annihilate(2)
            + 0.17
            * fermion.create(0)
            * fermion.annihilate(0)
            * fermion.create(3)
            * fermion.annihilate(3)
        )
        return operator, None, None
    if capability == "boson_full":
        space = tcp.OperatorSpace(bosons=3)
        boson = space.boson
        operator = (
            -0.43
            * (
                boson.create(0) * boson.annihilate(1)
                + boson.create(1) * boson.annihilate(0)
            )
            + 0.31 * boson.create(2) * boson.annihilate(2)
            + 0.13
            * boson.create(0)
            * boson.annihilate(0)
            * boson.create(1)
            * boson.annihilate(1)
            + 0.17 * (boson.create(0) + boson.annihilate(0))
        )
        return operator, None, {index: 2 for index in range(space.bosons)}
    if capability == "fermion_boson_full":
        space = tcp.OperatorSpace(fermions=3, bosons=2)
        fermion = space.fermion
        boson = space.boson
        number_zero = fermion.create(0) * fermion.annihilate(0)
        number_one = fermion.create(1) * fermion.annihilate(1)
        operator = (
            -0.51
            * (
                fermion.create(0) * fermion.annihilate(1)
                + fermion.create(1) * fermion.annihilate(0)
            )
            + 0.23
            * (
                boson.create(0) * boson.annihilate(1)
                + boson.create(1) * boson.annihilate(0)
            )
            + 0.37 * number_zero * (boson.create(0) + boson.annihilate(0))
            + 0.19 * number_one * boson.create(1) * boson.annihilate(1)
        )
        return operator, None, {index: 2 for index in range(space.bosons)}
    if capability == "fermion_boson_full_throughput":
        fermions = 10
        bosons = 4
        space = tcp.OperatorSpace(fermions=fermions, bosons=bosons)
        fermion = space.fermion
        boson = space.boson
        operator: Any = 0.11 * boson.create(0) * boson.annihilate(0)
        for index in range(fermions - 1):
            operator = operator - 0.41 * (
                fermion.create(index) * fermion.annihilate(index + 1)
                + fermion.create(index + 1) * fermion.annihilate(index)
            )
            operator = operator + 0.04 * fermion.create(index) * fermion.annihilate(
                index
            ) * (boson.create(index % bosons) + boson.annihilate(index % bosons))
        for index in range(bosons - 1):
            operator = operator + 0.17 * (
                boson.create(index) * boson.annihilate(index + 1)
                + boson.create(index + 1) * boson.annihilate(index)
            )
        return operator, None, {index: 3 for index in range(bosons)}
    if capability == "fermion_fixed_boson_spectators":
        space = tcp.OperatorSpace(fermions=4, bosons=2)
        fermion = space.fermion
        boson = space.boson
        number = tcp.AdditiveCharge(
            space, fermions={index: 1 for index in range(space.fermions)}
        )
        occupation = fermion.create(0) * fermion.annihilate(0)
        operator = (
            -0.61
            * (
                fermion.create(0) * fermion.annihilate(1)
                + fermion.create(1) * fermion.annihilate(0)
                + fermion.create(2) * fermion.annihilate(3)
                + fermion.create(3) * fermion.annihilate(2)
            )
            + 0.28 * occupation * (boson.create(0) + boson.annihilate(0))
            + 0.16 * boson.create(1) * boson.annihilate(1)
        )
        return (
            operator,
            lambda: number.sector(
                2, boson_cutoffs={index: 2 for index in range(space.bosons)}
            ),
            None,
        )
    if capability == "boson_fixed_fermion_spectators":
        space = tcp.OperatorSpace(fermions=3, bosons=3)
        fermion = space.fermion
        boson = space.boson
        number = tcp.AdditiveCharge(
            space, bosons={index: 1 for index in range(space.bosons)}
        )
        operator = (
            -0.55
            * (
                boson.create(0) * boson.annihilate(1)
                + boson.create(1) * boson.annihilate(0)
                + boson.create(1) * boson.annihilate(2)
                + boson.create(2) * boson.annihilate(1)
            )
            + 0.21 * fermion.create(0) * fermion.annihilate(0)
            - 0.14
            * (
                fermion.create(1) * fermion.annihilate(2)
                + fermion.create(2) * fermion.annihilate(1)
            )
            + 0.09
            * fermion.create(2)
            * fermion.annihilate(2)
            * boson.create(0)
            * boson.annihilate(0)
        )
        return (
            operator,
            lambda: number.sector(
                2, boson_cutoffs={index: 3 for index in range(space.bosons)}
            ),
            None,
        )
    if capability == "independent_spin_factors":
        space = tcp.OperatorSpace(qubits=7)
        qubit = space.qubit
        first = tcp.AdditiveCharge(space, qubits={index: (0, 1) for index in range(3)})
        second = tcp.AdditiveCharge(
            space, qubits={index: (0, 1) for index in range(3, 7)}
        )
        operator = (
            0.52 * _exchange(space, 0, 2)
            - 0.34 * _exchange(space, 3, 6)
            + 0.18 * qubit.z(1) * qubit.z(5)
            - 0.07 * qubit.z(4)
        )
        return operator, lambda: tcp.ChargeSector(((first, 1), (second, 2))), None
    if capability == "independent_spin_factors_throughput":
        factor_size = 11
        space = tcp.OperatorSpace(qubits=2 * factor_size)
        first = tcp.AdditiveCharge(
            space, qubits={index: (0, 1) for index in range(factor_size)}
        )
        second = tcp.AdditiveCharge(
            space,
            qubits={index: (0, 1) for index in range(factor_size, 2 * factor_size)},
        )
        operator: Any = 0.09 * space.qubit.z(0) * space.qubit.z(factor_size)
        for index in range(factor_size - 1):
            operator = operator + 0.21 * _exchange(space, index, index + 1)
            operator = operator - 0.17 * _exchange(
                space, factor_size + index, factor_size + index + 1
            )
            operator = operator + 0.03 * space.qubit.z(index) * space.qubit.z(
                factor_size + index
            )
        return (
            operator,
            lambda: tcp.ChargeSector(((first, 5), (second, 5))),
            None,
        )
    if capability == "reversed_boson_spin_factors":
        space = tcp.OperatorSpace(bosons=3, qubits=5)
        boson = space.boson
        qubit = space.qubit
        boson_number = tcp.AdditiveCharge(
            space, bosons={index: 1 for index in range(space.bosons)}
        )
        spin_number = tcp.AdditiveCharge(
            space, qubits={index: (0, 1) for index in range(space.qubits)}
        )
        operator = (
            -0.47
            * (
                boson.create(0) * boson.annihilate(1)
                + boson.create(1) * boson.annihilate(0)
            )
            + 0.39 * _exchange(space, 1, 4)
            + 0.16 * boson.create(2) * boson.annihilate(2) * qubit.z(0)
            - 0.12 * qubit.z(3)
        )
        return (
            operator,
            lambda: tcp.ChargeSector(
                ((boson_number, 2), (spin_number, 2)),
                boson_cutoffs={index: 3 for index in range(space.bosons)},
            ),
            None,
        )
    if capability == "generic_coupled_charge":
        space = tcp.OperatorSpace(bosons=2, qubits=3)
        boson = space.boson
        qubit = space.qubit
        total = tcp.AdditiveCharge(
            space,
            bosons={index: 1 for index in range(space.bosons)},
            qubits={index: (0, 1) for index in range(space.qubits)},
        )
        spin_up = 0.5 * (qubit.x(0) + 1j * qubit.y(0))
        spin_down = 0.5 * (qubit.x(0) - 1j * qubit.y(0))
        operator = (
            0.44 * spin_up * boson.create(0)
            + 0.44 * spin_down * boson.annihilate(0)
            + 0.18
            * (
                boson.create(0) * boson.annihilate(1)
                + boson.create(1) * boson.annihilate(0)
            )
            + 0.12 * qubit.z(2)
        )
        return (
            operator,
            lambda: total.sector(
                2, boson_cutoffs={index: 2 for index in range(space.bosons)}
            ),
            None,
        )
    if capability == "qudit_weyl":
        space = tcp.OperatorSpace(qudits=(3, 3))
        qudit = space.qudit
        operator = (
            0.31 * qudit.weyl(0, 1, 0)
            + 0.31 * qudit.weyl(0, 2, 0)
            + 0.17 * qudit.weyl(1, 0, 1)
            + 0.23 * qudit.weyl(0, 1, 1) * qudit.weyl(1, 2, 0)
        )
        return operator, None, None
    raise ValueError(f"unsupported capability {capability!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate and emit the canonical action artifact without timing",
    )
    return parser.parse_args()


def main() -> None:
    """Construct, validate, and time a single TenCirPauli capability plan."""
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    max_bytes = int(manifest.get("max_bytes", 16 * 1024**3))

    operator_start = time.perf_counter()
    operator, sector_factory, cartesian_cutoffs = _operator_and_sector(
        str(manifest["capability"])
    )
    operator_build_seconds = time.perf_counter() - operator_start
    plan_start = time.perf_counter()
    if sector_factory is None:
        sector_build_seconds = 0.0
        sector = None
        plan = operator.compile(
            "native_mvp",
            storage=str(manifest["storage"]),
            boson_cutoffs=cartesian_cutoffs,
            max_bytes=max_bytes,
        )
    else:
        sector_start = time.perf_counter()
        sector = sector_factory()
        sector_build_seconds = time.perf_counter() - sector_start
        restricted = operator.restrict_charge(
            sector, storage=str(manifest["storage"]), max_bytes=max_bytes
        )
        plan = restricted.mvp_plan(
            storage=str(manifest["storage"]), max_bytes=max_bytes
        )
    plan_build_seconds = time.perf_counter() - plan_start

    validation_start = time.perf_counter()
    validation_mode = str(manifest.get("action_validation", "dense"))
    if sector is None:
        states = _cartesian_states(operator.space, cartesian_cutoffs)
    else:
        states = sector.basis_states(max_bytes=max_bytes)
    if states.shape[0] != plan.dimension:
        raise RuntimeError("public basis states do not match the plan dimension")
    state = canonical_state(states, int(manifest["seed"]))
    output = np.empty_like(state)
    plan.apply_into(state, output, max_bytes=max_bytes)
    if validation_mode == "dense":
        dense_action = np.asarray(
            (
                operator.compile(
                    "dense", boson_cutoffs=cartesian_cutoffs, max_bytes=max_bytes
                )
                if sector is None
                else restricted.dense(max_bytes=max_bytes)
            ),
            dtype=np.complex128,
        )
        action_error = float(np.max(np.abs(output - dense_action @ state)))
        tolerance = float(manifest["action_tolerance"])
        if action_error > tolerance:
            raise RuntimeError(
                f"TenCirPauli dense-action mismatch {action_error} exceeds {tolerance}"
            )
        save_action_artifact(args.artifact, dense_action, states)
    else:
        action_error = 0.0
        save_action_vector_artifact(args.artifact, output, states)
    validation_seconds = time.perf_counter() - validation_start

    if args.preflight:
        print(
            json.dumps(
                {
                    "stage": "preflight",
                    "library": "tencirpauli",
                    "capability": manifest["capability"],
                    "dimension": int(plan.dimension),
                    "action_error": action_error,
                    "validation_seconds": validation_seconds,
                    "thread_environment": thread_environment(),
                    "workload_manifest": manifest["workload_manifest"],
                    "exact_command": exact_command(),
                    **worktree_fingerprint(ROOT),
                },
                sort_keys=True,
            )
        )
        return

    warmups = int(manifest["warmups"])
    repeats = int(manifest["repeats"])
    for _ in range(warmups):
        plan.apply_into(state, output, max_bytes=max_bytes)
    first_apply_into_start = time.perf_counter()
    plan.apply_into(state, output, max_bytes=max_bytes)
    first_apply_into_seconds = time.perf_counter() - first_apply_into_start
    apply_into_seconds: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        plan.apply_into(state, output, max_bytes=max_bytes)
        apply_into_seconds.append(time.perf_counter() - start)

    for _ in range(warmups):
        plan.apply(state, max_bytes=max_bytes)
    first_public_apply_start = time.perf_counter()
    public_output = plan.apply(state, max_bytes=max_bytes)
    first_public_apply_seconds = time.perf_counter() - first_public_apply_start
    public_apply_seconds: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        public_output = plan.apply(state, max_bytes=max_bytes)
        public_apply_seconds.append(time.perf_counter() - start)

    record: dict[str, Any] = {
        "library": "tencirpauli",
        "capability": manifest["capability"],
        "comparison": manifest["comparison"],
        "operator_family": manifest["operator_family"],
        "dtype": manifest["dtype"],
        "storage": plan.storage,
        "dimension": int(plan.dimension),
        "term_count": int(plan.term_count),
        "execution_family": str(plan.execution_family),
        "native_variant": str(plan.native_variant),
        "retained_format": str(plan.retained_format),
        "transition_count": plan.transition_count,
        "native_diagnostics": _native_diagnostics(plan),
        "operator_build_seconds": operator_build_seconds,
        "sector_build_seconds": sector_build_seconds,
        "plan_build_seconds": plan_build_seconds,
        "validation_seconds": validation_seconds,
        "action_error": action_error,
        "warmups": warmups,
        "sample_count": repeats,
        "first_apply_into_seconds": first_apply_into_seconds,
        "apply_into_seconds": apply_into_seconds,
        "apply_into_seconds_median": float(np.median(apply_into_seconds)),
        "first_public_apply_seconds": first_public_apply_seconds,
        "public_apply_seconds": public_apply_seconds,
        "public_apply_seconds_median": float(np.median(public_apply_seconds)),
        "caller_owned_apply_supported": True,
        "input_output_norm_error": float(
            abs(np.linalg.norm(output) - np.linalg.norm(public_output))
        ),
        "plan_estimated_bytes": int(plan.estimated_bytes),
        "plan_estimated_retained_bytes": int(plan.estimated_retained_bytes),
        "plan_estimated_construction_peak_bytes": int(
            plan.estimated_construction_peak_bytes
        ),
        "peak_rss_bytes": peak_rss_bytes(),
        "thread_environment": thread_environment(),
        "workload_manifest": manifest["workload_manifest"],
        "exact_command": exact_command(),
        **worktree_fingerprint(ROOT),
    }
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
