"""Manual one-dimensional Holstein electron--phonon study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

import tencirpauli as tcp


ROOT = Path(__file__).resolve().parents[3]
WORKLOAD_MANIFEST_VERSION = "holstein-v1"
BENCHMARK_HELPERS = ROOT / "benchmarks" / "manual"


def worktree_fingerprint() -> dict[str, str | bool]:
    """Load the shared manual benchmark provenance helper on demand."""
    spec = importlib.util.spec_from_file_location(
        "tencirpauli_manual_provenance", BENCHMARK_HELPERS / "_provenance.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("manual benchmark provenance helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.worktree_fingerprint(ROOT)


def number_operator(space: tcp.OperatorSpace, mode: int) -> tcp.HybridOperator:
    """Return the fermionic occupation operator for one spin-orbital."""
    return space.fermion.create(mode) * space.fermion.annihilate(mode)


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on supported local platforms."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def build_holstein(
    sites: int,
    hopping_strength: float,
    phonon_frequency: float,
    coupling_strength: float,
    interaction_strength: float,
) -> tcp.HybridOperator:
    """Build an open-boundary, spinful one-dimensional Holstein Hamiltonian."""
    space = tcp.OperatorSpace(fermions=2 * sites, bosons=sites)
    fermion = space.fermion
    boson = space.boson

    hopping: tcp.HybridOperator | None = None
    for left in range(sites - 1):
        for spin in range(2):
            first = 2 * left + spin
            second = 2 * (left + 1) + spin
            bond = fermion.create(first) * fermion.annihilate(second)
            bond = bond + fermion.create(second) * fermion.annihilate(first)
            hopping = bond if hopping is None else hopping + bond
    assert hopping is not None

    phonon_energy = boson.create(0) * boson.annihilate(0)
    for site in range(1, sites):
        phonon_energy = phonon_energy + boson.create(site) * boson.annihilate(site)

    site_density = [
        number_operator(space, 2 * site) + number_operator(space, 2 * site + 1)
        for site in range(sites)
    ]
    interaction = number_operator(space, 0) * number_operator(space, 1)
    for site in range(1, sites):
        interaction = interaction + number_operator(space, 2 * site) * number_operator(
            space, 2 * site + 1
        )

    coupling = site_density[0] * (boson.create(0) + boson.annihilate(0))
    for site in range(1, sites):
        displacement = boson.create(site) + boson.annihilate(site)
        coupling = coupling + site_density[site] * displacement

    result = (
        -hopping_strength * hopping
        + phonon_frequency * phonon_energy
        + interaction_strength * interaction
        + coupling_strength * coupling
    )
    if not isinstance(result, tcp.HybridOperator):
        raise TypeError("the Holstein model must remain a hybrid operator")
    return result


def build_spinful_sector(
    operator: tcp.HybridOperator,
    sites: int,
    particles_per_spin: int,
    cutoff: int,
    max_bytes: int,
) -> tcp.ChargeSector:
    """Select equal fixed up/down particle numbers with finite phonon levels."""
    total = tcp.AdditiveCharge(
        operator.space,
        fermions={mode: 1 for mode in range(2 * sites)},
    )
    balance = tcp.AdditiveCharge(
        operator.space,
        fermions={
            2 * site + spin: (1 if spin == 0 else -1)
            for site in range(sites)
            for spin in range(2)
        },
    )
    return tcp.ChargeSector(
        ((total, 2 * particles_per_spin), (balance, 0)),
        boson_cutoffs={site: cutoff for site in range(sites)},
        max_bytes=max_bytes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=7)
    parser.add_argument("--t", type=float, default=1.0, dest="hopping_strength")
    parser.add_argument("--omega", type=float, default=0.5, dest="phonon_frequency")
    parser.add_argument("--g", type=float, default=0.7, dest="coupling_strength")
    parser.add_argument("--u", type=float, default=1.5, dest="interaction_strength")
    parser.add_argument(
        "--cutoff",
        type=int,
        default=3,
        help="maximum local phonon occupation; default gives three Fock levels",
    )
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--storage", choices=("lazy", "eager"), default="lazy")
    parser.add_argument("--max-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--dense-check", action="store_true")
    parser.add_argument("--max-dense-dimension", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--eigsh", action="store_true")
    parser.add_argument(
        "--csr-eigsh",
        action="store_true",
        help=(
            "also materialize an explicit SciPy CSR matrix and run eigsh; "
            "implies --eigsh and requires --storage eager"
        ),
    )
    parser.add_argument(
        "--materialize-csr",
        action="store_true",
        help="materialize and time the eager CSR representation without eigsh",
    )
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--ncv", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sites < 2:
        raise SystemExit("--sites must be at least 2 for a one-dimensional chain")
    if args.cutoff < 0:
        raise SystemExit("--cutoff must be non-negative")
    if args.max_bytes < 0:
        raise SystemExit("--max-bytes must be non-negative")
    if (args.csr_eigsh or args.materialize_csr) and args.storage != "eager":
        raise SystemExit("CSR materialization requires --storage eager")
    particles = args.sites // 2 if args.particles is None else args.particles
    if not 0 <= particles <= args.sites:
        raise SystemExit("particles must be between zero and the number of sites")
    if args.max_dense_dimension < 2:
        raise SystemExit("--max-dense-dimension must be at least 2")
    if args.tol <= 0 or args.maxiter < 1 or args.ncv < 3 or args.repeats < 7:
        raise SystemExit(
            "tol positive, maxiter positive, ncv at least 3, and repeats at least "
            "seven are required"
        )

    end_to_end_start = time.perf_counter()
    operator_start = time.perf_counter()
    operator = build_holstein(
        args.sites,
        args.hopping_strength,
        args.phonon_frequency,
        args.coupling_strength,
        args.interaction_strength,
    )
    operator_build_seconds = time.perf_counter() - operator_start
    sector_start = time.perf_counter()
    sector = build_spinful_sector(
        operator, args.sites, particles, args.cutoff, args.max_bytes
    )
    sector_build_seconds = time.perf_counter() - sector_start
    plan_start = time.perf_counter()
    restricted = operator.restrict_charge(
        sector, storage=args.storage, max_bytes=args.max_bytes
    )
    native_plan = restricted.mvp_plan(storage=args.storage, max_bytes=args.max_bytes)
    plan_build_seconds = time.perf_counter() - plan_start
    if (args.eigsh or args.csr_eigsh) and native_plan.dimension <= args.ncv:
        raise SystemExit("ncv must be smaller than the finite Hilbert-space dimension")

    rng = np.random.default_rng(args.seed)
    start = rng.normal(size=native_plan.dimension) + 1j * rng.normal(
        size=native_plan.dimension
    )
    start = np.asarray(start / np.linalg.norm(start), dtype=np.complex128)
    for _ in range(2):
        native_plan(start)
    first_mvp_start = time.perf_counter()
    result = native_plan(start)
    first_mvp_seconds = time.perf_counter() - first_mvp_start
    mvp_seconds = []
    for _ in range(args.repeats):
        mvp_start = time.perf_counter()
        result = native_plan(start)
        mvp_seconds.append(time.perf_counter() - mvp_start)
    setup_end_to_end_seconds = time.perf_counter() - end_to_end_start

    output: dict[str, Any] = {
        "library": "tencirpauli",
        "model": "one_dimensional_spinful_holstein",
        "sites": args.sites,
        "particles_per_spin": particles,
        "term_count": operator.term_count,
        "dimension": native_plan.dimension,
        "cutoff": args.cutoff,
        "storage": native_plan.storage,
        "execution_family": native_plan.execution_family,
        "native_variant": getattr(native_plan, "native_variant", "unknown"),
        "retained_format": native_plan.retained_format,
        "operator_build_seconds": operator_build_seconds,
        "sector_build_seconds": sector_build_seconds,
        "plan_build_seconds": plan_build_seconds,
        "setup_end_to_end_seconds": setup_end_to_end_seconds,
        "first_mvp_seconds": first_mvp_seconds,
        "mvp_seconds": mvp_seconds,
        "mvp_seconds_median": float(np.median(mvp_seconds)),
        "output_norm": float(np.linalg.norm(result)),
        "plan_estimated_bytes": native_plan.estimated_bytes,
        "plan_estimated_retained_bytes": native_plan.estimated_retained_bytes,
        "plan_estimated_construction_peak_bytes": native_plan.estimated_construction_peak_bytes,
        "transition_count": native_plan.transition_count,
        "peak_rss_bytes": peak_rss_bytes(),
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
        "workload_manifest": WORKLOAD_MANIFEST_VERSION,
        "exact_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        **worktree_fingerprint(),
    }
    native_eager = getattr(native_plan, "_native_plan", None)
    if native_eager is not None and hasattr(native_eager, "construction_profile"):
        profile = native_eager.construction_profile()
        output["csr_construction_profile"] = {
            "generation_nanos": int(profile[0]),
            "chunk_count": int(profile[1]),
            "sorted_entries": int(profile[2]),
            "merged_entries": int(profile[3]),
            "heap_operations": int(profile[4]),
            "final_array_allocation_entries": int(profile[5]),
            "chunk_metadata_bytes": int(profile[6]),
            "sort_nanos": int(profile[7]),
            "merge_nanos": int(profile[8]),
            "final_array_allocation_nanos": int(profile[9]),
        }
    if args.eigsh or args.csr_eigsh:
        linear = native_plan.to_scipy_linear_operator(max_bytes=args.max_bytes)
        rss_before_eigsh = peak_rss_bytes()
        eigsh_start = time.perf_counter()
        ncv = min(args.ncv, native_plan.dimension - 1)
        values, vectors = eigsh(
            linear,
            k=1,
            which="SA",
            v0=start,
            tol=args.tol,
            maxiter=args.maxiter,
            ncv=ncv,
        )
        eigsh_seconds = time.perf_counter() - eigsh_start
        native_energy = float(values[0].real)
        ground_state = vectors[:, 0]
        residual = float(
            np.linalg.norm(native_plan(ground_state) - native_energy * ground_state)
        )
        output.update(
            {
                "eigsh_seconds": eigsh_seconds,
                "ground_state_end_to_end_seconds": time.perf_counter()
                - end_to_end_start,
                "native_ground_energy": native_energy,
                "native_ground_residual": residual,
                "ground_residual": residual,
                "peak_rss_before_eigsh_bytes": rss_before_eigsh,
            }
        )
    if args.csr_eigsh or args.materialize_csr:
        csr_materialize_start = time.perf_counter()
        csr = restricted.csr(max_bytes=args.max_bytes)
        scipy_csr = csr.to_scipy()
        csr_materialize_seconds = time.perf_counter() - csr_materialize_start
        output.update(
            {
                "csr_materialize_seconds": csr_materialize_seconds,
                "csr_nnz": int(scipy_csr.nnz),
                "csr_array_bytes": int(
                    csr.indptr.nbytes + csr.indices.nbytes + csr.data.nbytes
                ),
            }
        )
        if args.csr_eigsh:
            csr_rss_before_eigsh = peak_rss_bytes()
            csr_eigsh_start = time.perf_counter()
            csr_values, csr_vectors = eigsh(
                scipy_csr,
                k=1,
                which="SA",
                v0=start.copy(),
                tol=args.tol,
                maxiter=args.maxiter,
                ncv=ncv,
            )
            csr_eigsh_seconds = time.perf_counter() - csr_eigsh_start
            csr_energy = float(csr_values[0].real)
            csr_ground_state = csr_vectors[:, 0]
            csr_residual = float(
                np.linalg.norm(
                    scipy_csr @ csr_ground_state - csr_energy * csr_ground_state
                )
            )
            np.testing.assert_allclose(
                csr_energy, output["native_ground_energy"], rtol=1.0e-8, atol=1.0e-9
            )
            output.update(
                {
                    "csr_eigsh_seconds": csr_eigsh_seconds,
                    "csr_ground_energy": csr_energy,
                    "csr_ground_residual": csr_residual,
                    "peak_rss_before_csr_eigsh_bytes": csr_rss_before_eigsh,
                    "peak_rss_bytes_after_csr_eigsh": peak_rss_bytes(),
                    "csr_vs_native_energy_error": abs(
                        csr_energy - output["native_ground_energy"]
                    ),
                }
            )
    if args.dense_check:
        if native_plan.dimension > args.max_dense_dimension:
            raise SystemExit(
                "dense check exceeds --max-dense-dimension; lower sites/cutoff "
                "or raise the explicit guard"
            )
        dense = restricted.dense(max_bytes=args.max_bytes)
        dense_energy = float(np.linalg.eigvalsh(dense)[0])
        if not args.eigsh:
            output["dense_ground_energy"] = dense_energy
        else:
            np.testing.assert_allclose(
                native_energy, dense_energy, rtol=1.0e-8, atol=1.0e-9
            )
            output["dense_ground_energy"] = dense_energy
            output["ground_energy_error"] = abs(native_energy - dense_energy)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
