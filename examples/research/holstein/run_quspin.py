"""Manual QuSpin one-dimensional Holstein study."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from typing import Any

import numpy as np
from quspin.basis import boson_basis_1d, spinless_fermion_basis_1d, tensor_basis
from quspin.operators import hamiltonian, quantum_LinearOperator


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on supported local platforms."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def add_static(
    static: list[list[object]],
    opstr: str,
    coefficient: float,
    indices: list[int],
) -> None:
    """Append one QuSpin operator-string term."""
    static.append([opstr, [[coefficient, *indices]]])


def holstein_static(
    sites: int,
    hopping_strength: float,
    phonon_frequency: float,
    coupling_strength: float,
    interaction_strength: float,
) -> list[list[object]]:
    """Build the same open-boundary Holstein terms as the native runner."""
    static: list[list[object]] = []
    for left in range(sites - 1):
        right = left + 1
        add_static(static, "+-|I|I", -hopping_strength, [left, right, 0, 0])
        add_static(static, "+-|I|I", -hopping_strength, [right, left, 0, 0])
        add_static(static, "I|+-|I", -hopping_strength, [0, left, right, 0])
        add_static(static, "I|+-|I", -hopping_strength, [0, right, left, 0])
    for site in range(sites):
        add_static(static, "I|I|n", phonon_frequency, [0, 0, site])
        add_static(static, "n|n|I", interaction_strength, [site, site, 0])
        add_static(static, "n|I|+", coupling_strength, [site, 0, site])
        add_static(static, "n|I|-", coupling_strength, [site, 0, site])
        add_static(static, "I|n|+", coupling_strength, [0, site, site])
        add_static(static, "I|n|-", coupling_strength, [0, site, site])
    return static


def random_state(dimension: int, seed: int) -> np.ndarray[Any, Any]:
    """Return a deterministic normalized complex starting vector."""
    generator = np.random.default_rng(seed)
    state = generator.normal(size=dimension) + 1j * generator.normal(size=dimension)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=7)
    parser.add_argument("--t", type=float, default=1.0, dest="hopping_strength")
    parser.add_argument("--omega", type=float, default=0.5, dest="phonon_frequency")
    parser.add_argument("--g", type=float, default=0.7, dest="coupling_strength")
    parser.add_argument("--u", type=float, default=1.5, dest="interaction_strength")
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("matrix_free", "explicit"),
        default="matrix_free",
        help="QuSpin comparison path: matrix-free quantum_LinearOperator or CSR",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--eigsh", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--ncv", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sites < 2 or args.cutoff < 0:
        raise SystemExit("sites must be at least 2 and cutoff must be non-negative")
    particles = args.sites // 2 if args.particles is None else args.particles
    if not 0 <= particles <= args.sites:
        raise SystemExit("particles must be between zero and the number of sites")
    if args.tol <= 0 or args.maxiter < 1 or args.ncv < 3 or args.repeats < 7:
        raise SystemExit(
            "tol positive, maxiter positive, ncv at least 3, and repeats at least "
            "seven are required"
        )

    end_to_end_start = time.perf_counter()
    basis_start = time.perf_counter()
    # tensor_basis supplies the hybrid Fock space. Two spinless factors are
    # used for up/down electrons; every Holstein term is fermion-even, so this
    # commuting tensor representation has the same physical Hamiltonian and
    # spectrum as the spinful form. The basis ordering is implementation-specific.
    basis = tensor_basis(
        spinless_fermion_basis_1d(args.sites, Nf=particles),
        spinless_fermion_basis_1d(args.sites, Nf=particles),
        boson_basis_1d(args.sites, sps=args.cutoff + 1),
    )
    basis_build_seconds = time.perf_counter() - basis_start
    static_start = time.perf_counter()
    static = holstein_static(
        args.sites,
        args.hopping_strength,
        args.phonon_frequency,
        args.coupling_strength,
        args.interaction_strength,
    )
    static_build_seconds = time.perf_counter() - static_start
    operator_start = time.perf_counter()
    if args.mode == "matrix_free":
        operator = quantum_LinearOperator(
            static,
            basis=basis,
            dtype=np.complex128,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
    else:
        operator = hamiltonian(
            static,
            [],
            basis=basis,
            dtype=np.complex128,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
            static_fmt="csr",
        )
    explicit_sparse = operator.tocsr() if args.mode == "explicit" else None
    operator_build_seconds = time.perf_counter() - operator_start
    dimension = operator.Ns if explicit_sparse is None else explicit_sparse.shape[0]
    state = random_state(dimension, args.seed)

    for _ in range(2):
        operator.dot(state)
    first_mvp_start = time.perf_counter()
    result = operator.dot(state)
    first_mvp_seconds = time.perf_counter() - first_mvp_start
    mvp_seconds = []
    for _ in range(args.repeats):
        mvp_start = time.perf_counter()
        result = operator.dot(state)
        mvp_seconds.append(time.perf_counter() - mvp_start)
    setup_end_to_end_seconds = time.perf_counter() - end_to_end_start

    output: dict[str, object] = {
        "library": "quspin",
        "operator": "quantum_LinearOperator",
        "basis": "tensor_basis(spinless_fermion, spinless_fermion, boson)",
        "model": "one_dimensional_spinful_holstein",
        "sites": args.sites,
        "particles_per_spin": particles,
        "term_count": len(static),
        "dimension": dimension,
        "mode": args.mode,
        "cutoff": args.cutoff,
        "basis_build_seconds": basis_build_seconds,
        "static_build_seconds": static_build_seconds,
        "operator_build_seconds": operator_build_seconds,
        "setup_end_to_end_seconds": setup_end_to_end_seconds,
        "first_mvp_seconds": first_mvp_seconds,
        "mvp_seconds": mvp_seconds,
        "mvp_seconds_median": float(np.median(mvp_seconds)),
        "output_norm": float(np.linalg.norm(result)),
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
    }
    if explicit_sparse is not None:
        output.update(
            {
                "operator": "hamiltonian",
                "retained_format": "csr",
                "sparse_nnz": int(explicit_sparse.nnz),
                "sparse_array_bytes": int(
                    explicit_sparse.indptr.nbytes
                    + explicit_sparse.indices.nbytes
                    + explicit_sparse.data.nbytes
                ),
            }
        )
    if args.eigsh:
        rss_before_eigsh = peak_rss_bytes()
        eigsh_start = time.perf_counter()
        values, vectors = operator.eigsh(
            k=1,
            which="SA",
            v0=state,
            tol=args.tol,
            maxiter=args.maxiter,
            ncv=min(args.ncv, dimension - 1),
        )
        eigsh_seconds = time.perf_counter() - eigsh_start
        ground_energy = float(values[0].real)
        ground_state = vectors[:, 0]
        residual = float(
            np.linalg.norm(operator.dot(ground_state) - ground_energy * ground_state)
        )
        output.update(
            {
                "eigsh_seconds": eigsh_seconds,
                "ground_state_end_to_end_seconds": time.perf_counter()
                - end_to_end_start,
                "ground_energy": ground_energy,
                "ground_residual": residual,
                "peak_rss_before_eigsh_bytes": rss_before_eigsh,
            }
        )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
