"""Manual SYK ground-state study with matched native and explicit paths."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import resource
import sys
import time
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

import tencirpauli as tcp


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on supported local platforms."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def make_syk_terms(
    n_modes: int, coupling: float, seed: int
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Return independent quartic SYK couplings in Majorana-generator order."""
    n_majorana = 2 * n_modes
    scale = math.sqrt(6.0) * coupling / (n_majorana**1.5)
    generator = np.random.default_rng(seed)
    return tuple(
        (indices, float(generator.normal(loc=0.0, scale=scale)))
        for indices in itertools.combinations(range(n_majorana), 4)
    )


def random_state(dimension: int, seed: int) -> np.ndarray[Any, Any]:
    """Return a deterministic normalized complex starting vector."""
    generator = np.random.default_rng(seed)
    state = generator.normal(size=dimension) + 1j * generator.normal(size=dimension)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-modes",
        type=int,
        default=12,
        help="number of complex fermion modes, giving N=2*n_modes Majoranas",
    )
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--state-seed", type=int, default=20260807)
    parser.add_argument(
        "--mapping",
        choices=("jordan_wigner", "parity", "bravyi_kitaev"),
        default="jordan_wigner",
    )
    parser.add_argument(
        "--mode",
        choices=("matrix_free", "explicit"),
        default="matrix_free",
        help="match QuSpin's matrix-free MVP or explicit CSR construction",
    )
    parser.add_argument("--storage", choices=("lazy", "eager"), default="lazy")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--ncv", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024**2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_modes < 2:
        raise SystemExit("n-modes must be at least 2")
    if not math.isfinite(args.coupling) or args.coupling <= 0.0:
        raise SystemExit("coupling must be finite and positive")
    if args.repeats < 1:
        raise SystemExit("repeats must be positive")

    end_to_end_start = time.perf_counter()
    generation_start = time.perf_counter()
    raw_terms = make_syk_terms(args.n_modes, args.coupling, args.seed)
    generation_seconds = time.perf_counter() - generation_start

    build_start = time.perf_counter()
    operator = tcp.MajoranaOperator.from_terms(
        args.n_modes, ((indices, -coefficient) for indices, coefficient in raw_terms)
    )
    build_seconds = time.perf_counter() - build_start

    plan_seconds: float | None = None
    explicit_csr_seconds: float | None = None
    plan: Any = None
    matrix: Any = None
    if args.mode == "matrix_free":
        plan_start = time.perf_counter()
        plan = operator.compile(
            "native_mvp",
            storage=args.storage,
            mapping=args.mapping,
            max_bytes=args.max_bytes,
        )
        plan_seconds = time.perf_counter() - plan_start
        dimension = plan.dimension
        mapped_term_count = plan.term_count

        def apply(state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return plan.apply(state, max_bytes=args.max_bytes)

    else:
        csr_start = time.perf_counter()
        mapped = operator.map_fermions(args.mapping, max_bytes=args.max_bytes)
        matrix = mapped.csr(max_bytes=args.max_bytes).to_scipy()
        explicit_csr_seconds = time.perf_counter() - csr_start
        dimension = int(matrix.shape[0])
        mapped_term_count = mapped.term_count

        def apply(state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.asarray(matrix @ state, dtype=np.complex128)

    state = random_state(dimension, args.state_seed)
    first_start = time.perf_counter()
    result = apply(state)
    first_mvp_seconds = time.perf_counter() - first_start
    setup_end_to_end_seconds = time.perf_counter() - end_to_end_start
    mvp_seconds = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = apply(state)
        mvp_seconds.append(time.perf_counter() - start)

    ncv = min(max(args.ncv, 3), dimension - 1)
    if args.mode == "matrix_free":
        linear = plan.to_scipy_linear_operator(max_bytes=args.max_bytes)
    else:
        linear = LinearOperator(
            (dimension, dimension), matvec=apply, dtype=np.complex128
        )
    eigsh_start = time.perf_counter()
    values, vectors = eigsh(
        linear,
        k=1,
        which="SA",
        v0=state,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=ncv,
    )
    eigsh_seconds = time.perf_counter() - eigsh_start
    ground_state = vectors[:, 0]
    energy = float(values[0].real)
    residual = np.linalg.norm(apply(ground_state) - energy * ground_state)
    ground_state_end_to_end_seconds = time.perf_counter() - end_to_end_start

    output: dict[str, object] = {
        "library": "tencirpauli",
        "model": "SYK_q4",
        "n_modes": args.n_modes,
        "n_majorana": 2 * args.n_modes,
        "coupling": args.coupling,
        "seed": args.seed,
        "mapping": args.mapping,
        "mode": args.mode,
        "majorana_term_count": operator.term_count,
        "mapped_term_count": mapped_term_count,
        "dimension": dimension,
        "plan_storage": None if plan is None else plan.storage,
        "plan_execution_family": None if plan is None else plan.execution_family,
        "plan_retained_format": None if plan is None else plan.retained_format,
        "plan_estimated_bytes": None if plan is None else plan.estimated_bytes,
        "state_bytes": state.nbytes,
        "generation_seconds": generation_seconds,
        "majorana_build_seconds": build_seconds,
        "native_mvp_plan_seconds": plan_seconds,
        "explicit_csr_seconds": explicit_csr_seconds,
        "setup_end_to_end_seconds": setup_end_to_end_seconds,
        "first_mvp_seconds": first_mvp_seconds,
        "mvp_seconds": mvp_seconds,
        "mvp_seconds_median": float(np.median(mvp_seconds)),
        "eigsh_seconds": eigsh_seconds,
        "ground_state_end_to_end_seconds": ground_state_end_to_end_seconds,
        "ground_energy": energy,
        "ground_residual": float(residual),
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
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
