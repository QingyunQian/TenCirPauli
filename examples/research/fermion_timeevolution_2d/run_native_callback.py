"""Run raw-Fermion native MVP through a JAX pure callback.

The time-evolution recurrence is staged with ``jax.jit`` while each MVP is
executed by the Rust native plan on the host. This is a forward-only bridge;
the callback is intentionally not differentiated.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc
from common import hubbard_terms, model_info, spectral_radius_bound, spinful_neel_state

import tencirpauli as tcp


jax.config.update("jax_enable_x64", True)


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _timed(function: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = function()
    result.block_until_ready()
    return time.perf_counter() - start, result


def _warm_median(function: Callable[[], Any], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        elapsed, _ = _timed(function)
        samples.append(elapsed)
    return statistics.median(samples)


def _callback_mvp(plan: Any, dimension: int) -> Callable[[Any], Any]:
    result_spec = jax.ShapeDtypeStruct((dimension,), jnp.complex128)

    def callback(state: Any) -> np.ndarray[Any, Any]:
        value = np.ascontiguousarray(np.asarray(state, dtype=np.complex128))
        return np.ascontiguousarray(plan.apply(value), dtype=np.complex128)

    def mvp(state: Any) -> Any:
        return jax.pure_callback(
            callback,
            result_spec,
            state,
            vmap_method="sequential",
        )

    return mvp


def _run_solver(
    solver: str,
    mvp: Callable[[Any], Any],
    state: Any,
    times: np.ndarray[Any, Any],
    radius: float,
    subspace_dimension: int,
    chebyshev_terms: int,
    bessel_iterations: int,
    repeats: int,
) -> Dict[str, Any]:
    if solver == "krylov":

        def evolve(value: Any) -> Any:
            return tc.timeevol.krylov_evol(
                mvp,
                value,
                times,
                subspace_dimension=subspace_dimension,
                scan_impl=True,
            )

    else:

        def evolve(value: Any) -> Any:
            return tc.timeevol.chebyshev_evol(
                mvp,
                value,
                float(times[-1]),
                (radius, -radius),
                k=chebyshev_terms,
                M=bessel_iterations,
            )

    compiled = jax.jit(evolve)
    first_seconds, first_result = _timed(lambda: compiled(state))
    warm_seconds = _warm_median(lambda: compiled(state), repeats)
    final = np.asarray(first_result)
    return {
        "first_jit_seconds": first_seconds,
        "warm_median_seconds": warm_seconds,
        "result_shape": list(final.shape),
        "result_norm": float(np.linalg.norm(final)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    info = model_info(args.rows, args.cols)
    terms = hubbard_terms(args.rows, args.cols, args.hopping, args.interaction)

    operator_start = time.perf_counter()
    operator = tcp.FermionOperator.from_terms(info["n_modes"], terms)
    operator_seconds = time.perf_counter() - operator_start
    plan_start = time.perf_counter()
    plan = operator.compile("native_mvp")
    plan_seconds = time.perf_counter() - plan_start
    adapter_start = time.perf_counter()
    mvp = _callback_mvp(plan, info["dimension"])
    adapter_seconds = time.perf_counter() - adapter_start
    state = jnp.asarray(spinful_neel_state(args.rows, args.cols))
    times = np.asarray(args.times, dtype=np.float64)
    radius = spectral_radius_bound(args.rows, args.cols, args.hopping, args.interaction)
    chebyshev_terms = tc.timeevol.estimate_k(float(times[-1]), (radius, -radius))
    bessel_iterations = tc.timeevol.estimate_M(
        float(times[-1]), (radius, -radius), chebyshev_terms
    )
    results = {
        solver: _run_solver(
            solver,
            mvp,
            state,
            times,
            radius,
            args.subspace_dimension,
            chebyshev_terms,
            bessel_iterations,
            args.repeats,
        )
        for solver in args.solvers
    }
    return {
        "implementation": "raw_fermion_native_mvp_pure_callback",
        "initial_state": "spinful_neel",
        "backend": "tensorcircuit_ng_jax_with_host_callback",
        "jax": jax.__version__,
        "jax_platform": jax.default_backend(),
        "nqubits": info["n_modes"],
        "operator_term_count": operator.term_count,
        "plan_term_count": plan.term_count,
        "dimension": info["dimension"],
        "rows": args.rows,
        "cols": args.cols,
        "bonds": info["bonds"],
        "hopping": args.hopping,
        "interaction": args.interaction,
        "times": list(args.times),
        "subspace_dimension": args.subspace_dimension,
        "chebyshev_terms": chebyshev_terms,
        "bessel_iterations": bessel_iterations,
        "repeats": args.repeats,
        "operator_build_seconds": operator_seconds,
        "native_plan_seconds": plan_seconds,
        "callback_adapter_seconds": adapter_seconds,
        "state_bytes": int(state.size * 16),
        "peak_rss_bytes": peak_rss_bytes(),
        "results": results,
        "package": getattr(tcp, "__version__", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--interaction", type=float, default=4.0)
    parser.add_argument("--times", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=("krylov", "chebyshev"),
        default=["krylov", "chebyshev"],
    )
    parser.add_argument("--subspace-dimension", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.subspace_dimension < 1:
        raise SystemExit("repeats and subspace-dimension must be positive")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
