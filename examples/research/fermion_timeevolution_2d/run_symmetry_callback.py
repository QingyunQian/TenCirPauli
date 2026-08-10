"""Run a symmetry-restricted Hubbard MVP through a JAX pure callback."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc
from common import (
    hubbard_terms,
    model_info,
    spectral_radius_bound,
    spinful_neel_state,
)
from run_native_callback import _callback_mvp, _run_solver

import tencirpauli as tcp


jax.config.update("jax_enable_x64", True)


def _spinful_sector(sites: int) -> tcp.ChargeSector:
    space = tcp.OperatorSpace(fermions=2 * sites)
    particle_number = tcp.AdditiveCharge(
        space,
        name="particle_number",
        fermions={mode: 1 for mode in range(2 * sites)},
    )
    two_sz = tcp.AdditiveCharge(
        space,
        name="two_Sz",
        fermions={mode: (1 if mode < sites else -1) for mode in range(2 * sites)},
    )
    return tcp.ChargeSector(((particle_number, sites), (two_sz, 0)))


def _direct_callback_error(plan: Any, state: np.ndarray[Any, Any]) -> float:
    mvp = _callback_mvp(plan, state.size)
    compiled = jax.jit(mvp)
    callback_output = np.asarray(compiled(jnp.asarray(state)))
    native_output = np.asarray(plan.apply(state))
    return float(np.max(np.abs(callback_output - native_output)))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    info = model_info(args.rows, args.cols)
    sites = info["sites"]
    sector = _spinful_sector(sites)
    terms = hubbard_terms(args.rows, args.cols, args.hopping, args.interaction)

    operator_start = time.perf_counter()
    operator = tcp.FermionOperator.from_terms(info["n_modes"], terms)
    operator_seconds = time.perf_counter() - operator_start
    restrict_start = time.perf_counter()
    restricted = operator.restrict_charge(sector, storage=args.storage)
    restrict_seconds = time.perf_counter() - restrict_start
    plan_start = time.perf_counter()
    plan = restricted.mvp_plan(storage=args.storage)
    plan_seconds = time.perf_counter() - plan_start
    mvp = _callback_mvp(plan, sector.dimension)

    state = spinful_neel_state(args.rows, args.cols, sector=sector)
    direct_error = _direct_callback_error(plan, state)
    jax_state = jnp.asarray(state)
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
            jax_state,
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
        "implementation": "charge_restricted_fermion_native_mvp_pure_callback",
        "initial_state": "spinful_neel",
        "backend": "tensorcircuit_ng_jax_with_host_callback",
        "jax": jax.__version__,
        "jax_platform": jax.default_backend(),
        "rows": args.rows,
        "cols": args.cols,
        "sites": sites,
        "nqubits": info["n_modes"],
        "full_dimension": info["dimension"],
        "sector_dimension": sector.dimension,
        "sector_reduction": info["dimension"] / sector.dimension,
        "particle_number": sites,
        "two_sz": 0,
        "particles_up": sites // 2,
        "particles_down": sites // 2,
        "bonds": info["bonds"],
        "hopping": args.hopping,
        "interaction": args.interaction,
        "storage": args.storage,
        "times": list(args.times),
        "subspace_dimension": args.subspace_dimension,
        "chebyshev_terms": chebyshev_terms,
        "bessel_iterations": bessel_iterations,
        "repeats": args.repeats,
        "operator_term_count": operator.term_count,
        "operator_build_seconds": operator_seconds,
        "restriction_seconds": restrict_seconds,
        "native_plan_seconds": plan_seconds,
        "native_variant": plan.native_variant,
        "execution_family": plan.execution_family,
        "transition_count": plan.transition_count,
        "estimated_retained_bytes": plan.estimated_retained_bytes,
        "direct_callback_max_abs_error": direct_error,
        "state_bytes": int(state.nbytes),
        "results": results,
        "package": getattr(tcp, "__version__", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--interaction", type=float, default=4.0)
    parser.add_argument("--storage", choices=("lazy", "eager"), default="eager")
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
