# 2D Fermi–Hubbard time evolution: native, JAX callback, and JAX backend MVPs

This study compares two matrix-free time-evolution routes for the same open-boundary spinful Fermi–Hubbard Hamiltonian on a two-dimensional rectangular lattice. The default workload is a `3×3` square lattice with 9 sites, 18 spin-orbitals, 57 canonical raw Fermion terms, and a full-Fock state dimension of `2**18 = 262,144`.

The native route keeps raw creation/annihilation descriptors and uses `FermionOperator.compile("native_mvp")`. TensorCircuit-NG's NumPy Krylov/Chebyshev solver calls the Rust plan directly on every MVP. The callback route uses the same Rust plan behind `jax.pure_callback`, so the time-evolution recurrence is staged with JAX `jit` while each MVP executes on the host. The backend route uses `FermionOperator.compile("backend_mvp", mapping="jordan_wigner")`, constructs a Pauli backend plan, and runs both the recurrence and MVP through JAX. The JAX first call includes tracing and XLA compilation; the warm timing is measured after compilation.

The three routes have the same Fock-space semantics and are compared with the same Néel product state, lattice, coefficients, times, Krylov subspace dimension, and Chebyshev scaling bound. The full-space paths encode the Néel occupation as a one-hot Fock vector; the restricted path maps the same occupation tuple with `ChargeSector.rank()` before creating its one-hot sector vector. Chebyshev `k` and Bessel-estimation `M` are selected by TensorCircuit-NG's `estimate_k()` and `estimate_M()` helpers from the conservative spectral bound; they are not hand-tuned per implementation. The study reports operator/setup time, first evolution time, warm evolution time, mapped term count, state size, and peak RSS. The callback path is forward-only: `pure_callback` makes the native MVP callable from a JAX graph, but it remains a host-side opaque operation and is not an XLA-native kernel.

The symmetry-restricted companion fixes both total particle number and `2*Sz` for a half-filled `3×4` lattice. It uses the exact sector dimension `C(12, 6)**2 = 853,776` instead of the full `2**24 = 16,777,216` Fock dimension. This path is intentionally restricted-native only: the native charge-sector MVP is wrapped by `jax.pure_callback`, while TensorCircuit-NG JAX compiles the full Krylov or Chebyshev recurrence. The default `storage="eager"` precomputes the restricted CSR transition graph once, trading about 388 MiB of retained graph storage and a few seconds of setup for much faster repeated MVPs. Pass `--storage lazy` to measure the compact on-demand path.

Run both solvers on the default `3×3` workload:

```bash
conda run -p .conda python examples/research/fermion_timeevolution_2d/compare.py
```

Run the smaller `2×2` correctness-scale workload:

```bash
conda run -p .conda python examples/research/fermion_timeevolution_2d/compare.py --rows 2 --cols 2 --repeats 5
```

Try a larger rectangular pressure test only after checking the state memory (`3×4` has 24 spin-orbitals and a `2**24` state):

```bash
conda run -p .conda python examples/research/fermion_timeevolution_2d/compare.py --rows 3 --cols 4 --solvers krylov --repeats 1
```

Run the symmetry-restricted `3×4` evolution with both JAX-JIT algorithms:

```bash
conda run -p .conda python examples/research/fermion_timeevolution_2d/run_symmetry_callback.py
```

The JAX path currently defaults to Jordan–Wigner; pass `--mapping parity` or `--mapping bravyi_kitaev` to compare another mapped backend plan. Results are printed as JSON and are not written into the repository.
