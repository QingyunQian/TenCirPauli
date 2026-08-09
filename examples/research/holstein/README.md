# One-dimensional Holstein electron–phonon study

This manual study builds an open-boundary, spinful one-dimensional Holstein Hamiltonian from the structured fermion–boson API. It combines nearest-neighbor fermionic hopping and on-site interaction with local phonon energy and density–displacement coupling across a configurable chain, then computes the ground state with a native matrix-free plan and SciPy Lanczos.

The default seven-site chain has three maximum phonon quanta per site and a 20,070,400-dimensional fixed-spin mixed fermion–boson sector. Run the compact lazy comparison from the repository root:

```bash
conda run -p .conda python examples/research/holstein/run_tencirpauli.py
```

The sector fixes `N_up=N_down=sites//2`, so its dimension is `binom(sites, particles_per_spin)^2 * (cutoff + 1)^sites`. `storage=lazy` retains only the sector plan and generates transitions during each MVP; `storage=eager` precomputes and retains the complete destination-major CSR transition graph. The eager graph is the reusable native MVP representation, not a Python/SciPy matrix. Change `--sites`, `--cutoff`, `--particles`, `--t`, `--omega`, `--g`, or `--u` for a manual study. For a small independent dense check, use for example:

```bash
conda run -p .conda python examples/research/holstein/run_tencirpauli.py --sites 2 --cutoff 3 --dense-check
```

To compare the two eigensolver representations on a medium-size case, use eager storage and request both paths:

```bash
conda run -p .conda python examples/research/holstein/run_tencirpauli.py --sites 6 --cutoff 3 --storage eager --eigsh --csr-eigsh
```

This reports native eager-`LinearOperator` setup and Lanczos time separately from explicit CSR materialization and SciPy-CSR Lanczos time. They should have the same asymptotic sparse-MVP work and the same eigenvalues, but need not have identical timings: the native path executes the retained CSR graph in Rust, while the explicit path pays Python/NumPy array materialization and uses SciPy's sparse kernel. `--csr-eigsh` requires `--storage eager` because the comparison is specifically between two consumers of the eager graph and temporarily retains an additional Python/SciPy copy. Keep the large seven-site run to MVP measurements unless its estimated memory fits the local budget; adding `--eigsh --csr-eigsh` can require several additional gigabytes for Lanczos work vectors and the explicit CSR copy.

QuSpin can represent this hybrid Hilbert space with `tensor_basis` and `boson_basis_1d`. The companion uses two spinless fermion factors for the two electron spins plus one local boson basis; because every term in this Holstein Hamiltonian is even in each fermion species, it has the same physical Hamiltonian and spectrum as the spinful formulation. The basis ordering is implementation-specific, so raw state vectors are not compared elementwise. QuSpin is intentionally kept in a separate optional environment:

```bash
conda run -n quspin python examples/research/holstein/run_quspin.py
```

The QuSpin script defaults to basis/operator construction and MVP timings; add `--eigsh` when the Lanczos solve is wanted. Run the two commands with the same parameters and thread environment to compare operator setup, first and steady matrix-vector products, and, when requested, the complete Lanczos solve. The scripts report process-local end-to-end timings; Python startup/import overhead is not mixed into the algorithm timings. QuSpin's tensor-product basis ordering is different, so compare dimensions, energies, residuals, and timings rather than raw state-vector entries.
