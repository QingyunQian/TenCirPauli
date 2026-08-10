# SYK Majorana MVP study

This study builds a real, quartic Sachdev–Ye–Kitaev Hamiltonian directly in the Majorana algebra, maps it to qubits in one batched operation, and computes its lowest eigenvalue with SciPy's Lanczos solver. The default `matrix_free` mode compares TenCirPauli's reusable `native_mvp` plan with QuSpin's `quantum_LinearOperator`, so neither side constructs a Hamiltonian matrix. The optional `explicit` mode constructs CSR matrices in both environments and compares that representation instead.

The default case uses `N = 2 * n_modes = 24` Majorana generators and a 4096-dimensional qubit state space:

```bash
conda run -p .conda python examples/research/syk_majorana/run_tencirpauli.py
```

Use a larger or smaller Majorana system explicitly, and choose a different fermion-to-qubit mapping when comparing setup costs:

```bash
conda run -p .conda python examples/research/syk_majorana/run_tencirpauli.py --n-modes 12 --mapping bravyi_kitaev
```

QuSpin is intentionally kept in a separate environment. Run the matched matrix-free workload with the same `--n-modes`, `--coupling`, and `--seed`:

```bash
conda run -n quspin python examples/research/syk_majorana/run_quspin.py --n-modes 10
conda run -n quspin python examples/research/syk_majorana/run_quspin.py --n-modes 12
```

For a matched explicit-CSR comparison, use the same mode on both sides. Explicit TenCirPauli construction may require a larger memory budget than the native MVP default:

```bash
conda run -p .conda python examples/research/syk_majorana/run_tencirpauli.py --mode explicit --max-bytes $((16*1024**3))
conda run -n quspin python examples/research/syk_majorana/run_quspin.py --mode explicit
```

The TenCirPauli and QuSpin outputs use the same full Fock-space dimension `2**n_modes`, random couplings, and SciPy `eigsh` interface. In `matrix_free` mode, compare native plan construction, first MVP, steady MVP, and the complete ground-state solve. In `explicit` mode, compare CSR construction, CSR MVP, and the complete solve. Do not compare a matrix-free record from one mode with an explicit record from the other. The two processes should be run separately because their environments and dependency stacks are intentionally different.

The JSON result separates random-coupling generation, Majorana canonicalization, mapped setup, first and steady MVP application, and the ground-state solve. The native matrix-free script uses `storage="lazy"` by default; pass `--storage eager` to retain the grouped X-mask diagonals for a repeated-MVP comparison within the native family. `setup_end_to_end_seconds` covers workload generation through the first MVP, while `ground_state_end_to_end_seconds` covers the same setup plus `eigsh`; these are the matched end-to-end fields within a selected mode. It also reports the residual `||H|psi> - E|psi>||` and the retained plan metadata. The convention is `H = i^(q/2) sum J_abcd chi_a chi_b chi_c chi_d` with `q=4`, `chi_a^2=1`, and `Var(J_abcd) = 3! J^2 / N^3`.
