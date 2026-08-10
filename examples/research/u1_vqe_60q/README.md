# 60-qubit, two-particle U(1) VQE comparison

This study compares TensorCircuit's Python `U1Circuit` with JAX `jit(value_and_grad)` against TenCirPauli's native `U1Circuit.value_and_grad`. Both implementations use the same 60-qubit, two-particle initial state, the same deterministic 10-layer circuit, the same 900 independent gate angles, and the same `XX+YY` Hamiltonian.

Each circuit layer applies 30 disjoint periodic-brickwork iSWAP gates followed by 60 independent RZ gates, for 90 parameters per layer. The iSWAP layers alternate `(0,1), (2,3), ...` with `(1,2), (3,4), ..., (59,0)`. The Hamiltonian is the periodic nearest-neighbour operator `H = sum_i (X_i X_{i+1} + Y_i Y_{i+1})`, with indices modulo 60.

The state dimension is `binom(60, 2) = 1770`, so the comparison isolates compressed U(1) circuit execution and automatic differentiation rather than a full `2**60` statevector. The scripts use the same seeded 900-parameter vector and verify the energy and all 900 gradient entries before reporting timings. TensorCircuit evaluates the layer loop with `jax.lax.scan` to keep the compiled graph bounded without changing the independent-parameter workload.

Run the native implementation after a release build:

```bash
conda run -p .conda python examples/research/u1_vqe_60q/run_tencirpauli.py --repeats 7
```

Run TensorCircuit against the local checkout with JAX on CPU:

```bash
PYTHONPATH="/path/to/tensorcircuit:/path/to/TenCirPauli/python" conda run -p .conda python examples/research/u1_vqe_60q/run_tensorcircuit.py --repeats 7
```

Run both children and enforce full energy/gradient agreement:

```bash
conda run -p .conda python examples/research/u1_vqe_60q/compare.py --repeats 7
```

The JSON output separates circuit/setup time, first value-and-gradient time, and synchronized steady-state median. For TensorCircuit, the first call includes JAX tracing and XLA compilation. For TenCirPauli, the first call includes native plan construction. `end_to_end_first_seconds` additionally includes building the corresponding circuit object before the first call.
