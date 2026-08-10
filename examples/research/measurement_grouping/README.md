# Measurement grouping: TenCirPauli, a NumPy/Python baseline, and Qiskit

This study measures the measurement-ready qubit-wise commuting (QWC) path: deterministic grouping, group construction, and reconstruction of Pauli eigenvalues from rotated computational-basis bitstrings. It uses a 64-qubit local Pauli workload with 1,024 unique terms, support width at most 4, and 1,024 synthetic shots. The workload is generated once from a fixed seed and passed to every implementation.

TenCirPauli uses `PauliOperator.group_commuting(mode="qubit_wise", algorithm="largest_first")` and its native `QWCGroupingResult.reconstruct()` path. The independent baseline uses the same largest-first graph-coloring rule with a NumPy compatibility matrix and Python group assembly. Qiskit is an optional external comparison using `SparsePauliOp.group_commuting(qubit_wise=True)`; its coloring strategy is independent, so group membership need not match TenCirPauli. Correctness therefore checks full term coverage, QWC validity, and reconstruction values rather than requiring identical group partitions.

The benchmark reports grouping-only and end-to-end construction-plus-grouping times, group counts, largest group size, and reconstruction time for all groups. The TenCirPauli runner also verifies that its grouping matches the independent largest-first reference exactly on the canonical term order. It does not compare general commuting groups: those are algebraically valid but currently marked `measurement_ready=False` by the public API.

Run the native grouping and independent baseline after installing the local package in release mode:

```bash
conda run -p .conda python examples/research/measurement_grouping/run_tencirpauli.py
```

Qiskit is optional and belongs in a comparison environment. For example, if it is available in the `2602` Conda environment:

```bash
QISKIT_PYTHON=/opt/anaconda3/envs/2602/bin/python \
conda run -p .conda python examples/research/measurement_grouping/compare.py \
  --qiskit-python "$QISKIT_PYTHON"
```

Use `--term-count 2048` for a larger graph, or change `--support-width` to study denser Pauli supports. The scripts print JSON and do not write machine-specific results into the repository.
