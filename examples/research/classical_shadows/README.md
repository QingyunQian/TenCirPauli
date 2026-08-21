# Classical-shadow research comparison

This study measures the public classical-shadow workflow on a non-Clifford dense state. The Pauli path compares TenCirPauli with TensorCircuit-ng, while the global-Clifford path reports TenCirPauli's dense-state sampling and both fidelity post-processing paths separately. The new `Snapshots.fidelity_statevector` result uses an arbitrary dense non-stabilizer target, so both the input and target are outside the stabilizer shortcut; the existing `Snapshots.fidelity` result remains the polynomial stabilizer-target control. The Clifford numbers are deliberately not presented as a cross-library speedup: the public PyClifford implementation found during the survey is a stabilizer-state simulator and its measurement entry point accepts `StabilizerState`, so it is not an equivalent baseline for an arbitrary non-Clifford statevector.

The Pauli comparison separates sampling from estimator execution. TensorCircuit receives fixed setting and status arrays so its measurement workload is reproducible; TenCirPauli generates its own settings internally as required by its public sampling contract. For estimator-only timing, the TensorCircuit outcomes are imported into a TenCirPauli snapshot handle, so both estimators consume the same raw Pauli data. A warm-up call is timed separately, and the reported steady-state timing is the median of the requested repetitions.

The example also demonstrates `Snapshots.sample`, `Snapshots.from_pauli_measurements`, `expectation`, `rdm`, `shadow_bound`, stabilizer-target `fidelity`, and arbitrary-target `fidelity_statevector`. The JSON includes the exact values and a one-repeat normal-error model for the fixed 8,192-shot workload; this is a timing workload, not a high-precision accuracy budget. Smaller setting counts are useful for quick scans. The script emits one JSON record and does not write machine-specific results into the repository.

Run it from the repository root after building the package in release mode:

```bash
PYTHONPATH="$PWD/python:/Users/shixin/Nutstore Files/newwork/quantum-information/codebases/tensorcircuit" .conda/bin/python examples/research/classical_shadows/compare.py
```

The default dense workload is 20 qubits and 8,192 settings with one repeat, so it processes 8,192 shots using the JAX backend in batches of 64. Useful scan options are `--nqubits 16 --nsettings 256`, `--repetitions 7`, and `--tensorcircuit-backend numpy`. Native sampling in this example uses an explicit 16 GiB `max_bytes` budget; independent dense settings run in deterministic memory-bounded worker chunks, and `RAYON_NUM_THREADS` can pin the worker ceiling for reproducible scaling scans. Dense-target fidelity likewise evaluates independent Clifford snapshots in parallel above its small-workload crossover, then performs deterministic block reduction. Override the JAX batch size with `--tensorcircuit-batch-size` when scanning batch-size tradeoffs. The 20-qubit run is a genuine dense-target post-processing workload: the target has `2**20` complex amplitudes and each Clifford snapshot performs a dense overlap, while no full snapshot statevector is retained. The JAX result includes synchronization through `numpy.asarray`; compare cold setup separately if backend compilation is part of the question. Use `--skip-tensorcircuit` when only the native dense/stabilizer Clifford timing is desired.

## Recorded release-mode result

The default command was recorded on 2026-08-21 on an Apple arm64 macOS 15.3.1 host with Python 3.11.15, TenCirPauli 0.5.0, TensorCircuit-NG 1.9.1, JAX 0.10.2 on CPU, `RAYON_NUM_THREADS=4`, seed `20260820`, and one timed repetition. The main workload used 20 qubits, 8,192 settings, one repeat, and therefore 8,192 shots. These are single-machine informational measurements; the JAX sampling row includes synchronized device-to-host conversion and its cold JIT setup was 78.23 s.

| Path | Workload | Median result |
| --- | --- | --- |
| TenCirPauli local-Pauli sampling | Dense non-Clifford state, 20 qubits, 8,192 shots | 19.73 s total, 2.409 ms/shot |
| TensorCircuit/JAX local-Pauli sampling | Same state and shots, batch size 64 | 77.56 s total, 9.468 ms/shot; 3.93× slower |
| Pauli estimator | 20-qubit, 8,192 imported snapshots | TenCirPauli 67.2 μs vs TensorCircuit/JAX 1.855 ms; 27.3× lower |
| Global-Clifford sampling | Dense non-Clifford state, 20 qubits, 8,192 shots | 106.78 s total, 13.034 ms/shot |
| Dense-target fidelity | Same global-Clifford snapshots and a dense 20-qubit target | 28.54 s post-processing; estimate `2.72×10⁻⁴`, exact overlap `1.73×10⁻⁷` |

The same run also sampled a 64-qubit zero stabilizer state with 128 Clifford settings in 19.7 ms, or 153.6 μs per shot. The complete JSON record is intentionally printed to stdout rather than committed; rerun the command above to regenerate it on another machine.

The external Clifford candidate is [PyClifford](https://github.com/hongyehu/PyClifford). Its public research notebook demonstrates global Clifford shadow tomography, but the current repository has API drift between the old `povm` notebook and the current `ClassicalShadow` wrapper. It is therefore not installed or included in this comparison until a pinned, reproducible interface is selected.
