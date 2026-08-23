# Static stabilizer-code comparison: TenCirPauli and qLDPC

This manual Research Example compares TenCirPauli with qLDPC 0.3.3 on their shared static qubit stabilizer-code workflow: construction, rank and encoded dimension, physical logical operators, original-order batch syndromes, error classification, and correction-residual analysis. qLDPC is a broader QEC workbench with code families, distance search, decoders, circuit generation, qudit codes, and subsystem codes; those capabilities are outside the matched timing boundary here.

The example does not add qLDPC to TenCirPauli's runtime dependencies and intentionally has no `requirements.txt`. `run_tencirpauli.py` uses the public `StabilizerCode` API. `run_qldpc.py` imports qLDPC directly and uses its documented `CSSCode`/`QuditCode`, `rank`, `dimension`, and `get_logical_ops()` interfaces; the visible adapter performs only the shared GF(2) syndrome, classification, and residual products.

The workload owner is [`workloads.py`](workloads.py). It creates identical ordered X/Z symplectic matrices, deterministic error and correction batches, reference logical fixtures, and content hashes without importing either package. The matrix set includes repetition codes, independent five-qubit-code direct sums with a fixed qubit permutation, Steane's code, binary hypergraph products of open-boundary repetition checks, a reproducible sparse random HGP workload `random_hgp_14400`, and a fixed dense general stabilizer search-quality workload `random_general_256`. Redundant-check variants are included for the medium repetition and finite-rate families.

Both implementations use q0-first qubit order and phase-free I/X/Y/Z codes `0/1/2/3`. General checks are represented as `[x | z]`. CSS checks are represented as X-only rows followed by Z-only rows, matching TenCirPauli's public syndrome order; qLDPC's internal field arrays are not used as a substitute for the original ordered input matrix.

Correctness runs before timed runs. It checks matrix hashes and metadata, logical commutation and canonical pairings, equality of stabilizer-plus-logical quotient spans, an invertible GF(2) logical-basis transformation, original-order syndromes, classification kinds, and correction residuals. Raw logical arrays are not required to match because the two packages choose different canonical representatives.

Each timed child reports import-independent warmup, fresh construction-to-logicals, first and steady syndrome/classification/correction timings, round statistics, input/output payload sizes, source paths, thread variables, and process high-water RSS. Metadata is retained for correctness and workload identification but is not a performance comparison. The primary comparison fixes Rayon/OpenMP/OpenBLAS/MKL worker settings to one thread. Peak RSS is a process-level observation, not an exact operation allocation measurement.

The timed operations have distinct meanings. Construction-to-logicals builds a fresh code and materializes the physical logical operators; this is the construction boundary used for comparison because it includes the code-space basis needed by the shared workflow. Syndrome computes which ordered stabilizer checks anticommute with every batch error. Classification computes the syndrome and, for zero-syndrome rows, distinguishes stabilizers from non-trivial logical errors and returns logical coordinates. Correction forms the residual `correction * error` for each pair and classifies that residual; it does not search for a decoder correction.

The independent logical-witness quality benchmark is [`search_benchmark.py`](search_benchmark.py). It is not part of the bounded qLDPC comparison or the repository regression benchmark suite. It builds deterministic HGP and general-code workloads through the public TenCirPauli API, sweeps selected `sketch_rows` and `max_trials`, validates every returned witness, and reports the deterministic incumbent, best weight, improvement amount, first/steady end-to-end time, trials per second, native setup and initialization counters, resolved sketch rows, RSS, and source-tree native provenance. CSS workloads measure `auto`, explicit `x`, explicit `z`, and `general` separately by default, while `--sectors general` restricts a run to the general path. The default profile is capped at 180 seconds and uses HGP-8/16/24/32 with `sketch_rows=32,64` and `max_trials=16,64`; larger HGP-48, `random_hgp_14400`, `random_general_256`, or wider sweeps are opt-in.

The `random_hgp_14400` workload uses a fixed simple `(4,3)`-regular `72×96` binary matrix with full row rank as both classical HGP factors. Its HGP has 14,400 physical qubits and 576 logical qubits; the generated matrix hash is reported in every search record. The distance is not assumed known, so search results are witnessed upper bounds only.

The `random_general_256` workload is the algorithm-quality fixture. It starts from a deterministic random CSS code with parameters `[[256,64,3]]`, deliberately plants a weight-3 logical, and then applies a fixed independent single-qubit Clifford scramble so that the 192 checks are genuinely mixed X/Y/Z general stabilizer rows. Its matrix hash is `ec2129628227ff44f1c161c4bb4d3cc099231f476bf45b4a36280163fff2f827`. An exhaustive check of all 294,528 nonidentity weight-1/2 Paulis found no logical operator, while the search returns a weight-3 logical, so this fixture has exact distance 3. It is intentionally search-only: it demonstrates randomized quality at a moderate width and is not presented as an unbiased code ensemble or a sparse QLDPC family.

Run a bounded research profile with:

```bash
PYTHONPATH="$PWD/python:$PWD/examples/research/stabilizer_code_qldpc" .conda/bin/python examples/research/stabilizer_code_qldpc/search_benchmark.py \
  --workloads hgp_16,hgp_32 --sketch-rows 16,32,64 --trials 16,64,256 \
  --time-budget 180 --output .benchmarks/logical-search.json
```

The resulting JSON is machine-local and should remain under the ignored `.benchmarks/` directory. Use the known HGP repetition distance as a fixture-quality reference only; larger or unfamiliar constructions report witnessed upper bounds, not exact distance claims.

The large `random_hgp_14400` and `random_general_256` workloads are search-only and are intentionally excluded from `compare.py --workloads all`; request them explicitly through `search_benchmark.py`.

For the large random HGP, keep the first scan general-only and bounded:

```bash
PYTHONPATH="$PWD/python:$PWD/examples/research/stabilizer_code_qldpc" .conda/bin/python examples/research/stabilizer_code_qldpc/search_benchmark.py \
  --workloads random_hgp_14400 --sectors general --sketch-rows 16,32,64,128,256 \
  --trials 64,256,1024 --time-budget 180 --output .benchmarks/random-hgp-14400.json
```

Run the quality fixture with a bounded curve and require an actual improvement:

```bash
PYTHONPATH="$PWD/python:$PWD/examples/research/stabilizer_code_qldpc" .conda/bin/python examples/research/stabilizer_code_qldpc/search_benchmark.py \
  --workloads random_general_256 --sectors general --sketch-rows 64,256,320 \
  --trials 1,4,16,64 --time-budget 60 --require-improvement \
  --output .benchmarks/random-general-256.json
```

With matrix seed `20260823`, search seed `11400714819309886402`, and the current release build, the quality curve includes the following deterministic records:

| sketch rows | max trials | initial incumbent | best weight |
| ---: | ---: | ---: | ---: |
| 64 | 64 | 80 | 80 |
| 256 | 64 | 80 | 80 |
| 320 | 1 | 80 | 69 |
| 320 | 4 | 80 | 63 |
| 320 | 16 | 80 | 3 |

Thus the fixture directly distinguishes deterministic initialization from randomized search: the returned weight-3 witness is an upper bound `d ≤ 3`, and the separate exhaustive check establishes that the example's exact distance is 3.

TenCirPauli's logical-basis construction selects the required quotient directions in free-coordinate space and materializes only the `2k` logical quotient vectors; it does not materialize the full normalizer basis. This optimization preserves the deterministic logical representatives and all shared-workflow correctness checks.

The fixture generator uses the public I/X/Y/Z encoding `0/1/2/3`, keeps CSS X-only and Z-only stabilizers separate, and injects reference logical operators into the zero-syndrome error pool. In the HGP-8 and HGP-48 batch-size-256 checks, the resulting classification counts are 86 syndrome-visible, 85 stabilizer, and 85 logical samples.

Build the source-tree native extension in the project environment before running the TenCirPauli side:

```bash
PATH="$PWD/.conda/bin:$PATH" PYTHONPATH="$PWD/python" .conda/bin/maturin develop --release --skip-install
```

Run one side directly:

```bash
PYTHONPATH="$PWD/python" .conda/bin/python examples/research/stabilizer_code_qldpc/run_tencirpauli.py --workload repetition_64 --batch-size 256
<qldpc-python> examples/research/stabilizer_code_qldpc/run_qldpc.py --workload repetition_64 --batch-size 256
```

The selected qLDPC interpreter is only the local comparison environment used for this study. It is not required to install or use TenCirPauli. If qLDPC is unavailable, invoking its runner naturally raises Python's normal `ImportError`.

Run the correctness-first comparison with explicitly selected interpreters:

```bash
QLDPC_PYTHON=/path/to/qldpc/bin/python
PYTHONPATH="$PWD/python" .conda/bin/python examples/research/stabilizer_code_qldpc/compare.py \
  --tencirpauli-python "$PWD/.conda/bin/python" \
  --qldpc-python "$QLDPC_PYTHON" \
  --workloads repetition_64,hgp_8,five_qubit_16 \
  --batch-sizes 1,256,4096 \
  --repetitions 5
```

Use `--workloads all` to run every declared workload. qLDPC's HGP family constructor is checked outside the matched timer against the independent matrix formula; its family-construction time is reported separately and has no TenCirPauli speed ratio. Machine-specific JSON, logs, and profiles should remain in an ignored temporary directory.

The following post-dispatch one-thread local run used Python 3.11.15, TenCirPauli 0.5.0 from the source-tree native extension, Rust 1.97.1, NumPy 2.4.6 and SciPy 1.17.1 for both interpreters, qLDPC 0.3.3 with `galois` 0.4.11 and Stim 1.16.0, three steady rounds at batch size 256, and every declared workload. The ratios are qLDPC/TenCirPauli, so values above one mean that TenCirPauli is faster. The table uses construction-to-logicals as the only construction comparison and includes the CSS-specific X-Z validation; it is a local research record rather than a release claim.

| Workload | Qubits | Checks | Logical qubits | construction→logicals | syndrome | classification | correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repetition_3 | 3 | 2 | 1 | 11.82× | 0.82× | 1.08× | 0.98× |
| five_qubit_1 | 5 | 4 | 1 | 13.95× | 0.90× | 1.20× | 1.12× |
| steane_7 | 7 | 6 | 1 | 17.23× | 1.26× | 1.25× | 1.06× |
| repetition_64 | 64 | 63 | 1 | 72.70× | 7.68× | 5.29× | 4.19× |
| repetition_256 | 256 | 255 | 1 | 39.36× | 28.03× | 23.24× | 17.29× |
| repetition_1024 | 1024 | 1023 | 1 | 16.49× | 50.61× | 48.29× | 41.45× |
| hgp_8 | 113 | 112 | 1 | 52.28× | 13.55× | 10.33× | 8.72× |
| hgp_16 | 481 | 480 | 1 | 22.19× | 42.94× | 36.77× | 30.91× |
| hgp_24 | 1105 | 1104 | 1 | 11.25× | 45.85× | 43.38× | 40.96× |
| hgp_32 | 1985 | 1984 | 1 | 6.92× | 63.63× | 58.98× | 56.95× |
| hgp_48 | 4513 | 4512 | 1 | 4.43× | 63.12× | 61.96× | 61.15× |
| five_qubit_16 | 80 | 64 | 16 | 27.31× | 15.88× | 14.47× | 10.03× |
| five_qubit_64 | 320 | 256 | 64 | 11.62× | 58.47× | 59.53× | 47.05× |
| five_qubit_256 | 1280 | 1024 | 256 | 2.86× | 94.09× | 101.48× | 89.76× |
| repetition_256_redundant | 256 | 280 | 1 | 33.51× | 27.06× | 21.99× | 18.38× |
| five_qubit_64_redundant | 320 | 281 | 64 | 11.39× | 63.19× | 60.85× | 49.69× |

The corrected large HGP cases show the scaling split clearly: at HGP-32, TenCirPauli is about 6.92× faster through logical-basis construction and about 57–64× faster for batch analysis; at HGP-48, it is about 4.43× faster through logical-basis construction and about 61–63× faster for batch analysis. In the same run, process high-water RSS was about 443 MB versus 108 MB for HGP-32 and 785 MB versus 233 MB for HGP-48 for qLDPC versus TenCirPauli. qLDPC's HGP family-generation check is reported separately and is excluded from the matched timer.

These ratios are scoped to the exact local run and to the shared static algebra. They do not compare qLDPC distance algorithms, decoders, Stim circuits, or the broader qLDPC package against unsupported TenCirPauli functionality.
