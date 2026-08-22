# Local Benchmarks

TenCirPauli benchmarks are intentionally local and informational. They do not run as CI performance gates. The committed benchmark code defines stable workloads, while machine-specific measurements live under the ignored `.benchmarks/` directory.

The standard automated benchmark suite lives under `benchmarks/python/`. The algebra suites include eager Pauli BCH, native-backed lazy Pauli BCH, native Fermion/Boson BCH, plain export, explicit Python term materialization, and eager/materialized Structured BCH as separate cases. The handle-boundary suite records flat construction, handle-native mapping and conversion, and dense/COO/CSR/native-MVP terminal compilation with input/output term metadata. Scripts under `benchmarks/manual/` are opt-in workloads that may require an external TensorCircuit checkout, a release build, or a deliberately chosen large input; they are not part of the default pytest collection or commit smoke check.

The static stabilizer-code suite is deliberately a compact acceptance matrix rather than a family-by-size Cartesian product. `python benchmarks/run.py record --suite stabilizer_code` measures 21 Rust Criterion rows and 22 Python/FFI rows across representative 64-, 256-, and 1024-qubit inputs: small construction and syndrome, medium redundant/CSS classification and correction, and one large generic construction/classification/error-set endpoint. The matrix retains packed and unpacked paths, all three correction residual kinds, all three error-set grouping regimes, source-tree provenance, and exact-output metadata. Its bounded sampling profile is 10 Criterion samples with 250 ms measurement and 75 ms warmup per Rust row, plus 250 ms maximum and three minimum rounds per Python row; the first recorded run completed in about 20.9 seconds including release build/setup. This is the full QEC acceptance record and is hard-bounded far below the ten-minute budget. For daily regression checks, append `--quick`; it uses 100 ms per layer and completed the same 22-row Python/21-row Rust matrix in about 12.4 seconds on the reference machine. Do not expand this suite by adding every family at every width; add a row only when it covers a distinct algorithmic branch that the compact matrix cannot exercise.

The classical-shadow suite is `benchmarks/python/test_shadows_benchmark.py`. It records public sampling and estimator paths for dense statevectors and stabilizer tableaux, local-Pauli and global-Clifford protocols, repeated outcomes, batched observables, RDM, Rényi-2 entropy, and fidelity post-processing. The standalone research runner is `examples/research/classical_shadows/compare.py`; it keeps TensorCircuit-NG comparison, dense-target fidelity, and native-only stabilizer workloads separate.

The current reference release-mode record for the research runner uses 20 qubits, 8,192 settings, one repeat, JAX CPU batches of 64, and `RAYON_NUM_THREADS=4` on Apple arm64/macOS 15.3.1. With TenCirPauli 0.5.0, TensorCircuit-NG 1.9.1, JAX 0.10.2, Python 3.11.15, and seed `20260820`, local-Pauli sampling was 19.73 s (2.409 ms/shot) versus 77.56 s (9.468 ms/shot) for TensorCircuit/JAX; the native estimator was 67.2 μs versus 1.855 ms on the same imported snapshots. Native global-Clifford sampling was 106.78 s (13.034 ms/shot), and dense-target fidelity post-processing was 28.54 s. These are informational single-machine values; the detailed workload and exact outputs are recorded in `examples/research/classical_shadows/README.md`, while raw JSON remains untracked.

The actual-angle circuit boundary benchmark is `benchmarks/manual/circuit_differentiation_ab.py`. It records construction, private plan creation, public forward expectation, public occurrence-space native value-and-gradient, the private native endpoint, first execution, synchronized warm JAX `jit(value_and_grad)` for Propagation/U1/SPPS, and an applicable TensorCircuit/JAX circuit baseline. JAX timing synchronizes every value-and-gradient leaf; SPPS records observable terms, angle count, sample budget, gradient workspace, callback bytes, and process peak RSS. Its forward tape uses zero runtime parameters while its gradient tape uses one private slot per angle occurrence, making the performance boundary explicit.

The formal U1 parameterized-circuit cases are in `benchmarks/python/test_u1_circuit_benchmark.py`. The 16-qubit, four-particle, three-layer workload records native and JAX first/steady forward expectation plus value-and-gradient timings, and checks every result against the native numerical oracle. Native forward calls construct a fresh circuit per timed invocation so facade state caching cannot be mistaken for execution throughput.

The restricted-charge benchmark suite is `benchmarks/python/test_charge_factorization_benchmark.py`. It records public construction, caller-owned `apply_into`, and direct eager construction for representative Fermion, Qubit, Qudit, Boson, Hybrid, and coupled-constraint cases. `benchmarks/manual/factorized_charge_matrix.py` adds one coarse public sample of sector setup, lazy setup, both apply forms, direct eager construction, CSR read-back, retained bytes, representation metadata, and a numerical checksum for the same stable matrix. Its comparison command reads an already saved pytest-benchmark JSON, measures only the current checkout once, then writes a machine-readable per-case report. The restricted-charge comparison fails when an existing matching case is slower than the selected threshold (1.15× by default for regressions of at least 1 ms, or 1.30× for smaller regressions), so timing regressions cannot be hidden in a manually inspected table.

The isolated QuSpin capability matrix is `benchmarks/manual/quspin_capability_matrix.py`. It launches environment-specific TenCirPauli and QuSpin children, performs a separate canonical occupation-order numerical preflight before any timing child runs, then uses ABBA order for timing. Bounded workloads compare dense actions; larger throughput workloads compare one deterministic vector action to avoid materializing a dense matrix. The runner records construction/RSS/thread metadata and keeps TenCirPauli caller-owned `apply_into` separate from the allocation-inclusive public comparison. The committed manifests under `benchmarks/workloads/quspin/` cover full and fixed-Hamming spin, Fermion/Boson, mixed Fermion-Boson, fixed-sector spectators, independent/reversed Product factors, and explicit no-fair-baseline rows for coupled charge and finite Weyl semantics. Run it only against release builds, for example `QUSPIN_PYTHON=/path/to/quspin/python .conda/bin/python benchmarks/manual/quspin_capability_matrix.py --all --tencirpauli-python .conda/bin/python --quspin-python "$QUSPIN_PYTHON" --threads 1 4 --output .benchmarks/quspin-matrix.json`; raw result files remain local and must not be committed.

`benchmarks/manual/fermion_mask_phase_profile.py --throughput` is the matched raw-Fermion versus explicit Jordan--Wigner diagnostic. It uses 18 modes (`D=262,144`), two untimed warmups, and at least seven steady caller-owned samples, so it is suitable for millisecond-scale kernel conclusions rather than microsecond API preflight measurements.

`benchmarks/manual/fermion_mapping_topology_profile.py` extends that diagnostic across a local chain, long-range hopping, and nonlocal quartic pair transfer at the same 18-mode dimension. It reports the actual raw and Jordan--Wigner schedule counts before comparing construction and caller-owned application; do not infer the winner from the geometric range of a fermionic term alone.

## Setup

Install the benchmark extra in the active development environment, then make sure Rust, Cargo, and maturin come from the same selected toolchain:

```bash
python -m pip install -e ".[benchmark]"
```

## Record a run

Record both the Rust Criterion microbenchmarks and Python/PyO3 integration benchmarks:

```bash
python benchmarks/run.py record
```

The generated label combines UTC time, the current Git commit, and a `dirty` suffix when the worktree is not clean. Use `--label NAME` to choose a stable name and `--suite rust` or `--suite python` to run only one layer.

The runner stores Criterion baselines, pytest-benchmark JSON, and a metadata manifest containing the commit, dirty state, platform, and tool versions. It deliberately omits usernames, hostnames, and absolute project paths.

The repository pre-commit hook runs `python scripts/check.py --benchmark smoke` by default, after formatting, linting, typing, correctness tests, and doctests pass. Set `TENCIRPAULI_PRE_COMMIT_BENCHMARK=record` when a full benchmark record is wanted, or use `python scripts/check.py --benchmark skip` when benchmarking is intentionally handled separately. Restricted-charge records retain their benchmark identity and input metadata so that saved values remain comparable across implementation changes.

当前面向科学计算用户的性能覆盖包括：创建和检查 Pauli 项（权重、支集、对易性、乘法）；把大量 Pauli 项合并成确定性的 Pauli Hamiltonian；把 Hamiltonian 生成 dense、COO 或 CSR 矩阵；不生成矩阵而直接计算 `H|ψ⟩`；重复使用同一个 Hamiltonian 的 native MVP plan；把 Pauli 项分成可共同测量的 QWC groups；Z2 analysis/tapering setup；U1 restriction setup、restricted MVP steady apply 和 CSR materialization；以及在 TensorCircuit/JAX 中执行 Hamiltonian MVP 和 sparse matrix。当前还覆盖了 20-qubit 的随机与局域 Heisenberg Hamiltonian/MVP、26-qubit full-space native MVP（约 1 GiB complex128 statevector）、26-qubit `k=2` restricted MVP、16 GiB 默认预算下的显式 materialization，以及真正超过默认预算时的明确拒绝。deterministic propagation 还覆盖 GateTape setup、Rust scalar first/steady expectation、operator materialization、100-qubit Clifford/near-Clifford tape、128-qubit/12-layer native-only near-Clifford scaling、2D Heisenberg rotations、custom PTM 和同步的 matched complex128 JAX warm reference。general-commuting 分组、完整 operator algebra 吞吐、backend plan 的实际执行和更大规模 propagation 仍是后续 benchmark 候选，不把它们写成已经有性能结论的功能。
当前面向科学计算用户的性能覆盖包括：创建和检查 Pauli 项（权重、支集、对易性、乘法）；把大量 Pauli 项合并成确定性的 Pauli Hamiltonian；把 Hamiltonian 生成 dense、COO 或 CSR 矩阵；不生成矩阵而直接计算 `H|ψ⟩`；重复使用同一个 Hamiltonian 的 native MVP plan；把 Pauli 项分成可共同测量的 QWC groups；Z2 analysis/tapering setup；U1 restriction setup、restricted MVP steady apply 和 CSR materialization；以及在 TensorCircuit/JAX 中执行 Hamiltonian MVP 和 sparse matrix。当前还覆盖了 20-qubit 的随机与局域 Heisenberg Hamiltonian/MVP、26-qubit full-space native MVP（约 1 GiB complex128 statevector）、26-qubit `k=2` restricted MVP、16 GiB 默认预算下的显式 materialization，以及真正超过默认预算时的明确拒绝。deterministic propagation 还覆盖 GateTape setup、Rust scalar first/steady expectation、operator materialization、100-qubit Clifford/near-Clifford tape、128-qubit/12-layer native-only near-Clifford scaling、2D Heisenberg rotations、custom PTM 和同步的 matched complex128 JAX warm reference。stochastic propagation 新增 deterministic value-and-gradient 的 checkpoint/gradient-length cases、SPPS 的 12q/100q fixed-budget、adaptive、zero-factor 与 sample-budget scaling cases，并在可选 TensorCircuit 环境中从 Python 端对照 `examples/spps_pauli_path_vqe.py` 和 `PauliPropagationEngine` + JAX autodiff。general-commuting 分组、完整 operator algebra 吞吐、backend plan 的实际执行和更大规模 propagation 仍是后续 benchmark 候选，不把它们写成已经有性能结论的功能。
新增的 symmetry-aware JAX benchmark 使用相同 fixed-Hamming-weight transition table 和相同 tapered Pauli semantics，分别报告 setup、JIT steady apply、first-call/end-to-end 和 Rust native 结果；它不把对称性降维后的 Rust 结果与 full-space JAX 结果做不公平比较。

structured benchmarks also cover direct uniform-Weyl qudit Hamiltonians at dimensions 3, 5, and 7, timing COO, CSR, and native MVP construction with dense reconstruction or MVP differential checks. The same suite now measures Jordan–Wigner, parity, and Bravyi–Kitaev fermion mappings separately for COO, CSR, and native MVP compilation on a bounded Hubbard workload; each mapping is validated against its own dense encoded-basis result.

stochastic propagation 大规模 matched runner 位于 `benchmarks/manual/large_tensorcircuit_compare.py`，默认运行 12q/16q 的 TensorCircuit/JAX 与 native 对照；可用 `--only deterministic --case 20,3,256`、`--only deterministic --case 24,4,128` 和 `--only deterministic --case 32,3,128` 扩展 deterministic locality-3 workload，用 `--native-only --only spps --case 20,3,1024` 测量更高 SPPS budget，并用 `--only spps --case 20,1,64` 测量 TensorCircuit 示例仍可完成编译的较大 SPPS case。输出同时报告 first-call（包含 JAX compile）和 steady median；SPPS 的 Rust seed 与 TensorCircuit 外部 uniforms 不同，因此随机 value 不作逐样本相等断言。`test_propagation_engines_benchmark.py` 另外保留代表性的 `max_weight=2/4` 和 rotation-heavy SPPS cases，不展开完整参数笛卡尔积。

symmetry 的 Rust Criterion suite 同时覆盖 Z2 analysis/taper transform、U1 restriction setup、低粒子数与 central-sector restricted MVP，以及 CSR materialization；Python suite 记录相同 workload 的 public/FFI boundary、restricted output storage 和 central-sector scaling。运行前可显式设置 `RAYON_NUM_THREADS=1` 或目标线程数，runner 会把该配置写入本地 manifest；JAX setup 会在计时 callable 内同步 device arrays。

## Compare with an earlier run

List available local records and compare the current checkout with one baseline:

```bash
python benchmarks/run.py list
python benchmarks/run.py compare <baseline-label>
```

For restricted-charge acceptance, select that suite explicitly:

```bash
python benchmarks/run.py compare <baseline-label> --suite charge_factorization --threshold 1.15
```

The command consumes the persisted baseline below `.benchmarks/python/`; it does not check out or rerun the old source. It saves the fresh current result and a deterministic JSON report below `.benchmarks/runs/`, while the console prints only the coarse summary and rows needing attention. `BLOCK` and `MISSING_CURRENT` cause the restricted-charge command to return non-zero. At or above 1 ms absolute regression, the default block threshold is 1.15×. Below 1 ms, the relaxed but still enforced threshold is 1.30× (`--small-regression-threshold`): a microsecond-scale change is not ignored merely because its absolute duration is small. A completed current record can be re-compared without timing again through `--current <label>`. Benchmark names remain stable; the small, documented migration map in the runner is reserved for an explicitly equivalent rename and keeps the original history comparable. Existing pytest-benchmark files from before manifests can be compared directly by supplying `--suite`, or registered once with `python benchmarks/run.py import-baseline <baseline-label> --suite charge_factorization`.

Criterion continues to print statistical change estimates and write its HTML report below `.benchmarks/rust-target/criterion/`.

## Measurement discipline

- Compare runs on the same machine, power mode, toolchain, feature set, and thread configuration.
- Close competing CPU-heavy processes and allow the machine to reach a stable thermal state.
- Use one current suite run per comparison; pytest-benchmark performs its own internal sampling. Do not add an outer repeat matrix. When an isolated diagnosis is genuinely needed, record one extra same-workload run and inspect its coarse threshold outcome, absolute time, throughput, and memory together.
- Keep benchmark names and workload semantics stable. Add a new case instead of silently changing an old baseline.
- Validate outputs outside the timed region and use release builds for all performance claims.
- Do not commit `.benchmarks/`; archive it separately if long-term machine history matters.
