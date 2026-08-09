# Phase 10.5 acceptance evidence

This document records the current release-mode evidence for the structured-operator execution work. D29 remains the historical third-round decision; D30 records the fourth-round containment, Product public-index remediation, allocation evidence, current native-variant matrix, and renewed primary lazy measurement. Unsupported coupled constraints remain explicitly `generic` by design; clean-commit provenance is intentionally excluded from this technical acceptance record and may be supplied by the owner as a separate repository operation.

## Fifth-round remediation evidence

The fifth-round Product remediation compiles active/inactive compact-factor shapes per schedule group. A new structural differential expands an inactive fixed-weight factor from 2 to 4 to 6 qubits: dimensions grow from 4 to 12 to 40, while factor-action, group, and aggregate counts remain constant, only identity-segment applications grow, and every Product result equals its independent dense reference. The Product plan retains bounded factor metadata, emits zero fallback/global-ranker/destination-allocation counters, and uses direct contiguous suffix segments where public order permits.

The pure raw-Fermion contiguous binary path now computes packed source words once per source cache and reuses the shared descriptor for schedule members. The `fermion-mask-phase-v2` release record (`c50038bac50d-dirty-896134efcce85367`) has raw/mapped error `2.20e-14`, 4,096 packed-source units versus 139,264 member evaluations, and a default-thread median of `0.128 ms` raw versus `0.098 ms` explicit packed Pauli. The equivalent one-thread medians are `0.354 ms` and `0.243 ms`. This replaces the prior approximately 20x raw-path regression without reintroducing Jordan--Wigner execution or a duplicate sign implementation.

## Fourth-round remediation evidence

The current public selector retains measured capability-class winners before attempting the general Product plan: compact Qubit, Qudit, Boson, and balanced spinful Fermion layouts select their specialized native variants, including balanced Fermion sectors with Cartesian Boson spectators. The manual `product_variant_matrix.py` release record covers Generic, forced Product, and each eligible specialized variant across Qubit, Qudit, Boson, pure/mixed Fermion, two-factor, reversed-factor, and interleaved-axis cases. Every child asserts its actual `native_variant`, uses a dense numerical oracle, reports construction/first/seven-steady applies, retained bytes, RSS, counters, commands, thread environment, manifest version, and a commit-plus-dirty worktree fingerprint.

`FastProductMvpPlan` now uses factor-local prefix-completion metadata to recover public mixed-radix rank. It stores no selected-space-sized public/component permutation and does not invoke the global sector ranker in the apply loop. Structural regressions exercise independent two-factor, reversed, and interleaved Product cases; they require positive factor/group activity and zero Product ranker, fallback, and destination-key-allocation counts. A 512-level identity spectator regression guards the retained Product plan against reintroducing `O(D)` public-index storage.

Allocator-backed core regressions cover one-word and three-word `FermionMaskDescriptor` inner loops, constant-per-call compiled Fermion apply allocation, and repeated scalar/wide accumulator clear/insert/drain cycles. The current raw 12-mode Fermion profile record (`fermion-mask-phase-v1`, `c50038bac50d-dirty-86200bc8938d6843`, one thread) reports raw-versus-explicit-Jordan--Wigner action error `2.20e-14`, raw median `4.69 ms`, mapped median `0.242 ms`, and batched raw descriptor shares of 70.3% packed extraction, 6.5% parity, 21.4% index update, and 1.8% input/output. This explains the remaining raw/mapped gap without restoring a duplicate sign kernel.

The fresh coupled Generic profile (`generic-wide-profile-v1`, the same worktree fingerprint, one thread) reports scalar inserts `(99, 0, 0, 0, 10, 55, 3, 0)` and 65-axis wide inserts `(0, 2112, 0, 0, 10, 726, 3, 0)`. The zero fields are, respectively, capacity growth, owned keys, and post-publication destination-key allocation; lazy/eager actions agree exactly. `csr_chunk_scaling.py` now runs each capacity in a fresh child process and checks both authoritative CSR arrays and MVP action against its one-chunk reference; the `1/8/64/512` sweep passes with 1,120 sorted/merged entries and 2,240 heap operations in every record.

## Isolated QuSpin environment

The repository provides `benchmarks/environments/quspin.yml` with pinned Python, NumPy, SciPy, QuSpin, and QuSpin extension versions. The local acceptance run used the existing dedicated QuSpin Conda environment, which reports QuSpin `0.3.7` from package metadata and `quspin==1.0.0` from pip metadata. The parent runner launches TenCirPauli and QuSpin as separate subprocesses and exchanges only the frozen workload parameters and flat JSON records.

The small independent preflight used two sites, cutoff one, one particle per spin, and complex128. Matrix-free and explicit CSR QuSpin runs both matched the TenCirPauli dimension `16`; the ground-energy error was below `5e-15` in both paths and both residuals were below `9e-10`.

A post-Product-change matrix-free smoke through `benchmarks/manual/compare_quspin.py` used the same two-site workload, seven timed samples, two untimed warmups, and one thread. Dimensions matched (`16`); the TenCirPauli Product plan median was `14.5 us` and the isolated QuSpin median was `0.923 ms`. This small point is protocol evidence only, not a replacement for the large acceptance medians below.

## Product correctness

The native Product enum has compact Fermion and factor-component descriptors for independent compact constraints. Product construction builds factor-local ranks, prefix-completion public-index metadata, flat factor-rank transition tables, spectator digit tables, and shared FermionMask descriptors without retaining a full selected-space permutation. Apply evaluates factor actions outside the spectator traversal and never calls the global sector ranker; it no longer interprets raw charge terms per source. Fermion grouping explicitly supports both grouped-spin and site-interleaved mode order. The Qudit path retains source-dependent modular Weyl phases and uses the exact factor charge ranker; the Boson path uses the same factor executor with ladder amplitudes and finite-cutoff boundaries. Same-domain zero-charge spectators are lowered through active public-axis maps. Native metadata is authoritative: coupled constraints for which no Product descriptor is constructed are reported as `generic`.

Independent full-dense differential tests cover Qubit, Qudit, and Boson factors with Cartesian and same-domain zero-charge spectators, including caller-owned `apply_into(max_bytes=0)`. Independent two-constraint Product and coupled-charge Generic regressions now exercise the native family selector. Exact cancellation-before-leakage is checked for Product lazy and eager paths. Site-interleaved Holstein dense checks for two and three sites pass with maximum absolute error below `5e-13`. Existing Fermion, Boson, Qubit cancellation, Qudit wrap, leakage, eager, COO, and CSR tests remain passing.

The release runner `benchmarks/manual/product_lazy_precompile_ab.py` records a same-workload A/B at four spinful sites, cutoff two, one particle per spin, and seven steady repetitions. The forced Generic term-interpreter baseline measured median steady `apply_into` `0.780 ms`, while the Product factor-local descriptor path measured `0.064 ms`; first applies were `0.758 ms` and `0.115 ms`, respectively. Construction medians were `0.296 ms` and `0.273 ms`, dimensions were `1296` in both cases, and descriptor retained bytes were `16,720` (Generic baseline `15,468`). Both outputs had the same maximum magnitude (`3,641.600045870496`); this is a structural A/B diagnostic rather than a fixed speed gate or a runtime fallback policy.

## Third-round remediation evidence

`FastProductMvpPlan` compiles coefficient-free `ProductScheduleGroup` fingerprints with canonical member order, evaluates factor transitions before the spectator traversal, and uses inverse transitions for destination gather. Its counters are ordered as factor evaluations, destination applications, destination-key allocation, fallback invocation, ranker invocation, group evaluations, and aggregate insertions. The general Product executor is kept measurable through the variant matrix but is not allowed to displace a materially faster specialized winner.

The specialized pure-Fermion executor is retained by an explicit measured decision rather than being mislabeled as the general Product kernel: the matched release A/B measured approximately `9.11 ms` and `8.25 MiB` for Product versus `0.317 ms` and `105 KiB` for the specialized Fermion plan. Mixed Fermion×Boson sectors continue to use the shared Product descriptor. Benchmark records therefore report both the coarse `execution_family` and the authoritative `native_variant`; `execution_family="product"` alone is not treated as proof that the general Product implementation ran.

Structured compiled plans group schedules by active axes and use a suffix stride for contiguous spectator blocks. The shared `FermionMaskDescriptor` receives caller-owned one-word or multiword scratch, and the legacy raw Fermion operation interpreter is absent from the production path; the diagnostic legacy-invocation counter remains zero in the structured numerical matrix. Pure binary Structured plans report the packed native Pauli variant from the native plan itself. The fixed-width Generic charge path reports zero destination-key allocations after construction in both scalar and wide-layout regressions.

The release `csr_chunk_scaling.py` sweep produced 1, 8, 63, and 374 actual chunks for the same 1,120-transition workload. Sorted and merged entries remained 1,120 at every setting, heap operations remained 2,240, and output equality was deterministic. The fresh six-site eager profile recorded `D=1,638,400`, `E=22,527,980`, 1,375 chunks, generation `6.577 s`, chunk sorting `0.335 s`, heap merge `0.811 s`, final-array allocation timing `11.8 us`, chunk metadata `77,000 bytes`, retained CSR `553,778,728 bytes`, estimated construction peak `1,635,202,672 bytes`, and peak RSS approximately `1.45 GB`.

## Product thread A/B

The committed runner `benchmarks/manual/product_thread_ab.py` asserts its expected native variant and reports first apply, seven steady applies, retained bytes, peak RSS, explicit threads, command, manifest version, and worktree fingerprint. On the reviewed `c50038bac50d-dirty-86200bc8938d6843` worktree, the bounded five-site interleaved Holstein workload (`D=102,400`) selected the retained `fermion` variant and measured:

| Threads | Median apply | Peak RSS |
| ---: | ---: | ---: |
| 1 | 5.00 ms | 73.8 MB |
| 2 | 2.91 ms | 73.9 MB |
| 4 | 1.69 ms | 80.0 MB |

The primary seven-site lazy workload (`D=20,070,400`) was renewed with seven steady repetitions after two untimed warmups and one thread. The current public route selects `native_variant="fermion"`: TenCirPauli median is `1.534 s`, with 1.45 GB peak RSS and `133,972` retained plan bytes. The matched isolated QuSpin matrix-free run has median `1.730 s` and 2.90 GB peak RSS. The specialized Fermion executor owns disjoint output rows and does not allocate `O(pD)` worker vectors.

## Six-site eager and explicit sparse comparison

The six-site workload has `D=1,638,400` and exact `nnz=22,527,980`. The explicit comparison used the QuSpin `hamiltonian` CSR path and the TenCirPauli eager plan plus one explicit CSR expansion.

| Threads | TenCirPauli steady MVP | QuSpin steady MVP | TenCirPauli CSR bytes | QuSpin CSR bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21.28 ms | 20.55 ms | 553.8 MB | 457.1 MB |
| 4 | 11.64 ms | 20.74 ms | 553.8 MB | 457.1 MB |

The TenCirPauli eager retained graph reports `553,778,728` bytes and an estimated construction peak of `1,635,202,672` bytes. The fresh post-remediation measured TenCirPauli process peak RSS was approximately `1.45 GB`; the earlier matched QuSpin explicit process peak RSS was approximately `8.8 GB`. Ground-energy and residual checks passed on the small explicit preflight. The single-thread explicit speed comparison remains parity-level and is not claimed as a win.

## Eager representation evidence

The bounded runner `benchmarks/manual/representation_ab.py` exercises the currently retained production formats and records first/steady apply, construction, CSR expansion, peak RSS, retained bytes, and maximum apply error versus the emitted CSR. The Pauli XOR capability retains `compact_graph`; Cartesian shift/stride and Product additive-sector capabilities retain CSR under D24. The Pauli sample (`D=256`, eight terms) retained `4,352` estimated bytes versus `8,200` bytes for expanded CSR, with a `1.5–1.7 us` steady compact apply in the latest local runs. The runner includes Qubit (`D=6`), Qudit (`D=3`), Boson (`D=3`), independent two-factor (`D=2`), reversed factor-size Product (`D=6`), compact Fermion Product (`D=4` after adding the total-plus-spin-balance constraint), and an irregular coupled-charge Generic CSR (`D=4`) record. Every record has numerical error zero or below machine precision versus CSR. These records support the D24 owner selection; they do not introduce a runtime representation selector. The runner performs two untimed warmups and seven timed samples. Independent numerical coverage remains in the focused dense differential tests.

## Reproducible tooling

The manual entry points are `benchmarks/manual/compare_quspin.py`, `benchmarks/manual/product_thread_ab.py`, `benchmarks/manual/product_variant_matrix.py`, `benchmarks/manual/fermion_mask_phase_profile.py`, `benchmarks/manual/generic_wide_profile.py`, `benchmarks/manual/representation_ab.py`, `benchmarks/manual/product_u1_16_ab.py`, and `benchmarks/manual/csr_chunk_scaling.py`. The six-site phase profile is emitted by `examples/research/holstein/run_tencirpauli.py` through `construction_profile()`. The frozen Holstein parameter manifest is `benchmarks/workloads/holstein.json`. The QuSpin runner supports both `quantum_LinearOperator` matrix-free execution and explicit CSR execution; unsupported callback-heavy or nonequivalent baselines are not used.

## Project quality gate

The final native build passed `cargo fmt --check`, Clippy with warnings denied, 52 Rust tests, both release `maturin develop --release --locked` and the isolated `maturin develop --release --locked --skip-install` build, Black, Ruff, strict mypy, 416 Python tests, 8 doctests, 3 Rust benchmark smoke suites, and 353 bounded Python benchmark cases. The Python run produced only the existing 57 deprecation warnings for the compatibility `restrict_u1()` API. The local package environment already supplies the required runtime dependency, so both release validation forms are reproducible here.

## Remediation status

The fourth-round F4-1 through F4-7 items are technically closed by D30 in the current worktree: native metadata is authoritative; measured specialized winners are selected before Product; Product uses bounded factor-level public indexing and truthful activity counters; Structured/Fermion and Generic allocation regressions are allocator-backed; current variant, phase, CSR, and primary lazy records include the selected native route; and CSR equality/RSS claims are executed by the cited runner. D24 retains eager-format winners and D22 remains the owner-approved eager parity decision.

## Performance interpretation

The matched six-site eager/explicit single-thread comparison is closed as owner-approved parity under D22: the latest seven-sample medians are approximately `21.70 ms` for TenCirPauli and `21.16 ms` for QuSpin (about `2.6%` slower), numerical preflight passes, and no material memory or scaling regression is observed. This is not a speedup claim. The seven-site lazy gate, interleaved Product differential, memory accounting, and isolated QuSpin tooling remain useful evidence; they do not replace the source-level closure evidence recorded by D29. Deliberately unsupported coupled constraints remain `generic`; raw Fermion monomials use the bounded one-word or multiword mask descriptor.

## Remediation ledger

| Finding | Current state | Source and regression evidence |
| --- | --- | --- |
| R1 | Closed for the supported independent-factor envelope; Product groups coefficient-free transformations, reuses factor actions over spectator segments, and reports zero destination-key allocations in the representative diagnostic. | `python/tencirpauli/charge.py`, `crates/tencir-pauli-core/src/charge.rs`; `product_u1_16_ab.py`, Product dense differentials, caller-owned apply, and counters. |
| R2 | Closed by active-axis/suffix Structured schedules and the shared `FermionMaskDescriptor`; pure binary plans select the actual packed Pauli native variant, while the measured pure-Fermion A/B explicitly retains its faster specialized executor. | `crates/tencir-pauli-core/src/fermion_mask.rs`, `structured.rs`, `charge.rs`; ordered-CAR/mode-128 tests, structured counters, native-variant tests, and release A/B records. |
| R3 | Closed by ordered-row append plus D25's bounded source-major exception; the sink uses deterministic heap k-way merge with cross-chunk coalescing and exact-zero removal, and the required chunk/profile evidence is recorded above. | `crates/tencir-pauli-core/src/csr.rs`, `structured.rs`, `charge.rs`; `csr_chunk_scaling.py`, six-site `construction_profile()`, direct CSR/COO equality, and multi-chunk cancellation regression. |
| R4 | Closed by construction-time compiled schedules and reusable scalar/fixed-width accumulators; raw identifier conversion, descriptor validation, and destination-key allocation are absent from scalable apply loops. | `phase-10.5-trusted-handle-audit.md`, `phase-10.5-owner-decision-log.md` (D29); `tests/test_charge.py`, schedule compilation, conservative preflight, Clippy, and release suite. |
| R5 | Closed by D22/D24 and D29's native-variant/capability evidence; clean repository provenance remains an owner-side repository operation and is not part of this technical decision. | `phase-10.5-owner-decision-log.md` (D22/D24/D29), `representation_ab.py`, `product_u1_16_ab.py`, isolated Holstein/QuSpin runners, and this document. |
