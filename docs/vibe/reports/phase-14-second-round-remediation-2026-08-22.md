# Phase 14 Second-Round Remediation

Status: implementation and focused verification complete. The owner-selected eager logical-basis design remains accepted, the eager-versus-lazy A/B remains waived for the intended tens-to-thousands-of-qubits workload, and the QEC benchmark protocol is now bounded to a compact matrix whose measured total runtime is well below ten minutes.

## Review disposition

All four second-round findings were reasonable and were implemented. No finding was rejected. The only design item not implemented is lazy logical construction, because the owner decision explicitly selected eager construction and waived that A/B for this workload; the specification and status documents now record that decision instead of treating it as an open blocker.

## SR1 — construction resource estimate

`ConstructionEstimate` now separates the retained-input estimate from the post-rank estimate. The initial advisory check covers retained checks, signs, and statistics. After the actual rank is known, the second check covers the retained relation and eager logical payload. It no longer adds normalizer, quotient, elimination, and projected-row buffers from different construction intervals as if they were simultaneously allocated. Checked arithmetic and the best-effort `max_bytes` policy remain unchanged.

The resource regression tests retain both the zero-budget rejection and two nonzero cases that previously could be rejected by the conservative sum: an empty 64-qubit code under a 3,000-byte advisory limit and a heavily dependent 128-check/64-rank input under a 5,000-byte advisory limit.

## SR2 — independent correctness evidence

The QEC tests now independently bind signed projector trace/rank/action, reject the signed `-I` production case, verify nontrivial projected logical X/Z action and anticommutation on a repetition code, assert fixed canonical word arrays for the five-qubit and Steane codes, and differentially check PauliWord-batch correction residuals against singular classification. These checks bind numeric values and logical action; they do not rely on shape or metadata alone.

## SR3 — bounded benchmark evidence

The previous family-by-size Cartesian product was removed. The compact acceptance matrix deliberately samples one small, one medium, and one large representative workload rather than repeating every operation at every width.

| Layer | Rows | Representative coverage |
| --- | ---: | --- |
| Rust Criterion | 21 | Four construction cases; packed/unpacked mixed classification at 256/256/1024 qubits; packed/unpacked syndrome at 64 qubits; three correction residual kinds in both layouts at 256 qubits; three 1024-qubit error-set grouping regimes |
| Python/FFI | 22 | Four construction cases; the same three-width packed/unpacked classification representatives; packed/unpacked syndrome output at 64 qubits; three correction residual kinds in both layouts at 256 qubits; one PauliWord boundary case; three 1024-qubit error-set grouping regimes |

The default acceptance command is `python benchmarks/run.py record --suite stabilizer_code`. Rust uses 10 Criterion samples, 75 ms warmup, and 250 ms measurement per row. Python uses three minimum rounds, a 250 ms maximum per row, and warmup disabled. The release record `phase14-qec-compact-20260822-r1` completed with status `complete` in approximately 20.9 seconds of wall time including the release build/setup, far below the ten-minute QEC budget. The record contains 22 Python rows, source-tree `tencirpauli.__file__` and `_native.__file__` values, and `max_abs_error=0.0` with explicit singular or fixed-grouping accuracy references.

The daily command is `python benchmarks/run.py record --suite stabilizer_code --quick`. It keeps the same coverage matrix but uses 100 ms measurement windows and completed as `phase14-qec-compact-20260822-quick-r1` in approximately 12.4 seconds on the same machine. Both profiles are explicit so a quick result is not silently compared with a full acceptance result. A new benchmark row requires a distinct algorithmic branch that this compact matrix cannot exercise; a new size alone is not sufficient justification.

## SR4 — flat ABI and helper cleanup

Singular logical coordinates and error-set logical coordinates now use flat `uint8` NumPy arrays, with `None` for absent logical output. The packed and unpacked batch read-back paths have matching tuple arity, dtype, shape, contiguity, and numeric differential tests. The duplicate syndrome helper was removed; the singular path reuses the row-unpacked syndrome primitive.

## Verification

- `conda run -p .conda cargo test -p tencir-pauli-core stabilizer_code` — 4 passed.
- `conda run -p .conda cargo check --locked -p tencir-pauli-core --benches` — passed.
- `conda run -p .conda maturin develop --release --locked` — passed; the benchmark process resolved the package and native extension inside the source tree.
- `PYTHONPATH="$PWD/python" conda run -p .conda python -m pytest tests/test_qec.py -q` — 12 passed.
- `PYTHONPATH="$PWD/python" conda run -p .conda python -m pytest benchmarks/python/test_stabilizer_code_benchmark.py --benchmark-disable -q` — 22 passed.
- Rust formatting, targeted Ruff/Black checks, and `git diff --check` passed during this remediation slice.

The benchmark result files remain ignored under `.benchmarks/`; the repository retains the workload definitions, runner, and this evidence report rather than machine-specific measurements. The compact record supersedes the earlier exploratory expanded matrix as the QEC development and acceptance protocol.

## Follow-up review — paired retained check arrays

A post-closure review identified that the retained-input estimate counted `check_x_words` and `check_z_words` as one packed check array. The advisory estimate now explicitly multiplies that term by two. A 64-qubit full-rank CSS boundary test uses a budget of 1,800 bytes: the former estimate would have allowed it, while the corrected estimate rejects it. This is a narrow guard-tightening fix; it does not change numerical semantics, default behavior, or the phase acceptance status, and transient peak memory remains intentionally outside the exact guarantee.
