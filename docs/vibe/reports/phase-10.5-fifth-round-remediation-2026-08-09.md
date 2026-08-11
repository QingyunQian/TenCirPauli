# Phase 10.5 Fifth-Round Remediation, 2026-08-09

Status: implementation and focused verification complete; final Phase 10.5 acceptance remains subject to the full quality gate and owner review of the refreshed benchmark record.

## B5-1 — Product inactive compact-factor reuse

`FastProductMvpPlan` now compiles an explicit active/inactive component shape for every schedule group. The apply loop evaluates inverse actions and grouped amplitudes only over active factor and spectator tuples, then traverses inactive tuples. A trailing public-order Cartesian identity suffix is applied as a contiguous segment; interleaved layouts continue through the bounded factor-prefix public index without calling the global sector ranker or retaining a selected-space permutation.

`test_product_inactive_compact_factor_scales_segments_not_active_work` expands an inactive fixed-weight qubit factor from 2 to 4 to 6 qubits. The Product dimensions are 4, 12, and 40; factor-action, group, and aggregate counters remain constant, while applied identity-segment elements increase. Every case matches the independently materialized dense result, selects `native_variant="product"`, and reports zero fallback, global-ranker, and destination-allocation counters.

## B5-2 — Raw Fermion packed-source reuse

The pure contiguous binary raw-Fermion path now packs source occupations once into a transient one-word source cache and reuses the shared `FermionMaskDescriptor` for every schedule member. The same descriptor supplies one-word validity, parity, and flip operations; no Jordan--Wigner expansion or second CAR/sign implementation is introduced. Destination-owned cached gather writes directly into caller-owned output and uses the same work threshold as the packed-Pauli MVP path.

The refreshed `fermion-mask-phase-v2` record on `c50038bac50d-dirty-896134efcce85367`, with two warmups and seven release samples, reports raw/mapped maximum action error `2.20e-14`. With default Rayon configuration the raw median is `0.128 ms` and explicit Jordan--Wigner packed Pauli is `0.098 ms`; with `RAYON_NUM_THREADS=1` they are `0.354 ms` and `0.243 ms`. This removes the reported approximately 20x regression. The profile records 4,096 packed-source units versus 139,264 schedule-member evaluations; packed extraction is about 3.8% of batched descriptor work. Ordered CAR, complex hopping, and multiword descriptor regressions remain covered by the shared descriptor tests.

## Verification

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p tencir-pauli-core --lib` (55 passed)
- Release `maturin develop --release --locked`
- Focused Python numerical suite: 152 passed
