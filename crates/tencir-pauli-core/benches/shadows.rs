use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tencir_pauli_core::{
    Complex64, PauliOperator, PauliWord, SnapshotProtocol, Snapshots, StabilizerState,
};

fn basis_state(nqubits: usize) -> Vec<Complex64> {
    let mut state = vec![Complex64::default(); 1usize << nqubits];
    state[0] = Complex64::new(1.0, 0.0);
    state
}

fn benchmark_sampling(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("shadows/sampling");
    for nqubits in [4usize, 8, 10] {
        let state = basis_state(nqubits);
        let bytes = (state.len() * 16) as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::new("pauli_statevector", nqubits),
            &state,
            |bencher, input| {
                bencher.iter(|| {
                    black_box(Snapshots::sample_statevector(
                        black_box(input),
                        16,
                        SnapshotProtocol::Pauli,
                        1,
                        7,
                        u128::MAX,
                    ))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("clifford_statevector", nqubits),
            &state,
            |bencher, input| {
                bencher.iter(|| {
                    black_box(Snapshots::sample_statevector(
                        black_box(input),
                        16,
                        SnapshotProtocol::Clifford,
                        1,
                        7,
                        u128::MAX,
                    ))
                });
            },
        );
    }
    group.finish();
}

fn benchmark_stabilizer_and_estimators(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("shadows/stabilizer_estimators");
    for nqubits in [8usize, 32, 64] {
        let state = StabilizerState::zero_state(nqubits).expect("zero tableau");
        let snapshots =
            Snapshots::sample_stabilizer(&state, 128, SnapshotProtocol::Clifford, 1, 17, u128::MAX)
                .expect("stabilizer snapshots");
        let word = PauliWord::from_codes(nqubits, &vec![3; nqubits]).expect("word");
        let observable =
            PauliOperator::from_terms(nqubits, &[vec![3; nqubits]], &[Complex64::new(1.0, 0.0)])
                .expect("observable");
        let _ = word;
        group.bench_with_input(
            BenchmarkId::from_parameter(nqubits),
            &snapshots,
            |bencher, input| {
                bencher.iter(|| black_box(input.expectation(black_box(&observable), 1, false)))
            },
        );
    }
    group.finish();
}

fn dense_target(nqubits: usize) -> Vec<Complex64> {
    let mut target = (0..(1usize << nqubits))
        .map(|index| {
            let angle = (index as f64 + 1.0) * 0.173;
            Complex64::new(angle.sin(), (angle * 1.37).cos())
        })
        .collect::<Vec<_>>();
    let norm = target
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    for value in &mut target {
        *value /= norm;
    }
    target
}

fn benchmark_dense_target_fidelity(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("shadows/dense_target_fidelity");
    for nqubits in [8usize, 10, 12] {
        let state = basis_state(nqubits);
        let target = dense_target(nqubits);
        let snapshots =
            Snapshots::sample_statevector(&state, 64, SnapshotProtocol::Clifford, 1, 31, u128::MAX)
                .expect("dense Clifford snapshots");
        group.throughput(Throughput::Elements((1usize << nqubits) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(nqubits),
            &snapshots,
            |bencher, input| {
                bencher
                    .iter(|| black_box(input.fidelity_statevector(black_box(&target), 1, false)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_sampling,
    benchmark_stabilizer_and_estimators,
    benchmark_dense_target_fidelity
);
criterion_main!(benches);
