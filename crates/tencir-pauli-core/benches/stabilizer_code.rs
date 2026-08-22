use std::{env, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tencir_pauli_core::{pack_code_batch, StabilizerCode};

const BATCH: usize = 256;
const CLASSIFICATION_COMPONENT_BATCH: usize = 64;

fn repetition_code(nqubits: usize) -> StabilizerCode {
    let mut z_bits = vec![0_u8; (nqubits - 1) * nqubits];
    for row in 0..nqubits - 1 {
        z_bits[row * nqubits + row] = 1;
        z_bits[row * nqubits + row + 1] = 1;
    }
    StabilizerCode::from_css_binary(
        nqubits,
        &[],
        &z_bits,
        &[],
        &vec![1_i8; nqubits - 1],
        u128::MAX,
    )
    .expect("repetition code")
}

fn redundant_repetition_code(nqubits: usize) -> StabilizerCode {
    let mut z_bits = vec![0_u8; nqubits * nqubits];
    for row in 0..nqubits - 1 {
        z_bits[row * nqubits + row] = 1;
        z_bits[row * nqubits + row + 1] = 1;
    }
    z_bits[(nqubits - 1) * nqubits] = 1;
    z_bits[(nqubits - 1) * nqubits + 1] = 1;
    StabilizerCode::from_css_binary(nqubits, &[], &z_bits, &[], &vec![1_i8; nqubits], u128::MAX)
        .expect("redundant repetition code")
}

fn balanced_css_code(nqubits: usize) -> StabilizerCode {
    let half = nqubits / 4;
    let mut x_bits = vec![0_u8; half * nqubits];
    let mut z_bits = vec![0_u8; half * nqubits];
    for row in 0..half {
        x_bits[row * nqubits + row] = 1;
        z_bits[row * nqubits + half + row] = 1;
    }
    StabilizerCode::from_css_binary(
        nqubits,
        &x_bits,
        &z_bits,
        &vec![1_i8; half],
        &vec![1_i8; half],
        u128::MAX,
    )
    .expect("balanced CSS code")
}

fn general_finite_rate_code(nqubits: usize) -> StabilizerCode {
    let rank = nqubits / 4;
    let mut x_bits = vec![0_u8; rank * nqubits];
    let mut z_bits = vec![0_u8; rank * nqubits];
    for row in 0..rank {
        let qubit = 2 * row;
        x_bits[row * nqubits + qubit] = 1;
        z_bits[row * nqubits + qubit] = 1;
    }
    StabilizerCode::from_binary(nqubits, &x_bits, &z_bits, &vec![1_i8; rank], u128::MAX)
        .expect("general finite-rate code")
}

struct HgpCssInput {
    nqubits: usize,
    x_bits: Vec<u8>,
    z_bits: Vec<u8>,
    x_signs: Vec<i8>,
    z_signs: Vec<i8>,
}

struct HgpSymplecticInput {
    nqubits: usize,
    x_bits: Vec<u8>,
    z_bits: Vec<u8>,
    signs: Vec<i8>,
}

fn hgp_repetition_input(length: usize) -> HgpCssInput {
    let checks_per_sector = length * (length - 1);
    let nqubits = length * length + (length - 1) * (length - 1);
    let mut x_bits = vec![0_u8; checks_per_sector * nqubits];
    let mut z_bits = vec![0_u8; checks_per_sector * nqubits];
    let square = length * length;
    let m_square = (length - 1) * (length - 1);

    // H_X = [H tensor I, I tensor H^T], with rows indexed by (a, b).
    for a in 0..length - 1 {
        for b in 0..length {
            let row = a * length + b;
            for c in [a, a + 1] {
                x_bits[row * nqubits + c * length + b] = 1;
            }
            for d in 0..length - 1 {
                if b == d || b == d + 1 {
                    x_bits[row * nqubits + square + a * (length - 1) + d] = 1;
                }
            }
        }
    }

    // H_Z = [I tensor H, H^T tensor I], with rows indexed by (b, a).
    for b in 0..length {
        for a in 0..length - 1 {
            let row = b * (length - 1) + a;
            for c in [a, a + 1] {
                z_bits[row * nqubits + b * length + c] = 1;
            }
            for d in 0..length - 1 {
                if b == d || b == d + 1 {
                    z_bits[row * nqubits + square + d * (length - 1) + a] = 1;
                }
            }
        }
    }

    debug_assert_eq!(square + m_square, nqubits);
    HgpCssInput {
        nqubits,
        x_bits,
        z_bits,
        x_signs: vec![1_i8; checks_per_sector],
        z_signs: vec![1_i8; checks_per_sector],
    }
}

fn hgp_repetition_symplectic_input(input: &HgpCssInput) -> HgpSymplecticInput {
    let x_rows = input.x_signs.len();
    let z_rows = input.z_signs.len();
    let row_width = input.nqubits;
    let mut x_bits = vec![0_u8; (x_rows + z_rows) * row_width];
    let mut z_bits = vec![0_u8; (x_rows + z_rows) * row_width];
    x_bits[..x_rows * row_width].copy_from_slice(&input.x_bits);
    z_bits[z_rows * row_width..].copy_from_slice(&input.z_bits);
    HgpSymplecticInput {
        nqubits: input.nqubits,
        x_bits,
        z_bits,
        signs: vec![1_i8; x_rows + z_rows],
    }
}

fn hgp_repetition_code(input: &HgpCssInput) -> StabilizerCode {
    StabilizerCode::from_css_binary(
        input.nqubits,
        &input.x_bits,
        &input.z_bits,
        &input.x_signs,
        &input.z_signs,
        u128::MAX,
    )
    .expect("HGP repetition CSS code")
}

fn hgp_repetition_general_code(input: &HgpSymplecticInput) -> StabilizerCode {
    StabilizerCode::from_binary(
        input.nqubits,
        &input.x_bits,
        &input.z_bits,
        &input.signs,
        u128::MAX,
    )
    .expect("HGP repetition symplectic code")
}

fn pack_codes(codes: Vec<u8>, nqubits: usize, batch: usize) -> (Vec<u64>, Vec<u64>) {
    pack_code_batch(nqubits, &codes, batch).expect("code batch")
}

fn visible_batch(family: &str, nqubits: usize, batch: usize) -> (Vec<u64>, Vec<u64>) {
    let mut codes = vec![0_u8; batch * nqubits];
    for row in 0..batch {
        let qubit = match family {
            "balanced_css" => nqubits / 4 + row % (nqubits / 4),
            "finite_rate_general" => 2 * (row % (nqubits / 4)),
            "repetition" => row % nqubits,
            _ => row % nqubits,
        };
        codes[row * nqubits + qubit] = 1;
    }
    pack_codes(codes, nqubits, batch)
}

fn stabilizer_batch(family: &str, nqubits: usize, batch: usize) -> (Vec<u64>, Vec<u64>) {
    let mut codes = vec![0_u8; batch * nqubits];
    for row in 0..batch {
        let base = match family {
            "repetition" => row % (nqubits - 1),
            "balanced_css" => row % (nqubits / 4),
            "finite_rate_general" => 2 * (row % (nqubits / 4)),
            _ => row % nqubits,
        };
        if family == "repetition" {
            codes[row * nqubits + base] = 3;
            codes[row * nqubits + base + 1] = 3;
        } else if family == "finite_rate_general" {
            codes[row * nqubits + base] = 3;
        } else {
            codes[row * nqubits + base] = 1;
        }
    }
    pack_codes(codes, nqubits, batch)
}

fn logical_batch(family: &str, nqubits: usize, batch: usize) -> (Vec<u64>, Vec<u64>) {
    let mut codes = vec![0_u8; batch * nqubits];
    for row in 0..batch {
        match family {
            "repetition" => {
                for value in &mut codes[row * nqubits..(row + 1) * nqubits] {
                    *value = 1;
                }
            }
            "balanced_css" => {
                codes[row * nqubits + nqubits / 2 + row % (nqubits / 4)] = 1;
            }
            "finite_rate_general" => {
                codes[row * nqubits + nqubits - 1] = 1;
            }
            _ => unreachable!("unknown stabilizer-code family"),
        }
    }
    pack_codes(codes, nqubits, batch)
}

fn mixed_classification_batch(
    family: &str,
    nqubits: usize,
    component_batch: usize,
) -> (Vec<u64>, Vec<u64>, usize) {
    let mut x_words = Vec::new();
    let mut z_words = Vec::new();
    for make_batch in [visible_batch, stabilizer_batch, logical_batch] {
        let (x, z) = make_batch(family, nqubits, component_batch);
        x_words.extend(x);
        z_words.extend(z);
    }
    (x_words, z_words, component_batch * 3)
}

fn syndrome_batch(nqubits: usize, batch: usize) -> (Vec<u64>, Vec<u64>) {
    visible_batch("repetition", nqubits, batch)
}

fn zero_batch(code: &StabilizerCode, batch: usize) -> (Vec<u64>, Vec<u64>) {
    (
        vec![0_u64; batch * code.word_count()],
        vec![0_u64; batch * code.word_count()],
    )
}

fn error_set_codes(nqubits: usize, regime: &str) -> Vec<u8> {
    let mut codes = vec![0_u8; BATCH * nqubits];
    match regime {
        "unique_syndromes" => {
            for row in 0..BATCH {
                codes[row * nqubits + row] = 1;
            }
        }
        "repeated_same_coset" => {
            for row in 0..BATCH {
                codes[row * nqubits] = 1;
            }
        }
        "multiple_conflicting_cosets" => {
            let logical_qubit = nqubits / 2;
            codes[logical_qubit] = 1;
            codes[nqubits] = 1;
            codes[3 * nqubits] = 1;
            codes[3 * nqubits + logical_qubit] = 1;
        }
        _ => unreachable!("unknown error-set regime"),
    }
    codes
}

fn benchmark_stabilizer_code(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("stabilizer_code");

    // Four construction points cover a small sparse code, redundant checks, CSS
    // checks, and a large generic signed-check input without a family-by-size grid.
    for (family, builder, nqubits) in [
        (
            "repetition",
            repetition_code as fn(usize) -> StabilizerCode,
            64_usize,
        ),
        ("repetition_redundant", redundant_repetition_code, 256_usize),
        ("balanced_css", balanced_css_code, 256_usize),
        ("finite_rate_general", general_finite_rate_code, 1024_usize),
    ] {
        group.bench_with_input(
            BenchmarkId::new(format!("{family}/construction"), nqubits),
            &nqubits,
            |bencher, &n| bencher.iter(|| black_box(builder(n))),
        );
    }

    // These are the HGP-like workloads used by the QLDPC comparison. The
    // general-symplectic path is diagnostic only; both measurements include
    // binary-to-packed conversion, validation, elimination, and logical-basis
    // construction.
    for length in [8_usize, 16, 24] {
        let css_input = hgp_repetition_input(length);
        let symplectic_input = hgp_repetition_symplectic_input(&css_input);
        let nqubits = css_input.nqubits;
        group.bench_with_input(
            BenchmarkId::new("hgp_repetition/construction_css", nqubits),
            &css_input,
            |bencher, input| bencher.iter(|| black_box(hgp_repetition_code(input))),
        );
        group.bench_with_input(
            BenchmarkId::new("hgp_repetition/construction_general_symplectic", nqubits),
            &symplectic_input,
            |bencher, input| bencher.iter(|| black_box(hgp_repetition_general_code(input))),
        );
    }

    // The mixed batch contains visible, stabilizer, and logical rows. The
    // packed/unpacked pair is retained only at three representative widths.
    for (family, nqubits) in [
        ("repetition", 256_usize),
        ("balanced_css", 256_usize),
        ("finite_rate_general", 1024_usize),
    ] {
        let code = match family {
            "repetition" => repetition_code(nqubits),
            "balanced_css" => balanced_css_code(nqubits),
            "finite_rate_general" => general_finite_rate_code(nqubits),
            _ => unreachable!("unknown stabilizer-code family"),
        };
        let (x_words, z_words, batch) =
            mixed_classification_batch(family, nqubits, CLASSIFICATION_COMPONENT_BATCH);
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{family}/classify_unpacked"), nqubits),
            &x_words,
            |bencher, x| {
                bencher.iter(|| black_box(code.classify_many(x, &z_words, batch, u128::MAX)))
            },
        );
        group.bench_with_input(
            BenchmarkId::new(format!("{family}/classify_packed"), nqubits),
            &x_words,
            |bencher, x| {
                bencher.iter(|| black_box(code.classify_many_packed(x, &z_words, batch, u128::MAX)))
            },
        );
    }

    // Direct syndrome throughput keeps the packed and unpacked output paths
    // visible independently of classification read-back.
    let syndrome_code = repetition_code(64);
    let (syndrome_x, syndrome_z) = syndrome_batch(64, BATCH);
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_with_input(
        BenchmarkId::new("repetition/syndrome_unpacked", 64),
        &syndrome_x,
        |bencher, x| {
            bencher.iter(|| {
                black_box(syndrome_code.syndrome_many_unpacked(x, &syndrome_z, BATCH, u128::MAX))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("repetition/syndrome_packed", 64),
        &syndrome_x,
        |bencher, x| {
            bencher.iter(|| {
                black_box(syndrome_code.syndrome_many_packed(x, &syndrome_z, BATCH, u128::MAX))
            })
        },
    );

    // Correction analysis retains the three semantically distinct residuals,
    // while using one medium-width code and both syndrome ABIs.
    let correction_code = repetition_code(256);
    let zero = zero_batch(&correction_code, BATCH);
    for (kind, residual) in [
        ("visible", visible_batch("repetition", 256, BATCH)),
        ("stabilizer", stabilizer_batch("repetition", 256, BATCH)),
        ("logical", logical_batch("repetition", 256, BATCH)),
    ] {
        group.throughput(Throughput::Elements(BATCH as u64));
        for (packed, label) in [(false, "unpacked"), (true, "packed")] {
            let name = format!("repetition/correction_{kind}_{label}");
            if packed {
                group.bench_with_input(BenchmarkId::new(name, 256), &residual.0, |bencher, x| {
                    bencher.iter(|| {
                        black_box(correction_code.analyze_corrections_packed(
                            (x, &residual.1),
                            (&zero.0, &zero.1),
                            BATCH,
                            u128::MAX,
                        ))
                    })
                });
            } else {
                group.bench_with_input(BenchmarkId::new(name, 256), &residual.0, |bencher, x| {
                    bencher.iter(|| {
                        black_box(correction_code.analyze_corrections_unpacked(
                            (x, &residual.1),
                            (&zero.0, &zero.1),
                            BATCH,
                            u128::MAX,
                        ))
                    })
                });
            }
        }
    }

    // The grouping cases are intentionally one large code, not repeated at all
    // widths: unique, same-coset, and conflicting-coset behavior share the same
    // 1024-qubit finite-rate endpoint.
    let code = general_finite_rate_code(1024);
    for regime in [
        "unique_syndromes",
        "repeated_same_coset",
        "multiple_conflicting_cosets",
    ] {
        let codes = error_set_codes(1024, regime);
        let (x_words, z_words) = pack_codes(codes, 1024, BATCH);
        group.bench_function(
            BenchmarkId::new("finite_rate_general/error_set", regime),
            |bencher| {
                bencher.iter(|| {
                    black_box(code.analyze_error_set(&x_words, &z_words, BATCH, u128::MAX))
                })
            },
        );
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    let quick = env::var_os("TENCIRPAULI_BENCH_QUICK").is_some();
    let measurement = if quick { 100 } else { 250 };
    let warm_up = if quick { 25 } else { 75 };
    Criterion::default()
        .warm_up_time(Duration::from_millis(warm_up))
        .measurement_time(Duration::from_millis(measurement))
        .sample_size(10)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = benchmark_stabilizer_code
}
criterion_main!(benches);
