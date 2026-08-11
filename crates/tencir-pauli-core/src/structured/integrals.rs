/// Canonicalize a batch of raw fermion monomials with CAR rewriting.
pub fn canonicalize_fermion_terms(
    n_modes: usize,
    factors: &[Vec<(usize, u8)>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    if factors.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: factors.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for (index, (raw_factors, &coefficient)) in factors.iter().zip(coefficients).enumerate() {
        validate_coefficient(coefficient, index)?;
        check_structured_bytes_u128(
            raw_factors.len() as u128,
            max_bytes,
            "fermion raw expansion",
        )?;
        let mut sequence = Vec::with_capacity(raw_factors.len());
        for &(mode, action) in raw_factors {
            if mode >= n_modes {
                return Err(PauliError::InvalidIndex {
                    context: "fermion mode",
                });
            }
            if action > 1 {
                return Err(PauliError::InvalidCode {
                    code: action,
                    index,
                });
            }
            sequence.push((mode, action == 0));
        }
        for (key, integer) in fermion_rewrite(sequence, max_bytes)? {
            push_aggregate(
                &mut aggregate,
                key,
                coefficient * integer as f64,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    finish_fermion_aggregate(aggregate)
}

/// Ingest a complete or compact molecular Hamiltonian in one native batch.
///
/// The two-body coefficient is multiplied by one half exactly here. Values
/// related by Hermiticity are averaged before CAR canonicalization, which
/// removes harmless roundoff asymmetry without applying a numerical cutoff.
pub fn canonicalize_fermion_integrals(
    source: FermionIntegralSource<'_>,
    constant: Complex64,
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    let n_modes = match &source {
        FermionIntegralSource::SpinOrbital {
            n_modes,
            one_body,
            two_body,
        } => {
            let n_squared = n_modes.checked_mul(*n_modes).ok_or(PauliError::Overflow {
                context: "checking molecular integral dimensions",
            })?;
            let n_fourth = n_squared
                .checked_mul(n_squared)
                .ok_or(PauliError::Overflow {
                    context: "checking molecular integral dimensions",
                })?;
            validate_integral_length(one_body.len(), n_squared)?;
            validate_integral_length(two_body.len(), n_fourth)?;
            validate_spin_orbital_integrals(one_body, two_body, *n_modes)?;
            *n_modes
        }
        FermionIntegralSource::SpinBlocks(blocks) => {
            let n_squared =
                blocks
                    .n_spatial
                    .checked_mul(blocks.n_spatial)
                    .ok_or(PauliError::Overflow {
                        context: "checking molecular integral dimensions",
                    })?;
            let n_fourth = n_squared
                .checked_mul(n_squared)
                .ok_or(PauliError::Overflow {
                    context: "checking molecular integral dimensions",
                })?;
            validate_integral_length(blocks.one_alpha.len(), n_squared)?;
            validate_integral_length(blocks.one_beta.len(), n_squared)?;
            for length in [
                blocks.eri_aa.len(),
                blocks.eri_ab.len(),
                blocks.eri_ba.len(),
                blocks.eri_bb.len(),
            ] {
                validate_integral_length(length, n_fourth)?;
            }
            validate_spin_block_integrals(blocks)?;
            blocks
                .n_spatial
                .checked_mul(2)
                .ok_or(PauliError::Overflow {
                    context: "checking molecular spin-orbital dimensions",
                })?
        }
    };

    let mut factors = Vec::new();
    let mut coefficients = Vec::new();
    let mut raw_count = 0usize;
    if !is_exact_zero(constant) {
        push_integral_term(
            &mut factors,
            &mut coefficients,
            &mut raw_count,
            &[],
            constant,
            max_bytes,
        )?;
    }
    match source {
        FermionIntegralSource::SpinOrbital {
            n_modes,
            one_body,
            two_body,
        } => visit_spin_orbital_terms(n_modes, one_body, two_body, |word, coefficient| {
            push_integral_term(
                &mut factors,
                &mut coefficients,
                &mut raw_count,
                word,
                coefficient,
                max_bytes,
            )
        })?,
        FermionIntegralSource::SpinBlocks(blocks) => {
            visit_spin_block_terms(&blocks, |word, coefficient| {
                push_integral_term(
                    &mut factors,
                    &mut coefficients,
                    &mut raw_count,
                    word,
                    coefficient,
                    max_bytes,
                )
            })?
        }
    }
    canonicalize_fermion_terms(n_modes, &factors, &coefficients, max_bytes)
}

fn push_integral_term(
    factors: &mut Vec<Vec<(usize, u8)>>,
    coefficients: &mut Vec<Complex64>,
    raw_count: &mut usize,
    word: &[(usize, u8)],
    coefficient: Complex64,
    max_bytes: u128,
) -> Result<(), PauliError> {
    *raw_count = raw_count.checked_add(1).ok_or(PauliError::Overflow {
        context: "counting molecular integral terms",
    })?;
    let requested = (*raw_count as u128)
        .checked_mul(768)
        .ok_or(PauliError::Overflow {
            context: "estimating molecular integral ingestion",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    factors.push(word.to_vec());
    coefficients.push(coefficient);
    Ok(())
}

fn validate_integral_length(actual: usize, expected: usize) -> Result<(), PauliError> {
    if actual != expected {
        return Err(PauliError::InvalidStructureLength { expected, actual });
    }
    Ok(())
}

const INTEGRAL_RTOL: f64 = 1.0e-10;
const INTEGRAL_ATOL: f64 = 1.0e-12;

fn validate_spin_orbital_integrals(
    one_body: &[Complex64],
    two_body: &[Complex64],
    n_modes: usize,
) -> Result<(), PauliError> {
    for p in 0..n_modes {
        for q in 0..n_modes {
            let index = p * n_modes + q;
            let reverse = q * n_modes + p;
            validate_hermitian_pair(
                one_body[index],
                one_body[reverse],
                index,
                "one_body must be Hermitian",
            )?;
            for r in 0..n_modes {
                for s in 0..n_modes {
                    let index = spin_orbital_index(n_modes, p, q, r, s);
                    let reverse = spin_orbital_index(n_modes, r, s, p, q);
                    validate_hermitian_pair(
                        two_body[index],
                        two_body[reverse],
                        index,
                        "two_body must satisfy Hermitian pair symmetry",
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn validate_spin_block_integrals(blocks: &FermionSpinBlocks<'_>) -> Result<(), PauliError> {
    for (values, context) in [
        (blocks.one_alpha, "one_alpha must be Hermitian"),
        (blocks.one_beta, "one_beta must be Hermitian"),
    ] {
        validate_matrix_pairs(values, blocks.n_spatial, context)?;
    }
    for (values, context) in [
        (blocks.eri_aa, "eri_aa must satisfy Hermitian pair symmetry"),
        (blocks.eri_ab, "eri_ab must satisfy Hermitian pair symmetry"),
        (blocks.eri_ba, "eri_ba must satisfy Hermitian pair symmetry"),
        (blocks.eri_bb, "eri_bb must satisfy Hermitian pair symmetry"),
    ] {
        validate_chemist_pairs(values, blocks.n_spatial, context)?;
    }
    Ok(())
}

fn validate_matrix_pairs(
    values: &[Complex64],
    n: usize,
    context: &'static str,
) -> Result<(), PauliError> {
    for p in 0..n {
        for q in 0..n {
            let index = p * n + q;
            let reverse = q * n + p;
            validate_hermitian_pair(values[index], values[reverse], index, context)?;
        }
    }
    Ok(())
}

fn validate_chemist_pairs(
    values: &[Complex64],
    n_spatial: usize,
    context: &'static str,
) -> Result<(), PauliError> {
    for p in 0..n_spatial {
        for r in 0..n_spatial {
            for q in 0..n_spatial {
                for s in 0..n_spatial {
                    let index = chemist_index(n_spatial, p, r, q, s);
                    let reverse = chemist_index(n_spatial, r, p, s, q);
                    validate_hermitian_pair(values[index], values[reverse], index, context)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_hermitian_pair(
    left: Complex64,
    right: Complex64,
    index: usize,
    context: &'static str,
) -> Result<(), PauliError> {
    if !left.re.is_finite() || !left.im.is_finite() {
        return Err(PauliError::NonFiniteCoefficient { index });
    }
    if !right.re.is_finite() || !right.im.is_finite() {
        return Err(PauliError::NonFiniteCoefficient { index });
    }
    if (left - right.conj()).norm() > INTEGRAL_ATOL + INTEGRAL_RTOL * right.norm() {
        return Err(PauliError::NonHermitianIntegral { context });
    }
    Ok(())
}

fn visit_spin_orbital_terms<E>(
    n_modes: usize,
    one_body: &[Complex64],
    two_body: &[Complex64],
    mut visit: impl FnMut(&[(usize, u8)], Complex64) -> Result<(), E>,
) -> Result<(), E> {
    for p in 0..n_modes {
        for q in 0..n_modes {
            let value = hermitian_average(one_body[p * n_modes + q], one_body[q * n_modes + p]);
            if !is_exact_zero(value) {
                let word = [(p, 0), (q, 1)];
                visit(&word, value)?;
            }
            for r in 0..n_modes {
                for s in 0..n_modes {
                    let value = hermitian_average(
                        two_body[spin_orbital_index(n_modes, p, q, r, s)],
                        two_body[spin_orbital_index(n_modes, r, s, p, q)],
                    ) * 0.5;
                    if !is_exact_zero(value) {
                        let word = [(p, 0), (q, 0), (s, 1), (r, 1)];
                        visit(&word, value)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn visit_spin_block_terms<E>(
    blocks: &FermionSpinBlocks<'_>,
    mut visit: impl FnMut(&[(usize, u8)], Complex64) -> Result<(), E>,
) -> Result<(), E> {
    let n = blocks.n_spatial;
    for (spin, one_body) in [(0usize, blocks.one_alpha), (1usize, blocks.one_beta)] {
        for p in 0..n {
            for q in 0..n {
                let value = hermitian_average(one_body[p * n + q], one_body[q * n + p]);
                if !is_exact_zero(value) {
                    let word = [
                        (spin_mode(n, p, spin, blocks.ordering), 0),
                        (spin_mode(n, q, spin, blocks.ordering), 1),
                    ];
                    visit(&word, value)?;
                }
            }
        }
    }
    for (left_spin, right_spin, eri) in [
        (0usize, 0usize, blocks.eri_aa),
        (0usize, 1usize, blocks.eri_ab),
        (1usize, 0usize, blocks.eri_ba),
        (1usize, 1usize, blocks.eri_bb),
    ] {
        for p in 0..n {
            for r in 0..n {
                for q in 0..n {
                    for s in 0..n {
                        let index = chemist_index(n, p, r, q, s);
                        let reverse = chemist_index(n, r, p, s, q);
                        if !is_exact_zero(hermitian_average(eri[index], eri[reverse]) * 0.5) {
                            let value = hermitian_average(eri[index], eri[reverse]) * 0.5;
                            let word = [
                                (spin_mode(n, p, left_spin, blocks.ordering), 0),
                                (spin_mode(n, q, right_spin, blocks.ordering), 0),
                                (spin_mode(n, s, right_spin, blocks.ordering), 1),
                                (spin_mode(n, r, left_spin, blocks.ordering), 1),
                            ];
                            visit(&word, value)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn hermitian_average(left: Complex64, right: Complex64) -> Complex64 {
    (left + right.conj()) * 0.5
}

#[inline]
fn spin_orbital_index(n_modes: usize, p: usize, q: usize, r: usize, s: usize) -> usize {
    ((p * n_modes + q) * n_modes + r) * n_modes + s
}

#[inline]
fn chemist_index(n_spatial: usize, p: usize, r: usize, q: usize, s: usize) -> usize {
    ((p * n_spatial + r) * n_spatial + q) * n_spatial + s
}

#[inline]
fn spin_mode(
    n_spatial: usize,
    orbital: usize,
    spin: usize,
    ordering: FermionSpinOrdering,
) -> usize {
    match ordering {
        FermionSpinOrdering::Interleaved => 2 * orbital + spin,
        FermionSpinOrdering::AlphaThenBeta => spin * n_spatial + orbital,
    }
}
use super::*;
