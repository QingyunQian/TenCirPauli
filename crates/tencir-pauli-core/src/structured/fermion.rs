/// Multiply two canonical fermion operators in one coarse-grained call.
pub fn multiply_fermion_terms(
    n_modes: usize,
    left: FermionBatch<'_>,
    right: FermionBatch<'_>,
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    validate_fermion_arrays(n_modes, left.creation, left.annihilation, left.coefficients)?;
    validate_fermion_arrays(
        n_modes,
        right.creation,
        right.annihilation,
        right.coefficients,
    )?;
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating fermion product expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "fermion product expansion")?;
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            let mut sequence =
                fermion_sequence(&left.creation[left_index], &left.annihilation[left_index]);
            sequence.extend(fermion_sequence(
                &right.creation[right_index],
                &right.annihilation[right_index],
            ));
            for (key, integer) in fermion_rewrite(sequence, max_bytes)? {
                push_aggregate(
                    &mut aggregate,
                    key,
                    left.coefficients[left_index]
                        * right.coefficients[right_index]
                        * integer as f64,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_fermion_aggregate(aggregate)
}

/// Compute a fermion commutator or anticommutator without materializing the
/// two directional products first.
pub fn binary_fermion_terms(
    n_modes: usize,
    left: FermionBatch<'_>,
    right: FermionBatch<'_>,
    max_bytes: u128,
    reverse_sign: i8,
) -> Result<FermionCanonicalResult, PauliError> {
    validate_fermion_arrays(n_modes, left.creation, left.annihilation, left.coefficients)?;
    validate_fermion_arrays(
        n_modes,
        right.creation,
        right.annihilation,
        right.coefficients,
    )?;
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .and_then(|value| value.checked_mul(if reverse_sign == 0 { 1 } else { 2 }))
        .ok_or(PauliError::Overflow {
            context: "estimating fermion binary expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "fermion product expansion")?;
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            accumulate_fermion_product(
                &left,
                left_index,
                &right,
                right_index,
                1,
                &mut aggregate,
                &mut total_values,
                max_bytes,
            )?;
            if reverse_sign != 0 {
                accumulate_fermion_product(
                    &right,
                    right_index,
                    &left,
                    left_index,
                    reverse_sign,
                    &mut aggregate,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_fermion_aggregate(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_fermion_product(
    left: &FermionBatch<'_>,
    left_index: usize,
    right: &FermionBatch<'_>,
    right_index: usize,
    direction_sign: i8,
    aggregate: &mut FxHashMap<FermionKey, Vec<Complex64>>,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let mut sequence = fermion_sequence(&left.creation[left_index], &left.annihilation[left_index]);
    sequence.extend(fermion_sequence(
        &right.creation[right_index],
        &right.annihilation[right_index],
    ));
    for (key, integer) in fermion_rewrite(sequence, max_bytes)? {
        push_aggregate(
            aggregate,
            key,
            left.coefficients[left_index]
                * right.coefficients[right_index]
                * f64::from(direction_sign)
                * integer as f64,
            total_values,
            max_bytes,
        )?;
    }
    Ok(())
}

/// Map canonical fermion terms to Jordan-Wigner Pauli terms in one batch.
pub fn jordan_wigner_terms(
    n_modes: usize,
    creation: &[Vec<u32>],
    annihilation: &[Vec<u32>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<(Vec<Vec<u8>>, Vec<Complex64>), PauliError> {
    validate_fermion_arrays(n_modes, creation, annihilation, coefficients)?;
    let mut aggregate: FxHashMap<Vec<u8>, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for (index, coefficient) in coefficients.iter().copied().enumerate() {
        let current = jordan_wigner_word_expansion(
            n_modes,
            fermion_sequence(&creation[index], &annihilation[index]),
            max_bytes,
        )?;
        for (word, value) in current {
            push_pauli_aggregate(
                &mut aggregate,
                word,
                coefficient * value,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut words = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (word, contributions) in entries {
        let value = deterministic_sum(contributions);
        if !is_exact_zero(value) {
            words.push(word);
            values.push(value);
        }
    }
    Ok((words, values))
}

/// Map mixed-domain terms while retaining their non-fermionic factors.
pub fn jordan_wigner_hybrid_terms(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    validate_hybrid_batch(layout, &batch)?;
    jordan_wigner_hybrid_terms_trusted(layout, batch, max_bytes)
}

/// Map a canonical native hybrid handle without rescanning its storage.
pub fn jordan_wigner_hybrid_terms_trusted(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..batch.coefficients.len() {
        let expansions = if batch.fermion_present[index] {
            jordan_wigner_word_expansion(
                layout.n_modes,
                fermion_sequence(
                    &batch.fermion_creation[index],
                    &batch.fermion_annihilation[index],
                ),
                max_bytes,
            )?
        } else {
            vec![(vec![0_u8; layout.n_modes], Complex64::new(1.0, 0.0))]
        };
        let base_mapped = if batch.mapped_present[index] {
            batch.mapped_codes[index].clone()
        } else {
            vec![0_u8; layout.n_modes]
        };
        for (mapped_codes, mapped_coefficient) in expansions {
            let (combined_codes, mapped_phase) = multiply_pauli_codes(&base_mapped, &mapped_codes);
            push_aggregate(
                &mut aggregate,
                (
                    None,
                    if batch.boson_present[index] {
                        Some(batch.boson_blocks[index].clone())
                    } else {
                        None
                    },
                    batch.qubit_codes[index].clone(),
                    if batch.fermion_present[index] || batch.mapped_present[index] {
                        Some(combined_codes)
                    } else {
                        None
                    },
                    if batch.qudit_present[index] {
                        Some(batch.qudit_triples[index].clone())
                    } else {
                        None
                    },
                ),
                batch.coefficients[index] * mapped_coefficient * mapped_phase,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    finish_hybrid_aggregate(aggregate)
}

fn jordan_wigner_word_expansion(
    n_modes: usize,
    sequence: Vec<(usize, bool)>,
    max_bytes: u128,
) -> Result<Vec<(Vec<u8>, Complex64)>, PauliError> {
    // This convention is also used by Python's tensor-product adapter:
    // X=1/2, Y=+i/2 for annihilation and -i/2 for creation, with Z below the
    // active mode. Keep the two adapters covered by a differential test.
    let mut current: Vec<(Vec<u8>, Complex64)> = vec![(vec![0; n_modes], Complex64::new(1.0, 0.0))];
    for (mode, is_creation) in sequence {
        let y_coefficient = if is_creation {
            Complex64::new(0.0, -0.5)
        } else {
            Complex64::new(0.0, 0.5)
        };
        let expansion_count = current.len().checked_mul(2).ok_or(PauliError::Overflow {
            context: "estimating Jordan-Wigner expansion",
        })?;
        check_structured_bytes(expansion_count, max_bytes, "Jordan-Wigner expansion")?;
        let mut next: FxHashMap<Vec<u8>, Complex64> = FxHashMap::default();
        let mut right_word = vec![3_u8; n_modes];
        right_word[mode..].fill(0);
        for (left_word, left_coefficient) in current {
            for (code, right_coefficient) in
                [(1_u8, Complex64::new(0.5, 0.0)), (2_u8, y_coefficient)]
            {
                right_word[mode] = code;
                let (word, phase) = multiply_pauli_codes(&left_word, &right_word);
                *next.entry(word).or_default() += left_coefficient * right_coefficient * phase;
            }
        }
        current = next
            .into_iter()
            .filter(|(_, value)| !is_exact_zero(*value))
            .collect();
        current.sort_by(|left, right| left.0.cmp(&right.0));
    }
    Ok(current)
}
use super::*;
