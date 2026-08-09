pub(super) fn contraction_coefficient(q: u32, r: u32, k: u32) -> Result<i128, PauliError> {
    let mut binomial = 1_i128;
    for index in 0..k {
        binomial = binomial
            .checked_mul((q - index) as i128)
            .and_then(|value| value.checked_div((index + 1) as i128))
            .ok_or(PauliError::Overflow {
                context: "boson binomial coefficient",
            })?;
    }
    let mut falling = 1_i128;
    for index in 0..k {
        falling = falling
            .checked_mul((r - index) as i128)
            .ok_or(PauliError::Overflow {
                context: "boson falling factorial",
            })?;
    }
    binomial.checked_mul(falling).ok_or(PauliError::Overflow {
        context: "boson contraction coefficient",
    })
}

pub(super) fn fermion_sequence(creation: &[u32], annihilation: &[u32]) -> Vec<(usize, bool)> {
    creation
        .iter()
        .map(|&mode| (mode as usize, true))
        .chain(annihilation.iter().map(|&mode| (mode as usize, false)))
        .collect()
}

pub(super) fn raw_sequence(
    factors: &[(usize, u8)],
    n_modes: usize,
    index: usize,
    context: &'static str,
) -> Result<Vec<(usize, bool)>, PauliError> {
    let mut sequence = Vec::with_capacity(factors.len());
    for &(mode, action) in factors {
        if mode >= n_modes {
            return Err(PauliError::InvalidIndex { context });
        }
        if action > 1 {
            return Err(PauliError::InvalidCode {
                code: action,
                index,
            });
        }
        sequence.push((mode, action == 0));
    }
    Ok(sequence)
}

pub(super) fn add_integer<K: Ord>(
    target: &mut BTreeMap<K, i128>,
    key: K,
    value: i128,
) -> Result<(), PauliError> {
    let entry = target.entry(key).or_default();
    *entry = entry.checked_add(value).ok_or(PauliError::Overflow {
        context: "structured integer coefficient",
    })?;
    Ok(())
}

pub(super) fn add_integer_maps<K: Ord>(
    target: &mut BTreeMap<K, i128>,
    source: BTreeMap<K, i128>,
    factor: i128,
) -> Result<(), PauliError> {
    for (key, value) in source {
        add_integer(
            target,
            key,
            value.checked_mul(factor).ok_or(PauliError::Overflow {
                context: "structured integer coefficient",
            })?,
        )?;
    }
    Ok(())
}

pub(super) fn validate_coefficient(value: Complex64, index: usize) -> Result<(), PauliError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(PauliError::NonFiniteCoefficient { index })
    }
}

pub(super) fn validate_fermion_arrays(
    n_modes: usize,
    creation: &[Vec<u32>],
    annihilation: &[Vec<u32>],
    coefficients: &[Complex64],
) -> Result<(), PauliError> {
    if creation.len() != annihilation.len() || creation.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: creation.len(),
            actual: coefficients.len(),
        });
    }
    for index in 0..creation.len() {
        if creation[index].windows(2).any(|pair| pair[0] >= pair[1])
            || annihilation[index]
                .windows(2)
                .any(|pair| pair[0] <= pair[1])
            || creation[index]
                .iter()
                .chain(&annihilation[index])
                .any(|&mode| mode as usize >= n_modes)
        {
            return Err(PauliError::NonCanonicalTerms { index });
        }
    }
    Ok(())
}

pub(crate) fn validate_boson_blocks(
    n_modes: usize,
    blocks: &[(u32, u32, u32)],
) -> Result<(), PauliError> {
    if blocks.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || blocks.iter().any(|&(mode, _, _)| mode as usize >= n_modes)
    {
        return Err(PauliError::NonCanonicalTerms { index: 0 });
    }
    Ok(())
}

pub(super) fn check_structured_bytes(
    count: usize,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), PauliError> {
    check_structured_bytes_u128(count as u128, max_bytes, context)
}

pub(super) fn check_structured_bytes_u128(
    count: u128,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), PauliError> {
    let requested = count
        .checked_mul(192)
        .ok_or(PauliError::Overflow { context })?;
    if requested > max_bytes {
        Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        })
    } else {
        Ok(())
    }
}

pub(super) fn push_aggregate<K: Eq + std::hash::Hash>(
    aggregate: &mut FxHashMap<K, Vec<Complex64>>,
    key: K,
    value: Complex64,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let value_count = {
        let values = aggregate.entry(key).or_default();
        values.push(value);
        values.len()
    };
    *total_values = (*total_values).checked_add(1).ok_or(PauliError::Overflow {
        context: "estimating structured aggregation",
    })?;
    check_structured_bytes(
        (*total_values).max(value_count),
        max_bytes,
        "structured aggregation",
    )
}

pub(super) fn push_pauli_aggregate(
    aggregate: &mut FxHashMap<Vec<u8>, Vec<Complex64>>,
    key: Vec<u8>,
    value: Complex64,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    push_aggregate(aggregate, key, value, total_values, max_bytes)
}

pub(super) fn finish_fermion_aggregate(
    aggregate: FxHashMap<FermionKey, Vec<Complex64>>,
) -> Result<FermionCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut creation = Vec::with_capacity(entries.len());
    let mut annihilation = Vec::with_capacity(entries.len());
    let mut coefficients = Vec::with_capacity(entries.len());
    for ((left, right), values) in entries {
        let value = deterministic_sum(values);
        if !is_exact_zero(value) {
            creation.push(left);
            annihilation.push(right);
            coefficients.push(value);
        }
    }
    Ok((creation, annihilation, coefficients))
}

pub(super) fn finish_boson_aggregate(
    aggregate: FxHashMap<BosonKey, Vec<Complex64>>,
) -> Result<BosonCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut blocks = Vec::with_capacity(entries.len());
    let mut coefficients = Vec::with_capacity(entries.len());
    for (key, values) in entries {
        let value = deterministic_sum(values);
        if !is_exact_zero(value) {
            blocks.push(key);
            coefficients.push(value);
        }
    }
    Ok((blocks, coefficients))
}

pub(super) fn deterministic_sum(mut values: Vec<Complex64>) -> Complex64 {
    values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
    values
        .into_iter()
        .fold(Complex64::default(), |sum, value| sum + value)
}

pub(super) fn is_exact_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

pub(super) fn multiply_pauli_codes(left: &[u8], right: &[u8]) -> (Vec<u8>, Complex64) {
    let mut result = Vec::with_capacity(left.len());
    let mut phase = Complex64::new(1.0, 0.0);
    for (&left_code, &right_code) in left.iter().zip(right) {
        let (code, local_phase) = pauli_code_product(left_code, right_code);
        result.push(code);
        phase *= local_phase;
    }
    (result, phase)
}

pub(super) fn pauli_code_product(left: u8, right: u8) -> (u8, Complex64) {
    match (left, right) {
        (0, code) | (code, 0) => (code, Complex64::new(1.0, 0.0)),
        (1, 1) | (2, 2) | (3, 3) => (0, Complex64::new(1.0, 0.0)),
        (1, 2) => (3, Complex64::new(0.0, 1.0)),
        (1, 3) => (2, Complex64::new(0.0, -1.0)),
        (2, 1) => (3, Complex64::new(0.0, -1.0)),
        (2, 3) => (1, Complex64::new(0.0, 1.0)),
        (3, 1) => (2, Complex64::new(0.0, 1.0)),
        (3, 2) => (1, Complex64::new(0.0, -1.0)),
        _ => (0, Complex64::new(0.0, 0.0)),
    }
}
use super::*;
