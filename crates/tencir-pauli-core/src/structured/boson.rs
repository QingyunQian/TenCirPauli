/// Canonicalize a batch of raw boson monomials with CCR rewriting.
pub fn canonicalize_boson_terms(
    n_modes: usize,
    factors: &[Vec<(usize, u8)>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<BosonCanonicalResult, PauliError> {
    if factors.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: factors.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for (index, (raw_factors, &coefficient)) in factors.iter().zip(coefficients).enumerate() {
        validate_coefficient(coefficient, index)?;
        check_structured_bytes_u128(raw_factors.len() as u128, max_bytes, "boson raw expansion")?;
        let mut sequence = Vec::with_capacity(raw_factors.len());
        for &(mode, action) in raw_factors {
            if mode >= n_modes {
                return Err(PauliError::InvalidIndex {
                    context: "boson mode",
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
        for (key, integer) in boson_rewrite(sequence, max_bytes)? {
            push_aggregate(
                &mut aggregate,
                key,
                coefficient * integer as f64,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    finish_boson_aggregate(aggregate)
}

/// Multiply two canonical boson operators in one coarse-grained call.
pub fn multiply_boson_terms(
    n_modes: usize,
    left_blocks: &[Vec<(u32, u32, u32)>],
    left_coefficients: &[Complex64],
    right_blocks: &[Vec<(u32, u32, u32)>],
    right_coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<BosonCanonicalResult, PauliError> {
    if left_blocks.len() != left_coefficients.len()
        || right_blocks.len() != right_coefficients.len()
    {
        return Err(PauliError::InvalidStructureLength {
            expected: left_blocks.len(),
            actual: left_coefficients.len(),
        });
    }
    for blocks in left_blocks.iter().chain(right_blocks) {
        validate_boson_blocks(n_modes, blocks)?;
    }
    let pair_count = left_coefficients
        .len()
        .checked_mul(right_coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating boson product expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "boson product expansion")?;
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left in 0..left_blocks.len() {
        for right in 0..right_blocks.len() {
            for (key, integer) in
                boson_block_product(&left_blocks[left], &right_blocks[right], max_bytes)?
            {
                push_aggregate(
                    &mut aggregate,
                    key,
                    left_coefficients[left] * right_coefficients[right] * integer as f64,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_boson_aggregate(aggregate)
}

/// Compute a boson commutator or anticommutator in one shared aggregate.
pub fn binary_boson_terms(
    n_modes: usize,
    left_blocks: &[Vec<(u32, u32, u32)>],
    left_coefficients: &[Complex64],
    right_blocks: &[Vec<(u32, u32, u32)>],
    right_coefficients: &[Complex64],
    max_bytes: u128,
    reverse_sign: i8,
) -> Result<BosonCanonicalResult, PauliError> {
    if left_blocks.len() != left_coefficients.len()
        || right_blocks.len() != right_coefficients.len()
    {
        return Err(PauliError::InvalidStructureLength {
            expected: left_blocks.len(),
            actual: left_coefficients.len(),
        });
    }
    for blocks in left_blocks.iter().chain(right_blocks) {
        validate_boson_blocks(n_modes, blocks)?;
    }
    let pair_count = left_coefficients
        .len()
        .checked_mul(right_coefficients.len())
        .and_then(|value| value.checked_mul(if reverse_sign == 0 { 1 } else { 2 }))
        .ok_or(PauliError::Overflow {
            context: "estimating boson binary expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "boson product expansion")?;
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left_blocks.len() {
        for right_index in 0..right_blocks.len() {
            accumulate_boson_product(
                &left_blocks[left_index],
                left_coefficients[left_index],
                &right_blocks[right_index],
                right_coefficients[right_index],
                1,
                &mut aggregate,
                &mut total_values,
                max_bytes,
            )?;
            if reverse_sign != 0 {
                accumulate_boson_product(
                    &right_blocks[right_index],
                    right_coefficients[right_index],
                    &left_blocks[left_index],
                    left_coefficients[left_index],
                    reverse_sign,
                    &mut aggregate,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_boson_aggregate(aggregate)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_boson_product(
    left: &[(u32, u32, u32)],
    left_coefficient: Complex64,
    right: &[(u32, u32, u32)],
    right_coefficient: Complex64,
    direction_sign: i8,
    aggregate: &mut FxHashMap<BosonKey, Vec<Complex64>>,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    for (key, integer) in boson_block_product(left, right, max_bytes)? {
        push_aggregate(
            aggregate,
            key,
            left_coefficient * right_coefficient * f64::from(direction_sign) * integer as f64,
            total_values,
            max_bytes,
        )?;
    }
    Ok(())
}

pub(super) fn fermion_rewrite(
    sequence: Vec<(usize, bool)>,
    max_bytes: u128,
) -> Result<BTreeMap<FermionKey, i128>, PauliError> {
    if sequence.is_empty() {
        let mut result = BTreeMap::new();
        result.insert((Vec::new(), Vec::new()), 1);
        return Ok(result);
    }

    // Canonical/no-contraction inputs are common in bulk construction.  They
    // should not be rejected by a blanket 2**len estimate: the exact result
    // is one key (or zero for an adjacent nilpotent pair) and no branching is
    // needed.  The same fast path handles long inversion-only words below.
    if let Some(result) = canonical_without_contractions(&sequence) {
        return result;
    }

    fn recurse(
        sequence: &[(usize, bool)],
        max_bytes: u128,
    ) -> Result<BTreeMap<FermionKey, i128>, PauliError> {
        for index in 0..sequence.len().saturating_sub(1) {
            let (left_mode, left_creation) = sequence[index];
            let (right_mode, right_creation) = sequence[index + 1];
            if left_creation == right_creation && left_mode == right_mode {
                return Ok(BTreeMap::new());
            }
            let inversion = if left_creation && right_creation {
                left_mode > right_mode
            } else if !left_creation && !right_creation {
                left_mode < right_mode
            } else {
                !left_creation && right_creation
            };
            if !inversion {
                continue;
            }
            let mut swapped = sequence.to_vec();
            swapped.swap(index, index + 1);
            let mut result = BTreeMap::new();
            add_integer_maps(&mut result, recurse(&swapped, max_bytes)?, -1)?;
            check_structured_bytes_u128(result.len() as u128, max_bytes, "fermion expansion")?;
            if !left_creation && right_creation && left_mode == right_mode {
                let mut contracted = sequence.to_vec();
                contracted.drain(index..=index + 1);
                add_integer_maps(&mut result, recurse(&contracted, max_bytes)?, 1)?;
                check_structured_bytes_u128(result.len() as u128, max_bytes, "fermion expansion")?;
            }
            result.retain(|_, value| *value != 0);
            check_structured_bytes_u128(result.len() as u128, max_bytes, "fermion expansion")?;
            return Ok(result);
        }
        let creation = sequence
            .iter()
            .filter_map(|&(mode, is_creation)| is_creation.then_some(mode as u32))
            .collect();
        let annihilation = sequence
            .iter()
            .filter_map(|&(mode, is_creation)| (!is_creation).then_some(mode as u32))
            .collect();
        let mut result = BTreeMap::new();
        result.insert((creation, annihilation), 1);
        Ok(result)
    }

    recurse(&sequence, max_bytes)
}

/// Return the exact one-key/zero-key result for a non-branching CAR word.
///
/// `None` means that a contraction may be reachable and the general rewrite
/// must be used.  The scan is linear for the ordinary canonical case and the
/// quadratic inversion count is only used for words whose creation and
/// annihilation supports are disjoint.
pub(super) fn canonical_without_contractions(
    sequence: &[(usize, bool)],
) -> Option<Result<BTreeMap<FermionKey, i128>, PauliError>> {
    let mut canonical = true;
    for pair in sequence.windows(2) {
        let (left_mode, left_creation) = pair[0];
        let (right_mode, right_creation) = pair[1];
        if left_creation == right_creation && left_mode == right_mode {
            return Some(Ok(BTreeMap::new()));
        }
        let inversion = if left_creation && right_creation {
            left_mode > right_mode
        } else if !left_creation && !right_creation {
            left_mode < right_mode
        } else {
            !left_creation && right_creation
        };
        if inversion {
            canonical = false;
            break;
        }
    }
    if canonical {
        return Some(Ok(single_canonical_fermion_key(sequence, 1)));
    }

    let creation_modes: Vec<usize> = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| is_creation.then_some(mode))
        .collect();
    let annihilation_modes: Vec<usize> = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| (!is_creation).then_some(mode))
        .collect();
    if creation_modes
        .iter()
        .any(|mode| annihilation_modes.iter().any(|other| mode == other))
    {
        return None;
    }
    if has_duplicate(&creation_modes) || has_duplicate(&annihilation_modes) {
        return Some(Ok(BTreeMap::new()));
    }

    let inversions = sequence
        .iter()
        .enumerate()
        .map(|(left_index, &left)| {
            sequence[left_index + 1..]
                .iter()
                .filter(|&&right| fermion_inversion(left, right))
                .count()
        })
        .sum::<usize>();
    let sign = if inversions & 1 == 0 { 1 } else { -1 };
    Some(Ok(single_canonical_fermion_key(sequence, sign)))
}

pub(super) fn fermion_inversion(left: (usize, bool), right: (usize, bool)) -> bool {
    let (left_mode, left_creation) = left;
    let (right_mode, right_creation) = right;
    if left_creation && right_creation {
        left_mode > right_mode
    } else if !left_creation && !right_creation {
        left_mode < right_mode
    } else {
        !left_creation && right_creation
    }
}

pub(super) fn has_duplicate(values: &[usize]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[index + 1..].contains(value))
}

pub(super) fn single_canonical_fermion_key(
    sequence: &[(usize, bool)],
    coefficient: i128,
) -> BTreeMap<FermionKey, i128> {
    let mut creation = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| is_creation.then_some(mode as u32))
        .collect::<Vec<_>>();
    let mut annihilation = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| (!is_creation).then_some(mode as u32))
        .collect::<Vec<_>>();
    creation.sort_unstable();
    annihilation.sort_unstable_by(|left, right| right.cmp(left));
    let mut result = BTreeMap::new();
    result.insert((creation, annihilation), coefficient);
    result
}

pub(super) fn boson_rewrite(
    sequence: Vec<(usize, bool)>,
    max_bytes: u128,
) -> Result<BTreeMap<BosonKey, i128>, PauliError> {
    let mut by_mode: BTreeMap<usize, Vec<bool>> = BTreeMap::new();
    for (mode, is_creation) in sequence {
        by_mode.entry(mode).or_default().push(is_creation);
    }
    let mut aggregate: BTreeMap<BosonKey, i128> = BTreeMap::new();
    aggregate.insert(Vec::new(), 1);
    for (mode, factors) in by_mode {
        let mut local: BTreeMap<(u32, u32), i128> = BTreeMap::new();
        local.insert((0, 0), 1);
        for is_creation in factors {
            let estimated = (local.len() as u128)
                .checked_mul(2)
                .ok_or(PauliError::Overflow {
                    context: "estimating boson expansion",
                })?;
            check_structured_bytes_u128(estimated, max_bytes, "boson expansion")?;
            let mut next = BTreeMap::new();
            for (&(creation, annihilation), &value) in &local {
                if is_creation {
                    let next_creation = creation.checked_add(1).ok_or(PauliError::Overflow {
                        context: "boson creation power",
                    })?;
                    add_integer(&mut next, (next_creation, annihilation), value)?;
                    if annihilation != 0 {
                        let contracted = value.checked_mul(annihilation as i128).ok_or(
                            PauliError::Overflow {
                                context: "boson contraction coefficient",
                            },
                        )?;
                        add_integer(&mut next, (creation, annihilation - 1), contracted)?;
                    }
                } else {
                    let next_annihilation =
                        annihilation.checked_add(1).ok_or(PauliError::Overflow {
                            context: "boson annihilation power",
                        })?;
                    add_integer(&mut next, (creation, next_annihilation), value)?;
                }
            }
            local = next;
        }
        let product_count = (aggregate.len() as u128)
            .checked_mul(local.len() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating boson expansion",
            })?;
        check_structured_bytes_u128(product_count, max_bytes, "boson expansion")?;
        let mut next = BTreeMap::new();
        for (prefix, prefix_value) in aggregate {
            for (&(creation, annihilation), &local_value) in &local {
                let mut blocks = prefix.clone();
                if creation != 0 || annihilation != 0 {
                    blocks.push((mode as u32, creation, annihilation));
                }
                add_integer(
                    &mut next,
                    blocks,
                    prefix_value
                        .checked_mul(local_value)
                        .ok_or(PauliError::Overflow {
                            context: "boson expansion coefficient",
                        })?,
                )?;
            }
        }
        aggregate = next;
    }
    Ok(aggregate)
}

pub(super) fn boson_block_product(
    left: &[(u32, u32, u32)],
    right: &[(u32, u32, u32)],
    max_bytes: u128,
) -> Result<BTreeMap<BosonKey, i128>, PauliError> {
    let mut left_by_mode = BTreeMap::new();
    let mut right_by_mode = BTreeMap::new();
    for &(mode, creation, annihilation) in left {
        left_by_mode.insert(mode, (creation, annihilation));
    }
    for &(mode, creation, annihilation) in right {
        right_by_mode.insert(mode, (creation, annihilation));
    }
    let mut aggregate: BTreeMap<BosonKey, i128> = BTreeMap::new();
    aggregate.insert(Vec::new(), 1);
    let modes: BTreeSet<u32> = left_by_mode
        .keys()
        .chain(right_by_mode.keys())
        .copied()
        .collect();
    for mode in modes {
        let (left_creation, left_annihilation) = left_by_mode.get(&mode).copied().unwrap_or((0, 0));
        let (right_creation, right_annihilation) =
            right_by_mode.get(&mode).copied().unwrap_or((0, 0));
        let mut local = BTreeMap::new();
        let maximum = left_annihilation.min(right_creation);
        for contractions in 0..=maximum {
            let coefficient =
                contraction_coefficient(left_annihilation, right_creation, contractions)?;
            let creation = left_creation
                .checked_add(right_creation)
                .and_then(|value| value.checked_sub(contractions))
                .ok_or(PauliError::Overflow {
                    context: "boson product power",
                })?;
            let annihilation = left_annihilation
                .checked_add(right_annihilation)
                .and_then(|value| value.checked_sub(contractions))
                .ok_or(PauliError::Overflow {
                    context: "boson product power",
                })?;
            add_integer(&mut local, (creation, annihilation), coefficient)?;
        }
        let product_count = (aggregate.len() as u128)
            .checked_mul(local.len() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating boson product expansion",
            })?;
        check_structured_bytes_u128(product_count, max_bytes, "boson product expansion")?;
        let mut next = BTreeMap::new();
        for (prefix, prefix_value) in aggregate {
            for (&(creation, annihilation), &local_value) in &local {
                let mut blocks = prefix.clone();
                if creation != 0 || annihilation != 0 {
                    blocks.push((mode, creation, annihilation));
                }
                add_integer(
                    &mut next,
                    blocks,
                    prefix_value
                        .checked_mul(local_value)
                        .ok_or(PauliError::Overflow {
                            context: "boson product coefficient",
                        })?,
                )?;
            }
        }
        aggregate = next;
    }
    Ok(aggregate)
}
use super::*;
