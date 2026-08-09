/// Return ``n choose k`` without overflowing the intermediate product when
/// the final result still fits in ``u128``.
pub(in crate::charge::implementation) fn binomial(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1_u128;
    for index in 1..=k {
        let mut numerator = u128::from((n - k + index) as u64);
        let mut denominator = u128::from(index as u64);
        let common = gcd(numerator, denominator);
        numerator /= common;
        denominator /= common;
        let common = gcd(result, denominator);
        result /= common;
        denominator /= common;
        if denominator != 1 {
            return None;
        }
        result = result.checked_mul(numerator)?;
    }
    Some(result)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

pub(in crate::charge::implementation) fn unrank_combination(
    length: usize,
    particles: usize,
    mut rank: u128,
) -> Option<u128> {
    let mut mask = 0_u128;
    let mut remaining = particles;
    for position in 0..length {
        if remaining == 0 {
            break;
        }
        let zero_count = binomial(length - position - 1, remaining)?;
        if rank < zero_count {
            continue;
        }
        rank -= zero_count;
        mask |= 1_u128 << position;
        remaining -= 1;
    }
    if remaining == 0 && rank == 0 {
        Some(mask)
    } else {
        None
    }
}

pub(in crate::charge::implementation) fn rank_combination(
    length: usize,
    particles: usize,
    mask: u128,
) -> Option<u128> {
    if mask.count_ones() as usize != particles || (length < 128 && mask >> length != 0) {
        return None;
    }
    let mut rank = 0_u128;
    let mut remaining = particles;
    for position in 0..length {
        if mask & (1_u128 << position) != 0 {
            rank = rank.checked_add(binomial(length - position - 1, remaining)?)?;
            remaining -= 1;
        }
    }
    (remaining == 0).then_some(rank)
}

pub(in crate::charge::implementation) type FactorActions = Box<[Vec<(usize, Complex64)>]>;

pub(in crate::charge::implementation) fn build_qubit_factor_actions(
    terms: &[ChargeTransitionTerm],
    factor_length: usize,
    active_qubit_indices: &[usize],
    particles: usize,
    factor_dimension: usize,
) -> Result<Option<FactorActions>, PauliError> {
    if terms.iter().any(|term| {
        !term.fermion_creation.is_empty()
            || !term.fermion_annihilation.is_empty()
            || !term.boson_blocks.is_empty()
            || term.mapped_present
            || term.qudit_present
    }) {
        return Ok(None);
    }
    let mut actions = Vec::with_capacity(factor_dimension);
    for source_rank in 0..factor_dimension {
        let source_mask = unrank_combination(factor_length, particles, source_rank as u128).ok_or(
            PauliError::InvalidSector {
                context: "fast qubit factor unrank failed",
            },
        )?;
        let mut grouped = BTreeMap::<u128, Complex64>::new();
        for term in terms {
            if term.qubit_codes.len() <= active_qubit_indices.iter().copied().max().unwrap_or(0) {
                return Ok(None);
            }
            let mut destination = source_mask;
            let mut value = term.coefficient;
            if term
                .qubit_codes
                .iter()
                .enumerate()
                .any(|(index, &code)| !active_qubit_indices.contains(&index) && code != 0)
            {
                return Ok(None);
            }
            for (offset, &global_index) in active_qubit_indices.iter().enumerate() {
                let code = term.qubit_codes[global_index];
                let digit = (source_mask >> offset) & 1;
                match code {
                    0 => {}
                    1 => destination ^= 1_u128 << offset,
                    2 => {
                        destination ^= 1_u128 << offset;
                        value *= if digit == 0 {
                            Complex64::new(0.0, 1.0)
                        } else {
                            Complex64::new(0.0, -1.0)
                        };
                    }
                    3 => {
                        if digit != 0 {
                            value = -value;
                        }
                    }
                    _ => {
                        return Err(PauliError::InvalidCode {
                            code,
                            index: offset,
                        });
                    }
                }
            }
            grouped
                .entry(destination)
                .and_modify(|existing| *existing += value)
                .or_insert(value);
        }
        let mut row_actions = Vec::with_capacity(grouped.len());
        for (destination, value) in grouped {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let destination_rank = rank_combination(factor_length, particles, destination).ok_or(
                PauliError::InvalidSector {
                    context: "operator leaks outside the selected qubit sector",
                },
            )?;
            let destination_rank =
                usize::try_from(destination_rank).map_err(|_| PauliError::Overflow {
                    context: "converting fast qubit factor rank",
                })?;
            row_actions.push((destination_rank, value));
        }
        actions.push(row_actions);
    }
    Ok(Some(actions.into_boxed_slice()))
}
use super::*;
