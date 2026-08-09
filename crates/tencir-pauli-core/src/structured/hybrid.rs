fn validate_embedding_map(
    mapping: &[usize],
    source_count: usize,
    target_count: usize,
) -> Result<(), PauliError> {
    if mapping.len() != source_count || mapping.iter().any(|&value| value >= target_count) {
        return Err(PauliError::InvalidStructureLength {
            expected: source_count,
            actual: mapping.len(),
        });
    }
    let mut seen = FxHashMap::default();
    for &value in mapping {
        if seen.insert(value, ()).is_some() {
            return Err(PauliError::InvalidStructureLength {
                expected: source_count,
                actual: mapping.len(),
            });
        }
    }
    Ok(())
}

fn embedding_sign(values: &[u32], mapping: &[usize], descending: bool) -> f64 {
    let mut odd = false;
    for (index, &value) in values.iter().enumerate() {
        let left = mapping[value as usize];
        for &right_value in &values[index + 1..] {
            let right = mapping[right_value as usize];
            if if descending {
                left < right
            } else {
                left > right
            } {
                odd = !odd;
            }
        }
    }
    if odd {
        -1.0
    } else {
        1.0
    }
}

/// Embed a canonical mixed-domain operator into a larger ordered layout.
///
/// The source batch remains borrowed for the complete operation. Factors are
/// remapped, fermionic permutation signs are applied, and collisions are
/// aggregated before the canonical result crosses the FFI boundary.
#[allow(clippy::too_many_arguments)]
pub fn embed_hybrid_terms(
    source_layout: HybridLayout,
    target_layout: HybridLayout,
    batch: HybridBatch<'_>,
    fermion_map: &[usize],
    boson_map: &[usize],
    qubit_map: &[usize],
    qudit_map: &[usize],
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    validate_embedding_map(fermion_map, source_layout.n_modes, target_layout.n_modes)?;
    validate_embedding_map(boson_map, source_layout.n_bosons, target_layout.n_bosons)?;
    validate_embedding_map(qubit_map, source_layout.nqubits, target_layout.nqubits)?;
    validate_embedding_map(
        qudit_map,
        source_layout.n_qudit_sites,
        target_layout.n_qudit_sites,
    )?;
    if source_layout.n_qudit_sites != 0
        && source_layout.qudit_dimension != target_layout.qudit_dimension
    {
        return Err(PauliError::InvalidStructureLength {
            expected: source_layout.qudit_dimension,
            actual: target_layout.qudit_dimension,
        });
    }
    check_structured_bytes(
        batch.coefficients.len(),
        max_bytes,
        "structured embedding output",
    )?;
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..batch.coefficients.len() {
        let fermion = if batch.fermion_present[index] {
            let creation = batch.fermion_creation[index]
                .iter()
                .map(|&mode| fermion_map[mode as usize] as u32)
                .collect::<Vec<_>>();
            let annihilation = batch.fermion_annihilation[index]
                .iter()
                .map(|&mode| fermion_map[mode as usize] as u32)
                .collect::<Vec<_>>();
            let sign = embedding_sign(&batch.fermion_creation[index], fermion_map, false)
                * embedding_sign(&batch.fermion_annihilation[index], fermion_map, true);
            let mut creation = creation;
            let mut annihilation = annihilation;
            creation.sort_unstable();
            annihilation.sort_unstable_by(|left, right| right.cmp(left));
            Some((creation, annihilation, sign))
        } else {
            None
        };
        let boson = if batch.boson_present[index] {
            let mut blocks = batch.boson_blocks[index]
                .iter()
                .map(|&(mode, creation, annihilation)| {
                    (boson_map[mode as usize] as u32, creation, annihilation)
                })
                .collect::<Vec<_>>();
            blocks.sort_unstable_by_key(|block| block.0);
            Some(blocks)
        } else {
            None
        };
        let mut qubit = vec![0_u8; target_layout.nqubits];
        for (source, &code) in batch.qubit_codes[index].iter().enumerate() {
            qubit[qubit_map[source]] = code;
        }
        let mapped = if batch.mapped_present[index] {
            let mut codes = vec![0_u8; target_layout.n_modes];
            for (source, &code) in batch.mapped_codes[index].iter().enumerate() {
                codes[fermion_map[source]] = code;
            }
            Some(codes)
        } else {
            None
        };
        let qudit = if batch.qudit_present[index] {
            let mut triples = batch.qudit_triples[index]
                .iter()
                .map(|&(site, a, b)| (qudit_map[site as usize] as u32, a, b))
                .collect::<Vec<_>>();
            triples.sort_unstable_by_key(|triple| triple.0);
            Some(triples)
        } else {
            None
        };
        let fermion_sign = fermion.as_ref().map_or(1.0, |(_, _, sign)| *sign);
        let fermion_key = fermion.map(|(creation, annihilation, _)| (creation, annihilation));
        push_aggregate(
            &mut aggregate,
            (fermion_key, boson, qubit, mapped, qudit),
            batch.coefficients[index] * fermion_sign,
            &mut total_values,
            max_bytes,
        )?;
    }
    finish_hybrid_aggregate(aggregate)
}

/// Analyze a diagonal additive charge directly on canonical hybrid storage.
pub fn analyze_hybrid_charge(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    fermion_weights: &[f64],
    boson_weights: &[f64],
    qubit_levels: &[(f64, f64)],
    qudit_weights: &[f64],
    max_bytes: u128,
) -> Result<(bool, usize), PauliError> {
    if fermion_weights.len() != layout.n_modes
        || boson_weights.len() != layout.n_bosons
        || qubit_levels.len() != layout.nqubits
        || qudit_weights.len() != layout.n_qudit_sites
    {
        return Err(PauliError::InvalidStructureLength {
            expected: layout.n_modes + layout.n_bosons + layout.nqubits + layout.n_qudit_sites,
            actual: fermion_weights.len()
                + boson_weights.len()
                + qubit_levels.len()
                + qudit_weights.len(),
        });
    }
    if batch.mapped_present.iter().any(|&present| present) {
        return Err(PauliError::InvalidStructureLength {
            expected: 0,
            actual: 1,
        });
    }
    check_structured_bytes(
        batch.coefficients.len(),
        max_bytes,
        "estimating additive-charge analysis",
    )?;
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut nonconserving_qudit: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..batch.coefficients.len() {
        let fermion_key = if batch.fermion_present[index] {
            Some((
                batch.fermion_creation[index].clone(),
                batch.fermion_annihilation[index].clone(),
            ))
        } else {
            None
        };
        let boson_key = if batch.boson_present[index] {
            Some(batch.boson_blocks[index].clone())
        } else {
            None
        };
        let qubit_key = batch.qubit_codes[index].clone();
        let qudit_key = if batch.qudit_present[index] {
            Some(batch.qudit_triples[index].clone())
        } else {
            None
        };
        let base_key = (
            fermion_key.clone(),
            boson_key.clone(),
            qubit_key.clone(),
            None,
            qudit_key.clone(),
        );
        let mut raw_delta = 0.0;
        if batch.fermion_present[index] {
            raw_delta += batch.fermion_creation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
            raw_delta -= batch.fermion_annihilation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
        }
        if batch.boson_present[index] {
            raw_delta += batch.boson_blocks[index]
                .iter()
                .map(|&(mode, creation, annihilation)| {
                    boson_weights[mode as usize] * (creation as f64 - annihilation as f64)
                })
                .sum::<f64>();
        }
        if batch.qudit_present[index]
            && batch.qudit_triples[index]
                .iter()
                .any(|&(site, a, _)| a != 0 && qudit_weights[site as usize] != 0.0)
        {
            push_aggregate(
                &mut nonconserving_qudit,
                base_key.clone(),
                batch.coefficients[index],
                &mut total_values,
                max_bytes,
            )?;
        }
        if raw_delta != 0.0 {
            push_aggregate(
                &mut aggregate,
                base_key.clone(),
                batch.coefficients[index] * -raw_delta,
                &mut total_values,
                max_bytes,
            )?;
        }
        for (qubit, &code) in qubit_key.iter().enumerate() {
            if code != 1 && code != 2 {
                continue;
            }
            let difference = qubit_levels[qubit].0 - qubit_levels[qubit].1;
            if difference == 0.0 {
                continue;
            }
            let mut changed = qubit_key.clone();
            changed[qubit] = if code == 1 { 2 } else { 1 };
            let key = (
                fermion_key.clone(),
                boson_key.clone(),
                changed,
                None,
                qudit_key.clone(),
            );
            let scale = if code == 1 { -difference } else { difference };
            push_aggregate(
                &mut aggregate,
                key,
                batch.coefficients[index] * Complex64::new(0.0, scale),
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    let nonzero = aggregate
        .into_values()
        .chain(nonconserving_qudit.into_values())
        .filter_map(|mut values| {
            values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
            let value = deterministic_sum(values);
            (!is_exact_zero(value)).then_some(())
        })
        .count();
    Ok((nonzero == 0, nonzero))
}

/// Return whether every canonical hybrid term preserves a diagonal charge.
pub fn hybrid_terms_conserve_charge(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    fermion_weights: &[f64],
    boson_weights: &[f64],
    qubit_levels: &[(f64, f64)],
    qudit_weights: &[f64],
) -> Result<bool, PauliError> {
    if fermion_weights.len() != layout.n_modes
        || boson_weights.len() != layout.n_bosons
        || qubit_levels.len() != layout.nqubits
        || qudit_weights.len() != layout.n_qudit_sites
    {
        return Err(PauliError::InvalidStructureLength {
            expected: layout.n_modes + layout.n_bosons + layout.nqubits + layout.n_qudit_sites,
            actual: fermion_weights.len()
                + boson_weights.len()
                + qubit_levels.len()
                + qudit_weights.len(),
        });
    }
    for index in 0..batch.coefficients.len() {
        if batch.mapped_present[index] {
            return Ok(false);
        }
        if batch.qudit_present[index]
            && batch.qudit_triples[index]
                .iter()
                .any(|&(site, a, _)| a != 0 && qudit_weights[site as usize] != 0.0)
        {
            return Ok(false);
        }
        let mut delta = 0.0;
        if batch.fermion_present[index] {
            delta += batch.fermion_creation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
            delta -= batch.fermion_annihilation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
        }
        if batch.boson_present[index] {
            delta += batch.boson_blocks[index]
                .iter()
                .map(|&(mode, creation, annihilation)| {
                    boson_weights[mode as usize] * (creation as f64 - annihilation as f64)
                })
                .sum::<f64>();
        }
        if delta != 0.0
            || batch.qubit_codes[index]
                .iter()
                .enumerate()
                .any(|(qubit, &code)| {
                    (code == 1 || code == 2) && qubit_levels[qubit].0 != qubit_levels[qubit].1
                })
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Multiply complete canonical hybrid operators in one coarse-grained call.
pub fn multiply_hybrid_terms(
    layout: HybridLayout,
    left: HybridBatch<'_>,
    right: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    binary_hybrid_terms(layout, left, right, max_bytes, 1, 0)
}

/// Compute a commutator or anticommutator of canonical hybrid operators in a
/// single aggregate. `reverse_sign` is `-1` for a commutator and `1` for an
/// anticommutator; `forward_sign` is kept explicit so the same kernel also
/// serves the ordinary product.
pub fn binary_hybrid_terms(
    layout: HybridLayout,
    left: HybridBatch<'_>,
    right: HybridBatch<'_>,
    max_bytes: u128,
    forward_sign: i8,
    reverse_sign: i8,
) -> Result<HybridCanonicalResult, PauliError> {
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating hybrid product expansion",
        })?;
    let directional_count = if reverse_sign == 0 {
        pair_count
    } else {
        pair_count.checked_mul(2).ok_or(PauliError::Overflow {
            context: "estimating hybrid binary expansion",
        })?
    };
    check_structured_bytes(directional_count, max_bytes, "hybrid product expansion")?;

    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            accumulate_hybrid_product(
                layout,
                &left,
                left_index,
                &right,
                right_index,
                forward_sign,
                &mut aggregate,
                &mut total_values,
                max_bytes,
            )?;
            if reverse_sign != 0 {
                accumulate_hybrid_product(
                    layout,
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
    finish_hybrid_aggregate(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_hybrid_product(
    layout: HybridLayout,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
    direction_sign: i8,
    aggregate: &mut FxHashMap<HybridKey, Vec<Complex64>>,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let fermion_products =
        hybrid_fermion_products(left, left_index, right, right_index, max_bytes)?;
    let boson_products = hybrid_boson_products(left, left_index, right, right_index, max_bytes)?;
    let (qubit_codes, qubit_phase) = multiply_pauli_codes(
        &left.qubit_codes[left_index],
        &right.qubit_codes[right_index],
    );
    let (mapped_codes, mapped_phase) =
        hybrid_mapped_product(layout.n_modes, left, left_index, right, right_index);
    let (qudit_triples, qudit_phase) =
        hybrid_qudit_product(layout.qudit_dimension, left, left_index, right, right_index);
    let scalar = left.coefficients[left_index]
        * right.coefficients[right_index]
        * qubit_phase
        * mapped_phase
        * qudit_phase
        * f64::from(direction_sign);
    for (fermion, fermion_integer) in &fermion_products {
        for (boson, boson_integer) in &boson_products {
            let value = scalar * *fermion_integer as f64 * *boson_integer as f64;
            push_aggregate(
                aggregate,
                (
                    fermion.clone(),
                    boson.clone(),
                    qubit_codes.clone(),
                    mapped_codes.clone(),
                    qudit_triples.clone(),
                ),
                value,
                total_values,
                max_bytes,
            )?;
        }
    }
    Ok(())
}

/// Canonicalize raw mixed-domain product specifications in one batch.
pub fn canonicalize_hybrid_terms(
    layout: HybridLayout,
    batch: HybridRawBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    let count = batch.coefficients.len();
    if batch.fermion_factors.len() != count
        || batch.boson_factors.len() != count
        || batch.qubit_codes.len() != count
        || batch.qudit_present.len() != count
        || batch.qudit_triples.len() != count
    {
        return Err(PauliError::InvalidStructureLength {
            expected: count,
            actual: batch.fermion_factors.len(),
        });
    }
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..count {
        if batch.qubit_codes[index].len() != layout.nqubits {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.nqubits,
                actual: batch.qubit_codes[index].len(),
            });
        }
        if batch.qubit_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        let fermion_products = if batch.fermion_factors[index].is_empty() {
            vec![(None, 1)]
        } else {
            check_structured_bytes_u128(
                batch.fermion_factors[index].len() as u128,
                max_bytes,
                "hybrid fermion raw expansion",
            )?;
            let sequence = raw_sequence(
                &batch.fermion_factors[index],
                layout.n_modes,
                index,
                "fermion mode",
            )?;
            fermion_rewrite(sequence, max_bytes)?
                .into_iter()
                .map(|(key, value)| (Some(key), value))
                .collect()
        };
        let boson_products = if batch.boson_factors[index].is_empty() {
            vec![(None, 1)]
        } else {
            check_structured_bytes_u128(
                batch.boson_factors[index].len() as u128,
                max_bytes,
                "hybrid boson raw expansion",
            )?;
            let sequence = raw_sequence(
                &batch.boson_factors[index],
                layout.n_bosons,
                index,
                "boson mode",
            )?;
            boson_rewrite(sequence, max_bytes)?
                .into_iter()
                .map(|(key, value)| (Some(key), value))
                .collect()
        };
        if batch.qudit_triples[index]
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
            || batch.qudit_triples[index].iter().any(|&(site, a, b)| {
                site as usize >= layout.n_qudit_sites
                    || a as usize >= layout.qudit_dimension
                    || b as usize >= layout.qudit_dimension
            })
        {
            return Err(PauliError::NonCanonicalTerms { index });
        }
        for (fermion, fermion_integer) in &fermion_products {
            for (boson, boson_integer) in &boson_products {
                push_aggregate(
                    &mut aggregate,
                    (
                        fermion.clone(),
                        boson.clone(),
                        batch.qubit_codes[index].clone(),
                        None,
                        if batch.qudit_present[index] {
                            Some(batch.qudit_triples[index].clone())
                        } else {
                            None
                        },
                    ),
                    batch.coefficients[index] * *fermion_integer as f64 * *boson_integer as f64,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_hybrid_aggregate(aggregate)
}

fn hybrid_fermion_products(
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
    max_bytes: u128,
) -> Result<Vec<(Option<FermionKey>, i128)>, PauliError> {
    if !left.fermion_present[left_index] && !right.fermion_present[right_index] {
        return Ok(vec![(None, 1)]);
    }
    let mut sequence = fermion_sequence(
        &left.fermion_creation[left_index],
        &left.fermion_annihilation[left_index],
    );
    sequence.extend(fermion_sequence(
        &right.fermion_creation[right_index],
        &right.fermion_annihilation[right_index],
    ));
    Ok(fermion_rewrite(sequence, max_bytes)?
        .into_iter()
        .map(|(key, value)| (Some(key), value))
        .collect())
}

fn hybrid_boson_products(
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
    max_bytes: u128,
) -> Result<Vec<(Option<BosonKey>, i128)>, PauliError> {
    if !left.boson_present[left_index] && !right.boson_present[right_index] {
        return Ok(vec![(None, 1)]);
    }
    boson_block_product(
        &left.boson_blocks[left_index],
        &right.boson_blocks[right_index],
        max_bytes,
    )
    .map(|values| {
        values
            .into_iter()
            .map(|(key, value)| (Some(key), value))
            .collect()
    })
}

fn hybrid_mapped_product(
    n_modes: usize,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> (Option<Vec<u8>>, Complex64) {
    if !left.mapped_present[left_index] && !right.mapped_present[right_index] {
        return (None, Complex64::new(1.0, 0.0));
    }
    let identity = vec![0_u8; n_modes];
    let left_codes = if left.mapped_present[left_index] {
        &left.mapped_codes[left_index]
    } else {
        &identity
    };
    let right_codes = if right.mapped_present[right_index] {
        &right.mapped_codes[right_index]
    } else {
        &identity
    };
    let (codes, phase) = multiply_pauli_codes(left_codes, right_codes);
    (Some(codes), phase)
}

pub(super) fn hybrid_qudit_product(
    dimension: usize,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> (Option<WeylKey>, Complex64) {
    if !left.qudit_present[left_index] && !right.qudit_present[right_index] {
        return (None, Complex64::new(1.0, 0.0));
    }
    let left_triples: &[(u32, u32, u32)] = if left.qudit_present[left_index] {
        &left.qudit_triples[left_index]
    } else {
        &[]
    };
    let right_triples: &[(u32, u32, u32)] = if right.qudit_present[right_index] {
        &right.qudit_triples[right_index]
    } else {
        &[]
    };
    let mut by_site: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut phase_exponent = 0_u128;
    let mut left_by_site = BTreeMap::new();
    let mut right_by_site = BTreeMap::new();
    for &(site, a, b) in left_triples {
        left_by_site.insert(site, (a, b));
    }
    for &(site, a, b) in right_triples {
        right_by_site.insert(site, (a, b));
    }
    for site in left_by_site
        .keys()
        .chain(right_by_site.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let (a, b) = left_by_site.get(&site).copied().unwrap_or((0, 0));
        let (c, e) = right_by_site.get(&site).copied().unwrap_or((0, 0));
        phase_exponent =
            (phase_exponent + u128::from(b) * u128::from(c)) % u128::from(dimension as u64);
        let aa = (u128::from(a) + u128::from(c)) % u128::from(dimension as u64);
        let bb = (u128::from(b) + u128::from(e)) % u128::from(dimension as u64);
        if aa != 0 || bb != 0 {
            by_site.insert(site, (aa as u32, bb as u32));
        }
    }
    let triples: Vec<(u32, u32, u32)> = by_site
        .into_iter()
        .map(|(site, (a, b))| (site, a, b))
        .collect();
    let angle = 2.0 * std::f64::consts::PI * phase_exponent as f64 / dimension as f64;
    let triples = if triples.is_empty() {
        None
    } else {
        Some(triples)
    };
    (triples, Complex64::new(angle.cos(), angle.sin()))
}

pub(super) fn validate_hybrid_batch(
    layout: HybridLayout,
    batch: &HybridBatch<'_>,
) -> Result<(), PauliError> {
    let count = batch.coefficients.len();
    let lengths = [
        batch.fermion_present.len(),
        batch.fermion_creation.len(),
        batch.fermion_annihilation.len(),
        batch.boson_present.len(),
        batch.boson_blocks.len(),
        batch.qubit_codes.len(),
        batch.mapped_present.len(),
        batch.mapped_codes.len(),
        batch.qudit_present.len(),
        batch.qudit_triples.len(),
    ];
    if lengths.iter().any(|&length| length != count) {
        return Err(PauliError::InvalidStructureLength {
            expected: count,
            actual: lengths
                .into_iter()
                .find(|&length| length != count)
                .unwrap_or(count),
        });
    }
    for index in 0..count {
        validate_coefficient(batch.coefficients[index], index)?;
        validate_fermion_arrays(
            layout.n_modes,
            std::slice::from_ref(&batch.fermion_creation[index]),
            std::slice::from_ref(&batch.fermion_annihilation[index]),
            std::slice::from_ref(&batch.coefficients[index]),
        )?;
        validate_boson_blocks(layout.n_bosons, &batch.boson_blocks[index])?;
        if batch.qubit_codes[index].len() != layout.nqubits {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.nqubits,
                actual: batch.qubit_codes[index].len(),
            });
        }
        if batch.qubit_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        if batch.mapped_codes[index].len() != layout.n_modes {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.n_modes,
                actual: batch.mapped_codes[index].len(),
            });
        }
        if batch.mapped_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        if batch.qudit_triples[index]
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
            || batch.qudit_triples[index].iter().any(|&(site, a, b)| {
                site as usize >= layout.n_qudit_sites
                    || a as usize >= layout.qudit_dimension
                    || b as usize >= layout.qudit_dimension
            })
        {
            return Err(PauliError::NonCanonicalTerms { index });
        }
    }
    if layout.n_qudit_sites != 0 && layout.qudit_dimension < 3 {
        return Err(PauliError::InvalidStructureLength {
            expected: 3,
            actual: layout.qudit_dimension,
        });
    }
    Ok(())
}

pub(super) fn finish_hybrid_aggregate(
    aggregate: FxHashMap<HybridKey, Vec<Complex64>>,
) -> Result<HybridCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let capacity = entries.len();
    let mut result = HybridCanonicalResult {
        fermion_present: Vec::with_capacity(capacity),
        fermion_creation: Vec::with_capacity(capacity),
        fermion_annihilation: Vec::with_capacity(capacity),
        boson_present: Vec::with_capacity(capacity),
        boson_blocks: Vec::with_capacity(capacity),
        qubit_codes: Vec::with_capacity(capacity),
        mapped_present: Vec::with_capacity(capacity),
        mapped_codes: Vec::with_capacity(capacity),
        qudit_present: Vec::with_capacity(capacity),
        qudit_triples: Vec::with_capacity(capacity),
        coefficients: Vec::with_capacity(capacity),
    };
    for ((fermion, boson, qubit, mapped, qudit), values) in entries {
        let coefficient = deterministic_sum(values);
        if is_exact_zero(coefficient) {
            continue;
        }
        if let Some((creation, annihilation)) = fermion {
            result.fermion_present.push(true);
            result.fermion_creation.push(creation);
            result.fermion_annihilation.push(annihilation);
        } else {
            result.fermion_present.push(false);
            result.fermion_creation.push(Vec::new());
            result.fermion_annihilation.push(Vec::new());
        }
        if let Some(blocks) = boson {
            result.boson_present.push(true);
            result.boson_blocks.push(blocks);
        } else {
            result.boson_present.push(false);
            result.boson_blocks.push(Vec::new());
        }
        result.qubit_codes.push(qubit);
        if let Some(codes) = mapped {
            result.mapped_present.push(true);
            result.mapped_codes.push(codes);
        } else {
            result.mapped_present.push(false);
            result.mapped_codes.push(Vec::new());
        }
        if let Some(triples) = qudit {
            result.qudit_present.push(true);
            result.qudit_triples.push(triples);
        } else {
            result.qudit_present.push(false);
            result.qudit_triples.push(Vec::new());
        }
        result.coefficients.push(coefficient);
    }
    Ok(result)
}
use super::*;
