fn build_additive_factor_actions(
    factor: &ChargeSectorPlan,
    terms: &[ChargeTransitionTerm],
    factor_dimensions: &[usize],
    qudit_dimension: usize,
    boson_mode: bool,
    domain_to_factor: Option<&[usize]>,
) -> Result<Option<FactorActions>, PauliError> {
    if terms.iter().any(|term| {
        !term.fermion_creation.is_empty()
            || !term.fermion_annihilation.is_empty()
            || term.mapped_present
            || term.qubit_codes.iter().any(|&code| code != 0)
            || if boson_mode {
                term.qudit_present || !term.qudit_triples.is_empty()
            } else {
                !term.boson_blocks.is_empty()
            }
    }) {
        return Ok(None);
    }
    let factor_dimension = factor.dimension();
    let mut actions = Vec::with_capacity(factor_dimension);
    let mut source = vec![0_u64; factor_dimensions.len()];
    let mut destination = vec![0_u64; factor_dimensions.len()];
    let mut remaining = vec![0_i128; 1];
    let mut candidate = vec![0_i128; 1];
    for source_rank in 0..factor_dimension {
        factor.unrank_into_with_scratch(
            source_rank as u64,
            &mut source,
            &mut remaining,
            &mut candidate,
        )?;
        let mut grouped = BTreeMap::<Vec<u64>, Complex64>::new();
        for term in terms {
            destination.copy_from_slice(&source);
            let mut value = term.coefficient;
            let mut valid = true;
            if boson_mode {
                for &(axis, creation, annihilation) in &term.boson_blocks {
                    let axis = usize::try_from(axis).map_err(|_| PauliError::Overflow {
                        context: "converting fast boson factor axis",
                    })?;
                    let axis = domain_to_factor
                        .map(|mapping| mapping.get(axis).copied().unwrap_or(usize::MAX))
                        .unwrap_or(axis);
                    if axis == usize::MAX {
                        return Ok(None);
                    }
                    let creation = usize::try_from(creation).map_err(|_| PauliError::Overflow {
                        context: "converting fast boson creation power",
                    })?;
                    let annihilation =
                        usize::try_from(annihilation).map_err(|_| PauliError::Overflow {
                            context: "converting fast boson annihilation power",
                        })?;
                    if axis >= destination.len() || source[axis] < annihilation as u64 {
                        valid = false;
                        break;
                    }
                    let level =
                        usize::try_from(source[axis]).map_err(|_| PauliError::Overflow {
                            context: "converting fast boson factor level",
                        })?;
                    let Some(new_level) = level
                        .checked_sub(annihilation)
                        .and_then(|value| value.checked_add(creation))
                    else {
                        valid = false;
                        break;
                    };
                    if new_level >= factor_dimensions[axis] {
                        valid = false;
                        break;
                    }
                    let mut ladder = 1.0;
                    for offset in 0..annihilation {
                        ladder *= ((level - offset) as f64).sqrt();
                    }
                    let remaining_level = level - annihilation;
                    for offset in 0..creation {
                        ladder *= ((remaining_level + offset + 1) as f64).sqrt();
                    }
                    destination[axis] = new_level as u64;
                    value *= Complex64::new(ladder, 0.0);
                }
            } else {
                for &(site, shift, phase_power) in &term.qudit_triples {
                    let site = usize::try_from(site).map_err(|_| PauliError::Overflow {
                        context: "converting fast qudit factor axis",
                    })?;
                    let site = domain_to_factor
                        .map(|mapping| mapping.get(site).copied().unwrap_or(usize::MAX))
                        .unwrap_or(site);
                    if site == usize::MAX {
                        return Ok(None);
                    }
                    if site >= destination.len() || qudit_dimension == 0 {
                        return Err(invalid_sector());
                    }
                    let digit =
                        usize::try_from(source[site]).map_err(|_| PauliError::Overflow {
                            context: "converting fast qudit factor level",
                        })?;
                    let shift = usize::try_from(shift).map_err(|_| PauliError::Overflow {
                        context: "converting fast qudit shift",
                    })?;
                    destination[site] = ((digit + shift) % qudit_dimension) as u64;
                    let exponent =
                        (u128::from(phase_power) * digit as u128) % qudit_dimension as u128;
                    let angle =
                        2.0 * std::f64::consts::PI * exponent as f64 / qudit_dimension as f64;
                    value *= Complex64::from_polar(1.0, angle);
                }
            }
            if valid {
                grouped
                    .entry(destination.clone())
                    .and_modify(|existing| *existing += value)
                    .or_insert(value);
            }
        }
        let mut row_actions = Vec::with_capacity(grouped.len());
        for (destination, value) in grouped {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let rank = factor.rank_into(&destination, &mut remaining, &mut candidate)?;
            row_actions.push((
                usize::try_from(rank).map_err(|_| PauliError::Overflow {
                    context: "converting fast additive factor rank",
                })?,
                value,
            ));
        }
        actions.push(row_actions);
    }
    Ok(Some(actions.into_boxed_slice()))
}

/// Compact additive spin-z qudit factor with arbitrary Cartesian spectators.
pub struct FastAdditiveMvpPlan {
    layout: FastProductLayout,
    factor_actions: FactorActions,
    estimated_bytes: u128,
}

impl FastAdditiveMvpPlan {
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        _max_bytes: u128,
    ) -> Result<(), PauliError> {
        if state.len() != output.len() || state.len() != self.layout.expected_dimension()? {
            return Err(invalid_sector());
        }
        output.fill(Complex64::default());
        let factor_dimension = self.layout.factor_dimension;
        let suffix_dimension = self.layout.suffix_dimension;
        let block_dimension =
            factor_dimension
                .checked_mul(suffix_dimension)
                .ok_or(PauliError::Overflow {
                    context: "indexing fast additive product block",
                })?;
        for (column, state_value) in state.iter().enumerate() {
            let factor_rank = (column / suffix_dimension) % factor_dimension;
            let prefix = column / block_dimension;
            let suffix = column % suffix_dimension;
            for &(destination_rank, value) in &self.factor_actions[factor_rank] {
                let row = prefix
                    .checked_mul(block_dimension)
                    .and_then(|base| {
                        base.checked_add(destination_rank.checked_mul(suffix_dimension)?)
                    })
                    .and_then(|base| base.checked_add(suffix))
                    .ok_or(PauliError::Overflow {
                        context: "indexing fast additive product destination",
                    })?;
                output[row] += value * *state_value;
            }
        }
        Ok(())
    }
}

/// Build an exact canonical-spin-z qudit factor with Cartesian spectators.
pub fn build_fast_qudit_mvp_plan(
    context: &ChargeBuildContext<'_>,
    weights: Option<&[i128]>,
    target: Option<i128>,
    factor_positions_override: Option<&[u64]>,
) -> Result<Option<FastAdditiveMvpPlan>, PauliError> {
    let plan = context.sector();
    let local_dimensions = context.local_dimensions();
    let fermion_positions = context.fermion_positions();
    let boson_positions = context.boson_positions();
    let qubit_positions = context.qubit_positions();
    let all_qudit_positions = context.qudit_positions();
    let qudit_dimension = context.qudit_dimension();
    let terms = context.terms();
    let max_bytes = context.max_bytes();
    let (Some(weights), Some(target)) = (weights, target) else {
        return Ok(None);
    };
    let qudit_positions = factor_positions_override
        .map(|values| positions(values, local_dimensions.len(), max_bytes))
        .transpose()?
        .unwrap_or_else(|| all_qudit_positions.to_vec());
    if qudit_positions.is_empty()
        || weights.len() != qudit_positions.len()
        || weights.contains(&0)
        || qudit_positions
            .iter()
            .enumerate()
            .any(|(offset, &position)| {
                position != qudit_positions[0] + offset
                    || local_dimensions[position] != qudit_dimension
                    || qudit_dimension < 3
            })
    {
        return Ok(None);
    }
    let factor_dimensions = vec![
        usize::try_from(qudit_dimension).map_err(|_| {
            PauliError::Overflow {
                context: "converting fast qudit dimension",
            }
        })?;
        qudit_positions.len()
    ];
    let contributions = factor_dimensions
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            (0..dimension)
                .map(|level| {
                    let level = i128::try_from(level).map_err(|_| PauliError::Overflow {
                        context: "converting fast qudit level",
                    })?;
                    let spectrum = if dimension % 2 == 1 {
                        i128::try_from((dimension - 1) / 2).map_err(|_| PauliError::Overflow {
                            context: "computing fast qudit spectrum",
                        })? - level
                    } else {
                        i128::try_from(dimension - 1).map_err(|_| PauliError::Overflow {
                            context: "computing fast qudit spectrum",
                        })? - level.checked_mul(2).ok_or(PauliError::Overflow {
                            context: "computing fast qudit spectrum",
                        })?
                    };
                    Ok(vec![weights[axis].checked_mul(spectrum).ok_or(
                        PauliError::Overflow {
                            context: "computing fast qudit contribution",
                        },
                    )?])
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let factor =
        build_charge_sector_plan(factor_dimensions, contributions, vec![target], max_bytes)?;
    let factor_dimension = factor.dimension();
    let Some(layout) = FastProductLayout::new(
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        all_qudit_positions,
        &qudit_positions,
        factor_dimension,
    )?
    else {
        return Ok(None);
    };
    if plan.dimension() != layout.expected_dimension()? {
        return Ok(None);
    }
    let factor_actions = build_additive_factor_actions(
        &factor,
        terms,
        factor.local_dimensions(),
        usize::try_from(qudit_dimension).map_err(|_| PauliError::Overflow {
            context: "converting fast qudit dimension",
        })?,
        false,
        Some(
            &all_qudit_positions
                .iter()
                .map(|position| {
                    qudit_positions
                        .iter()
                        .position(|candidate| candidate == position)
                        .unwrap_or(usize::MAX)
                })
                .collect::<Vec<_>>(),
        ),
    )?;
    if factor_actions.is_none() {
        return Ok(None);
    }
    let action_bytes = factor_actions
        .as_ref()
        .map(|actions| actions.iter().map(Vec::len).sum::<usize>())
        .unwrap_or(0)
        .checked_mul(std::mem::size_of::<(usize, Complex64)>())
        .ok_or(PauliError::Overflow {
            context: "estimating fast qudit factor actions",
        })?;
    let term_bytes = estimate_charge_transition_terms_bytes(terms)?;
    let estimated_bytes = layout
        .descriptor_bytes()
        .checked_add(factor.estimated_bytes())
        .and_then(|value| value.checked_add(term_bytes))
        .and_then(|value| value.checked_add(action_bytes as u128))
        .ok_or(PauliError::Overflow {
            context: "estimating fast qudit plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(Some(FastAdditiveMvpPlan {
        layout,
        factor_actions: factor_actions.expect("validated additive factor actions"),
        estimated_bytes,
    }))
}

/// Build a fixed-number boson factor with arbitrary Cartesian spectators.
pub fn build_fast_boson_mvp_plan(
    context: &ChargeBuildContext<'_>,
    weights: Option<&[i128]>,
    target: Option<i128>,
    factor_positions_override: Option<&[u64]>,
) -> Result<Option<FastAdditiveMvpPlan>, PauliError> {
    let plan = context.sector();
    let local_dimensions = context.local_dimensions();
    let fermion_positions = context.fermion_positions();
    let all_boson_positions = context.boson_positions();
    let qubit_positions = context.qubit_positions();
    let qudit_positions = context.qudit_positions();
    let terms = context.terms();
    let max_bytes = context.max_bytes();
    let (Some(weights), Some(target)) = (weights, target) else {
        return Ok(None);
    };
    let boson_positions = factor_positions_override
        .map(|values| positions(values, local_dimensions.len(), max_bytes))
        .transpose()?
        .unwrap_or_else(|| all_boson_positions.to_vec());
    if boson_positions.is_empty()
        || weights.len() != boson_positions.len()
        || weights.contains(&0)
        || boson_positions
            .iter()
            .enumerate()
            .any(|(offset, &position)| position != boson_positions[0] + offset)
    {
        return Ok(None);
    }
    let factor_dimensions = boson_positions
        .iter()
        .map(|&position| usize::try_from(local_dimensions[position]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PauliError::Overflow {
            context: "converting fast boson dimensions",
        })?;
    if factor_dimensions.contains(&0) {
        return Ok(None);
    }
    let contributions = factor_dimensions
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            (0..dimension)
                .map(|level| {
                    let level = i128::try_from(level).map_err(|_| PauliError::Overflow {
                        context: "converting fast boson level",
                    })?;
                    Ok(vec![weights[axis].checked_mul(level).ok_or(
                        PauliError::Overflow {
                            context: "computing fast boson contribution",
                        },
                    )?])
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let factor =
        build_charge_sector_plan(factor_dimensions, contributions, vec![target], max_bytes)?;
    let factor_dimension = factor.dimension();
    let factor_positions = boson_positions.clone();
    let Some(layout) = FastProductLayout::new(
        local_dimensions,
        fermion_positions,
        all_boson_positions,
        qubit_positions,
        qudit_positions,
        &factor_positions,
        factor_dimension,
    )?
    else {
        return Ok(None);
    };
    if plan.dimension() != layout.expected_dimension()? {
        return Ok(None);
    }
    let factor_actions = build_additive_factor_actions(
        &factor,
        terms,
        factor.local_dimensions(),
        0,
        true,
        Some(
            &all_boson_positions
                .iter()
                .map(|position| {
                    boson_positions
                        .iter()
                        .position(|candidate| candidate == position)
                        .unwrap_or(usize::MAX)
                })
                .collect::<Vec<_>>(),
        ),
    )?;
    if factor_actions.is_none() {
        return Ok(None);
    }
    let action_bytes = factor_actions
        .as_ref()
        .map(|actions| actions.iter().map(Vec::len).sum::<usize>())
        .unwrap_or(0)
        .checked_mul(std::mem::size_of::<(usize, Complex64)>())
        .ok_or(PauliError::Overflow {
            context: "estimating fast boson factor actions",
        })?;
    let term_bytes = estimate_charge_transition_terms_bytes(terms)?;
    let estimated_bytes = layout
        .descriptor_bytes()
        .checked_add(factor.estimated_bytes())
        .and_then(|value| value.checked_add(term_bytes))
        .and_then(|value| value.checked_add(action_bytes as u128))
        .ok_or(PauliError::Overflow {
            context: "estimating fast boson plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(Some(FastAdditiveMvpPlan {
        layout,
        factor_actions: factor_actions.expect("validated additive factor actions"),
        estimated_bytes,
    }))
}
use super::*;
