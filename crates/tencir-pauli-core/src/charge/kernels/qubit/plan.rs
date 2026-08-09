/// Compact fixed-Hamming-weight qubit factor with arbitrary Cartesian spectators.
pub struct FastQubitMvpPlan {
    layout: FastProductLayout,
    factor_actions: FactorActions,
    estimated_bytes: u128,
}

impl FastQubitMvpPlan {
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
                    context: "indexing fast qubit product block",
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
                        context: "indexing fast qubit product destination",
                    })?;
                output[row] += value * *state_value;
            }
        }
        Ok(())
    }
}

/// Build a fixed-Hamming-weight qubit factor with Cartesian spectators.
pub fn build_fast_qubit_mvp_plan(
    context: &ChargeBuildContext<'_>,
    particles: Option<usize>,
    factor_positions_override: Option<&[u64]>,
) -> Result<Option<FastQubitMvpPlan>, PauliError> {
    let plan = context.sector();
    let local_dimensions = context.local_dimensions();
    let fermion_positions = context.fermion_positions();
    let boson_positions = context.boson_positions();
    let qubit_positions = context.qubit_positions();
    let qudit_positions = context.qudit_positions();
    let terms = context.terms();
    let max_bytes = context.max_bytes();
    let Some(particles) = particles else {
        return Ok(None);
    };
    let factor_positions = factor_positions_override
        .map(|values| positions(values, local_dimensions.len(), max_bytes))
        .transpose()?
        .unwrap_or_else(|| qubit_positions.to_vec());
    if factor_positions.is_empty()
        || factor_positions.len() > 128
        || factor_positions
            .iter()
            .any(|&position| local_dimensions[position] != 2)
        || factor_positions
            .iter()
            .enumerate()
            .any(|(offset, &position)| position != factor_positions[0] + offset)
        || particles > factor_positions.len()
    {
        return Ok(None);
    }
    let factor_dimension = usize::try_from(binomial(factor_positions.len(), particles).ok_or(
        PauliError::Overflow {
            context: "counting fast qubit configurations",
        },
    )?)
    .map_err(|_| PauliError::Overflow {
        context: "converting fast qubit configurations",
    })?;
    let Some(layout) = FastProductLayout::new(
        local_dimensions,
        fermion_positions,
        boson_positions,
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
    let active_qubit_indices = factor_positions
        .iter()
        .map(|position| {
            qubit_positions
                .iter()
                .position(|candidate| candidate == position)
                .ok_or(PauliError::InvalidIndex {
                    context: "fast qubit factor axis",
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let factor_actions = build_qubit_factor_actions(
        terms,
        factor_positions.len(),
        &active_qubit_indices,
        particles,
        factor_dimension,
    )?;
    if factor_actions.is_none() {
        return Ok(None);
    }
    let action_count = factor_actions
        .as_ref()
        .map(|actions| actions.iter().map(Vec::len).sum::<usize>())
        .unwrap_or(0);
    let action_bytes = action_count
        .checked_mul(std::mem::size_of::<(usize, Complex64)>())
        .ok_or(PauliError::Overflow {
            context: "estimating fast qubit factor actions",
        })?;
    let estimated_bytes = layout
        .descriptor_bytes()
        .checked_add(estimate_charge_transition_terms_bytes(terms)?)
        .and_then(|value| value.checked_add(action_bytes as u128))
        .ok_or(PauliError::Overflow {
            context: "estimating fast qubit plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(Some(FastQubitMvpPlan {
        layout,
        factor_actions: factor_actions.expect("validated qubit factor actions"),
        estimated_bytes,
    }))
}
use super::*;
