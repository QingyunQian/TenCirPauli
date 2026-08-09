/// Build the cached spinful fast path when the native descriptors are eligible.
pub fn build_fast_fermion_mvp_plan(
    context: &ChargeBuildContext<'_>,
    particles: Option<usize>,
    mode_order: Option<&[u64]>,
) -> Result<Option<FastFermionMvpPlan>, PauliError> {
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
    if terms.iter().any(|term| {
        !term.qubit_codes.is_empty()
            || term.mapped_present
            || !term.qudit_triples.is_empty()
            || term.qudit_present
    }) {
        return Ok(None);
    }
    if !qubit_positions.is_empty() || !qudit_positions.is_empty() {
        return Ok(None);
    }
    if fermion_positions.len() % 2 != 0
        || fermion_positions
            .iter()
            .enumerate()
            .any(|(mode, &position)| mode != position)
        || boson_positions
            .iter()
            .enumerate()
            .any(|(mode, &position)| position != fermion_positions.len() + mode)
        || local_dimensions
            .iter()
            .enumerate()
            .any(|(position, &dimension)| {
                if position < fermion_positions.len() {
                    dimension != 2
                } else {
                    !boson_positions.contains(&position)
                }
            })
    {
        return Ok(None);
    }
    let sites = fermion_positions.len() / 2;
    if sites == 0 || particles > sites {
        return Ok(None);
    }
    let combination_count =
        usize::try_from(binomial(sites, particles).ok_or(PauliError::Overflow {
            context: "counting fast product fermion configurations",
        })?)
        .map_err(|_| PauliError::Overflow {
            context: "converting fast product fermion configurations",
        })?;
    let fermion_dimension =
        combination_count
            .checked_mul(combination_count)
            .ok_or(PauliError::Overflow {
                context: "sizing fast product fermion factor",
            })?;
    let factor_dimensions = vec![2_u64; fermion_positions.len()];
    let factor_positions = (0..fermion_positions.len()).collect::<Vec<_>>();
    let empty_positions: Vec<usize> = Vec::new();
    let Some(index) = FastFermionSectorIndex::new(
        fermion_dimension,
        &factor_dimensions,
        &factor_positions,
        &empty_positions,
        &empty_positions,
        &empty_positions,
        particles,
        max_bytes,
    ) else {
        return Ok(None);
    };
    let boson_dimensions = boson_positions
        .iter()
        .map(|&position| {
            local_dimensions
                .get(position)
                .copied()
                .and_then(|dimension| usize::try_from(dimension).ok())
                .ok_or_else(invalid_sector)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let spectator_dimension = boson_dimensions
        .iter()
        .try_fold(1usize, |value, &dimension| {
            value.checked_mul(dimension).ok_or(PauliError::Overflow {
                context: "sizing fast product spectator factor",
            })
        })?;
    let full_dimension =
        fermion_dimension
            .checked_mul(spectator_dimension)
            .ok_or(PauliError::Overflow {
                context: "sizing fast product layout",
            })?;
    if plan.dimension() != full_dimension {
        return Ok(None);
    }
    let boson_strides = mixed_radix_strides(&boson_dimensions)?;
    let mode_count = index.sites * 2;
    if mode_count > 128 {
        return Ok(None);
    }
    let mode_order = mode_order
        .map(|values| {
            values
                .iter()
                .map(|&value| {
                    usize::try_from(value).map_err(|_| PauliError::InvalidSector {
                        context: "fast fermion mode grouping is invalid",
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let mode_order = mode_order.unwrap_or_else(|| (0..mode_count).collect());
    if mode_order.len() != mode_count
        || mode_order
            .iter()
            .enumerate()
            .any(|(offset, &mode)| mode >= mode_count || mode_order[..offset].contains(&mode))
    {
        return Ok(None);
    }
    let to_modes = |values: &[usize]| -> Result<Box<[u8]>, PauliError> {
        values
            .iter()
            .map(|&mode| {
                u8::try_from(mode).map_err(|_| PauliError::InvalidSector {
                    context: "fast fermion mode grouping does not fit descriptor",
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    };
    let up_modes = to_modes(&mode_order[..sites])?;
    let down_modes = to_modes(&mode_order[sites..])?;
    let identity_order = mode_order
        .iter()
        .enumerate()
        .all(|(index, &mode)| index == mode);
    let (factor_to_public, public_to_factor) = if identity_order {
        (Vec::new(), Vec::new())
    } else {
        let mapping_bytes = (fermion_dimension as u128)
            .checked_mul(2)
            .and_then(|value| value.checked_mul(std::mem::size_of::<usize>() as u128))
            .ok_or(PauliError::Overflow {
                context: "estimating fast fermion public-order mapping",
            })?;
        if mapping_bytes > max_bytes {
            return Ok(None);
        }
        let mut factor_to_public = vec![usize::MAX; fermion_dimension];
        let mut public_to_factor = vec![usize::MAX; fermion_dimension];
        let mut occupations = vec![0_u64; local_dimensions.len()];
        for (factor_column, factor_to_public_entry) in factor_to_public.iter_mut().enumerate() {
            let up = index.unrank(factor_column / combination_count).ok_or(
                PauliError::InvalidSector {
                    context: "fast fermion factor unrank failed",
                },
            )?;
            let down = index.unrank(factor_column % combination_count).ok_or(
                PauliError::InvalidSector {
                    context: "fast fermion factor unrank failed",
                },
            )?;
            occupations.fill(0);
            for (offset, &mode) in up_modes.iter().enumerate() {
                if up & (1_u128 << offset) != 0 {
                    occupations[usize::from(mode)] = 1;
                }
            }
            for (offset, &mode) in down_modes.iter().enumerate() {
                if down & (1_u128 << offset) != 0 {
                    occupations[usize::from(mode)] = 1;
                }
            }
            let public_rank =
                usize::try_from(plan.rank(&occupations)?).map_err(|_| PauliError::Overflow {
                    context: "converting fast fermion public-order rank",
                })?;
            if public_rank % spectator_dimension != 0 {
                return Ok(None);
            }
            let public_factor = public_rank / spectator_dimension;
            if public_factor >= fermion_dimension || public_to_factor[public_factor] != usize::MAX {
                return Ok(None);
            }
            *factor_to_public_entry = public_factor;
            public_to_factor[public_factor] = factor_column;
        }
        (factor_to_public, public_to_factor)
    };
    let mut compact_terms = Vec::with_capacity(terms.len());
    for term in terms.iter() {
        if term
            .fermion_creation
            .iter()
            .chain(&term.fermion_annihilation)
            .any(|&mode| usize::try_from(mode).map_or(true, |mode| mode >= mode_count))
        {
            return Err(PauliError::InvalidSector {
                context: "fast fermion term mode is outside the sector layout",
            });
        }
        let mode_axes = (0..mode_count).collect::<Vec<_>>();
        let descriptor = FermionMaskDescriptor::from_creation_annihilation(
            &mode_axes,
            &term.fermion_creation,
            &term.fermion_annihilation,
        );
        let inverse = FermionMaskDescriptor::from_creation_annihilation(
            &mode_axes,
            &term.fermion_annihilation,
            &term.fermion_creation,
        );
        let boson_blocks = term
            .boson_blocks
            .iter()
            .map(|&(mode, creation, annihilation)| {
                if usize::try_from(mode).map_or(true, |mode| mode >= boson_dimensions.len()) {
                    return Err(PauliError::InvalidSector {
                        context: "fast product boson mode is outside the sector layout",
                    });
                }
                Ok((mode, creation, annihilation))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        compact_terms.push(FastProductTerm {
            descriptor,
            inverse,
            coefficient: term.coefficient,
            boson_blocks,
        });
    }
    let descriptor_bytes = compact_terms.iter().try_fold(0_u128, |total, term| {
        let mask_bytes = [&term.descriptor, &term.inverse]
            .into_iter()
            .filter_map(Option::as_ref)
            .try_fold(0_u128, |bytes, mask| {
                let words = mask.required_occupied.len();
                let payload = words
                    .checked_mul(4 * std::mem::size_of::<u64>())
                    .and_then(|value| {
                        value.checked_add(mask.mode_axes.len() * std::mem::size_of::<usize>())
                    })
                    .ok_or(PauliError::Overflow {
                        context: "estimating fast FermionMask descriptors",
                    })? as u128;
                bytes.checked_add(payload).ok_or(PauliError::Overflow {
                    context: "estimating fast FermionMask descriptors",
                })
            })?;
        let boson_payload = term
            .boson_blocks
            .len()
            .checked_mul(std::mem::size_of::<(u32, u32, u32)>())
            .ok_or(PauliError::Overflow {
                context: "estimating fast product boson descriptors",
            })?;
        total
            .checked_add(std::mem::size_of::<FastProductTerm>() as u128)
            .and_then(|value| value.checked_add(mask_bytes))
            .and_then(|value| value.checked_add(boson_payload as u128))
            .ok_or(PauliError::Overflow {
                context: "estimating fast fermion descriptors",
            })
    })?;
    let estimated_bytes = index
        .estimated_bytes()
        .checked_add(descriptor_bytes)
        .and_then(|value| {
            value.checked_add(
                ((boson_dimensions.len() + boson_strides.len()) * std::mem::size_of::<usize>())
                    as u128,
            )
        })
        .and_then(|value| {
            value.checked_add(
                ((up_modes.len()
                    + down_modes.len()
                    + factor_to_public.len()
                    + public_to_factor.len())
                    * std::mem::size_of::<usize>()) as u128,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating fast fermion plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(Some(FastFermionMvpPlan {
        index,
        up_modes,
        down_modes,
        factor_to_public: factor_to_public.into_boxed_slice(),
        public_to_factor: public_to_factor.into_boxed_slice(),
        terms: compact_terms,
        boson_dimensions: boson_dimensions.into_boxed_slice(),
        boson_strides: boson_strides.into_boxed_slice(),
        spectator_dimension,
        estimated_bytes,
    }))
}

fn mixed_radix_strides(dimensions: &[usize]) -> Result<Vec<usize>, PauliError> {
    let mut strides = vec![1_usize; dimensions.len()];
    let mut stride = 1_usize;
    for position in (0..dimensions.len()).rev() {
        strides[position] = stride;
        stride = stride
            .checked_mul(dimensions[position])
            .ok_or(PauliError::Overflow {
                context: "computing fast product spectator strides",
            })?;
    }
    Ok(strides)
}

pub(super) fn apply_fast_boson_blocks(
    source: usize,
    dimensions: &[usize],
    strides: &[usize],
    blocks: &[(u32, u32, u32)],
    value: Complex64,
) -> Result<Option<(usize, Complex64)>, PauliError> {
    let Some((destination, amplitude)) = fast_boson_action(source, dimensions, strides, blocks)?
    else {
        return Ok(None);
    };
    let value = value * Complex64::new(amplitude, 0.0);
    if value.re == 0.0 && value.im == 0.0 {
        Ok(None)
    } else {
        Ok(Some((destination, value)))
    }
}

fn fast_boson_action(
    source: usize,
    dimensions: &[usize],
    strides: &[usize],
    blocks: &[(u32, u32, u32)],
) -> Result<Option<(usize, f64)>, PauliError> {
    let mut destination = source;
    let mut amplitude = 1.0;
    for &(mode, creation, annihilation) in blocks {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        let dimension = *dimensions.get(mode).ok_or_else(invalid_sector)?;
        let stride = *strides.get(mode).ok_or_else(invalid_sector)?;
        let digit = (source / stride) % dimension;
        let annihilation = usize::try_from(annihilation).map_err(|_| invalid_sector())?;
        let creation = usize::try_from(creation).map_err(|_| invalid_sector())?;
        if digit < annihilation {
            return Ok(None);
        }
        let Some(new_digit) = digit
            .checked_sub(annihilation)
            .and_then(|value| value.checked_add(creation))
        else {
            return Ok(None);
        };
        if new_digit >= dimension {
            return Ok(None);
        }
        for offset in 0..annihilation {
            amplitude *= ((digit - offset) as f64).sqrt();
        }
        let remaining = digit - annihilation;
        for offset in 0..creation {
            amplitude *= ((remaining + offset + 1) as f64).sqrt();
        }
        let delta = if new_digit >= digit {
            (new_digit - digit).checked_mul(stride)
        } else {
            (digit - new_digit).checked_mul(stride)
        }
        .ok_or(PauliError::Overflow {
            context: "computing fast product boson destination",
        })?;
        destination = if new_digit >= digit {
            destination.checked_add(delta)
        } else {
            destination.checked_sub(delta)
        }
        .ok_or(PauliError::Overflow {
            context: "computing fast product boson destination",
        })?;
    }
    Ok(Some((destination, amplitude)))
}

pub(super) fn apply_fermion_mask_words(
    descriptor: &FermionMaskDescriptor,
    source: u128,
    coefficient: Complex64,
) -> Option<(u128, Complex64)> {
    let words = descriptor.required_occupied.len();
    if words > 2 {
        return None;
    }
    let source_words = [source as u64, (source >> 64) as u64];
    let mut destination_words = [0_u64; 2];
    let amplitude = descriptor.apply_to_words(
        &source_words[..words],
        &mut destination_words[..words],
        coefficient,
    )?;
    let destination = u128::from(destination_words[0])
        | if words == 2 {
            u128::from(destination_words[1]) << 64
        } else {
            0
        };
    Some((destination, amplitude))
}

pub(super) fn apply_fast_boson_inverse(
    destination: usize,
    dimensions: &[usize],
    strides: &[usize],
    blocks: &[(u32, u32, u32)],
) -> Result<Option<(usize, f64)>, PauliError> {
    match blocks {
        [] => Ok(Some((destination, 1.0))),
        [(mode, creation, annihilation)] => {
            let inverse = [(*mode, *annihilation, *creation)];
            fast_boson_action(destination, dimensions, strides, &inverse)
        }
        _ => Ok(None),
    }
}
use super::*;
