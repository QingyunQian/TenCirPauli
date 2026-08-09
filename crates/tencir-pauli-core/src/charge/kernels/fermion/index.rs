pub(super) struct FastFermionSectorIndex {
    pub(super) sites: usize,
    pub(super) particles: usize,
    pub(super) combination_count: usize,
    pub(super) combination_masks: Option<Vec<u128>>,
    pub(super) rank_table: Option<Vec<u32>>,
}

const FAST_COMBINATION_CACHE_LIMIT: u128 = 64 * 1024 * 1024;
const FAST_RANK_CACHE_LIMIT: u128 = 4 * 1024 * 1024;

impl FastFermionSectorIndex {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        dimension: usize,
        local_dimensions: &[u64],
        fermion_positions: &[usize],
        boson_positions: &[usize],
        qubit_positions: &[usize],
        qudit_positions: &[usize],
        particles: usize,
        scratch_budget: u128,
    ) -> Option<Self> {
        if fermion_positions.len() % 2 != 0
            || !boson_positions.is_empty()
            || !qubit_positions.is_empty()
            || !qudit_positions.is_empty()
            || local_dimensions.iter().any(|&value| value != 2)
            || fermion_positions
                .iter()
                .enumerate()
                .any(|(mode, &position)| mode != position)
        {
            return None;
        }
        let sites = fermion_positions.len() / 2;
        if sites == 0 || sites > 64 || particles > sites {
            return None;
        }
        let combination_count = usize::try_from(binomial(sites, particles)?).ok()?;
        let expected_dimension = combination_count.checked_mul(combination_count)?;
        if dimension != expected_dimension {
            return None;
        }

        // These tables are optional. The direct combinatorial routines remain
        // the bounded-memory fallback for larger sectors.
        let mut remaining_budget = scratch_budget;
        let combination_masks = if (combination_count as u128)
            .checked_mul(std::mem::size_of::<u128>() as u128)?
            <= FAST_COMBINATION_CACHE_LIMIT
            && (combination_count as u128).checked_mul(16)? <= remaining_budget
        {
            let mut masks = Vec::with_capacity(combination_count);
            for rank in 0..combination_count {
                masks.push(unrank_combination(sites, particles, rank as u128)?);
            }
            remaining_budget -= (combination_count as u128) * 16;
            Some(masks)
        } else {
            None
        };
        let rank_table = if sites < usize::BITS as usize
            && sites <= 20
            && combination_count <= u32::MAX as usize
        {
            let table_len = 1_usize << sites;
            let table_bytes = (table_len as u128).checked_mul(4)?;
            if table_bytes <= FAST_RANK_CACHE_LIMIT && table_bytes <= remaining_budget {
                let mut table = vec![u32::MAX; table_len];
                for rank in 0..combination_count {
                    let mask = if let Some(masks) = &combination_masks {
                        masks[rank]
                    } else {
                        unrank_combination(sites, particles, rank as u128)?
                    };
                    table[mask as usize] = rank as u32;
                }
                Some(table)
            } else {
                None
            }
        } else {
            None
        };
        Some(Self {
            sites,
            particles,
            combination_count,
            combination_masks,
            rank_table,
        })
    }

    pub(super) fn unrank(&self, rank: usize) -> Option<u128> {
        if rank >= self.combination_count {
            return None;
        }
        self.combination_masks
            .as_ref()
            .map(|masks| masks[rank])
            .or_else(|| unrank_combination(self.sites, self.particles, rank as u128))
    }

    pub(super) fn rank(&self, mask: u128) -> Option<usize> {
        if let Some(table) = &self.rank_table {
            let index = usize::try_from(mask).ok()?;
            let rank = *table.get(index)?;
            (rank != u32::MAX).then_some(rank as usize)
        } else {
            usize::try_from(rank_combination(self.sites, self.particles, mask)?).ok()
        }
    }

    pub(super) fn estimated_bytes(&self) -> u128 {
        let combinations = self
            .combination_masks
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<u128>());
        let ranks = self
            .rank_table
            .as_ref()
            .map_or(0, |values| values.len() * std::mem::size_of::<u32>());
        (combinations + ranks) as u128
    }
}

/// Cached specialized execution for a balanced pure spinful fermion sector.
///
/// The index and validated term descriptors are constructed once and shared by
/// all applications. The kernel writes directly into caller-owned output and
/// never allocates a state-sized temporary.
pub struct FastFermionMvpPlan {
    pub(super) index: FastFermionSectorIndex,
    pub(super) up_modes: Box<[u8]>,
    pub(super) down_modes: Box<[u8]>,
    pub(super) factor_to_public: Box<[usize]>,
    pub(super) public_to_factor: Box<[usize]>,
    pub(super) terms: Vec<FastProductTerm>,
    pub(super) boson_dimensions: Box<[usize]>,
    pub(super) boson_strides: Box<[usize]>,
    pub(super) spectator_dimension: usize,
    pub(super) estimated_bytes: u128,
}

#[derive(Clone, Debug)]
pub(super) struct FastProductTerm {
    pub(super) descriptor: Option<FermionMaskDescriptor>,
    pub(super) inverse: Option<FermionMaskDescriptor>,
    pub(super) coefficient: Complex64,
    pub(super) boson_blocks: Box<[(u32, u32, u32)]>,
}

impl FastFermionMvpPlan {
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    fn can_gather(&self) -> bool {
        self.terms
            .iter()
            .all(|term| term.inverse.is_some() && term.boson_blocks.len() <= 1)
    }

    fn pack_source(&self, up: u128, down: u128) -> u128 {
        let mut source = 0_u128;
        for (offset, &mode) in self.up_modes.iter().enumerate() {
            if up & (1_u128 << offset) != 0 {
                source |= 1_u128 << mode;
            }
        }
        for (offset, &mode) in self.down_modes.iter().enumerate() {
            if down & (1_u128 << offset) != 0 {
                source |= 1_u128 << mode;
            }
        }
        source
    }

    fn extract_group(&self, source: u128, modes: &[u8]) -> u128 {
        let mut result = 0_u128;
        for (offset, &mode) in modes.iter().enumerate() {
            if source & (1_u128 << mode) != 0 {
                result |= 1_u128 << offset;
            }
        }
        result
    }

    fn apply_gather_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        let combinations = self.index.combination_count;
        let spectator_dimension = self.spectator_dimension;
        let boson_actions = self
            .terms
            .iter()
            .map(|term| {
                (0..spectator_dimension)
                    .map(|destination_boson| {
                        Ok(apply_fast_boson_inverse(
                            destination_boson,
                            &self.boson_dimensions,
                            &self.boson_strides,
                            &term.boson_blocks,
                        )?
                        .unwrap_or((usize::MAX, 0.0)))
                    })
                    .collect::<Result<Vec<_>, PauliError>>()
            })
            .collect::<Result<Vec<_>, PauliError>>()?;
        output
            .par_chunks_mut(spectator_dimension)
            .enumerate()
            .for_each_init(
                || vec![None; self.terms.len()],
                |fermion_actions, (destination_fermion, output_row)| {
                    let factor_column = if self.public_to_factor.is_empty() {
                        destination_fermion
                    } else {
                        self.public_to_factor[destination_fermion]
                    };
                    let up_rank = factor_column / combinations;
                    let down_rank = factor_column % combinations;
                    let Some(destination_up) = self.index.unrank(up_rank) else {
                        return;
                    };
                    let Some(destination_down) = self.index.unrank(down_rank) else {
                        return;
                    };
                    let destination = self.pack_source(destination_up, destination_down);

                    // The fermion inverse action and compact-sector rank are
                    // independent of the Cartesian spectator state.  Compute
                    // them once per destination row; doing this in the inner
                    // boson loop previously repeated the bit extraction and
                    // combinatorial rank for every spectator value.
                    for (action, term) in fermion_actions.iter_mut().zip(&self.terms) {
                        *action = term
                            .inverse
                            .as_ref()
                            .and_then(|descriptor| {
                                apply_fermion_mask_words(descriptor, destination, term.coefficient)
                            })
                            .and_then(|(source, value)| {
                                let source_up = self.extract_group(source, &self.up_modes);
                                let source_down = self.extract_group(source, &self.down_modes);
                                let source_up_rank = self.index.rank(source_up)?;
                                let source_down_rank = self.index.rank(source_down)?;
                                let source_fermion =
                                    source_up_rank * combinations + source_down_rank;
                                let source_public = if self.factor_to_public.is_empty() {
                                    source_fermion
                                } else {
                                    self.factor_to_public[source_fermion]
                                };
                                Some((source_public * spectator_dimension, value))
                            });
                    }

                    for destination_boson in 0..spectator_dimension {
                        let mut result = Complex64::default();
                        for (term_index, (action, _term)) in
                            fermion_actions.iter().zip(&self.terms).enumerate()
                        {
                            let Some(&(source_base, value)) = action.as_ref() else {
                                continue;
                            };
                            let (source_boson, boson_amplitude) =
                                boson_actions[term_index][destination_boson];
                            if source_boson == usize::MAX {
                                continue;
                            }
                            let source_index = source_base + source_boson;
                            result += value * state[source_index] * boson_amplitude;
                        }
                        output_row[destination_boson] = result;
                    }
                },
            );
        Ok(())
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        if state.len() != output.len() {
            return Err(invalid_sector());
        }
        let combinations = self.index.combination_count;
        let fermion_dimension = self
            .index
            .combination_count
            .checked_mul(self.index.combination_count)
            .ok_or(PauliError::Overflow {
                context: "sizing fast product fermion factor",
            })?;
        let expected_dimension = fermion_dimension
            .checked_mul(self.spectator_dimension)
            .ok_or(PauliError::Overflow {
                context: "sizing fast product layout",
            })?;
        if state.len() != expected_dimension {
            return Err(invalid_sector());
        }
        if self.can_gather() {
            return self.apply_gather_into(state, output);
        }
        output.fill(Complex64::default());
        let table_bytes = self
            .terms
            .len()
            .checked_mul(self.spectator_dimension)
            .and_then(|entries| entries.checked_mul(std::mem::size_of::<(usize, Complex64)>()))
            .ok_or(PauliError::Overflow {
                context: "estimating fast product action tables",
            })? as u128;
        let output_bytes = (output.len() as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating fast product output",
            })?;
        let use_boson_tables = table_bytes <= max_bytes.saturating_sub(output_bytes);
        let boson_tables = if use_boson_tables {
            self.terms
                .iter()
                .map(|term| {
                    if term.boson_blocks.is_empty() {
                        return Ok(None);
                    }
                    let mut actions = Vec::with_capacity(self.spectator_dimension);
                    for boson_column in 0..self.spectator_dimension {
                        let action = apply_fast_boson_blocks(
                            boson_column,
                            &self.boson_dimensions,
                            &self.boson_strides,
                            &term.boson_blocks,
                            Complex64::new(1.0, 0.0),
                        )?
                        .unwrap_or((usize::MAX, Complex64::default()));
                        actions.push(action);
                    }
                    Ok(Some(actions))
                })
                .collect::<Result<Vec<_>, PauliError>>()?
        } else {
            Vec::new()
        };
        // Lower each fermion action once per compact state and reuse it over
        // the contiguous Cartesian spectator segment.
        for fermion_column in 0..fermion_dimension {
            let public_column = if self.factor_to_public.is_empty() {
                fermion_column
            } else {
                self.factor_to_public[fermion_column]
            };
            let up_rank = fermion_column / combinations;
            let down_rank = fermion_column % combinations;
            let source = self.pack_source(
                self.index
                    .unrank(up_rank)
                    .ok_or(PauliError::InvalidSector {
                        context: "fast fermion sector unrank failed",
                    })?,
                self.index
                    .unrank(down_rank)
                    .ok_or(PauliError::InvalidSector {
                        context: "fast fermion sector unrank failed",
                    })?,
            );
            for (term_index, term) in self.terms.iter().enumerate() {
                let Some((destination, value)) = term.descriptor.as_ref().and_then(|descriptor| {
                    apply_fermion_mask_words(descriptor, source, term.coefficient)
                }) else {
                    continue;
                };
                let destination_fermion = if destination == source {
                    fermion_column
                } else {
                    let destination_up = self.extract_group(destination, &self.up_modes);
                    let destination_down = self.extract_group(destination, &self.down_modes);
                    let source_up = self.extract_group(source, &self.up_modes);
                    let source_down = self.extract_group(source, &self.down_modes);
                    let up_index = if destination_up == source_up {
                        up_rank
                    } else {
                        self.index
                            .rank(destination_up)
                            .ok_or(PauliError::InvalidSector {
                                context: "fast fermion sector rank failed",
                            })?
                    };
                    let down_index = if destination_down == source_down {
                        down_rank
                    } else {
                        self.index
                            .rank(destination_down)
                            .ok_or(PauliError::InvalidSector {
                                context: "fast fermion sector rank failed",
                            })?
                    };
                    up_index * combinations + down_index
                };
                let destination_public = if self.factor_to_public.is_empty() {
                    destination_fermion
                } else {
                    self.factor_to_public[destination_fermion]
                };
                let source_base = public_column.checked_mul(self.spectator_dimension).ok_or(
                    PauliError::Overflow {
                        context: "indexing fast product source",
                    },
                )?;
                let destination_base = destination_public
                    .checked_mul(self.spectator_dimension)
                    .ok_or(PauliError::Overflow {
                        context: "indexing fast product destination",
                    })?;
                if term.boson_blocks.is_empty() {
                    for offset in 0..self.spectator_dimension {
                        output[destination_base + offset] += value * state[source_base + offset];
                    }
                    continue;
                }
                if use_boson_tables {
                    for (boson_column, &(destination_boson, boson_value)) in boson_tables
                        [term_index]
                        .as_ref()
                        .expect("table exists")
                        .iter()
                        .enumerate()
                    {
                        if destination_boson != usize::MAX {
                            output[destination_base + destination_boson] +=
                                value * boson_value * state[source_base + boson_column];
                        }
                    }
                } else {
                    for boson_column in 0..self.spectator_dimension {
                        let Some((destination_boson, boson_value)) = apply_fast_boson_blocks(
                            boson_column,
                            &self.boson_dimensions,
                            &self.boson_strides,
                            &term.boson_blocks,
                            value,
                        )?
                        else {
                            continue;
                        };
                        output[destination_base + destination_boson] +=
                            boson_value * state[source_base + boson_column];
                    }
                }
            }
        }
        Ok(())
    }
}
use super::*;
