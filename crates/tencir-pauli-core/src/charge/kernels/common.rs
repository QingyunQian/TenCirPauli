/// Public-order layout shared by compact independent finite factors.
///
/// The factor axes must form one contiguous segment in the public mixed-radix
/// layout.  This is the representation already used by ordinary
/// ``OperatorSpace`` layouts; rejecting a non-contiguous factor keeps the
/// fast path honest until a bounded interleaved ranker is needed.
pub(super) struct FastProductLayout {
    pub(super) local_dimensions: Box<[u64]>,
    pub(super) fermion_positions: Box<[usize]>,
    pub(super) boson_positions: Box<[usize]>,
    pub(super) qubit_positions: Box<[usize]>,
    pub(super) qudit_positions: Box<[usize]>,
    pub(super) factor_positions: Box<[usize]>,
    pub(super) factor_dimension: usize,
    pub(super) prefix_dimension: usize,
    pub(super) suffix_dimension: usize,
}

impl FastProductLayout {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        local_dimensions: &[u64],
        fermion_positions: &[usize],
        boson_positions: &[usize],
        qubit_positions: &[usize],
        qudit_positions: &[usize],
        factor_positions: &[usize],
        factor_dimension: usize,
    ) -> Result<Option<Self>, PauliError> {
        if factor_positions.is_empty()
            || factor_positions
                .iter()
                .enumerate()
                .any(|(offset, &position)| position != factor_positions[0] + offset)
        {
            return Ok(None);
        }
        let factor_start = factor_positions[0];
        let factor_end =
            factor_start
                .checked_add(factor_positions.len())
                .ok_or(PauliError::Overflow {
                    context: "sizing fast product factor range",
                })?;
        if factor_end > local_dimensions.len()
            || factor_positions
                .iter()
                .any(|&position| local_dimensions[position] == 0)
        {
            return Ok(None);
        }
        let prefix_dimension =
            local_dimensions[..factor_start]
                .iter()
                .try_fold(1usize, |value, &dimension| {
                    value
                        .checked_mul(usize::try_from(dimension).map_err(|_| {
                            PauliError::Overflow {
                                context: "converting fast product prefix dimension",
                            }
                        })?)
                        .ok_or(PauliError::Overflow {
                            context: "sizing fast product prefix dimension",
                        })
                })?;
        let suffix_dimension =
            local_dimensions[factor_end..]
                .iter()
                .try_fold(1usize, |value, &dimension| {
                    value
                        .checked_mul(usize::try_from(dimension).map_err(|_| {
                            PauliError::Overflow {
                                context: "converting fast product suffix dimension",
                            }
                        })?)
                        .ok_or(PauliError::Overflow {
                            context: "sizing fast product suffix dimension",
                        })
                })?;
        let expected_dimension = prefix_dimension
            .checked_mul(factor_dimension)
            .and_then(|value| value.checked_mul(suffix_dimension))
            .ok_or(PauliError::Overflow {
                context: "sizing fast product layout dimension",
            })?;
        let local_dimensions = local_dimensions.to_vec().into_boxed_slice();
        if expected_dimension == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            local_dimensions,
            fermion_positions: fermion_positions.to_vec().into_boxed_slice(),
            boson_positions: boson_positions.to_vec().into_boxed_slice(),
            qubit_positions: qubit_positions.to_vec().into_boxed_slice(),
            qudit_positions: qudit_positions.to_vec().into_boxed_slice(),
            factor_positions: factor_positions.to_vec().into_boxed_slice(),
            factor_dimension,
            prefix_dimension,
            suffix_dimension,
        }))
    }

    pub(super) fn expected_dimension(&self) -> Result<usize, PauliError> {
        self.prefix_dimension
            .checked_mul(self.factor_dimension)
            .and_then(|value| value.checked_mul(self.suffix_dimension))
            .ok_or(PauliError::Overflow {
                context: "sizing fast product layout dimension",
            })
    }

    pub(super) fn descriptor_bytes(&self) -> u128 {
        ((self.local_dimensions.len()
            + self.fermion_positions.len()
            + self.boson_positions.len()
            + self.qubit_positions.len()
            + self.qudit_positions.len()
            + self.factor_positions.len())
            * std::mem::size_of::<usize>()) as u128
    }
}
use super::*;
