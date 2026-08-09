//! Width-generic lowering and application of ordered Fermion monomials.

use crate::{Complex64, PauliError};

/// A one-word Fermion action expressed directly in public basis-index bits.
///
/// Logical Fermion mode zero is the least-significant bit in the descriptor,
/// while public axis zero is the most-significant bit of a contiguous binary
/// basis index.  This view performs that reversal once during lowering so the
/// scalable full-space and Fermion-prefix kernels need neither source packing
/// nor a per-call cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ContiguousFermionAction {
    required_occupied: usize,
    required_empty: usize,
    flip_mask: usize,
    source_parity_mask: usize,
    fixed_sign: i8,
}

impl ContiguousFermionAction {
    #[inline]
    pub(crate) fn source_for_destination(self, destination: usize) -> usize {
        destination ^ self.flip_mask
    }

    #[inline]
    pub(crate) fn amplitude(self, source: usize, coefficient: Complex64) -> Option<Complex64> {
        if source & self.required_occupied != self.required_occupied
            || source & self.required_empty != 0
        {
            return None;
        }
        let mut amplitude = if self.fixed_sign < 0 {
            -coefficient
        } else {
            coefficient
        };
        if ((source & self.source_parity_mask).count_ones() & 1) != 0 {
            amplitude = -amplitude;
        }
        Some(amplitude)
    }
}

/// A canonical source-state Fermion action.
///
/// The descriptor is independent of the consumer. `mode_axes` maps logical
/// Fermion modes to public mixed-radix axes; the four packed masks encode the
/// source requirements, destination flip, and Jordan--Wigner source parity.
/// All vectors are allocated during lowering and are borrowed or caller-owned
/// during application.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FermionMaskDescriptor {
    pub(crate) required_occupied: Box<[u64]>,
    pub(crate) required_empty: Box<[u64]>,
    pub(crate) flip_mask: Box<[u64]>,
    pub(crate) source_parity_mask: Box<[u64]>,
    pub(crate) fixed_sign: i8,
    pub(crate) mode_axes: Box<[usize]>,
    contiguous_public: Option<ContiguousFermionAction>,
}

impl FermionMaskDescriptor {
    /// Lower an ordered `(mode, create)` sequence. `false` is annihilation.
    pub(crate) fn from_steps(mode_axes: &[usize], steps: &[(usize, bool)]) -> Option<Self> {
        if mode_axes.is_empty() || steps.iter().any(|&(mode, _)| mode >= mode_axes.len()) {
            return None;
        }
        let word_count = mode_axes.len().checked_add(63)?.checked_div(64)?;
        let mut required_occupied = vec![0_u64; word_count];
        let mut required_empty = vec![0_u64; word_count];
        let mut flip_mask = vec![0_u64; word_count];
        let mut source_parity_mask = vec![0_u64; word_count];
        let mut fixed_parity = false;
        for &(mode, create) in steps {
            let word = mode / 64;
            let bit = 1_u64 << (mode % 64);
            for lower in 0..word {
                source_parity_mask[lower] ^= u64::MAX;
                fixed_parity ^= (flip_mask[lower].count_ones() & 1) != 0;
            }
            let lower_bits = mode % 64;
            if lower_bits != 0 {
                let prefix = (1_u64 << lower_bits) - 1;
                source_parity_mask[word] ^= prefix;
                fixed_parity ^= (flip_mask[word] & prefix).count_ones() & 1 != 0;
            }
            let source_must_be_occupied = !create ^ (flip_mask[word] & bit != 0);
            if source_must_be_occupied {
                if required_empty[word] & bit != 0 {
                    return None;
                }
                required_occupied[word] |= bit;
            } else {
                if required_occupied[word] & bit != 0 {
                    return None;
                }
                required_empty[word] |= bit;
            }
            flip_mask[word] ^= bit;
        }
        let fixed_sign = if fixed_parity { -1 } else { 1 };
        let contiguous_public = (word_count == 1
            && mode_axes.len() <= usize::BITS as usize
            && mode_axes
                .iter()
                .enumerate()
                .all(|(mode, &axis)| mode == axis))
        .then(|| {
            let public_mask = |mask: u64| (mask.reverse_bits() >> (64 - mode_axes.len())) as usize;
            ContiguousFermionAction {
                required_occupied: public_mask(required_occupied[0]),
                required_empty: public_mask(required_empty[0]),
                flip_mask: public_mask(flip_mask[0]),
                source_parity_mask: public_mask(source_parity_mask[0]),
                fixed_sign,
            }
        });
        Some(Self {
            required_occupied: required_occupied.into_boxed_slice(),
            required_empty: required_empty.into_boxed_slice(),
            flip_mask: flip_mask.into_boxed_slice(),
            source_parity_mask: source_parity_mask.into_boxed_slice(),
            fixed_sign,
            mode_axes: mode_axes.to_vec().into_boxed_slice(),
            contiguous_public,
        })
    }

    pub(crate) fn from_creation_annihilation(
        mode_axes: &[usize],
        creation: &[u32],
        annihilation: &[u32],
    ) -> Option<Self> {
        let mut steps = Vec::with_capacity(creation.len().saturating_add(annihilation.len()));
        for &mode in annihilation.iter().rev() {
            steps.push((usize::try_from(mode).ok()?, false));
        }
        for &mode in creation.iter().rev() {
            steps.push((usize::try_from(mode).ok()?, true));
        }
        Self::from_steps(mode_axes, &steps)
    }

    pub(crate) fn pack_source(
        &self,
        occupations: &[u64],
        words: &mut [u64],
    ) -> Result<(), PauliError> {
        if words.len() != self.required_occupied.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.required_occupied.len(),
                actual: words.len(),
            });
        }
        words.fill(0);
        for (mode, &axis) in self.mode_axes.iter().enumerate() {
            let occupation = *occupations.get(axis).ok_or(PauliError::InvalidIndex {
                context: "FermionMask public-axis map",
            })?;
            if occupation > 1 {
                return Err(PauliError::InvalidIndex {
                    context: "FermionMask occupation",
                });
            }
            if occupation != 0 {
                words[mode / 64] |= 1_u64 << (mode % 64);
            }
        }
        Ok(())
    }

    pub(crate) fn apply_to_occupations(
        &self,
        source: &[u64],
        destination: &mut [u64],
        words: &mut [u64],
        coefficient: Complex64,
    ) -> Result<Option<Complex64>, PauliError> {
        self.pack_source(source, words)?;
        let Some(amplitude) = self.amplitude_from_words(words, coefficient) else {
            return Ok(None);
        };
        if destination.len() != source.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: source.len(),
                actual: destination.len(),
            });
        }
        destination.copy_from_slice(source);
        for (mode, &axis) in self.mode_axes.iter().enumerate() {
            if self.flip_mask[mode / 64] & (1_u64 << (mode % 64)) != 0 {
                destination[axis] = 1 - destination[axis];
            }
        }
        Ok(Some(amplitude))
    }

    pub(crate) fn pack_index(
        &self,
        index: usize,
        strides: &[usize],
        words: &mut [u64],
    ) -> Result<(), PauliError> {
        if words.len() != self.mode_axes.len().div_ceil(64) {
            return Err(PauliError::InvalidStructureLength {
                expected: self.mode_axes.len().div_ceil(64),
                actual: words.len(),
            });
        }
        words.fill(0);
        for (mode, &axis) in self.mode_axes.iter().enumerate() {
            let stride = strides.get(axis).copied().ok_or(PauliError::InvalidIndex {
                context: "FermionMask public-axis stride",
            })?;
            if if stride.is_power_of_two() {
                index & stride != 0
            } else {
                (index / stride) % 2 != 0
            } {
                words[mode / 64] |= 1_u64 << (mode % 64);
            }
        }
        Ok(())
    }

    pub(crate) fn pack_contiguous_binary_index(
        &self,
        index: usize,
        words: &mut [u64],
    ) -> Result<(), PauliError> {
        let mode_count = self.mode_axes.len();
        if mode_count > usize::BITS as usize || words.len() != mode_count.div_ceil(64) {
            return Err(PauliError::InvalidStructureLength {
                expected: mode_count.div_ceil(64),
                actual: words.len(),
            });
        }
        debug_assert!(self
            .mode_axes
            .iter()
            .enumerate()
            .all(|(mode, &axis)| mode == axis));
        words.fill(0);
        if mode_count != 0 {
            words[0] = (index.reverse_bits() >> (usize::BITS as usize - mode_count)) as u64;
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn amplitude_from_words(
        &self,
        source: &[u64],
        coefficient: Complex64,
    ) -> Option<Complex64> {
        if source.len() != self.required_occupied.len()
            || source
                .iter()
                .zip(&self.required_occupied)
                .any(|(&word, &required)| word & required != required)
            || source
                .iter()
                .zip(&self.required_empty)
                .any(|(&word, &required)| word & required != 0)
        {
            return None;
        }
        let mut amplitude = if self.fixed_sign < 0 {
            -coefficient
        } else {
            coefficient
        };
        let parity = source
            .iter()
            .zip(&self.source_parity_mask)
            .map(|(&word, &mask)| (word & mask).count_ones())
            .sum::<u32>();
        if parity & 1 != 0 {
            amplitude = -amplitude;
        }
        Some(amplitude)
    }

    #[inline]
    pub(crate) fn amplitude_from_single_word(
        &self,
        source: u64,
        coefficient: Complex64,
    ) -> Option<Complex64> {
        if self.required_occupied.len() != 1
            || source & self.required_occupied[0] != self.required_occupied[0]
            || source & self.required_empty[0] != 0
        {
            return None;
        }
        let mut amplitude = if self.fixed_sign < 0 {
            -coefficient
        } else {
            coefficient
        };
        if ((source & self.source_parity_mask[0]).count_ones() & 1) != 0 {
            amplitude = -amplitude;
        }
        Some(amplitude)
    }

    #[inline]
    pub(crate) fn contiguous_public_action(&self) -> Option<ContiguousFermionAction> {
        self.contiguous_public
    }

    #[cfg(test)]
    pub(crate) fn apply_to_words(
        &self,
        source: &[u64],
        destination: &mut [u64],
        coefficient: Complex64,
    ) -> Option<Complex64> {
        let amplitude = self.amplitude_from_words(source, coefficient)?;
        if destination.len() != source.len() {
            return None;
        }
        for ((destination, &source), &flip) in
            destination.iter_mut().zip(source).zip(&self.flip_mask)
        {
            *destination = source ^ flip;
        }
        Some(amplitude)
    }

    pub(crate) fn apply_flip_to_index(
        &self,
        index: usize,
        strides: &[usize],
    ) -> Result<usize, PauliError> {
        let mut result = index;
        for (mode, &axis) in self.mode_axes.iter().enumerate() {
            if self.flip_mask[mode / 64] & (1_u64 << (mode % 64)) == 0 {
                continue;
            }
            let stride = strides.get(axis).copied().ok_or(PauliError::InvalidIndex {
                context: "FermionMask public-axis stride",
            })?;
            let digit = (result / stride) % 2;
            result = if digit == 0 {
                result.checked_add(stride).ok_or(PauliError::Overflow {
                    context: "computing FermionMask destination",
                })?
            } else {
                result.checked_sub(stride).ok_or(PauliError::Overflow {
                    context: "computing FermionMask destination",
                })?
            };
        }
        Ok(result)
    }

    #[inline]
    pub(crate) fn apply_flip_to_contiguous_binary_index(
        &self,
        index: usize,
    ) -> Result<usize, PauliError> {
        let Some(action) = self.contiguous_public_action() else {
            return Err(PauliError::InvalidIndex {
                context: "FermionMask contiguous binary index",
            });
        };
        Ok(action.source_for_destination(index))
    }
}

#[cfg(test)]
mod tests {
    use super::FermionMaskDescriptor;
    use crate::{allocation_count, Complex64};

    #[test]
    fn ordered_car_lowering_matches_source_requirements_and_sign() {
        let mask =
            FermionMaskDescriptor::from_steps(&[0, 1, 2], &[(1, false), (2, false), (0, true)])
                .unwrap();
        let source = [0b110_u64];
        let mut destination = [0_u64];
        assert!(mask
            .apply_to_words(&source, &mut destination, Complex64::new(1.0, 0.0))
            .is_some());
        let source = [0_u64];
        assert!(mask
            .apply_to_words(&source, &mut destination, Complex64::new(1.0, 0.0))
            .is_none());
    }

    #[test]
    fn wide_application_uses_caller_owned_destination() {
        let mask =
            FermionMaskDescriptor::from_steps(&[0, 1, 2, 3, 4, 5, 6, 7, 8], &[(8, true)]).unwrap();
        let source = [0_u64];
        let mut destination = [u64::MAX];
        let amplitude = mask
            .apply_to_words(&source, &mut destination, Complex64::new(2.0, -1.0))
            .unwrap();
        assert_eq!(destination[0], 1_u64 << 8);
        assert_eq!(amplitude, Complex64::new(2.0, -1.0));
    }

    #[test]
    fn contiguous_public_action_matches_packed_descriptor_orientation_and_phase() {
        for mode_count in [1_usize, 8, usize::BITS as usize] {
            let axes = (0..mode_count).collect::<Vec<_>>();
            let mask = FermionMaskDescriptor::from_steps(&axes, &[(0, true)]).unwrap();
            let action = mask.contiguous_public_action().unwrap();
            let flipped_public_bit = 1_usize << (mode_count - 1);
            assert_eq!(action.source_for_destination(flipped_public_bit), 0);
            assert_eq!(
                action.amplitude(0, Complex64::new(2.0, -1.0)),
                Some(Complex64::new(2.0, -1.0))
            );
            assert_eq!(
                action.amplitude(flipped_public_bit, Complex64::new(1.0, 0.0)),
                None
            );
        }

        let axes = (0..8).collect::<Vec<_>>();
        let mask =
            FermionMaskDescriptor::from_steps(&axes, &[(1, false), (2, false), (0, true)]).unwrap();
        let action = mask.contiguous_public_action().unwrap();
        for public_index in 0..256 {
            let mut packed = [0_u64];
            mask.pack_contiguous_binary_index(public_index, &mut packed)
                .unwrap();
            let coefficient = Complex64::new(0.25, -0.5);
            assert_eq!(
                action.amplitude(public_index, coefficient),
                mask.amplitude_from_single_word(packed[0], coefficient),
            );
            assert_eq!(
                action.source_for_destination(public_index),
                mask.apply_flip_to_contiguous_binary_index(public_index)
                    .unwrap(),
            );
        }

        assert!(FermionMaskDescriptor::from_steps(&[1, 0], &[(0, true)])
            .unwrap()
            .contiguous_public_action()
            .is_none());
    }

    #[test]
    fn one_and_three_word_inner_loops_allocate_nothing() {
        for &mode_count in &[8_usize, 129] {
            let axes = (0..mode_count).collect::<Vec<_>>();
            let steps = vec![(0, false), (mode_count - 1, true)];
            let mask = FermionMaskDescriptor::from_steps(&axes, &steps).unwrap();
            let word_count = mode_count.div_ceil(64);
            let strides = vec![1_usize; mode_count];
            let mut packed = vec![0_u64; word_count];
            let source = {
                let mut words = vec![0_u64; word_count];
                words[0] = 1;
                words
            };
            let mut destination = vec![0_u64; word_count];
            let (_, allocations) = allocation_count(|| {
                for index in 0..512 {
                    mask.pack_index(index, &strides, &mut packed).unwrap();
                    let _ = mask.amplitude_from_words(&packed, Complex64::new(1.0, -0.5));
                    let _ = mask.apply_flip_to_index(index, &strides).unwrap();
                    let _ =
                        mask.apply_to_words(&source, &mut destination, Complex64::new(1.0, -0.5));
                }
            });
            assert_eq!(allocations, 0, "mode_count={mode_count}");
        }
    }
}
