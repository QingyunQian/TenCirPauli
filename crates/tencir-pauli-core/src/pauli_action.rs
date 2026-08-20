//! Allocation-free action of one Hermitian Pauli word on a dense statevector.
//!
//! Matrix-producing APIs in this crate use qubit zero as the most-significant
//! bit.  This small compiled action is the single place where that convention
//! and the canonical ``Y = i X Z`` phase are applied.

use crate::error::PauliError;
use crate::scalar::Complex64;
use crate::word::{packed_word_count, PauliWord};

#[derive(Clone, Copy, Debug)]
pub(crate) struct StatevectorPauliAction {
    dimension: usize,
    x_mask: usize,
    z_mask: usize,
    phase: Complex64,
}

impl StatevectorPauliAction {
    pub(crate) fn from_word(word: &PauliWord) -> Result<Self, PauliError> {
        Self::from_packed(word.nqubits(), word.x_words(), word.z_words(), 1.0)
    }

    pub(crate) fn from_packed(
        nqubits: usize,
        x_words: &[u64],
        z_words: &[u64],
        sign: f64,
    ) -> Result<Self, PauliError> {
        if !sign.is_finite() || (sign != 1.0 && sign != -1.0) {
            return Err(PauliError::InvalidClifford {
                context: "a signed Pauli action requires sign +1 or -1",
            });
        }
        let expected = packed_word_count(nqubits);
        if x_words.len() != expected {
            return Err(PauliError::InvalidWordLength {
                expected,
                actual: x_words.len(),
            });
        }
        if z_words.len() != expected {
            return Err(PauliError::InvalidWordLength {
                expected,
                actual: z_words.len(),
            });
        }
        if nqubits >= usize::BITS as usize {
            return Err(PauliError::Overflow {
                context: "computing dense statevector dimension",
            });
        }
        let dimension = 1usize
            .checked_shl(nqubits as u32)
            .ok_or(PauliError::Overflow {
                context: "computing dense statevector dimension",
            })?;
        let mut x_mask = 0usize;
        let mut z_mask = 0usize;
        let mut y_count = 0u32;
        for qubit in 0..nqubits {
            let packed_mask = 1u64 << (qubit % 64);
            let x = x_words[qubit / 64] & packed_mask != 0;
            let z = z_words[qubit / 64] & packed_mask != 0;
            let matrix_mask = 1usize << (nqubits - 1 - qubit);
            if x {
                x_mask |= matrix_mask;
            }
            if z {
                z_mask |= matrix_mask;
            }
            if x && z {
                y_count += 1;
            }
        }
        let y_phase = match y_count % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };
        Ok(Self {
            dimension,
            x_mask,
            z_mask,
            phase: y_phase * sign,
        })
    }

    pub(crate) fn x_mask(&self) -> usize {
        self.x_mask
    }

    pub(crate) fn z_mask(&self) -> usize {
        self.z_mask
    }

    pub(crate) fn phase(&self) -> Complex64 {
        self.phase
    }

    pub(crate) fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        if input.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: input.len(),
            });
        }
        if output.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: output.len(),
            });
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.value_at(input, row);
        }
        Ok(())
    }

    pub(crate) fn value_at(&self, input: &[Complex64], row: usize) -> Complex64 {
        let column = row ^ self.x_mask;
        let mut value = self.phase * input[column];
        if (self.z_mask & column).count_ones() & 1 != 0 {
            value = -value;
        }
        value
    }

    pub(crate) fn expectation_from_action(
        &self,
        input: &[Complex64],
        applied: &[Complex64],
    ) -> Result<f64, PauliError> {
        if input.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: input.len(),
            });
        }
        if applied.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: applied.len(),
            });
        }
        let mut real = 0.0;
        let mut imaginary = 0.0;
        for (left, right) in input.iter().zip(applied) {
            let product = left.conj() * *right;
            real += product.re;
            imaginary += product.im;
        }
        if !real.is_finite() || !imaginary.is_finite() {
            return Err(PauliError::InvalidClifford {
                context: "non-finite statevector Pauli expectation",
            });
        }
        if imaginary.abs() > 1e-9 {
            return Err(PauliError::InvalidClifford {
                context: "Hermitian Pauli expectation has a non-negligible imaginary part",
            });
        }
        Ok(real.clamp(-1.0, 1.0))
    }

    pub(crate) fn project_from_action_in_place(
        &self,
        state: &mut [Complex64],
        applied: &[Complex64],
        eigenvalue: f64,
        expectation: f64,
    ) -> Result<(), PauliError> {
        if state.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if applied.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: applied.len(),
            });
        }
        if eigenvalue != 1.0 && eigenvalue != -1.0 {
            return Err(PauliError::InvalidClifford {
                context: "a Pauli projector requires eigenvalue +1 or -1",
            });
        }
        let denominator = 2.0 * (1.0 + eigenvalue * expectation);
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(PauliError::InvalidClifford {
                context: "Pauli projector has zero or non-finite probability",
            });
        }
        let scale = denominator.sqrt().recip();
        for (value, applied_value) in state.iter_mut().zip(applied) {
            *value = (*value + eigenvalue * *applied_value) * scale;
        }
        Ok(())
    }
}
