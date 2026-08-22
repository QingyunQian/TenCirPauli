//! Static qubit stabilizer-code algebra.

use std::collections::BTreeMap;
use std::mem::size_of;

use crate::error::PauliError;
use crate::gf2::{get_bit, is_zero, null_space, null_space_basis, packed_count, set_bit};
use crate::gf2::{xor_assign, RrefBasis};
use crate::word::{packed_product_phase_exponent, packed_word_count, PauliWord};

/// The three mutually exclusive classifications of a phase-free Pauli error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StabilizerErrorKind {
    /// At least one supplied check anticommutes with the error.
    SyndromeVisible = 0,
    /// The error is in the supplied stabilizer group.
    Stabilizer = 1,
    /// The error is an undetectable non-trivial logical Pauli.
    Logical = 2,
}

/// One singular Pauli classification result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorClassification {
    pub syndrome: Vec<u8>,
    pub kind: StabilizerErrorKind,
    pub logical_x: Option<Vec<u8>>,
    pub logical_z: Option<Vec<u8>>,
}

/// Flat output of an unpacked batched classification or correction analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchClassification {
    pub syndromes: Vec<u8>,
    pub kinds: Vec<u8>,
    pub zero_syndrome_indices: Vec<usize>,
    pub logical_x: Vec<u8>,
    pub logical_z: Vec<u8>,
}

/// Flat output of a packed batched classification or correction analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedBatchClassification {
    pub syndromes: Vec<u64>,
    pub kinds: Vec<u8>,
    pub zero_syndrome_indices: Vec<usize>,
    pub logical_x: Vec<u8>,
    pub logical_z: Vec<u8>,
}

/// Batched decoder-correction output has the same representation as an
/// unpacked batch classification. The alias names the semantic operation at
/// call sites.
pub type CorrectionAnalysis = BatchClassification;
/// Packed batched decoder-correction output.
pub type PackedCorrectionAnalysis = PackedBatchClassification;

/// Result of applying the stabilizer Knill--Laflamme condition to an input set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorSetAnalysis {
    pub is_correctable: bool,
    pub conflicting_pair: Option<(usize, usize)>,
    pub logical_x: Option<Vec<u8>>,
    pub logical_z: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct EliminationRow {
    bits: Vec<u64>,
    provenance: Vec<u64>,
    sign: i8,
}

#[derive(Clone, Debug)]
struct ErrorSetGroup {
    reference_index: usize,
    reference_x: Vec<u64>,
    reference_z: Vec<u64>,
    conflict_found: bool,
}

type LogicalWordPair = (Vec<Vec<u64>>, Vec<Vec<u64>>);

#[derive(Clone, Copy, Debug)]
struct ConstructionEstimate {
    bytes: u128,
}

impl ConstructionEstimate {
    fn retained_input(
        nqubits: usize,
        check_count: usize,
        word_count: usize,
    ) -> Result<Self, PauliError> {
        let packed_check_words = checked_product(
            checked_product(check_count, word_count, "estimating one packed check array")?,
            2,
            "estimating packed X/Z check arrays",
        )?;
        let bytes = checked_sum(&[
            checked_bytes(
                packed_check_words,
                size_of::<u64>(),
                "estimating retained packed checks",
            )?,
            checked_bytes(
                check_count,
                size_of::<i8>(),
                "estimating retained check signs",
            )?,
            checked_bytes(check_count, size_of::<usize>(), "estimating check weights")?,
            checked_bytes(nqubits, size_of::<usize>(), "estimating qubit degrees")?,
        ])?;
        Ok(Self { bytes })
    }

    fn retained_after_rank(
        nqubits: usize,
        check_count: usize,
        word_count: usize,
        provenance_words: usize,
        rank: usize,
    ) -> Result<Self, PauliError> {
        let input = Self::retained_input(nqubits, check_count, word_count)?;
        let relation_count = check_count.checked_sub(rank).ok_or(PauliError::Overflow {
            context: "estimating stabilizer relations after rank",
        })?;
        let relation_words = checked_product(
            relation_count,
            provenance_words,
            "estimating retained stabilizer relations",
        )?;
        // Both logical families retain one packed X row and one packed Z row.
        // Nested Vec headers and transient construction workspaces are
        // intentionally outside this advisory estimate.
        let logical_count = nqubits.checked_sub(rank).ok_or(PauliError::Overflow {
            context: "estimating logical representatives after rank",
        })?;
        let logical_words = checked_product(
            checked_product(
                logical_count,
                4,
                "estimating retained logical representative words",
            )?,
            word_count,
            "estimating retained logical representatives",
        )?;
        let additional = checked_sum(&[
            checked_bytes(
                relation_words,
                size_of::<u64>(),
                "estimating retained stabilizer relations",
            )?,
            checked_bytes(
                logical_words,
                size_of::<u64>(),
                "estimating retained logical representatives",
            )?,
        ])?;
        Ok(Self {
            bytes: input
                .bytes
                .checked_add(additional)
                .ok_or(PauliError::Overflow {
                    context: "summing retained construction estimate",
                })?,
        })
    }
}

/// An immutable signed qubit stabilizer code.
#[derive(Clone, Debug)]
pub struct StabilizerCode {
    nqubits: usize,
    word_count: usize,
    check_word_count: usize,
    check_x_words: Vec<u64>,
    check_z_words: Vec<u64>,
    check_signs: Vec<i8>,
    check_weights: Vec<usize>,
    qubit_degrees: Vec<usize>,
    rank: usize,
    relations: Vec<Vec<u64>>,
    logical_x_words: Vec<Vec<u64>>,
    logical_z_words: Vec<Vec<u64>>,
}

impl StabilizerCode {
    /// Construct from packed phase-free check rows and signed eigenvalues.
    pub fn from_packed(
        nqubits: usize,
        x_words: &[u64],
        z_words: &[u64],
        signs: &[i8],
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        Self::from_packed_inner(nqubits, x_words, z_words, signs, max_bytes, true)
    }

    fn from_packed_inner(
        nqubits: usize,
        x_words: &[u64],
        z_words: &[u64],
        signs: &[i8],
        max_bytes: u128,
        validate_commutation: bool,
    ) -> Result<Self, PauliError> {
        let word_count = packed_word_count(nqubits);
        let check_count = signs.len();
        let expected = checked_product(
            check_count,
            word_count,
            "checking packed stabilizer row dimensions",
        )?;
        if x_words.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: x_words.len(),
            });
        }
        if z_words.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: z_words.len(),
            });
        }
        let variable_count = nqubits.checked_mul(2).ok_or(PauliError::Overflow {
            context: "computing stabilizer symplectic dimension",
        })?;
        let provenance_words = packed_count(check_count);
        let input_estimate =
            ConstructionEstimate::retained_input(nqubits, check_count, word_count)?;
        enforce_limit(input_estimate.bytes, max_bytes)?;

        let tail_mask = tail_mask(nqubits);
        let mut owned_x = x_words.to_vec();
        let mut owned_z = z_words.to_vec();
        if let Some(mask) = tail_mask {
            if word_count != 0 {
                for row in 0..check_count {
                    owned_x[row * word_count + word_count - 1] &= mask;
                    owned_z[row * word_count + word_count - 1] &= mask;
                }
            }
        }

        for (index, &sign) in signs.iter().enumerate() {
            if sign != 1 && sign != -1 {
                return Err(PauliError::InvalidStabilizerSign { index, sign });
            }
            let x_row = &owned_x[index * word_count..(index + 1) * word_count];
            let z_row = &owned_z[index * word_count..(index + 1) * word_count];
            if x_row.iter().all(|&word| word == 0) && z_row.iter().all(|&word| word == 0) {
                return Err(PauliError::InvalidStabilizer {
                    context: "identity checks are not allowed",
                });
            }
        }
        if validate_commutation {
            for left in 0..check_count {
                let left_x = &owned_x[left * word_count..(left + 1) * word_count];
                let left_z = &owned_z[left * word_count..(left + 1) * word_count];
                for right in 0..left {
                    let right_x = &owned_x[right * word_count..(right + 1) * word_count];
                    let right_z = &owned_z[right * word_count..(right + 1) * word_count];
                    if symplectic_words(left_x, left_z, right_x, right_z) {
                        return Err(PauliError::NoncommutingStabilizers { left, right });
                    }
                }
            }
        }
        let mut check_weights = Vec::with_capacity(check_count);
        let mut qubit_degrees = vec![0_usize; nqubits];
        for row in 0..check_count {
            let x_row = &owned_x[row * word_count..(row + 1) * word_count];
            let z_row = &owned_z[row * word_count..(row + 1) * word_count];
            check_weights.push(
                x_row
                    .iter()
                    .zip(z_row)
                    .map(|(&x, &z)| (x | z).count_ones() as usize)
                    .sum(),
            );
            for (qubit, degree) in qubit_degrees.iter_mut().enumerate() {
                if packed_bit(x_row, qubit) || packed_bit(z_row, qubit) {
                    *degree += 1;
                }
            }
        }
        let mut rows = Vec::with_capacity(check_count);
        for index in 0..check_count {
            let x_row = &owned_x[index * word_count..(index + 1) * word_count];
            let z_row = &owned_z[index * word_count..(index + 1) * word_count];
            let mut provenance = vec![0_u64; provenance_words];
            set_bit(&mut provenance, index, true);
            rows.push(EliminationRow {
                bits: bits_from_words(x_row, z_row, nqubits),
                provenance,
                sign: signs[index],
            });
        }
        let mut rank = 0;
        for column in 0..variable_count {
            let Some(pivot) = (rank..rows.len()).find(|&row| get_bit(&rows[row].bits, column))
            else {
                continue;
            };
            rows.swap(rank, pivot);
            let pivot_bits = rows[rank].bits.clone();
            let pivot_provenance = rows[rank].provenance.clone();
            let pivot_sign = rows[rank].sign;
            for (row_index, row) in rows.iter_mut().enumerate() {
                if row_index != rank && get_bit(&row.bits, column) {
                    combine_elimination_row(
                        row,
                        &pivot_bits,
                        &pivot_provenance,
                        pivot_sign,
                        nqubits,
                    )?;
                }
            }
            rank += 1;
            if rank == rows.len() {
                break;
            }
        }
        if rank > nqubits {
            return Err(PauliError::InvalidStabilizer {
                context: "commuting stabilizer rank cannot exceed the qubit count",
            });
        }
        for row in rows.iter().skip(rank) {
            if !is_zero(&row.bits) {
                return Err(PauliError::InvalidStabilizer {
                    context: "stabilizer elimination left a nonzero dependent row",
                });
            }
            if row.sign != 1 {
                return Err(PauliError::InvalidStabilizer {
                    context: "signed dependent checks multiply to -I",
                });
            }
        }

        let canonical_bits = rows[..rank]
            .iter()
            .map(|row| row.bits.clone())
            .collect::<Vec<_>>();
        let relations = rows[rank..]
            .iter()
            .map(|row| row.provenance.clone())
            .collect::<Vec<_>>();
        if relations.len() != check_count - rank {
            return Err(PauliError::InvalidStabilizer {
                context: "stabilizer relation rank is inconsistent",
            });
        }
        let retained_estimate = ConstructionEstimate::retained_after_rank(
            nqubits,
            check_count,
            word_count,
            provenance_words,
            rank,
        )?;
        enforce_limit(retained_estimate.bytes, max_bytes)?;
        let (logical_x_words, logical_z_words) =
            build_logical_basis(&canonical_bits, nqubits, rank)?;

        Ok(Self {
            nqubits,
            word_count,
            check_word_count: packed_count(check_count),
            check_x_words: owned_x,
            check_z_words: owned_z,
            check_signs: signs.to_vec(),
            check_weights,
            qubit_degrees,
            rank,
            relations,
            logical_x_words,
            logical_z_words,
        })
    }

    /// Construct from binary `X` and `Z` check arrays flattened row-major.
    pub fn from_binary(
        nqubits: usize,
        x_bits: &[u8],
        z_bits: &[u8],
        signs: &[i8],
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        let row_count = signs.len();
        let expected = checked_product(row_count, nqubits, "checking binary stabilizer rows")?;
        if x_bits.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: x_bits.len(),
            });
        }
        if z_bits.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: z_bits.len(),
            });
        }
        let (packed_x, packed_z) = pack_binary_rows(nqubits, x_bits, z_bits, row_count)?;
        Self::from_packed(nqubits, &packed_x, &packed_z, signs, max_bytes)
    }

    /// Construct from CSS X-only rows followed by Z-only rows.
    pub fn from_css_binary(
        nqubits: usize,
        x_bits: &[u8],
        z_bits: &[u8],
        x_signs: &[i8],
        z_signs: &[i8],
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        let x_rows = x_signs.len();
        let z_rows = z_signs.len();
        let x_expected = checked_product(x_rows, nqubits, "checking CSS X rows")?;
        let z_expected = checked_product(z_rows, nqubits, "checking CSS Z rows")?;
        if x_bits.len() != x_expected || z_bits.len() != z_expected {
            return Err(PauliError::InvalidStructureLength {
                expected: x_expected,
                actual: x_bits.len(),
            });
        }
        validate_binary_values(x_bits)?;
        validate_binary_values(z_bits)?;

        let word_count = packed_word_count(nqubits);
        let total_rows = x_rows.checked_add(z_rows).ok_or(PauliError::Overflow {
            context: "combining CSS stabilizer row counts",
        })?;
        let total_words = checked_product(total_rows, word_count, "packing CSS checks")?;
        let mut packed_x = vec![0_u64; total_words];
        let mut packed_z = vec![0_u64; total_words];
        for row in 0..x_rows {
            for qubit in 0..nqubits {
                if x_bits[row * nqubits + qubit] != 0 {
                    set_bit(
                        &mut packed_x[row * word_count..(row + 1) * word_count],
                        qubit,
                        true,
                    );
                }
            }
        }
        for row in 0..z_rows {
            let target = x_rows + row;
            for qubit in 0..nqubits {
                if z_bits[row * nqubits + qubit] != 0 {
                    set_bit(
                        &mut packed_z[target * word_count..(target + 1) * word_count],
                        qubit,
                        true,
                    );
                }
            }
        }
        let mut signs = Vec::with_capacity(total_rows);
        signs.extend_from_slice(x_signs);
        signs.extend_from_slice(z_signs);
        validate_css_commutation(&packed_x, &packed_z, x_rows, z_rows, word_count)?;
        Self::from_packed_inner(nqubits, &packed_x, &packed_z, &signs, max_bytes, false)
    }

    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    pub fn nchecks(&self) -> usize {
        self.check_signs.len()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn nlogical(&self) -> usize {
        self.logical_x_words.len()
    }

    pub fn nrelations(&self) -> usize {
        self.relations.len()
    }

    pub fn word_count(&self) -> usize {
        self.word_count
    }

    pub fn check_word_count(&self) -> usize {
        self.check_word_count
    }

    pub fn check_x_words(&self) -> &[u64] {
        &self.check_x_words
    }

    pub fn check_z_words(&self) -> &[u64] {
        &self.check_z_words
    }

    pub fn check_signs(&self) -> &[i8] {
        &self.check_signs
    }

    pub fn check_weights(&self) -> &[usize] {
        &self.check_weights
    }

    pub fn qubit_degrees(&self) -> &[usize] {
        &self.qubit_degrees
    }

    pub fn relation_words(&self) -> &[Vec<u64>] {
        &self.relations
    }

    pub fn logical_operator_words(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
        let mut logical_x_x = Vec::new();
        let mut logical_x_z = Vec::new();
        let mut logical_z_x = Vec::new();
        let mut logical_z_z = Vec::new();
        for words in &self.logical_x_words {
            logical_x_x.extend_from_slice(&words[..self.word_count]);
            logical_x_z.extend_from_slice(&words[self.word_count..]);
        }
        for words in &self.logical_z_words {
            logical_z_x.extend_from_slice(&words[..self.word_count]);
            logical_z_z.extend_from_slice(&words[self.word_count..]);
        }
        (logical_x_x, logical_x_z, logical_z_x, logical_z_z)
    }

    pub fn logical_operators(&self) -> Result<(Vec<PauliWord>, Vec<PauliWord>), PauliError> {
        let x = self
            .logical_x_words
            .iter()
            .map(|words| {
                PauliWord::from_words(
                    self.nqubits,
                    words[..self.word_count].to_vec(),
                    words[self.word_count..].to_vec(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let z = self
            .logical_z_words
            .iter()
            .map(|words| {
                PauliWord::from_words(
                    self.nqubits,
                    words[..self.word_count].to_vec(),
                    words[self.word_count..].to_vec(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((x, z))
    }

    pub fn syndrome_one(&self, x_words: &[u64], z_words: &[u64]) -> Result<Vec<u8>, PauliError> {
        self.validate_word_pair(x_words, z_words)?;
        let mut syndrome = vec![0_u8; self.nchecks()];
        self.syndrome_row_unpacked_into(x_words, z_words, &mut syndrome);
        Ok(syndrome)
    }

    pub fn syndrome_many_unpacked(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        batch: usize,
        max_bytes: u128,
    ) -> Result<Vec<u8>, PauliError> {
        self.validate_batch_words(x_words, z_words, batch)?;
        let output = checked_product(batch, self.nchecks(), "allocating unpacked syndromes")?;
        enforce_limit(
            checked_bytes(output, size_of::<u8>(), "allocating unpacked syndromes")?,
            max_bytes,
        )?;
        let mut result = vec![0_u8; output];
        for row in 0..batch {
            self.syndrome_row_unpacked_into(
                &x_words[row * self.word_count..(row + 1) * self.word_count],
                &z_words[row * self.word_count..(row + 1) * self.word_count],
                &mut result[row * self.nchecks()..(row + 1) * self.nchecks()],
            );
        }
        Ok(result)
    }

    pub fn syndrome_many_packed(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        batch: usize,
        max_bytes: u128,
    ) -> Result<Vec<u64>, PauliError> {
        self.validate_batch_words(x_words, z_words, batch)?;
        let output_words = packed_count(self.nchecks());
        let output = checked_product(batch, output_words, "allocating packed syndromes")?;
        enforce_limit(
            checked_bytes(output, size_of::<u64>(), "allocating packed syndromes")?,
            max_bytes,
        )?;
        let mut result = vec![0_u64; output];
        for row in 0..batch {
            self.syndrome_row_packed_into(
                &x_words[row * self.word_count..(row + 1) * self.word_count],
                &z_words[row * self.word_count..(row + 1) * self.word_count],
                &mut result[row * output_words..(row + 1) * output_words],
            );
        }
        Ok(result)
    }

    pub fn classify_one(
        &self,
        x_words: &[u64],
        z_words: &[u64],
    ) -> Result<ErrorClassification, PauliError> {
        self.validate_word_pair(x_words, z_words)?;
        let mut syndrome = vec![0_u8; self.nchecks()];
        if self.syndrome_row_unpacked_into(x_words, z_words, &mut syndrome) {
            return Ok(ErrorClassification {
                syndrome,
                kind: StabilizerErrorKind::SyndromeVisible,
                logical_x: None,
                logical_z: None,
            });
        }
        let (logical_x, logical_z) = self.logical_coordinates_words(x_words, z_words);
        let is_logical =
            logical_x.iter().any(|&value| value != 0) || logical_z.iter().any(|&value| value != 0);
        Ok(ErrorClassification {
            syndrome,
            kind: if is_logical {
                StabilizerErrorKind::Logical
            } else {
                StabilizerErrorKind::Stabilizer
            },
            logical_x: Some(logical_x),
            logical_z: Some(logical_z),
        })
    }

    pub fn classify_many(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        batch: usize,
        max_bytes: u128,
    ) -> Result<BatchClassification, PauliError> {
        self.validate_batch_words(x_words, z_words, batch)?;
        let syndrome_len =
            checked_product(batch, self.nchecks(), "allocating classification syndromes")?;
        let syndrome_bytes = checked_bytes(
            syndrome_len,
            size_of::<u8>(),
            "allocating classification syndromes",
        )?;
        self.check_classification_budget(batch, syndrome_bytes, max_bytes)?;
        let logical_capacity =
            checked_product(batch, self.nlogical(), "allocating logical coordinates")?;
        let mut syndromes = vec![0_u8; syndrome_len];
        let mut kinds = Vec::with_capacity(batch);
        let mut zero_indices = Vec::with_capacity(batch);
        let mut logical_x = Vec::with_capacity(logical_capacity);
        let mut logical_z = Vec::with_capacity(logical_capacity);
        for row in 0..batch {
            let x_row = &x_words[row * self.word_count..(row + 1) * self.word_count];
            let z_row = &z_words[row * self.word_count..(row + 1) * self.word_count];
            let syndrome_row = &mut syndromes[row * self.nchecks()..(row + 1) * self.nchecks()];
            if self.syndrome_row_unpacked_into(x_row, z_row, syndrome_row) {
                kinds.push(StabilizerErrorKind::SyndromeVisible as u8);
            } else {
                let (kind, x, z) = self.classify_zero_words(x_row, z_row);
                kinds.push(kind);
                zero_indices.push(row);
                logical_x.extend(x);
                logical_z.extend(z);
            }
        }
        Ok(BatchClassification {
            syndromes,
            kinds,
            zero_syndrome_indices: zero_indices,
            logical_x,
            logical_z,
        })
    }

    pub fn classify_many_packed(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        batch: usize,
        max_bytes: u128,
    ) -> Result<PackedBatchClassification, PauliError> {
        self.validate_batch_words(x_words, z_words, batch)?;
        let syndrome_words = packed_count(self.nchecks());
        let syndrome_len = checked_product(
            batch,
            syndrome_words,
            "allocating packed classification syndromes",
        )?;
        let syndrome_bytes = checked_bytes(
            syndrome_len,
            size_of::<u64>(),
            "allocating packed classification syndromes",
        )?;
        self.check_classification_budget(batch, syndrome_bytes, max_bytes)?;
        let logical_capacity =
            checked_product(batch, self.nlogical(), "allocating logical coordinates")?;
        let mut syndromes = vec![0_u64; syndrome_len];
        let mut kinds = Vec::with_capacity(batch);
        let mut zero_indices = Vec::with_capacity(batch);
        let mut logical_x = Vec::with_capacity(logical_capacity);
        let mut logical_z = Vec::with_capacity(logical_capacity);
        for row in 0..batch {
            let x_row = &x_words[row * self.word_count..(row + 1) * self.word_count];
            let z_row = &z_words[row * self.word_count..(row + 1) * self.word_count];
            let syndrome_row = &mut syndromes[row * syndrome_words..(row + 1) * syndrome_words];
            if self.syndrome_row_packed_into(x_row, z_row, syndrome_row) {
                kinds.push(StabilizerErrorKind::SyndromeVisible as u8);
            } else {
                let (kind, x, z) = self.classify_zero_words(x_row, z_row);
                kinds.push(kind);
                zero_indices.push(row);
                logical_x.extend(x);
                logical_z.extend(z);
            }
        }
        Ok(PackedBatchClassification {
            syndromes,
            kinds,
            zero_syndrome_indices: zero_indices,
            logical_x,
            logical_z,
        })
    }

    pub fn analyze_error_set(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        batch: usize,
        max_bytes: u128,
    ) -> Result<ErrorSetAnalysis, PauliError> {
        self.validate_batch_words(x_words, z_words, batch)?;
        let syndrome_words = packed_count(self.nchecks());
        let workspace = checked_sum(&[
            checked_bytes(
                checked_product(batch, self.word_count, "analyzing error-set workspace")?,
                size_of::<u64>() * 2,
                "analyzing error-set workspace",
            )?,
            checked_bytes(
                checked_product(batch, syndrome_words, "analyzing syndrome-group keys")?,
                size_of::<u64>(),
                "analyzing syndrome-group keys",
            )?,
            checked_bytes(batch, size_of::<usize>(), "analyzing error-set indices")?,
        ])?;
        enforce_limit(workspace, max_bytes)?;
        let mut groups: BTreeMap<Vec<u64>, ErrorSetGroup> = BTreeMap::new();
        let mut best_conflict: Option<(usize, usize, Vec<u8>, Vec<u8>)> = None;
        let mut syndrome = vec![0_u64; syndrome_words];
        for index in 0..batch {
            let x_row = &x_words[index * self.word_count..(index + 1) * self.word_count];
            let z_row = &z_words[index * self.word_count..(index + 1) * self.word_count];
            self.syndrome_row_packed_into(x_row, z_row, &mut syndrome);
            let group = groups
                .entry(syndrome.clone())
                .or_insert_with(|| ErrorSetGroup {
                    reference_index: index,
                    reference_x: x_row.to_vec(),
                    reference_z: z_row.to_vec(),
                    conflict_found: false,
                });
            if group.conflict_found || index == group.reference_index {
                continue;
            }
            let relative_x = xor_words(x_row, &group.reference_x);
            let relative_z = xor_words(z_row, &group.reference_z);
            let (logical_x, logical_z) = self.logical_coordinates_words(&relative_x, &relative_z);
            if logical_x.iter().any(|&value| value != 0)
                || logical_z.iter().any(|&value| value != 0)
            {
                let candidate = (group.reference_index, index, logical_x, logical_z);
                if best_conflict
                    .as_ref()
                    .is_none_or(|best| (candidate.0, candidate.1) < (best.0, best.1))
                {
                    best_conflict = Some(candidate);
                }
                group.conflict_found = true;
            }
        }
        if let Some((left, right, logical_x, logical_z)) = best_conflict {
            return Ok(ErrorSetAnalysis {
                is_correctable: false,
                conflicting_pair: Some((left, right)),
                logical_x: Some(logical_x),
                logical_z: Some(logical_z),
            });
        }
        Ok(ErrorSetAnalysis {
            is_correctable: true,
            conflicting_pair: None,
            logical_x: None,
            logical_z: None,
        })
    }

    pub fn analyze_corrections_unpacked(
        &self,
        errors: (&[u64], &[u64]),
        corrections: (&[u64], &[u64]),
        batch: usize,
        max_bytes: u128,
    ) -> Result<CorrectionAnalysis, PauliError> {
        let (error_x_words, error_z_words) = errors;
        let (correction_x_words, correction_z_words) = corrections;
        self.validate_batch_words(error_x_words, error_z_words, batch)?;
        self.validate_batch_words(correction_x_words, correction_z_words, batch)?;
        let syndrome_len =
            checked_product(batch, self.nchecks(), "allocating correction syndromes")?;
        let syndrome_bytes = checked_bytes(
            syndrome_len,
            size_of::<u8>(),
            "allocating correction syndromes",
        )?;
        let scratch_bytes = checked_bytes(
            checked_product(self.word_count, 2, "allocating correction scratch")?,
            size_of::<u64>(),
            "allocating correction scratch",
        )?;
        self.check_classification_budget_with_scratch(
            batch,
            syndrome_bytes,
            scratch_bytes,
            max_bytes,
        )?;
        let logical_capacity =
            checked_product(batch, self.nlogical(), "allocating logical coordinates")?;
        let mut residual_x = vec![0_u64; self.word_count];
        let mut residual_z = vec![0_u64; self.word_count];
        let mut syndromes = vec![0_u8; syndrome_len];
        let mut kinds = Vec::with_capacity(batch);
        let mut zero_indices = Vec::with_capacity(batch);
        let mut logical_x = Vec::with_capacity(logical_capacity);
        let mut logical_z = Vec::with_capacity(logical_capacity);
        for row in 0..batch {
            for word in 0..self.word_count {
                residual_x[word] = correction_x_words[row * self.word_count + word]
                    ^ error_x_words[row * self.word_count + word];
                residual_z[word] = correction_z_words[row * self.word_count + word]
                    ^ error_z_words[row * self.word_count + word];
            }
            let syndrome_row = &mut syndromes[row * self.nchecks()..(row + 1) * self.nchecks()];
            if self.syndrome_row_unpacked_into(&residual_x, &residual_z, syndrome_row) {
                kinds.push(StabilizerErrorKind::SyndromeVisible as u8);
            } else {
                let (kind, x, z) = self.classify_zero_words(&residual_x, &residual_z);
                kinds.push(kind);
                zero_indices.push(row);
                logical_x.extend(x);
                logical_z.extend(z);
            }
        }
        Ok(BatchClassification {
            syndromes,
            kinds,
            zero_syndrome_indices: zero_indices,
            logical_x,
            logical_z,
        })
    }

    pub fn analyze_corrections_packed(
        &self,
        errors: (&[u64], &[u64]),
        corrections: (&[u64], &[u64]),
        batch: usize,
        max_bytes: u128,
    ) -> Result<PackedCorrectionAnalysis, PauliError> {
        let (error_x_words, error_z_words) = errors;
        let (correction_x_words, correction_z_words) = corrections;
        self.validate_batch_words(error_x_words, error_z_words, batch)?;
        self.validate_batch_words(correction_x_words, correction_z_words, batch)?;
        let syndrome_words = packed_count(self.nchecks());
        let syndrome_len = checked_product(
            batch,
            syndrome_words,
            "allocating packed correction syndromes",
        )?;
        let syndrome_bytes = checked_bytes(
            syndrome_len,
            size_of::<u64>(),
            "allocating packed correction syndromes",
        )?;
        let scratch_bytes = checked_bytes(
            checked_product(self.word_count, 2, "allocating correction scratch")?,
            size_of::<u64>(),
            "allocating correction scratch",
        )?;
        self.check_classification_budget_with_scratch(
            batch,
            syndrome_bytes,
            scratch_bytes,
            max_bytes,
        )?;
        let logical_capacity =
            checked_product(batch, self.nlogical(), "allocating logical coordinates")?;
        let mut residual_x = vec![0_u64; self.word_count];
        let mut residual_z = vec![0_u64; self.word_count];
        let mut syndromes = vec![0_u64; syndrome_len];
        let mut kinds = Vec::with_capacity(batch);
        let mut zero_indices = Vec::with_capacity(batch);
        let mut logical_x = Vec::with_capacity(logical_capacity);
        let mut logical_z = Vec::with_capacity(logical_capacity);
        for row in 0..batch {
            for word in 0..self.word_count {
                residual_x[word] = correction_x_words[row * self.word_count + word]
                    ^ error_x_words[row * self.word_count + word];
                residual_z[word] = correction_z_words[row * self.word_count + word]
                    ^ error_z_words[row * self.word_count + word];
            }
            let syndrome_row = &mut syndromes[row * syndrome_words..(row + 1) * syndrome_words];
            if self.syndrome_row_packed_into(&residual_x, &residual_z, syndrome_row) {
                kinds.push(StabilizerErrorKind::SyndromeVisible as u8);
            } else {
                let (kind, x, z) = self.classify_zero_words(&residual_x, &residual_z);
                kinds.push(kind);
                zero_indices.push(row);
                logical_x.extend(x);
                logical_z.extend(z);
            }
        }
        Ok(PackedCorrectionAnalysis {
            syndromes,
            kinds,
            zero_syndrome_indices: zero_indices,
            logical_x,
            logical_z,
        })
    }

    fn validate_word_pair(&self, x_words: &[u64], z_words: &[u64]) -> Result<(), PauliError> {
        if x_words.len() != self.word_count {
            return Err(PauliError::InvalidWordLength {
                expected: self.word_count,
                actual: x_words.len(),
            });
        }
        if z_words.len() != self.word_count {
            return Err(PauliError::InvalidWordLength {
                expected: self.word_count,
                actual: z_words.len(),
            });
        }
        Ok(())
    }

    fn validate_batch_words(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        batch: usize,
    ) -> Result<(), PauliError> {
        let expected = checked_product(batch, self.word_count, "checking Pauli batch dimensions")?;
        if x_words.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: x_words.len(),
            });
        }
        if z_words.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: z_words.len(),
            });
        }
        Ok(())
    }

    fn check_classification_budget(
        &self,
        batch: usize,
        syndrome_bytes: u128,
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        self.check_classification_budget_with_scratch(batch, syndrome_bytes, 0, max_bytes)
    }

    fn check_classification_budget_with_scratch(
        &self,
        batch: usize,
        syndrome_bytes: u128,
        scratch_bytes: u128,
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        let logical_bytes = checked_bytes(
            checked_product(
                checked_product(batch, self.nlogical(), "allocating logical coordinates")?,
                2,
                "allocating logical coordinates",
            )?,
            size_of::<u8>(),
            "allocating logical coordinates",
        )?;
        let index_bytes = checked_bytes(
            batch,
            size_of::<usize>(),
            "allocating zero-syndrome indices",
        )?;
        enforce_limit(
            checked_sum(&[
                syndrome_bytes,
                checked_bytes(batch, size_of::<u8>(), "allocating classification kinds")?,
                logical_bytes,
                index_bytes,
                scratch_bytes,
            ])?,
            max_bytes,
        )
    }

    fn syndrome_row_unpacked_into(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        output: &mut [u8],
    ) -> bool {
        debug_assert_eq!(output.len(), self.nchecks());
        let mut visible = false;
        for (check, value) in output.iter_mut().enumerate() {
            let check_x =
                &self.check_x_words[check * self.word_count..(check + 1) * self.word_count];
            let check_z =
                &self.check_z_words[check * self.word_count..(check + 1) * self.word_count];
            let anticommutes = symplectic_words(check_x, check_z, x_words, z_words);
            *value = u8::from(anticommutes);
            visible |= anticommutes;
        }
        visible
    }

    fn syndrome_row_packed_into(
        &self,
        x_words: &[u64],
        z_words: &[u64],
        output: &mut [u64],
    ) -> bool {
        debug_assert_eq!(output.len(), packed_count(self.nchecks()));
        output.fill(0);
        let mut visible = false;
        for check in 0..self.nchecks() {
            let check_x =
                &self.check_x_words[check * self.word_count..(check + 1) * self.word_count];
            let check_z =
                &self.check_z_words[check * self.word_count..(check + 1) * self.word_count];
            let anticommutes = symplectic_words(check_x, check_z, x_words, z_words);
            if anticommutes {
                set_bit(output, check, true);
                visible = true;
            }
        }
        visible
    }

    fn classify_zero_words(&self, x_words: &[u64], z_words: &[u64]) -> (u8, Vec<u8>, Vec<u8>) {
        let (logical_x, logical_z) = self.logical_coordinates_words(x_words, z_words);
        let is_logical =
            logical_x.iter().any(|&value| value != 0) || logical_z.iter().any(|&value| value != 0);
        let kind = if is_logical {
            StabilizerErrorKind::Logical
        } else {
            StabilizerErrorKind::Stabilizer
        };
        (kind as u8, logical_x, logical_z)
    }

    fn logical_coordinates_words(&self, x_words: &[u64], z_words: &[u64]) -> (Vec<u8>, Vec<u8>) {
        let logical_x = self
            .logical_z_words
            .iter()
            .map(|logical_z| {
                u8::from(symplectic_words(
                    x_words,
                    z_words,
                    &logical_z[..self.word_count],
                    &logical_z[self.word_count..],
                ))
            })
            .collect();
        let logical_z = self
            .logical_x_words
            .iter()
            .map(|logical_x| {
                u8::from(symplectic_words(
                    x_words,
                    z_words,
                    &logical_x[..self.word_count],
                    &logical_x[self.word_count..],
                ))
            })
            .collect();
        (logical_x, logical_z)
    }
}

/// Pack a batch of public I/X/Y/Z codes in q0-first order.
pub fn pack_code_batch(
    nqubits: usize,
    codes: &[u8],
    batch: usize,
) -> Result<(Vec<u64>, Vec<u64>), PauliError> {
    let expected = checked_product(batch, nqubits, "checking Pauli-code batch dimensions")?;
    if codes.len() != expected {
        return Err(PauliError::InvalidStructureLength {
            expected,
            actual: codes.len(),
        });
    }
    let word_count = packed_word_count(nqubits);
    let output = checked_product(batch, word_count, "packing Pauli-code batch")?;
    let mut x_words = vec![0_u64; output];
    let mut z_words = vec![0_u64; output];
    for row in 0..batch {
        for qubit in 0..nqubits {
            let code = codes[row * nqubits + qubit];
            let (x, z) = match code {
                0 => (false, false),
                1 => (true, false),
                2 => (true, true),
                3 => (false, true),
                _ => return Err(PauliError::InvalidCode { code, index: qubit }),
            };
            if x {
                set_bit(
                    &mut x_words[row * word_count..(row + 1) * word_count],
                    qubit,
                    true,
                );
            }
            if z {
                set_bit(
                    &mut z_words[row * word_count..(row + 1) * word_count],
                    qubit,
                    true,
                );
            }
        }
    }
    Ok((x_words, z_words))
}

fn build_logical_basis(
    canonical_bits: &[Vec<u64>],
    nqubits: usize,
    rank: usize,
) -> Result<LogicalWordPair, PauliError> {
    let variable_count = nqubits.checked_mul(2).ok_or(PauliError::Overflow {
        context: "computing logical symplectic dimension",
    })?;
    let constraints = canonical_bits
        .iter()
        .map(|row| symplectic_constraint(row, nqubits))
        .collect::<Vec<_>>();
    let (normalizer_basis, constraint_rank) = null_space_basis(&constraints, variable_count);
    if constraint_rank != rank {
        return Err(PauliError::InvalidStabilizer {
            context: "normalizer constraints have an inconsistent rank",
        });
    }
    let logical_count = nqubits - rank;
    let target = logical_count.checked_mul(2).ok_or(PauliError::Overflow {
        context: "computing logical operator count",
    })?;
    if target == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let free_columns = normalizer_basis.free_columns();
    let mut projected_stabilizers = Vec::with_capacity(canonical_bits.len());
    for row in canonical_bits {
        let mut projected = vec![0_u64; packed_count(free_columns.len())];
        for (free_index, &column) in free_columns.iter().enumerate() {
            if get_bit(row, column) {
                set_bit(&mut projected, free_index, true);
            }
        }
        projected_stabilizers.push(projected);
    }
    let (dual_basis, projected_rank) = null_space(&projected_stabilizers, free_columns.len());
    if projected_rank != rank || dual_basis.len() != target {
        return Err(PauliError::InvalidStabilizer {
            context: "stabilizer projection has an inconsistent rank",
        });
    }

    let mut dual_span = RrefBasis::from_reduced_rows(Vec::new(), Vec::new());
    let dual_word_count = packed_count(dual_basis.len());
    let mut quotient = Vec::with_capacity(target);
    for (free_index, &free_column) in free_columns.iter().enumerate() {
        let mut coordinate = vec![0_u64; dual_word_count];
        for (dual_index, row) in dual_basis.iter().enumerate() {
            if get_bit(row, free_index) {
                set_bit(&mut coordinate, dual_index, true);
            }
        }
        if dual_span.insert(&coordinate) {
            quotient.push(normalizer_basis.vector_for_free_column(free_column));
            if quotient.len() == target {
                break;
            }
        }
    }
    if quotient.len() != target {
        return Err(PauliError::InvalidStabilizer {
            context: "failed to construct the logical quotient basis",
        });
    }

    let mut current = quotient;
    let mut logical_x = Vec::with_capacity(logical_count);
    let mut logical_z = Vec::with_capacity(logical_count);
    while !current.is_empty() {
        let x = current.remove(0);
        let Some(partner_index) = current
            .iter()
            .position(|candidate| symplectic_bits(&x, candidate, nqubits))
        else {
            return Err(PauliError::InvalidStabilizer {
                context: "logical quotient has a degenerate symplectic form",
            });
        };
        let z = current.remove(partner_index);
        logical_x.push(x.clone());
        logical_z.push(z.clone());

        let mut projected = Vec::with_capacity(current.len());
        for mut value in current {
            let x_pairing = symplectic_bits(&value, &z, nqubits);
            let z_pairing = symplectic_bits(&value, &x, nqubits);
            if x_pairing {
                xor_assign(&mut value, &x);
            }
            if z_pairing {
                xor_assign(&mut value, &z);
            }
            projected.push(value);
        }
        current = projected;
    }
    if logical_x.len() != logical_count || logical_z.len() != logical_count {
        return Err(PauliError::InvalidStabilizer {
            context: "logical pairing returned the wrong number of pairs",
        });
    }
    let logical_x_words = logical_x
        .into_iter()
        .map(|bits| flat_words_from_bits(&bits, nqubits))
        .collect();
    let logical_z_words = logical_z
        .into_iter()
        .map(|bits| flat_words_from_bits(&bits, nqubits))
        .collect();
    Ok((logical_x_words, logical_z_words))
}

fn combine_elimination_row(
    target: &mut EliminationRow,
    pivot_bits: &[u64],
    pivot_provenance: &[u64],
    pivot_sign: i8,
    nqubits: usize,
) -> Result<(), PauliError> {
    let phase = commuting_product_sign(&target.bits, pivot_bits, nqubits)?;
    target.sign *= pivot_sign * phase;
    xor_assign(&mut target.bits, pivot_bits);
    xor_assign(&mut target.provenance, pivot_provenance);
    Ok(())
}

fn pack_binary_rows(
    nqubits: usize,
    x_bits: &[u8],
    z_bits: &[u8],
    row_count: usize,
) -> Result<(Vec<u64>, Vec<u64>), PauliError> {
    validate_binary_values(x_bits)?;
    validate_binary_values(z_bits)?;
    let word_count = packed_word_count(nqubits);
    let output = checked_product(row_count, word_count, "packing binary stabilizer rows")?;
    let mut x_words = vec![0_u64; output];
    let mut z_words = vec![0_u64; output];
    for row in 0..row_count {
        for qubit in 0..nqubits {
            if x_bits[row * nqubits + qubit] != 0 {
                set_bit(
                    &mut x_words[row * word_count..(row + 1) * word_count],
                    qubit,
                    true,
                );
            }
            if z_bits[row * nqubits + qubit] != 0 {
                set_bit(
                    &mut z_words[row * word_count..(row + 1) * word_count],
                    qubit,
                    true,
                );
            }
        }
    }
    Ok((x_words, z_words))
}

fn validate_binary_values(values: &[u8]) -> Result<(), PauliError> {
    if values.iter().any(|&value| value > 1) {
        return Err(PauliError::InvalidStabilizer {
            context: "binary check arrays must contain only zero and one",
        });
    }
    Ok(())
}

fn bits_from_words(x_words: &[u64], z_words: &[u64], nqubits: usize) -> Vec<u64> {
    let mut bits = vec![0_u64; packed_count(nqubits * 2)];
    bits_from_words_into(x_words, z_words, nqubits, &mut bits);
    bits
}

fn bits_from_words_into(x_words: &[u64], z_words: &[u64], nqubits: usize, bits: &mut [u64]) {
    debug_assert_eq!(bits.len(), packed_count(nqubits * 2));
    bits.fill(0);
    for qubit in 0..nqubits {
        if packed_bit(x_words, qubit) {
            set_bit(bits, qubit, true);
        }
        if packed_bit(z_words, qubit) {
            set_bit(bits, nqubits + qubit, true);
        }
    }
}

fn words_from_bits(bits: &[u64], nqubits: usize) -> (Vec<u64>, Vec<u64>) {
    let word_count = packed_word_count(nqubits);
    let mut x_words = vec![0_u64; word_count];
    let mut z_words = vec![0_u64; word_count];
    for qubit in 0..nqubits {
        if get_bit(bits, qubit) {
            set_bit(&mut x_words, qubit, true);
        }
        if get_bit(bits, nqubits + qubit) {
            set_bit(&mut z_words, qubit, true);
        }
    }
    (x_words, z_words)
}

fn flat_words_from_bits(bits: &[u64], nqubits: usize) -> Vec<u64> {
    let (x_words, z_words) = words_from_bits(bits, nqubits);
    x_words.into_iter().chain(z_words).collect()
}

fn symplectic_constraint(row: &[u64], nqubits: usize) -> Vec<u64> {
    let mut constraint = vec![0_u64; packed_count(nqubits * 2)];
    for qubit in 0..nqubits {
        if get_bit(row, nqubits + qubit) {
            set_bit(&mut constraint, qubit, true);
        }
        if get_bit(row, qubit) {
            set_bit(&mut constraint, nqubits + qubit, true);
        }
    }
    constraint
}

fn symplectic_bits(left: &[u64], right: &[u64], nqubits: usize) -> bool {
    let mut parity = false;
    let word_count = packed_word_count(nqubits);
    let mask = tail_mask(nqubits).unwrap_or(u64::MAX);
    for word in 0..word_count {
        let last = word + 1 == word_count;
        let left_x = interleaved_segment_word(left, word, 0, last, mask);
        let left_z = interleaved_segment_word(left, word, nqubits, last, mask);
        let right_x = interleaved_segment_word(right, word, 0, last, mask);
        let right_z = interleaved_segment_word(right, word, nqubits, last, mask);
        parity ^= (left_x & right_z).count_ones() & 1 != 0;
        parity ^= (left_z & right_x).count_ones() & 1 != 0;
    }
    parity
}

fn symplectic_words(left_x: &[u64], left_z: &[u64], right_x: &[u64], right_z: &[u64]) -> bool {
    left_x.iter().zip(left_z).zip(right_x).zip(right_z).fold(
        false,
        |parity, (((x_left, z_left), x_right), z_right)| {
            parity
                ^ ((x_left & z_right).count_ones() & 1 != 0)
                ^ ((z_left & x_right).count_ones() & 1 != 0)
        },
    )
}

fn validate_css_commutation(
    packed_x: &[u64],
    packed_z: &[u64],
    x_rows: usize,
    z_rows: usize,
    word_count: usize,
) -> Result<(), PauliError> {
    for x_row in 0..x_rows {
        let x_words = &packed_x[x_row * word_count..(x_row + 1) * word_count];
        for z_row in 0..z_rows {
            let check_row = x_rows + z_row;
            let z_words = &packed_z[check_row * word_count..(check_row + 1) * word_count];
            let anticommutes = x_words
                .iter()
                .zip(z_words)
                .fold(false, |parity, (&x_word, &z_word)| {
                    parity ^ ((x_word & z_word).count_ones() & 1 != 0)
                });
            if anticommutes {
                return Err(PauliError::NoncommutingStabilizers {
                    left: x_row,
                    right: check_row,
                });
            }
        }
    }
    Ok(())
}

fn commuting_product_sign(left: &[u64], right: &[u64], nqubits: usize) -> Result<i8, PauliError> {
    let word_count = packed_word_count(nqubits);
    let mask = tail_mask(nqubits).unwrap_or(u64::MAX);
    let mut exponent = 0_u8;
    for word in 0..word_count {
        let last = word + 1 == word_count;
        let left_x = interleaved_segment_word(left, word, 0, last, mask);
        let left_z = interleaved_segment_word(left, word, nqubits, last, mask);
        let right_x = interleaved_segment_word(right, word, 0, last, mask);
        let right_z = interleaved_segment_word(right, word, nqubits, last, mask);
        exponent = (exponent
            + packed_product_phase_exponent(&[left_x], &[left_z], &[right_x], &[right_z]))
            % 4;
    }
    match exponent {
        0 => Ok(1),
        2 => Ok(-1),
        _ => Err(PauliError::InvalidStabilizer {
            context: "commuting stabilizer multiplication produced a non-Hermitian phase",
        }),
    }
}

fn interleaved_segment_word(
    bits: &[u64],
    word: usize,
    start: usize,
    last: bool,
    tail_mask: u64,
) -> u64 {
    let bit = start + word * 64;
    let base = bit / 64;
    let shift = bit % 64;
    let mut value = bits.get(base).copied().unwrap_or(0) >> shift;
    if shift != 0 {
        value |= bits.get(base + 1).copied().unwrap_or(0) << (64 - shift);
    }
    if last {
        value &= tail_mask;
    }
    value
}

fn packed_bit(words: &[u64], bit: usize) -> bool {
    words[bit / 64] & (1_u64 << (bit % 64)) != 0
}

fn xor_words(left: &[u64], right: &[u64]) -> Vec<u64> {
    left.iter().zip(right).map(|(&a, &b)| a ^ b).collect()
}

fn tail_mask(nqubits: usize) -> Option<u64> {
    let remainder = nqubits % 64;
    (remainder != 0).then(|| (1_u64 << remainder) - 1)
}

fn checked_product(left: usize, right: usize, context: &'static str) -> Result<usize, PauliError> {
    left.checked_mul(right)
        .ok_or(PauliError::Overflow { context })
}

fn checked_bytes(
    count: usize,
    element_size: usize,
    context: &'static str,
) -> Result<u128, PauliError> {
    (count as u128)
        .checked_mul(element_size as u128)
        .ok_or(PauliError::Overflow { context })
}

fn checked_sum(values: &[u128]) -> Result<u128, PauliError> {
    values.iter().try_fold(0_u128, |sum, &value| {
        sum.checked_add(value).ok_or(PauliError::Overflow {
            context: "estimating stabilizer-code memory",
        })
    })
}

fn enforce_limit(requested: u128, limit: u128) -> Result<(), PauliError> {
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{packed_product_phase_exponent, StabilizerCode, StabilizerErrorKind};
    use crate::error::PauliError;
    use crate::word::PauliWord;

    fn repetition() -> StabilizerCode {
        StabilizerCode::from_css_binary(3, &[], &[1, 1, 0, 0, 1, 1], &[], &[1, 1], u128::MAX)
            .expect("repetition code")
    }

    #[test]
    fn repetition_classifies_numeric_vectors() {
        let code = repetition();
        let x = super::pack_code_batch(3, &[1, 0, 0], 1).expect("pack");
        let result = code.classify_one(&x.0, &x.1).expect("classify");
        assert_eq!(result.syndrome, vec![1, 0]);
        assert_eq!(result.kind, StabilizerErrorKind::SyndromeVisible);

        let x = super::pack_code_batch(3, &[1, 1, 1], 1).expect("pack");
        let result = code.classify_one(&x.0, &x.1).expect("classify");
        assert_eq!(result.kind, StabilizerErrorKind::Logical);
        assert_eq!(result.logical_x, Some(vec![1]));
        assert_eq!(result.logical_z, Some(vec![0]));
    }

    #[test]
    fn css_constructor_matches_equivalent_binary_constructor() {
        let h = [
            1_u8, 0, 0, 1, 0, 1, 1, // Hamming check row 0
            0, 1, 0, 1, 1, 0, 1, // Hamming check row 1
            0, 0, 1, 0, 1, 1, 1, // Hamming check row 2
        ];
        let zeros = vec![0_u8; h.len()];
        let css = StabilizerCode::from_css_binary(7, &h, &h, &[1, 1, 1], &[1, 1, 1], u128::MAX)
            .expect("CSS constructor");

        let mut x_bits = h.to_vec();
        x_bits.extend_from_slice(&zeros);
        let mut z_bits = zeros;
        z_bits.extend_from_slice(&h);
        let binary = StabilizerCode::from_binary(7, &x_bits, &z_bits, &[1; 6], u128::MAX)
            .expect("equivalent binary constructor");

        assert_eq!(css.rank(), binary.rank());
        assert_eq!(css.nlogical(), binary.nlogical());
        assert_eq!(css.nrelations(), binary.nrelations());
        assert_eq!(css.check_x_words(), binary.check_x_words());
        assert_eq!(css.check_z_words(), binary.check_z_words());
        assert_eq!(css.check_signs(), binary.check_signs());
        assert_eq!(css.check_weights(), binary.check_weights());
        assert_eq!(css.qubit_degrees(), binary.qubit_degrees());
        assert_eq!(css.relation_words(), binary.relation_words());
        assert_eq!(
            css.logical_operator_words(),
            binary.logical_operator_words()
        );

        let error = super::pack_code_batch(7, &[1, 0, 0, 0, 0, 0, 0], 1).expect("pack error");
        assert_eq!(
            css.syndrome_one(&error.0, &error.1),
            Ok(vec![0, 0, 0, 1, 0, 0])
        );
        assert_eq!(
            css.syndrome_one(&error.0, &error.1),
            binary.syndrome_one(&error.0, &error.1)
        );
        assert_eq!(
            css.classify_one(&error.0, &error.1),
            binary.classify_one(&error.0, &error.1)
        );
    }

    #[test]
    fn css_constructor_still_rejects_noncommuting_checks() {
        let result = StabilizerCode::from_css_binary(1, &[1], &[1], &[1], &[1], u128::MAX);
        assert!(matches!(
            result,
            Err(PauliError::NoncommutingStabilizers { left: 0, right: 1 })
        ));
    }

    #[test]
    fn dependent_signed_checks_are_checked() {
        let code = StabilizerCode::from_packed(2, &[3, 3], &[0, 0], &[1, 1], u128::MAX)
            .expect("valid duplicate X checks");
        assert_eq!(code.rank(), 1);
        assert_eq!(code.nrelations(), 1);
        assert_eq!(code.relation_words()[0], vec![3]);
        assert!(StabilizerCode::from_packed(2, &[3, 3], &[0, 0], &[1, -1], u128::MAX).is_err());
    }

    #[test]
    fn empty_code_has_canonical_logicals() {
        let code = StabilizerCode::from_packed(2, &[], &[], &[], u128::MAX).expect("empty code");
        let (x, z) = code.logical_operators().expect("logical operators");
        assert_eq!(x[0].codes(), vec![1, 0]);
        assert_eq!(z[0].codes(), vec![3, 0]);
        assert_eq!(x[1].codes(), vec![0, 1]);
        assert_eq!(z[1].codes(), vec![0, 3]);
    }

    #[test]
    fn packed_phase_kernel_matches_authoritative_word_multiplication() {
        for left_code in 0..4_u8 {
            for right_code in 0..4_u8 {
                let left = PauliWord::from_codes(65, &[left_code; 65]).expect("left word");
                let right = PauliWord::from_codes(65, &[right_code; 65]).expect("right word");
                let (_, phase) = left.multiply(&right).expect("multiply words");
                assert_eq!(
                    packed_product_phase_exponent(
                        left.x_words(),
                        left.z_words(),
                        right.x_words(),
                        right.z_words(),
                    ),
                    phase.exponent(),
                    "codes {left_code} * {right_code}",
                );
            }
        }
    }
}
