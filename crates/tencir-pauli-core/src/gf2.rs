//! Deterministic packed binary-linear-algebra primitives used by code paths
//! whose rows live in a fixed little-endian bit layout.

/// Return the number of `u64` words needed for `bits` binary columns.
pub(crate) const fn packed_count(bits: usize) -> usize {
    bits / 64 + if bits % 64 == 0 { 0 } else { 1 }
}

pub(crate) fn get_bit(row: &[u64], column: usize) -> bool {
    row[column / 64] & (1_u64 << (column % 64)) != 0
}

pub(crate) fn set_bit(row: &mut [u64], column: usize, value: bool) {
    let mask = 1_u64 << (column % 64);
    if value {
        row[column / 64] |= mask;
    } else {
        row[column / 64] &= !mask;
    }
}

pub(crate) fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (lhs, &rhs) in left.iter_mut().zip(right) {
        *lhs ^= rhs;
    }
}

pub(crate) fn is_zero(row: &[u64]) -> bool {
    row.iter().all(|&word| word == 0)
}

pub(crate) fn first_set_bit(row: &[u64]) -> Option<usize> {
    row.iter().enumerate().find_map(|(word, &value)| {
        (value != 0).then(|| word * 64 + value.trailing_zeros() as usize)
    })
}

pub(crate) fn parity_and(left: &[u64], right: &[u64]) -> bool {
    left.iter().zip(right).fold(false, |parity, (&lhs, &rhs)| {
        parity ^ ((lhs & rhs).count_ones() & 1 != 0)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RowEchelon {
    rows: Vec<Vec<u64>>,
    pivots: Vec<usize>,
}

/// Reduce binary rows below each ascending pivot.
fn echelon(mut rows: Vec<Vec<u64>>, variable_count: usize) -> RowEchelon {
    let mut pivots = Vec::new();
    let mut rank = 0;
    for column in 0..variable_count {
        let Some(pivot) = (rank..rows.len()).find(|&row| get_bit(&rows[row], column)) else {
            continue;
        };
        rows.swap(rank, pivot);
        let (prefix, suffix) = rows.split_at_mut(rank + 1);
        let pivot_row = &prefix[rank];
        for row in suffix {
            if get_bit(row, column) {
                xor_assign(row, pivot_row);
            }
        }
        pivots.push(column);
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    RowEchelon { rows, pivots }
}

/// Return a deterministic null-space basis and the row rank.
pub(crate) fn null_space(rows: &[Vec<u64>], variable_count: usize) -> (Vec<Vec<u64>>, usize) {
    let (basis, rank) = null_space_basis(rows, variable_count);
    (basis.collect(), rank)
}

/// A deterministic null-space basis that materializes one vector on demand.
///
/// The stabilizer logical-basis path only needs enough normalizer vectors to
/// span the logical quotient. Keeping the echelon form and producing vectors
/// lazily avoids constructing the rest of the normalizer basis.
pub(crate) struct NullSpaceBasis {
    echelon: RowEchelon,
    pivot_flags: Vec<bool>,
    next_free: usize,
}

impl Iterator for NullSpaceBasis {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_free < self.pivot_flags.len() {
            let free = self.next_free;
            self.next_free += 1;
            if self.pivot_flags[free] {
                continue;
            }
            return Some(self.vector_for_free_column(free));
        }
        None
    }
}

impl NullSpaceBasis {
    /// Return free columns in the same order used by the eager basis.
    pub(crate) fn free_columns(&self) -> Vec<usize> {
        self.pivot_flags
            .iter()
            .enumerate()
            .filter_map(|(column, &is_pivot)| (!is_pivot).then_some(column))
            .collect()
    }

    /// Materialize the basis vector associated with one free column.
    pub(crate) fn vector_for_free_column(&self, free: usize) -> Vec<u64> {
        debug_assert!(free < self.pivot_flags.len());
        debug_assert!(!self.pivot_flags[free]);
        let mut vector = vec![0_u64; packed_count(self.pivot_flags.len())];
        set_bit(&mut vector, free, true);
        for (row, &pivot) in self.echelon.pivots.iter().enumerate().rev() {
            let value = parity_and(&self.echelon.rows[row], &vector);
            set_bit(&mut vector, pivot, value);
        }
        vector
    }
}

/// Return a deterministic lazy null-space basis and the row rank.
pub(crate) fn null_space_basis(
    rows: &[Vec<u64>],
    variable_count: usize,
) -> (NullSpaceBasis, usize) {
    let echelon = echelon(rows.to_vec(), variable_count);
    let rank = echelon.pivots.len();
    let mut pivot_flags = vec![false; variable_count];
    for &pivot in &echelon.pivots {
        pivot_flags[pivot] = true;
    }
    (
        NullSpaceBasis {
            echelon,
            pivot_flags,
            next_free: 0,
        },
        rank,
    )
}

pub(crate) fn rank(rows: &[Vec<u64>], variable_count: usize) -> usize {
    echelon(rows.to_vec(), variable_count).pivots.len()
}

/// A mutable reduced basis used for deterministic quotient-space selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RrefBasis {
    rows: Vec<Vec<u64>>,
    pivots: Vec<usize>,
}

impl RrefBasis {
    /// Build a basis from rows already reduced by the same deterministic RREF
    /// elimination used by the caller. The constructor is intentionally
    /// internal: callers must preserve ascending pivots and reduced rows.
    pub(crate) fn from_reduced_rows(rows: Vec<Vec<u64>>, pivots: Vec<usize>) -> Self {
        debug_assert_eq!(rows.len(), pivots.len());
        debug_assert!(pivots.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(rows
            .iter()
            .zip(&pivots)
            .all(|(row, &pivot)| get_bit(row, pivot)));
        Self { rows, pivots }
    }

    /// Insert a vector if it increases the span and return whether it did.
    pub(crate) fn insert(&mut self, value: &[u64]) -> bool {
        let mut residual = value.to_vec();
        for (row, &pivot) in self.rows.iter().zip(&self.pivots) {
            if get_bit(&residual, pivot) {
                xor_assign(&mut residual, row);
            }
        }
        let Some(pivot) = first_set_bit(&residual) else {
            return false;
        };

        for row in &mut self.rows {
            if get_bit(row, pivot) {
                xor_assign(row, &residual);
            }
        }
        let insertion = self.pivots.partition_point(|&column| column < pivot);
        self.pivots.insert(insertion, pivot);
        self.rows.insert(insertion, residual);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{null_space, null_space_basis, parity_and};

    #[test]
    fn lazy_null_space_matches_eager_numeric_basis() {
        let rows = vec![vec![0b0011_u64], vec![0b0110_u64]];
        let (eager, eager_rank) = null_space(&rows, 4);
        let (lazy, lazy_rank) = null_space_basis(&rows, 4);
        let materialized = lazy.collect::<Vec<_>>();

        assert_eq!(eager_rank, 2);
        assert_eq!(lazy_rank, eager_rank);
        assert_eq!(materialized, vec![vec![0b0111_u64], vec![0b1000_u64]]);
        assert_eq!(materialized, eager);
        assert!(materialized
            .iter()
            .all(|vector| { rows.iter().all(|row| !parity_and(row, vector)) }));
    }
}
