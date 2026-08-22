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
    let echelon = echelon(rows.to_vec(), variable_count);
    let rank = echelon.pivots.len();
    let mut pivot_flags = vec![false; variable_count];
    for &pivot in &echelon.pivots {
        pivot_flags[pivot] = true;
    }

    let mut result = Vec::with_capacity(variable_count - rank);
    for (free, &is_pivot) in pivot_flags.iter().enumerate() {
        if is_pivot {
            continue;
        }
        let mut vector = vec![0_u64; packed_count(variable_count)];
        set_bit(&mut vector, free, true);
        for (row, &pivot) in echelon.pivots.iter().enumerate().rev() {
            let value = parity_and(&echelon.rows[row], &vector);
            set_bit(&mut vector, pivot, value);
        }
        result.push(vector);
    }
    (result, rank)
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
