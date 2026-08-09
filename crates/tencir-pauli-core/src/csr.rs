//! Shared deterministic destination-row CSR construction sink.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::{Complex64, PauliError};

type CsrArrays = (Vec<u64>, Vec<u64>, Vec<Complex64>, usize);
pub(crate) const CSR_CHUNK_CAPACITY: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CsrBuildStats {
    pub(crate) chunk_count: usize,
    pub(crate) sorted_entries: usize,
    pub(crate) merged_entries: usize,
    pub(crate) heap_operations: usize,
    pub(crate) final_array_allocation_entries: usize,
    pub(crate) chunk_metadata_bytes: u128,
    pub(crate) sort_nanos: u128,
    pub(crate) merge_nanos: u128,
    pub(crate) final_array_allocation_nanos: u128,
}

enum CsrSinkMode {
    /// Source-major producers append bounded `(row, column, value)` runs.
    Chunked {
        chunks: Vec<Vec<(usize, u64, Complex64)>>,
        chunk_capacity: usize,
    },
    /// Destination-row producers append already sorted row payloads directly.
    RowOrdered {
        indptr: Vec<u64>,
        columns: Vec<u64>,
        values: Vec<Complex64>,
        next_row: usize,
    },
}

/// Shared deterministic CSR sink.
///
/// Source-major producers use fixed-capacity sorted runs and a binary-heap
/// k-way merge. Destination-row producers use [`Self::new_row_ordered`] and
/// publish each sorted row directly, so neither route retains a global COO
/// array or a vector of row vectors.
pub(crate) struct CsrSink {
    dimension: usize,
    mode: CsrSinkMode,
    entries: usize,
}

impl CsrSink {
    pub(crate) fn with_chunk_capacity(dimension: usize, chunk_capacity: usize) -> Self {
        Self::with_chunk_capacity_impl(dimension, chunk_capacity)
    }

    fn with_chunk_capacity_impl(dimension: usize, chunk_capacity: usize) -> Self {
        assert!(chunk_capacity != 0, "CSR chunk capacity must be nonzero");
        Self {
            dimension,
            mode: CsrSinkMode::Chunked {
                chunks: vec![Vec::with_capacity(chunk_capacity)],
                chunk_capacity,
            },
            entries: 0,
        }
    }

    pub(crate) fn new_row_ordered(dimension: usize) -> Result<Self, PauliError> {
        let pointer_count = dimension.checked_add(1).ok_or(PauliError::Overflow {
            context: "sizing CSR row pointers",
        })?;
        Ok(Self {
            dimension,
            mode: CsrSinkMode::RowOrdered {
                indptr: vec![0_u64; pointer_count],
                columns: Vec::new(),
                values: Vec::new(),
                next_row: 0,
            },
            entries: 0,
        })
    }

    pub(crate) fn push(
        &mut self,
        row: usize,
        column: u64,
        value: Complex64,
    ) -> Result<(), PauliError> {
        if row >= self.dimension {
            return Err(PauliError::InvalidIndex {
                context: "CSR destination row",
            });
        }
        if value.re == 0.0 && value.im == 0.0 {
            return Ok(());
        }
        match &mut self.mode {
            CsrSinkMode::Chunked {
                chunks,
                chunk_capacity,
            } => {
                if chunks
                    .last()
                    .is_some_and(|chunk| chunk.len() == *chunk_capacity)
                {
                    chunks.push(Vec::with_capacity(*chunk_capacity));
                }
                chunks
                    .last_mut()
                    .expect("CSR sink has an active chunk")
                    .push((row, column, value));
                self.entries = self.entries.checked_add(1).ok_or(PauliError::Overflow {
                    context: "counting CSR transitions",
                })?;
                Ok(())
            }
            CsrSinkMode::RowOrdered { .. } => {
                self.finish_row(row, std::slice::from_ref(&(column, value)))
            }
        }
    }

    /// Append one complete destination row in strictly increasing column order.
    pub(crate) fn finish_row(
        &mut self,
        row: usize,
        entries: &[(u64, Complex64)],
    ) -> Result<(), PauliError> {
        if row >= self.dimension {
            return Err(PauliError::InvalidIndex {
                context: "CSR destination row",
            });
        }
        let CsrSinkMode::RowOrdered {
            indptr,
            columns,
            values,
            next_row,
        } = &mut self.mode
        else {
            return Err(PauliError::InvalidStructureLength {
                expected: 0,
                actual: 1,
            });
        };
        if row != *next_row {
            return Err(PauliError::InvalidIndex {
                context: "CSR rows must be finalized in order",
            });
        }
        let mut previous = None;
        for &(column, value) in entries {
            if column >= self.dimension as u64 {
                return Err(PauliError::InvalidIndex {
                    context: "CSR source column",
                });
            }
            if let Some(previous) = previous {
                if column <= previous {
                    return Err(PauliError::InvalidStructureLength {
                        expected: previous as usize + 1,
                        actual: column as usize,
                    });
                }
            }
            previous = Some(column);
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            columns.push(column);
            values.push(value);
        }
        *next_row = next_row.checked_add(1).ok_or(PauliError::Overflow {
            context: "advancing CSR row",
        })?;
        indptr[*next_row] = u64::try_from(columns.len()).map_err(|_| PauliError::Overflow {
            context: "building CSR row pointer",
        })?;
        self.entries = columns.len();
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.entries
    }

    pub(crate) fn into_csr(self) -> Result<CsrArrays, PauliError> {
        self.into_csr_with_stats().map(|(arrays, _)| arrays)
    }

    pub(crate) fn into_csr_with_stats(self) -> Result<(CsrArrays, CsrBuildStats), PauliError> {
        match self.mode {
            CsrSinkMode::RowOrdered {
                mut indptr,
                columns,
                values,
                next_row,
            } => {
                let final_count = columns.len();
                let final_count_u64 =
                    u64::try_from(final_count).map_err(|_| PauliError::Overflow {
                        context: "building CSR row pointer",
                    })?;
                for pointer in indptr
                    .iter_mut()
                    .take(self.dimension + 1)
                    .skip(next_row + 1)
                {
                    *pointer = final_count_u64;
                }
                Ok((
                    (indptr, columns, values, final_count),
                    CsrBuildStats {
                        final_array_allocation_entries: final_count,
                        ..CsrBuildStats::default()
                    },
                ))
            }
            CsrSinkMode::Chunked {
                mut chunks,
                chunk_capacity: _,
            } => Self::merge_chunks(self.dimension, self.entries, &mut chunks),
        }
    }

    fn merge_chunks(
        dimension: usize,
        entries: usize,
        chunks: &mut [Vec<(usize, u64, Complex64)>],
    ) -> Result<(CsrArrays, CsrBuildStats), PauliError> {
        let pointer_count = dimension.checked_add(1).ok_or(PauliError::Overflow {
            context: "sizing CSR row pointers",
        })?;
        let mut indptr = vec![0_u64; pointer_count];
        let sort_start = Instant::now();
        for chunk in chunks.iter_mut() {
            chunk.sort_by_key(|&(row, column, _)| (row, column));
        }
        let sort_nanos = sort_start.elapsed().as_nanos();

        // The heap compares `(row, column, chunk_index)`, so equal entries are
        // accumulated in deterministic producer/chunk order. Each run cursor
        // advances once and each heap operation costs O(log K).
        let mut heap = BinaryHeap::<Reverse<(usize, u64, usize)>>::new();
        let mut cursors = vec![0usize; chunks.len()];
        let mut heap_operations = 0usize;
        for (chunk_index, chunk) in chunks.iter().enumerate() {
            if let Some(&(row, column, _)) = chunk.first() {
                heap.push(Reverse((row, column, chunk_index)));
                heap_operations = heap_operations.checked_add(1).ok_or(PauliError::Overflow {
                    context: "counting CSR heap operations",
                })?;
            }
        }
        let final_array_allocation_start = Instant::now();
        let mut columns = Vec::with_capacity(entries);
        let mut values = Vec::with_capacity(entries);
        let final_array_allocation_nanos = final_array_allocation_start.elapsed().as_nanos();
        let mut current: Option<(usize, u64, Complex64)> = None;
        let merge_start = Instant::now();
        let mut merged_entries = 0usize;

        while let Some(Reverse((row, column, chunk_index))) = heap.pop() {
            heap_operations = heap_operations.checked_add(1).ok_or(PauliError::Overflow {
                context: "counting CSR heap operations",
            })?;
            let cursor = cursors[chunk_index];
            let (_, _, value) = chunks[chunk_index][cursor];
            cursors[chunk_index] = cursor + 1;
            if let Some(&(next_row, next_column, _)) = chunks[chunk_index].get(cursor + 1) {
                heap.push(Reverse((next_row, next_column, chunk_index)));
                heap_operations = heap_operations.checked_add(1).ok_or(PauliError::Overflow {
                    context: "counting CSR heap operations",
                })?;
            }
            merged_entries = merged_entries.checked_add(1).ok_or(PauliError::Overflow {
                context: "counting CSR merged entries",
            })?;
            match current.as_mut() {
                Some((current_row, current_column, current_value))
                    if *current_row == row && *current_column == column =>
                {
                    *current_value += value;
                }
                Some((current_row, current_column, current_value)) => {
                    Self::publish_entry(
                        &mut indptr,
                        &mut columns,
                        &mut values,
                        *current_row,
                        *current_column,
                        *current_value,
                    )?;
                    *current_row = row;
                    *current_column = column;
                    *current_value = value;
                }
                None => current = Some((row, column, value)),
            }
        }
        let merge_nanos = merge_start.elapsed().as_nanos();
        if let Some((row, column, value)) = current {
            Self::publish_entry(&mut indptr, &mut columns, &mut values, row, column, value)?;
        }
        for row in 1..pointer_count {
            indptr[row] = indptr[row]
                .checked_add(indptr[row - 1])
                .ok_or(PauliError::Overflow {
                    context: "building CSR row pointers",
                })?;
        }
        let final_count = columns.len();
        let chunk_metadata_bytes = (chunks.len() as u128)
            .checked_mul(std::mem::size_of::<Vec<(usize, u64, Complex64)>>() as u128)
            .and_then(|bytes| {
                bytes.checked_add((cursors.len() * std::mem::size_of::<usize>()) as u128)
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    (chunks.len() * std::mem::size_of::<Reverse<(usize, u64, usize)>>()) as u128,
                )
            })
            .ok_or(PauliError::Overflow {
                context: "estimating CSR chunk metadata",
            })?;
        Ok((
            (indptr, columns, values, final_count),
            CsrBuildStats {
                chunk_count: chunks.len(),
                sorted_entries: entries,
                merged_entries,
                heap_operations,
                final_array_allocation_entries: final_count,
                chunk_metadata_bytes,
                sort_nanos,
                merge_nanos,
                // The source-major sink allocates the final arrays before the
                // merge and writes them directly; the measured interval is
                // therefore allocation time, with no hidden second copy.
                final_array_allocation_nanos,
            },
        ))
    }

    fn publish_entry(
        indptr: &mut [u64],
        columns: &mut Vec<u64>,
        values: &mut Vec<Complex64>,
        row: usize,
        column: u64,
        value: Complex64,
    ) -> Result<(), PauliError> {
        if value.re == 0.0 && value.im == 0.0 {
            return Ok(());
        }
        indptr[row + 1] = indptr[row + 1].checked_add(1).ok_or(PauliError::Overflow {
            context: "counting CSR row entries",
        })?;
        columns.push(column);
        values.push(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CsrSink;
    use crate::Complex64;

    #[test]
    fn row_ordered_sink_publishes_empty_rows_without_a_coo_buffer() {
        let mut sink = CsrSink::new_row_ordered(4).unwrap();
        sink.finish_row(0, &[(3, Complex64::new(2.0, 0.0))])
            .unwrap();
        sink.finish_row(1, &[]).unwrap();
        sink.finish_row(
            2,
            &[(1, Complex64::new(1.0, 0.0)), (3, Complex64::new(3.0, 0.0))],
        )
        .unwrap();
        sink.finish_row(3, &[(0, Complex64::new(0.0, 0.0))])
            .unwrap();
        let (indptr, columns, values, entries) = sink.into_csr().unwrap();
        assert_eq!(indptr, vec![0, 1, 1, 3, 3]);
        assert_eq!(columns, vec![3, 1, 3]);
        assert_eq!(values.len(), entries);
        assert_eq!(values[2], Complex64::new(3.0, 0.0));
    }

    #[test]
    fn chunked_sink_merges_multiple_runs_with_cross_run_cancellation() {
        let build = || {
            let mut sink = CsrSink::with_chunk_capacity(4, 2);
            sink.push(2, 4, Complex64::new(1.0, 0.0)).unwrap();
            sink.push(0, 3, Complex64::new(2.0, 0.0)).unwrap();
            sink.push(0, 1, Complex64::new(1.0, 0.0)).unwrap();
            sink.push(0, 1, Complex64::new(-1.0, 0.0)).unwrap();
            sink.push(2, 2, Complex64::new(3.0, 0.0)).unwrap();
            sink.push(2, 4, Complex64::new(-1.0, 0.0)).unwrap();
            sink.into_csr().unwrap()
        };
        let first = build();
        let second = build();
        assert_eq!(first, second);
        let (indptr, columns, values, _) = first;
        assert_eq!(indptr, vec![0, 1, 1, 2, 2]);
        assert_eq!(columns, vec![3, 2]);
        assert_eq!(
            values,
            vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)]
        );
    }

    #[test]
    fn chunked_sink_reports_bounded_merge_metadata() {
        let mut sink = CsrSink::with_chunk_capacity(2, 2);
        for column in 0..6 {
            sink.push(0, column, Complex64::new(1.0, 0.0)).unwrap();
        }
        let ((indptr, columns, values, entries), stats) = sink.into_csr_with_stats().unwrap();
        assert_eq!(indptr, vec![0, 6, 6]);
        assert_eq!(columns.len(), entries);
        assert_eq!(values.len(), entries);
        assert_eq!(stats.chunk_count, 3);
        assert_eq!(stats.sorted_entries, 6);
        assert_eq!(stats.merged_entries, 6);
        assert_eq!(stats.final_array_allocation_entries, 6);
        assert!(stats.chunk_metadata_bytes > 0);
    }
}
