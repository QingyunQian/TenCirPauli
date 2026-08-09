//! Generic finite charge-sector transition compilation.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::charge_sector::{build_charge_sector_plan, ChargeSectorPlan};
use crate::csr::{CsrSink, CSR_CHUNK_CAPACITY};
use crate::fermion_mask::FermionMaskDescriptor;
use crate::{Complex64, PauliError};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

// Release profiling keeps the serial gather for medium graphs (where Rayon
// scheduling dominates) and activates disjoint-row parallelism only for the
// large all-to-all graphs that showed a material end-to-end benefit.
const CSR_PARALLEL_TRANSITION_THRESHOLD: usize = 1 << 19;

/// One canonical structured term in a finite charge-sector layout.
#[derive(Clone, Debug)]
pub struct ChargeTransitionTerm {
    pub fermion_creation: Vec<u32>,
    pub fermion_annihilation: Vec<u32>,
    pub boson_blocks: Vec<(u32, u32, u32)>,
    pub qubit_codes: Vec<u8>,
    pub mapped_present: bool,
    pub mapped_codes: Vec<u8>,
    pub qudit_present: bool,
    pub qudit_triples: Vec<(u32, u32, u32)>,
    pub coefficient: Complex64,
}

type ChargeTransitionShape = (
    Vec<u32>,
    Vec<u32>,
    Vec<(u32, u32, u32)>,
    Vec<u8>,
    bool,
    Vec<u8>,
    bool,
    Vec<(u32, u32, u32)>,
);

/// Group identical coefficient-free charge schedules once at construction.
///
/// Native apply and eager construction consume the grouped descriptors, so
/// duplicate symbolic terms cannot force repeated semantic lowering in every
/// source column. Exact-zero cancellation is removed only after aggregation.
pub fn group_charge_transition_terms(
    terms: &[ChargeTransitionTerm],
) -> Result<Vec<ChargeTransitionTerm>, PauliError> {
    let mut grouped = BTreeMap::<ChargeTransitionShape, Complex64>::new();
    for term in terms {
        let shape = (
            term.fermion_creation.clone(),
            term.fermion_annihilation.clone(),
            term.boson_blocks.clone(),
            term.qubit_codes.clone(),
            term.mapped_present,
            term.mapped_codes.clone(),
            term.qudit_present,
            term.qudit_triples.clone(),
        );
        grouped
            .entry(shape)
            .and_modify(|coefficient| *coefficient += term.coefficient)
            .or_insert(term.coefficient);
    }
    grouped
        .into_iter()
        .filter_map(|(shape, coefficient)| {
            if coefficient.re == 0.0 && coefficient.im == 0.0 {
                return None;
            }
            let (
                fermion_creation,
                fermion_annihilation,
                boson_blocks,
                qubit_codes,
                mapped_present,
                mapped_codes,
                qudit_present,
                qudit_triples,
            ) = shape;
            Some(Ok(ChargeTransitionTerm {
                fermion_creation,
                fermion_annihilation,
                boson_blocks,
                qubit_codes,
                mapped_present,
                mapped_codes,
                qudit_present,
                qudit_triples,
                coefficient,
            }))
        })
        .collect()
}

/// Estimate the logical native storage for a batch of term descriptors.
///
/// Vec headers are counted together with their logical payloads; allocator
/// capacity and Python object overhead remain outside this best-effort policy.
pub fn estimate_charge_transition_terms_bytes(
    terms: &[ChargeTransitionTerm],
) -> Result<u128, PauliError> {
    terms.iter().try_fold(0_u128, |total, term| {
        let payload = (term.fermion_creation.len() as u128)
            .checked_mul(std::mem::size_of::<u32>() as u128)
            .and_then(|value| {
                value.checked_add(
                    (term.fermion_annihilation.len() as u128)
                        .checked_mul(std::mem::size_of::<u32>() as u128)?,
                )
            })
            .and_then(|value| {
                value.checked_add(
                    (term.boson_blocks.len() as u128).checked_mul(std::mem::size_of::<(
                        u32,
                        u32,
                        u32,
                    )>()
                        as u128)?,
                )
            })
            .and_then(|value| value.checked_add(term.qubit_codes.len() as u128))
            .and_then(|value| value.checked_add(term.mapped_codes.len() as u128))
            .and_then(|value| {
                value.checked_add(
                    (term.qudit_triples.len() as u128).checked_mul(std::mem::size_of::<(
                        u32,
                        u32,
                        u32,
                    )>()
                        as u128)?,
                )
            })
            .ok_or(PauliError::Overflow {
                context: "estimating charge transition descriptors",
            })?;
        total
            .checked_add(std::mem::size_of::<ChargeTransitionTerm>() as u128)
            .and_then(|value| value.checked_add(payload))
            .ok_or(PauliError::Overflow {
                context: "estimating charge transition descriptors",
            })
    })
}

/// Deterministic destination-major CSR graph produced by native eager builds.
pub struct ChargeCsrResult {
    pub dimension: usize,
    pub indptr: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
    pub estimated_bytes: u128,
    pub estimated_construction_peak_bytes: u128,
    pub schedule_count: usize,
    pub generation_nanos: u128,
    pub chunk_count: usize,
    pub sorted_entries: usize,
    pub merged_entries: usize,
    pub heap_operations: usize,
    pub final_array_allocation_entries: usize,
    pub chunk_metadata_bytes: u128,
    pub sort_nanos: u128,
    pub merge_nanos: u128,
    pub final_array_allocation_nanos: u128,
}

#[inline]
fn apply_charge_csr_row(columns: &[usize], values: &[Complex64], state: &[Complex64]) -> Complex64 {
    let mut result = Complex64::default();
    for (&column, coefficient) in columns.iter().zip(values) {
        let input = state[column];
        result.re += coefficient.re * input.re - coefficient.im * input.im;
        result.im += coefficient.re * input.im + coefficient.im * input.re;
    }
    result
}

/// Apply a validated destination-major CSR graph without allocating
/// state-sized worker buffers. Parallel rows have deterministic per-row
/// accumulation order because each worker owns one complete output row.
pub fn apply_charge_csr_into(
    indptr: &[usize],
    columns: &[usize],
    values: &[Complex64],
    state: &[Complex64],
    output: &mut [Complex64],
    parallel: bool,
) -> Result<(), PauliError> {
    if indptr.len()
        != output.len().checked_add(1).ok_or(PauliError::Overflow {
            context: "validating charge CSR output length",
        })?
        || columns.len() != values.len()
        || indptr.last().copied() != Some(columns.len())
        || indptr.windows(2).any(|window| window[0] > window[1])
        || columns.iter().any(|&column| column >= state.len())
    {
        return Err(PauliError::InvalidSector {
            context: "invalid charge CSR execution graph",
        });
    }
    let use_parallel = parallel
        && rayon::current_num_threads() > 1
        && values.len() >= CSR_PARALLEL_TRANSITION_THRESHOLD;
    if use_parallel {
        indptr
            .par_windows(2)
            .zip(output.par_iter_mut())
            .for_each(|(window, result)| {
                *result = apply_charge_csr_row(
                    &columns[window[0]..window[1]],
                    &values[window[0]..window[1]],
                    state,
                );
            });
    } else {
        for (window, result) in indptr.windows(2).zip(output.iter_mut()) {
            *result = apply_charge_csr_row(
                &columns[window[0]..window[1]],
                &values[window[0]..window[1]],
                state,
            );
        }
    }
    Ok(())
}

/// Owned charge-sector layout validated once for repeated native execution.
pub struct PreparedChargeTransitionPlanLayout {
    dimension: usize,
    local_dimensions: Vec<u64>,
    fermion_positions: Vec<usize>,
    boson_positions: Vec<usize>,
    qubit_positions: Vec<usize>,
    qudit_positions: Vec<usize>,
    qudit_dimension: u64,
    schedules: Vec<CompiledChargeSchedule>,
    full_strides: Option<Box<[usize]>>,
    estimated_bytes: u128,
    counters: Arc<ChargeApplyCounters>,
}

#[derive(Clone)]
struct CompiledChargeSchedule {
    coefficient: Complex64,
    zero: bool,
    fermion: Option<FermionMaskDescriptor>,
    boson_blocks: Box<[(usize, u32, u32)]>,
    qubit_ops: Box<[(usize, u8)]>,
    mapped_ops: Box<[(usize, u8)]>,
    qudit_ops: Box<[(usize, u32, u32)]>,
}

#[derive(Default)]
struct ChargeApplyCounters {
    scalar_insertions: AtomicUsize,
    wide_insertions: AtomicUsize,
    accumulator_capacity_growths: AtomicUsize,
    owned_destination_key_creations: AtomicUsize,
    fallback_invocations: AtomicUsize,
    ranker_invocations: AtomicUsize,
    construction_descriptor_validations: AtomicUsize,
    destination_key_allocations: AtomicUsize,
}

impl ChargeApplyCounters {
    fn snapshot(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        (
            self.scalar_insertions.load(Ordering::Relaxed),
            self.wide_insertions.load(Ordering::Relaxed),
            self.accumulator_capacity_growths.load(Ordering::Relaxed),
            self.owned_destination_key_creations.load(Ordering::Relaxed),
            self.fallback_invocations.load(Ordering::Relaxed),
            self.ranker_invocations.load(Ordering::Relaxed),
            self.construction_descriptor_validations
                .load(Ordering::Relaxed),
            self.destination_key_allocations.load(Ordering::Relaxed),
        )
    }
}

fn full_radix_strides(local_dimensions: &[u64]) -> Option<Box<[usize]>> {
    let mut strides = vec![1_usize; local_dimensions.len()];
    let mut stride = 1_usize;
    for position in (0..local_dimensions.len()).rev() {
        strides[position] = stride;
        let dimension = usize::try_from(local_dimensions[position]).ok()?;
        stride = stride.checked_mul(dimension)?;
    }
    Some(strides.into_boxed_slice())
}

fn encode_full_radix(values: &[u64], strides: &[usize]) -> Result<usize, PauliError> {
    if values.len() != strides.len() {
        return Err(invalid_sector());
    }
    values
        .iter()
        .zip(strides)
        .try_fold(0_usize, |index, (&value, &stride)| {
            let value = usize::try_from(value).map_err(|_| invalid_sector())?;
            index
                .checked_add(value.checked_mul(stride).ok_or(PauliError::Overflow {
                    context: "encoding full charge occupation",
                })?)
                .ok_or(PauliError::Overflow {
                    context: "encoding full charge occupation",
                })
        })
}

fn decode_full_radix(
    mut index: usize,
    local_dimensions: &[u64],
    strides: &[usize],
    values: &mut [u64],
) -> Result<(), PauliError> {
    if local_dimensions.len() != strides.len() || values.len() != strides.len() {
        return Err(invalid_sector());
    }
    for axis in 0..local_dimensions.len() {
        let dimension = usize::try_from(local_dimensions[axis]).map_err(|_| invalid_sector())?;
        if dimension == 0 {
            return Err(invalid_sector());
        }
        let digit = (index / strides[axis]) % dimension;
        values[axis] = u64::try_from(digit).map_err(|_| PauliError::Overflow {
            context: "decoding full charge occupation",
        })?;
        index %= strides[axis]
            .checked_mul(dimension)
            .ok_or(PauliError::Overflow {
                context: "decoding full charge occupation",
            })?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ScalarAggregateEntry {
    key: usize,
    value: Complex64,
}

/// A bounded destination aggregate whose storage is allocated once.
///
/// Hash-map `clear()` drops its nodes, so a later source would allocate them
/// again. The transition count is the upper bound on distinct destinations for
/// one source; a compact linear table therefore keeps the hot path allocation
/// free while preserving canonical insertion order.
struct ReusableScalarAccumulator {
    entries: Vec<ScalarAggregateEntry>,
}

impl ReusableScalarAccumulator {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn insert(&mut self, key: usize, value: Complex64) -> Result<bool, PauliError> {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) {
            entry.value += value;
            return Ok(false);
        }
        if self.entries.len() == self.entries.capacity() {
            return Err(PauliError::MemoryLimit {
                requested: (self.entries.len() + 1) as u128
                    * std::mem::size_of::<ScalarAggregateEntry>() as u128,
                limit: self.entries.capacity() as u128
                    * std::mem::size_of::<ScalarAggregateEntry>() as u128,
            });
        }
        self.entries.push(ScalarAggregateEntry { key, value });
        Ok(true)
    }
}

struct WideDestinationAccumulator {
    entries: Vec<ScalarAggregateEntry>,
    values: Vec<u64>,
    width: usize,
}

impl WideDestinationAccumulator {
    fn with_capacity(capacity: usize, width: usize) -> Result<Self, PauliError> {
        let value_capacity = capacity.checked_mul(width).ok_or(PauliError::Overflow {
            context: "sizing wide charge destination scratch",
        })?;
        Ok(Self {
            entries: Vec::with_capacity(capacity),
            values: Vec::with_capacity(value_capacity),
            width,
        })
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.values.clear();
    }

    fn insert(&mut self, destination: &[u64], value: Complex64) -> Result<(), PauliError> {
        if destination.len() != self.width {
            return Err(invalid_sector());
        }
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| self.values[entry.key..entry.key + self.width] == *destination)
        {
            entry.value += value;
            return Ok(());
        }
        if self.entries.len() == self.entries.capacity() {
            return Err(PauliError::MemoryLimit {
                requested: (self.entries.len() + 1) as u128
                    * std::mem::size_of::<ScalarAggregateEntry>() as u128,
                limit: self.entries.capacity() as u128
                    * std::mem::size_of::<ScalarAggregateEntry>() as u128,
            });
        }
        if self.values.capacity()
            < self
                .entries
                .len()
                .checked_add(1)
                .and_then(|entries| entries.checked_mul(self.width))
                .ok_or(PauliError::Overflow {
                    context: "sizing wide charge destination key",
                })?
        {
            return Err(PauliError::MemoryLimit {
                requested: (self.values.len() + self.width) as u128
                    * std::mem::size_of::<u64>() as u128,
                limit: self.values.capacity() as u128 * std::mem::size_of::<u64>() as u128,
            });
        }
        let offset = self.values.len();
        self.values.extend_from_slice(destination);
        self.entries
            .push(ScalarAggregateEntry { key: offset, value });
        Ok(())
    }

    fn drain<F>(&mut self, destination: &mut [u64], mut consume: F) -> Result<(), PauliError>
    where
        F: FnMut(&[u64], Complex64) -> Result<(), PauliError>,
    {
        let values = &self.values;
        for entry in self.entries.drain(..) {
            destination.copy_from_slice(&values[entry.key..entry.key + self.width]);
            consume(destination, entry.value)?;
        }
        self.values.clear();
        Ok(())
    }
}

struct ChargeDestinationAccumulator<'a> {
    scalar: Option<ReusableScalarAccumulator>,
    wide: Option<WideDestinationAccumulator>,
    strides: Option<&'a [usize]>,
    counters: Option<&'a ChargeApplyCounters>,
}

impl<'a> ChargeDestinationAccumulator<'a> {
    fn new(
        strides: Option<&'a [usize]>,
        capacity: usize,
        width: usize,
        counters: Option<&'a ChargeApplyCounters>,
    ) -> Result<Self, PauliError> {
        Ok(Self {
            scalar: strides.map(|_| ReusableScalarAccumulator::with_capacity(capacity)),
            wide: strides
                .is_none()
                .then(|| WideDestinationAccumulator::with_capacity(capacity, width))
                .transpose()?,
            strides,
            counters,
        })
    }

    fn clear(&mut self) {
        if let Some(scalar) = self.scalar.as_mut() {
            scalar.clear();
        }
        if let Some(wide) = self.wide.as_mut() {
            wide.clear();
        }
    }

    fn insert(&mut self, destination: &[u64], value: Complex64) -> Result<(), PauliError> {
        if let (Some(scalar), Some(strides)) = (self.scalar.as_mut(), self.strides) {
            let key = encode_full_radix(destination, strides)?;
            let is_new = !scalar.entries.iter().any(|entry| entry.key == key);
            if is_new && scalar.entries.len() == scalar.entries.capacity() {
                if let Some(counters) = self.counters {
                    counters
                        .accumulator_capacity_growths
                        .fetch_add(1, Ordering::Relaxed);
                    counters
                        .destination_key_allocations
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            if let Some(counters) = self.counters {
                counters.scalar_insertions.fetch_add(1, Ordering::Relaxed);
            }
            scalar.insert(key, value)?;
        } else if let Some(wide) = self.wide.as_mut() {
            let is_new = !wide
                .entries
                .iter()
                .any(|entry| wide.values[entry.key..entry.key + wide.width] == *destination);
            let required_values = wide
                .entries
                .len()
                .checked_add(1)
                .and_then(|entries| entries.checked_mul(wide.width))
                .unwrap_or(usize::MAX);
            if is_new
                && (wide.entries.len() == wide.entries.capacity()
                    || wide.values.capacity() < required_values)
            {
                if let Some(counters) = self.counters {
                    counters
                        .accumulator_capacity_growths
                        .fetch_add(1, Ordering::Relaxed);
                    counters
                        .owned_destination_key_creations
                        .fetch_add(1, Ordering::Relaxed);
                    counters
                        .destination_key_allocations
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            if let Some(counters) = self.counters {
                counters.wide_insertions.fetch_add(1, Ordering::Relaxed);
            }
            wide.insert(destination, value)?;
        } else {
            return Err(invalid_sector());
        }
        Ok(())
    }

    fn drain<F>(
        &mut self,
        destination: &mut [u64],
        local_dimensions: &[u64],
        mut consume: F,
    ) -> Result<(), PauliError>
    where
        F: FnMut(&[u64], Complex64) -> Result<(), PauliError>,
    {
        if let (Some(scalar), Some(strides)) = (self.scalar.as_mut(), self.strides) {
            for entry in scalar.entries.drain(..) {
                decode_full_radix(entry.key, local_dimensions, strides, destination)?;
                consume(destination, entry.value)?;
            }
        } else if let Some(wide) = self.wide.as_mut() {
            wide.drain(destination, consume)?;
        } else {
            return Err(invalid_sector());
        }
        Ok(())
    }
}

impl PreparedChargeTransitionPlanLayout {
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn debug_counters(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        self.counters.snapshot()
    }
}

fn invalid_sector() -> PauliError {
    PauliError::InvalidSector {
        context: "invalid charge-sector transition input",
    }
}

fn check_bytes(entries: usize, bytes_per_entry: usize, limit: u128) -> Result<(), PauliError> {
    let requested = (entries as u128)
        .checked_mul(bytes_per_entry as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector transition storage",
        })?;
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}

fn charge_csr_memory_estimate(
    dimension: usize,
    term_count: usize,
    axis_count: usize,
    buffered_entries: usize,
    chunk_capacity: usize,
) -> Result<(u128, u128), PauliError> {
    let pointer_count = dimension.checked_add(1).ok_or(PauliError::Overflow {
        context: "estimating charge CSR row pointers",
    })?;
    let pointer_bytes = (pointer_count as u128)
        .checked_mul(std::mem::size_of::<u64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR row pointers",
        })?;
    if chunk_capacity == 0 {
        return Err(PauliError::InvalidStructureLength {
            expected: 1,
            actual: 0,
        });
    }
    let chunk_count = buffered_entries
        .checked_add(chunk_capacity - 1)
        .and_then(|value| value.checked_div(chunk_capacity))
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR chunk count",
        })?;
    let chunk_headers = (chunk_count as u128)
        .checked_mul(std::mem::size_of::<Vec<(usize, u64, Complex64)>>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR chunk headers",
        })?;
    let cursor_bytes = (chunk_count as u128)
        .checked_mul(std::mem::size_of::<usize>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR chunk cursors",
        })?;
    let heap_bytes = (chunk_count as u128)
        .checked_mul(std::mem::size_of::<(usize, u64, usize)>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR merge heap",
        })?;
    let entry_bytes = (buffered_entries as u128)
        .checked_mul(std::mem::size_of::<(usize, u64, Complex64)>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR row buffers",
        })?;
    let final_entry_bytes = (buffered_entries as u128)
        .checked_mul((std::mem::size_of::<u64>() + std::mem::size_of::<Complex64>()) as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR retained graph",
        })?;
    let scratch_bytes = (axis_count as u128)
        .checked_mul(std::mem::size_of::<u64>() as u128)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add((term_count as u128).checked_mul(64)?))
        .and_then(|value| value.checked_add(32))
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR row scratch",
        })?;
    let retained_bytes =
        pointer_bytes
            .checked_add(final_entry_bytes)
            .ok_or(PauliError::Overflow {
                context: "estimating charge CSR retained graph",
            })?;
    // Count the row-local candidate buffers and the final CSR arrays while
    // the builder's bounded scratch is live. Explicit COO materialization is
    // derived from the retained CSR graph after construction.
    let construction_peak_bytes = chunk_headers
        .checked_add(cursor_bytes)
        .and_then(|value| value.checked_add(heap_bytes))
        .and_then(|value| value.checked_add(entry_bytes))
        .and_then(|value| value.checked_add((buffered_entries as u128).checked_mul(32)?))
        .and_then(|value| value.checked_add(pointer_bytes))
        .and_then(|value| value.checked_add((buffered_entries as u128).checked_mul(8)?))
        .and_then(|value| value.checked_add(scratch_bytes))
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR construction peak",
        })?;
    Ok((retained_bytes, construction_peak_bytes))
}

fn charge_csr_memory_preflight(
    dimension: usize,
    term_count: usize,
    axis_count: usize,
    max_bytes: u128,
) -> Result<u128, PauliError> {
    let raw_entries = dimension
        .checked_mul(term_count)
        .ok_or(PauliError::Overflow {
            context: "estimating charge CSR candidate transitions",
        })?;
    let (_, construction_peak_bytes) = charge_csr_memory_estimate(
        dimension,
        term_count,
        axis_count,
        raw_entries,
        CSR_CHUNK_CAPACITY,
    )?;
    if construction_peak_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: construction_peak_bytes,
            limit: max_bytes,
        });
    }
    Ok(construction_peak_bytes)
}

fn positions(values: &[u64], axis_count: usize, limit: u128) -> Result<Vec<usize>, PauliError> {
    check_bytes(values.len(), std::mem::size_of::<usize>(), limit)?;
    let mut result = Vec::with_capacity(values.len());
    for &value in values {
        let position = usize::try_from(value).map_err(|_| invalid_sector())?;
        if position >= axis_count {
            return Err(invalid_sector());
        }
        result.push(position);
    }
    if result.iter().collect::<FxHashSet<_>>().len() != result.len() {
        return Err(invalid_sector());
    }
    Ok(result)
}

fn validate_qudit_term(
    term: &ChargeTransitionTerm,
    index: usize,
    site_count: usize,
    dimension: u64,
) -> Result<(), PauliError> {
    if (!term.qudit_present && !term.qudit_triples.is_empty())
        || term
            .qudit_triples
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        || term.qudit_triples.iter().any(|&(site, a, b)| {
            usize::try_from(site).map_or(true, |site| site >= site_count)
                || u64::from(a) >= dimension
                || u64::from(b) >= dimension
        })
    {
        return Err(PauliError::NonCanonicalTerms { index });
    }
    Ok(())
}

fn compile_charge_pauli_ops(
    codes: &[u8],
    positions: &[usize],
) -> Result<Box<[(usize, u8)]>, PauliError> {
    if codes.len() != positions.len() {
        return Err(invalid_sector());
    }
    let mut result = Vec::with_capacity(codes.len());
    for (&code, &axis) in codes.iter().zip(positions) {
        if code > 3 {
            return Err(invalid_sector());
        }
        result.push((axis, code));
    }
    Ok(result.into_boxed_slice())
}

fn compile_charge_schedules(
    local_dimensions: &[u64],
    fermion_positions: &[usize],
    boson_positions: &[usize],
    qubit_positions: &[usize],
    qudit_positions: &[usize],
    qudit_dimension: u64,
    terms: &[ChargeTransitionTerm],
) -> Result<Vec<CompiledChargeSchedule>, PauliError> {
    let mode_axes = fermion_positions;
    terms
        .iter()
        .map(|term| {
            if !term.coefficient.re.is_finite() || !term.coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index: 0 });
            }
            let fermion =
                if term.fermion_creation.is_empty() && term.fermion_annihilation.is_empty() {
                    None
                } else {
                    FermionMaskDescriptor::from_creation_annihilation(
                        mode_axes,
                        &term.fermion_creation,
                        &term.fermion_annihilation,
                    )
                };
            let zero = (!term.fermion_creation.is_empty() || !term.fermion_annihilation.is_empty())
                && fermion.is_none();
            let mut boson_blocks = Vec::with_capacity(term.boson_blocks.len());
            for &(mode, creation, annihilation) in &term.boson_blocks {
                let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
                let axis = *boson_positions.get(mode).ok_or_else(invalid_sector)?;
                boson_blocks.push((axis, creation, annihilation));
            }
            let qubit_ops = compile_charge_pauli_ops(&term.qubit_codes, qubit_positions)?;
            let mapped_ops = if term.mapped_present {
                compile_charge_pauli_ops(&term.mapped_codes, fermion_positions)?
            } else {
                Vec::new().into_boxed_slice()
            };
            let mut qudit_ops = Vec::with_capacity(term.qudit_triples.len());
            for &(site, shift, phase_power) in &term.qudit_triples {
                let site = usize::try_from(site).map_err(|_| invalid_sector())?;
                if u64::from(shift) >= qudit_dimension || u64::from(phase_power) >= qudit_dimension
                {
                    return Err(invalid_sector());
                }
                let axis = *qudit_positions.get(site).ok_or_else(invalid_sector)?;
                qudit_ops.push((axis, shift, phase_power));
            }
            if qudit_dimension == 0 && !qudit_ops.is_empty() {
                return Err(invalid_sector());
            }
            if local_dimensions.contains(&0) {
                return Err(invalid_sector());
            }
            Ok(CompiledChargeSchedule {
                coefficient: term.coefficient,
                zero,
                fermion,
                boson_blocks: boson_blocks.into_boxed_slice(),
                qubit_ops,
                mapped_ops,
                qudit_ops: qudit_ops.into_boxed_slice(),
            })
        })
        .collect()
}

fn apply_compiled_charge_schedule(
    schedule: &CompiledChargeSchedule,
    source: &[u64],
    destination: &mut [u64],
    local_dimensions: &[u64],
    qudit_dimension: u64,
    fermion_words: &mut [u64],
) -> Result<Option<Complex64>, PauliError> {
    if schedule.zero {
        return Ok(None);
    }
    let mut value = schedule.coefficient;
    if let Some(fermion) = schedule.fermion.as_ref() {
        let Some(amplitude) =
            fermion.apply_to_occupations(source, destination, fermion_words, value)?
        else {
            return Ok(None);
        };
        value = amplitude;
    } else {
        destination.copy_from_slice(source);
    }
    for &(axis, creation, annihilation) in schedule.boson_blocks.iter().rev() {
        let mut occupation = destination[axis];
        for _ in 0..annihilation {
            if occupation == 0 {
                return Ok(None);
            }
            value *= (occupation as f64).sqrt();
            occupation -= 1;
        }
        for _ in 0..creation {
            let Some(next) = occupation.checked_add(1) else {
                return Ok(None);
            };
            if next >= local_dimensions[axis] {
                return Ok(None);
            }
            value *= (next as f64).sqrt();
            occupation = next;
        }
        destination[axis] = occupation;
    }
    for &(axis, code) in schedule.qubit_ops.iter().chain(schedule.mapped_ops.iter()) {
        let bit = destination[axis];
        if bit > 1 {
            return Err(invalid_sector());
        }
        match code {
            0 => {}
            1 => destination[axis] = 1 - bit,
            2 => {
                destination[axis] = 1 - bit;
                value *= if bit == 0 {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(0.0, -1.0)
                };
            }
            3 if bit == 1 => value = -value,
            3 => {}
            _ => unreachable!("charge Pauli code was validated during compilation"),
        }
    }
    for &(axis, shift, phase_power) in &schedule.qudit_ops {
        let input = destination[axis];
        if input >= qudit_dimension {
            return Err(invalid_sector());
        }
        let exponent = (u128::from(phase_power) * u128::from(input)) % u128::from(qudit_dimension);
        let angle = 2.0 * std::f64::consts::PI * exponent as f64 / qudit_dimension as f64;
        value *= Complex64::from_polar(1.0, angle);
        destination[axis] = (input + u64::from(shift)) % qudit_dimension;
    }
    if value.re == 0.0 && value.im == 0.0 {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

/// Convert and validate immutable charge layout and term metadata once.
#[allow(clippy::too_many_arguments)]
pub fn prepare_charge_transition_plan_layout(
    plan: &ChargeSectorPlan,
    dimension: usize,
    local_dimensions: Vec<u64>,
    fermion_positions: &[u64],
    boson_positions: &[u64],
    qubit_positions: &[u64],
    qudit_positions: &[u64],
    qudit_dimension: u64,
    terms: &[ChargeTransitionTerm],
    max_bytes: u128,
) -> Result<PreparedChargeTransitionPlanLayout, PauliError> {
    let axis_count = local_dimensions.len();
    if dimension != plan.dimension()
        || plan.local_dimensions().len() != axis_count
        || plan
            .local_dimensions()
            .iter()
            .zip(&local_dimensions)
            .any(|(&left, &right)| u64::try_from(left).ok() != Some(right))
        || local_dimensions.contains(&0)
    {
        return Err(invalid_sector());
    }
    let retained_positions = fermion_positions
        .len()
        .checked_add(boson_positions.len())
        .and_then(|value| value.checked_add(qubit_positions.len()))
        .and_then(|value| value.checked_add(qudit_positions.len()))
        .ok_or(PauliError::Overflow {
            context: "estimating prepared charge layout storage",
        })?;
    let estimated_bytes = (local_dimensions.len() as u128)
        .checked_mul(std::mem::size_of::<u64>() as u128)
        .and_then(|local_bytes| {
            (retained_positions as u128)
                .checked_mul(std::mem::size_of::<usize>() as u128)
                .and_then(|position_bytes| local_bytes.checked_add(position_bytes))
        })
        .ok_or(PauliError::Overflow {
            context: "estimating prepared charge layout storage",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    let fermion_positions = positions(fermion_positions, axis_count, max_bytes)?;
    let boson_positions = positions(boson_positions, axis_count, max_bytes)?;
    let qubit_positions = positions(qubit_positions, axis_count, max_bytes)?;
    let qudit_positions = positions(qudit_positions, axis_count, max_bytes)?;
    for (index, term) in terms.iter().enumerate() {
        if term.qubit_codes.len() != qubit_positions.len()
            || term.mapped_codes.len() != fermion_positions.len()
            || (term.mapped_present
                && (!term.fermion_creation.is_empty() || !term.fermion_annihilation.is_empty()))
        {
            return Err(invalid_sector());
        }
        validate_qudit_term(term, index, qudit_positions.len(), qudit_dimension)?;
    }
    let schedules = compile_charge_schedules(
        &local_dimensions,
        &fermion_positions,
        &boson_positions,
        &qubit_positions,
        &qudit_positions,
        qudit_dimension,
        terms,
    )?;
    let full_strides = full_radix_strides(&local_dimensions);
    let schedule_bytes = schedules.iter().try_fold(0_u128, |total, schedule| {
        let mask_bytes = schedule
            .fermion
            .as_ref()
            .map(|mask| {
                let words = mask.required_occupied.len();
                (words * 4 * std::mem::size_of::<u64>()
                    + mask.mode_axes.len() * std::mem::size_of::<usize>()) as u128
            })
            .unwrap_or(0);
        total
            .checked_add(std::mem::size_of::<CompiledChargeSchedule>() as u128)
            .and_then(|value| {
                value.checked_add(
                    (schedule.boson_blocks.len() * std::mem::size_of::<(usize, u32, u32)>()
                        + schedule.qubit_ops.len() * std::mem::size_of::<(usize, u8)>()
                        + schedule.mapped_ops.len() * std::mem::size_of::<(usize, u8)>()
                        + schedule.qudit_ops.len() * std::mem::size_of::<(usize, u32, u32)>())
                        as u128,
                )
            })
            .and_then(|value| value.checked_add(mask_bytes))
            .ok_or(PauliError::Overflow {
                context: "estimating compiled charge schedules",
            })
    })?;
    let estimated_bytes = estimated_bytes
        .checked_add(schedule_bytes)
        .and_then(|value| {
            value.checked_add(
                full_strides
                    .as_ref()
                    .map_or(0, |strides| strides.len() * std::mem::size_of::<usize>())
                    as u128,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating compiled charge schedules",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    let counters = Arc::new(ChargeApplyCounters::default());
    counters
        .construction_descriptor_validations
        .fetch_add(schedules.len(), Ordering::Relaxed);
    Ok(PreparedChargeTransitionPlanLayout {
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        qudit_dimension,
        schedules,
        full_strides,
        estimated_bytes,
        counters,
    })
}

/// Return ``n choose k`` without overflowing the intermediate product when
/// the final result still fits in ``u128``.
fn binomial(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1_u128;
    for index in 1..=k {
        let mut numerator = u128::from((n - k + index) as u64);
        let mut denominator = u128::from(index as u64);
        let common = gcd(numerator, denominator);
        numerator /= common;
        denominator /= common;
        let common = gcd(result, denominator);
        result /= common;
        denominator /= common;
        if denominator != 1 {
            return None;
        }
        result = result.checked_mul(numerator)?;
    }
    Some(result)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn unrank_combination(length: usize, particles: usize, mut rank: u128) -> Option<u128> {
    let mut mask = 0_u128;
    let mut remaining = particles;
    for position in 0..length {
        if remaining == 0 {
            break;
        }
        let zero_count = binomial(length - position - 1, remaining)?;
        if rank < zero_count {
            continue;
        }
        rank -= zero_count;
        mask |= 1_u128 << position;
        remaining -= 1;
    }
    if remaining == 0 && rank == 0 {
        Some(mask)
    } else {
        None
    }
}

fn rank_combination(length: usize, particles: usize, mask: u128) -> Option<u128> {
    if mask.count_ones() as usize != particles || (length < 128 && mask >> length != 0) {
        return None;
    }
    let mut rank = 0_u128;
    let mut remaining = particles;
    for position in 0..length {
        if mask & (1_u128 << position) != 0 {
            rank = rank.checked_add(binomial(length - position - 1, remaining)?)?;
            remaining -= 1;
        }
    }
    (remaining == 0).then_some(rank)
}

type FactorActions = Box<[Vec<(usize, Complex64)>]>;

fn build_qubit_factor_actions(
    terms: &[ChargeTransitionTerm],
    factor_length: usize,
    active_qubit_indices: &[usize],
    particles: usize,
    factor_dimension: usize,
) -> Result<Option<FactorActions>, PauliError> {
    if terms.iter().any(|term| {
        !term.fermion_creation.is_empty()
            || !term.fermion_annihilation.is_empty()
            || !term.boson_blocks.is_empty()
            || term.mapped_present
            || term.qudit_present
    }) {
        return Ok(None);
    }
    let mut actions = Vec::with_capacity(factor_dimension);
    for source_rank in 0..factor_dimension {
        let source_mask = unrank_combination(factor_length, particles, source_rank as u128).ok_or(
            PauliError::InvalidSector {
                context: "fast qubit factor unrank failed",
            },
        )?;
        let mut grouped = BTreeMap::<u128, Complex64>::new();
        for term in terms {
            if term.qubit_codes.len() <= active_qubit_indices.iter().copied().max().unwrap_or(0) {
                return Ok(None);
            }
            let mut destination = source_mask;
            let mut value = term.coefficient;
            if term
                .qubit_codes
                .iter()
                .enumerate()
                .any(|(index, &code)| !active_qubit_indices.contains(&index) && code != 0)
            {
                return Ok(None);
            }
            for (offset, &global_index) in active_qubit_indices.iter().enumerate() {
                let code = term.qubit_codes[global_index];
                let digit = (source_mask >> offset) & 1;
                match code {
                    0 => {}
                    1 => destination ^= 1_u128 << offset,
                    2 => {
                        destination ^= 1_u128 << offset;
                        value *= if digit == 0 {
                            Complex64::new(0.0, 1.0)
                        } else {
                            Complex64::new(0.0, -1.0)
                        };
                    }
                    3 => {
                        if digit != 0 {
                            value = -value;
                        }
                    }
                    _ => {
                        return Err(PauliError::InvalidCode {
                            code,
                            index: offset,
                        });
                    }
                }
            }
            grouped
                .entry(destination)
                .and_modify(|existing| *existing += value)
                .or_insert(value);
        }
        let mut row_actions = Vec::with_capacity(grouped.len());
        for (destination, value) in grouped {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let destination_rank = rank_combination(factor_length, particles, destination).ok_or(
                PauliError::InvalidSector {
                    context: "operator leaks outside the selected qubit sector",
                },
            )?;
            let destination_rank =
                usize::try_from(destination_rank).map_err(|_| PauliError::Overflow {
                    context: "converting fast qubit factor rank",
                })?;
            row_actions.push((destination_rank, value));
        }
        actions.push(row_actions);
    }
    Ok(Some(actions.into_boxed_slice()))
}

struct FastFermionSectorIndex {
    sites: usize,
    particles: usize,
    combination_count: usize,
    combination_masks: Option<Vec<u128>>,
    rank_table: Option<Vec<u32>>,
}

const FAST_COMBINATION_CACHE_LIMIT: u128 = 64 * 1024 * 1024;
const FAST_RANK_CACHE_LIMIT: u128 = 4 * 1024 * 1024;

impl FastFermionSectorIndex {
    #[allow(clippy::too_many_arguments)]
    fn new(
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

    fn unrank(&self, rank: usize) -> Option<u128> {
        if rank >= self.combination_count {
            return None;
        }
        self.combination_masks
            .as_ref()
            .map(|masks| masks[rank])
            .or_else(|| unrank_combination(self.sites, self.particles, rank as u128))
    }

    fn rank(&self, mask: u128) -> Option<usize> {
        if let Some(table) = &self.rank_table {
            let index = usize::try_from(mask).ok()?;
            let rank = *table.get(index)?;
            (rank != u32::MAX).then_some(rank as usize)
        } else {
            usize::try_from(rank_combination(self.sites, self.particles, mask)?).ok()
        }
    }

    fn estimated_bytes(&self) -> u128 {
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
    index: FastFermionSectorIndex,
    up_modes: Box<[u8]>,
    down_modes: Box<[u8]>,
    factor_to_public: Box<[usize]>,
    public_to_factor: Box<[usize]>,
    terms: Vec<FastProductTerm>,
    boson_dimensions: Box<[usize]>,
    boson_strides: Box<[usize]>,
    spectator_dimension: usize,
    estimated_bytes: u128,
}

#[derive(Clone, Debug)]
struct FastProductTerm {
    descriptor: Option<FermionMaskDescriptor>,
    inverse: Option<FermionMaskDescriptor>,
    coefficient: Complex64,
    boson_blocks: Box<[(u32, u32, u32)]>,
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

/// Public-order layout shared by compact independent finite factors.
///
/// The factor axes must form one contiguous segment in the public mixed-radix
/// layout.  This is the representation already used by ordinary
/// ``OperatorSpace`` layouts; rejecting a non-contiguous factor keeps the
/// fast path honest until a bounded interleaved ranker is needed.
struct FastProductLayout {
    local_dimensions: Box<[u64]>,
    fermion_positions: Box<[usize]>,
    boson_positions: Box<[usize]>,
    qubit_positions: Box<[usize]>,
    qudit_positions: Box<[usize]>,
    factor_positions: Box<[usize]>,
    factor_dimension: usize,
    prefix_dimension: usize,
    suffix_dimension: usize,
}

impl FastProductLayout {
    #[allow(clippy::too_many_arguments)]
    fn new(
        local_dimensions: &[u64],
        fermion_positions: &[u64],
        boson_positions: &[u64],
        qubit_positions: &[u64],
        qudit_positions: &[u64],
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
        let axis_count = local_dimensions.len();
        let local_dimensions = local_dimensions.to_vec().into_boxed_slice();
        if expected_dimension == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            local_dimensions,
            fermion_positions: positions(fermion_positions, axis_count, u128::MAX)?
                .into_boxed_slice(),
            boson_positions: positions(boson_positions, axis_count, u128::MAX)?.into_boxed_slice(),
            qubit_positions: positions(qubit_positions, axis_count, u128::MAX)?.into_boxed_slice(),
            qudit_positions: positions(qudit_positions, axis_count, u128::MAX)?.into_boxed_slice(),
            factor_positions: factor_positions.to_vec().into_boxed_slice(),
            factor_dimension,
            prefix_dimension,
            suffix_dimension,
        }))
    }

    fn expected_dimension(&self) -> Result<usize, PauliError> {
        self.prefix_dimension
            .checked_mul(self.factor_dimension)
            .and_then(|value| value.checked_mul(self.suffix_dimension))
            .ok_or(PauliError::Overflow {
                context: "sizing fast product layout dimension",
            })
    }

    fn descriptor_bytes(&self) -> u128 {
        ((self.local_dimensions.len()
            + self.fermion_positions.len()
            + self.boson_positions.len()
            + self.qubit_positions.len()
            + self.qudit_positions.len()
            + self.factor_positions.len())
            * std::mem::size_of::<usize>()) as u128
    }
}

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
#[allow(clippy::too_many_arguments)]
pub fn build_fast_qubit_mvp_plan(
    plan: &ChargeSectorPlan,
    local_dimensions: &[u64],
    fermion_positions: &[u64],
    boson_positions: &[u64],
    qubit_positions: &[u64],
    qudit_positions: &[u64],
    _qudit_dimension: u64,
    terms: &[ChargeTransitionTerm],
    particles: Option<usize>,
    factor_positions_override: Option<&[u64]>,
    max_bytes: u128,
) -> Result<Option<FastQubitMvpPlan>, PauliError> {
    let Some(particles) = particles else {
        return Ok(None);
    };
    let qubit_positions = positions(qubit_positions, local_dimensions.len(), max_bytes)?;
    let factor_positions = factor_positions_override
        .map(|values| positions(values, local_dimensions.len(), max_bytes))
        .transpose()?
        .unwrap_or_else(|| qubit_positions.clone());
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
        qubit_positions
            .iter()
            .map(|&value| value as u64)
            .collect::<Vec<_>>()
            .as_slice(),
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

fn build_additive_factor_actions(
    factor: &ChargeSectorPlan,
    terms: &[ChargeTransitionTerm],
    factor_dimensions: &[usize],
    qudit_dimension: usize,
    boson_mode: bool,
    domain_to_factor: Option<&[usize]>,
) -> Result<Option<FactorActions>, PauliError> {
    if terms.iter().any(|term| {
        !term.fermion_creation.is_empty()
            || !term.fermion_annihilation.is_empty()
            || term.mapped_present
            || term.qubit_codes.iter().any(|&code| code != 0)
            || if boson_mode {
                term.qudit_present || !term.qudit_triples.is_empty()
            } else {
                !term.boson_blocks.is_empty()
            }
    }) {
        return Ok(None);
    }
    let factor_dimension = factor.dimension();
    let mut actions = Vec::with_capacity(factor_dimension);
    let mut source = vec![0_u64; factor_dimensions.len()];
    let mut destination = vec![0_u64; factor_dimensions.len()];
    let mut remaining = vec![0_i128; 1];
    let mut candidate = vec![0_i128; 1];
    for source_rank in 0..factor_dimension {
        factor.unrank_into_with_scratch(
            source_rank as u64,
            &mut source,
            &mut remaining,
            &mut candidate,
        )?;
        let mut grouped = BTreeMap::<Vec<u64>, Complex64>::new();
        for term in terms {
            destination.copy_from_slice(&source);
            let mut value = term.coefficient;
            let mut valid = true;
            if boson_mode {
                for &(axis, creation, annihilation) in &term.boson_blocks {
                    let axis = usize::try_from(axis).map_err(|_| PauliError::Overflow {
                        context: "converting fast boson factor axis",
                    })?;
                    let axis = domain_to_factor
                        .map(|mapping| mapping.get(axis).copied().unwrap_or(usize::MAX))
                        .unwrap_or(axis);
                    if axis == usize::MAX {
                        return Ok(None);
                    }
                    let creation = usize::try_from(creation).map_err(|_| PauliError::Overflow {
                        context: "converting fast boson creation power",
                    })?;
                    let annihilation =
                        usize::try_from(annihilation).map_err(|_| PauliError::Overflow {
                            context: "converting fast boson annihilation power",
                        })?;
                    if axis >= destination.len() || source[axis] < annihilation as u64 {
                        valid = false;
                        break;
                    }
                    let level =
                        usize::try_from(source[axis]).map_err(|_| PauliError::Overflow {
                            context: "converting fast boson factor level",
                        })?;
                    let Some(new_level) = level
                        .checked_sub(annihilation)
                        .and_then(|value| value.checked_add(creation))
                    else {
                        valid = false;
                        break;
                    };
                    if new_level >= factor_dimensions[axis] {
                        valid = false;
                        break;
                    }
                    let mut ladder = 1.0;
                    for offset in 0..annihilation {
                        ladder *= ((level - offset) as f64).sqrt();
                    }
                    let remaining_level = level - annihilation;
                    for offset in 0..creation {
                        ladder *= ((remaining_level + offset + 1) as f64).sqrt();
                    }
                    destination[axis] = new_level as u64;
                    value *= Complex64::new(ladder, 0.0);
                }
            } else {
                for &(site, shift, phase_power) in &term.qudit_triples {
                    let site = usize::try_from(site).map_err(|_| PauliError::Overflow {
                        context: "converting fast qudit factor axis",
                    })?;
                    let site = domain_to_factor
                        .map(|mapping| mapping.get(site).copied().unwrap_or(usize::MAX))
                        .unwrap_or(site);
                    if site == usize::MAX {
                        return Ok(None);
                    }
                    if site >= destination.len() || qudit_dimension == 0 {
                        return Err(invalid_sector());
                    }
                    let digit =
                        usize::try_from(source[site]).map_err(|_| PauliError::Overflow {
                            context: "converting fast qudit factor level",
                        })?;
                    let shift = usize::try_from(shift).map_err(|_| PauliError::Overflow {
                        context: "converting fast qudit shift",
                    })?;
                    destination[site] = ((digit + shift) % qudit_dimension) as u64;
                    let exponent =
                        (u128::from(phase_power) * digit as u128) % qudit_dimension as u128;
                    let angle =
                        2.0 * std::f64::consts::PI * exponent as f64 / qudit_dimension as f64;
                    value *= Complex64::from_polar(1.0, angle);
                }
            }
            if valid {
                grouped
                    .entry(destination.clone())
                    .and_modify(|existing| *existing += value)
                    .or_insert(value);
            }
        }
        let mut row_actions = Vec::with_capacity(grouped.len());
        for (destination, value) in grouped {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let rank = factor.rank_into(&destination, &mut remaining, &mut candidate)?;
            row_actions.push((
                usize::try_from(rank).map_err(|_| PauliError::Overflow {
                    context: "converting fast additive factor rank",
                })?,
                value,
            ));
        }
        actions.push(row_actions);
    }
    Ok(Some(actions.into_boxed_slice()))
}

/// Compact additive spin-z qudit factor with arbitrary Cartesian spectators.
pub struct FastAdditiveMvpPlan {
    layout: FastProductLayout,
    factor_actions: FactorActions,
    estimated_bytes: u128,
}

impl FastAdditiveMvpPlan {
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
                    context: "indexing fast additive product block",
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
                        context: "indexing fast additive product destination",
                    })?;
                output[row] += value * *state_value;
            }
        }
        Ok(())
    }
}

/// Build an exact canonical-spin-z qudit factor with Cartesian spectators.
#[allow(clippy::too_many_arguments)]
pub fn build_fast_qudit_mvp_plan(
    plan: &ChargeSectorPlan,
    local_dimensions: &[u64],
    fermion_positions: &[u64],
    boson_positions: &[u64],
    qubit_positions: &[u64],
    qudit_positions: &[u64],
    qudit_dimension: u64,
    terms: &[ChargeTransitionTerm],
    weights: Option<&[i128]>,
    target: Option<i128>,
    factor_positions_override: Option<&[u64]>,
    max_bytes: u128,
) -> Result<Option<FastAdditiveMvpPlan>, PauliError> {
    let (Some(weights), Some(target)) = (weights, target) else {
        return Ok(None);
    };
    let all_qudit_positions = positions(qudit_positions, local_dimensions.len(), max_bytes)?;
    let qudit_positions = factor_positions_override
        .map(|values| positions(values, local_dimensions.len(), max_bytes))
        .transpose()?
        .unwrap_or_else(|| all_qudit_positions.clone());
    if qudit_positions.is_empty()
        || weights.len() != qudit_positions.len()
        || weights.contains(&0)
        || qudit_positions
            .iter()
            .enumerate()
            .any(|(offset, &position)| {
                position != qudit_positions[0] + offset
                    || local_dimensions[position] != qudit_dimension
                    || qudit_dimension < 3
            })
    {
        return Ok(None);
    }
    let factor_dimensions = vec![
        usize::try_from(qudit_dimension).map_err(|_| {
            PauliError::Overflow {
                context: "converting fast qudit dimension",
            }
        })?;
        qudit_positions.len()
    ];
    let contributions = factor_dimensions
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            (0..dimension)
                .map(|level| {
                    let level = i128::try_from(level).map_err(|_| PauliError::Overflow {
                        context: "converting fast qudit level",
                    })?;
                    let spectrum = if dimension % 2 == 1 {
                        i128::try_from((dimension - 1) / 2).map_err(|_| PauliError::Overflow {
                            context: "computing fast qudit spectrum",
                        })? - level
                    } else {
                        i128::try_from(dimension - 1).map_err(|_| PauliError::Overflow {
                            context: "computing fast qudit spectrum",
                        })? - level.checked_mul(2).ok_or(PauliError::Overflow {
                            context: "computing fast qudit spectrum",
                        })?
                    };
                    Ok(vec![weights[axis].checked_mul(spectrum).ok_or(
                        PauliError::Overflow {
                            context: "computing fast qudit contribution",
                        },
                    )?])
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let factor =
        build_charge_sector_plan(factor_dimensions, contributions, vec![target], max_bytes)?;
    let factor_dimension = factor.dimension();
    let Some(layout) = FastProductLayout::new(
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        all_qudit_positions
            .iter()
            .map(|&value| value as u64)
            .collect::<Vec<_>>()
            .as_slice(),
        &qudit_positions,
        factor_dimension,
    )?
    else {
        return Ok(None);
    };
    if plan.dimension() != layout.expected_dimension()? {
        return Ok(None);
    }
    let factor_actions = build_additive_factor_actions(
        &factor,
        terms,
        factor.local_dimensions(),
        usize::try_from(qudit_dimension).map_err(|_| PauliError::Overflow {
            context: "converting fast qudit dimension",
        })?,
        false,
        Some(
            &all_qudit_positions
                .iter()
                .map(|position| {
                    qudit_positions
                        .iter()
                        .position(|candidate| candidate == position)
                        .unwrap_or(usize::MAX)
                })
                .collect::<Vec<_>>(),
        ),
    )?;
    if factor_actions.is_none() {
        return Ok(None);
    }
    let action_bytes = factor_actions
        .as_ref()
        .map(|actions| actions.iter().map(Vec::len).sum::<usize>())
        .unwrap_or(0)
        .checked_mul(std::mem::size_of::<(usize, Complex64)>())
        .ok_or(PauliError::Overflow {
            context: "estimating fast qudit factor actions",
        })?;
    let term_bytes = estimate_charge_transition_terms_bytes(terms)?;
    let estimated_bytes = layout
        .descriptor_bytes()
        .checked_add(factor.estimated_bytes())
        .and_then(|value| value.checked_add(term_bytes))
        .and_then(|value| value.checked_add(action_bytes as u128))
        .ok_or(PauliError::Overflow {
            context: "estimating fast qudit plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(Some(FastAdditiveMvpPlan {
        layout,
        factor_actions: factor_actions.expect("validated additive factor actions"),
        estimated_bytes,
    }))
}

/// Build a fixed-number boson factor with arbitrary Cartesian spectators.
#[allow(clippy::too_many_arguments)]
pub fn build_fast_boson_mvp_plan(
    plan: &ChargeSectorPlan,
    local_dimensions: &[u64],
    fermion_positions: &[u64],
    boson_positions: &[u64],
    qubit_positions: &[u64],
    qudit_positions: &[u64],
    _qudit_dimension: u64,
    terms: &[ChargeTransitionTerm],
    weights: Option<&[i128]>,
    target: Option<i128>,
    factor_positions_override: Option<&[u64]>,
    max_bytes: u128,
) -> Result<Option<FastAdditiveMvpPlan>, PauliError> {
    let (Some(weights), Some(target)) = (weights, target) else {
        return Ok(None);
    };
    let all_boson_positions = positions(boson_positions, local_dimensions.len(), max_bytes)?;
    let boson_positions = factor_positions_override
        .map(|values| positions(values, local_dimensions.len(), max_bytes))
        .transpose()?
        .unwrap_or_else(|| all_boson_positions.clone());
    if boson_positions.is_empty()
        || weights.len() != boson_positions.len()
        || weights.contains(&0)
        || boson_positions
            .iter()
            .enumerate()
            .any(|(offset, &position)| position != boson_positions[0] + offset)
    {
        return Ok(None);
    }
    let factor_dimensions = boson_positions
        .iter()
        .map(|&position| usize::try_from(local_dimensions[position]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PauliError::Overflow {
            context: "converting fast boson dimensions",
        })?;
    if factor_dimensions.contains(&0) {
        return Ok(None);
    }
    let contributions = factor_dimensions
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            (0..dimension)
                .map(|level| {
                    let level = i128::try_from(level).map_err(|_| PauliError::Overflow {
                        context: "converting fast boson level",
                    })?;
                    Ok(vec![weights[axis].checked_mul(level).ok_or(
                        PauliError::Overflow {
                            context: "computing fast boson contribution",
                        },
                    )?])
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let factor =
        build_charge_sector_plan(factor_dimensions, contributions, vec![target], max_bytes)?;
    let factor_dimension = factor.dimension();
    let factor_positions = boson_positions.clone();
    let Some(layout) = FastProductLayout::new(
        local_dimensions,
        fermion_positions,
        all_boson_positions
            .iter()
            .map(|&value| value as u64)
            .collect::<Vec<_>>()
            .as_slice(),
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
    let factor_actions = build_additive_factor_actions(
        &factor,
        terms,
        factor.local_dimensions(),
        0,
        true,
        Some(
            &all_boson_positions
                .iter()
                .map(|position| {
                    boson_positions
                        .iter()
                        .position(|candidate| candidate == position)
                        .unwrap_or(usize::MAX)
                })
                .collect::<Vec<_>>(),
        ),
    )?;
    if factor_actions.is_none() {
        return Ok(None);
    }
    let action_bytes = factor_actions
        .as_ref()
        .map(|actions| actions.iter().map(Vec::len).sum::<usize>())
        .unwrap_or(0)
        .checked_mul(std::mem::size_of::<(usize, Complex64)>())
        .ok_or(PauliError::Overflow {
            context: "estimating fast boson factor actions",
        })?;
    let term_bytes = estimate_charge_transition_terms_bytes(terms)?;
    let estimated_bytes = layout
        .descriptor_bytes()
        .checked_add(factor.estimated_bytes())
        .and_then(|value| value.checked_add(term_bytes))
        .and_then(|value| value.checked_add(action_bytes as u128))
        .ok_or(PauliError::Overflow {
            context: "estimating fast boson plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(Some(FastAdditiveMvpPlan {
        layout,
        factor_actions: factor_actions.expect("validated additive factor actions"),
        estimated_bytes,
    }))
}

/// Native compact charge execution selected once during plan construction.
pub enum FastChargeMvpPlan {
    Fermion(FastFermionMvpPlan),
    Qubit(FastQubitMvpPlan),
    Qudit(FastAdditiveMvpPlan),
    Boson(FastAdditiveMvpPlan),
    Product(FastProductMvpPlan),
}

impl FastChargeMvpPlan {
    pub fn native_variant(&self) -> &'static str {
        match self {
            Self::Fermion(_) => "fermion",
            Self::Qubit(_) => "qubit",
            Self::Qudit(_) => "qudit",
            Self::Boson(_) => "boson",
            Self::Product(_) => "product",
        }
    }

    pub fn debug_counters(&self) -> Option<Vec<usize>> {
        match self {
            Self::Product(plan) => plan.debug_counters().map(|counters| {
                let (
                    factor_evaluations,
                    spectator_segments,
                    destination_allocations,
                    fallbacks,
                    rankers,
                    groups,
                    insertions,
                ) = counters;
                vec![
                    factor_evaluations,
                    spectator_segments,
                    destination_allocations,
                    fallbacks,
                    rankers,
                    groups,
                    insertions,
                ]
            }),
            _ => None,
        }
    }

    pub fn estimated_bytes(&self) -> u128 {
        match self {
            Self::Fermion(plan) => plan.estimated_bytes(),
            Self::Qubit(plan) => plan.estimated_bytes(),
            Self::Qudit(plan) => plan.estimated_bytes(),
            Self::Boson(plan) => plan.estimated_bytes(),
            Self::Product(plan) => plan.estimated_bytes(),
        }
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        match self {
            Self::Fermion(plan) => plan.apply_into(state, output, max_bytes),
            Self::Qubit(plan) => plan.apply_into(state, output, max_bytes),
            Self::Qudit(plan) => plan.apply_into(state, output, max_bytes),
            Self::Boson(plan) => plan.apply_into(state, output, max_bytes),
            Self::Product(plan) => plan.apply_into(state, output, max_bytes),
        }
    }
}

#[derive(Clone, Debug)]
enum ProductLocalOp {
    Fermion {
        axis: usize,
        create: bool,
    },
    Boson {
        axis: usize,
        creation: u32,
        annihilation: u32,
    },
    Pauli {
        axis: usize,
        code: u8,
    },
    Qudit {
        axis: usize,
        shift: u32,
        phase_power: u32,
    },
}

#[derive(Clone, Debug)]
struct ProductRankTransition {
    destination_rank: usize,
    value: Complex64,
}

#[derive(Clone, Debug)]
struct ProductDigitTransition {
    destination: u64,
    value: Complex64,
}

#[derive(Clone, Debug)]
struct ProductFactorAction {
    transitions: Box<[Option<ProductRankTransition>]>,
    inverse_transitions: Box<[Option<ProductInverseTransition>]>,
}

#[derive(Clone, Debug)]
struct ProductInverseTransition {
    source_rank: usize,
}

#[derive(Clone, Debug)]
struct ProductSpectatorAction {
    transitions: Box<[Option<ProductDigitTransition>]>,
    inverse_transitions: Box<[Option<ProductInverseTransition>]>,
}

#[derive(Clone, Debug)]
struct ProductLazyActionDescriptor {
    coefficient: Complex64,
    factor_actions: Box<[Option<ProductFactorAction>]>,
    spectator_actions: Box<[Option<ProductSpectatorAction>]>,
    fermion_mask: Option<FermionMaskDescriptor>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ProductScheduleKey(Vec<(bool, bool, usize)>);

struct ProductScheduleGroup {
    members: Box<[usize]>,
    representative: usize,
    active_factors: ProductComponentShape,
    inactive_factors: ProductComponentShape,
    active_spectators: ProductComponentShape,
    inactive_spectators: ProductComponentShape,
    trailing_identity_spectators: Box<[usize]>,
    trailing_identity_dimension: usize,
}

struct ProductComponentShape {
    components: Box<[ProductComponentAxis]>,
    dimension: usize,
}

struct ProductComponentAxis {
    index: usize,
    stride: usize,
}

impl ProductComponentShape {
    fn new(indices: Vec<usize>, dimensions: &[usize]) -> Result<Self, PauliError> {
        let mut components = Vec::with_capacity(indices.len());
        let mut dimension = 1usize;
        for &index in indices.iter().rev() {
            let local_dimension = *dimensions.get(index).ok_or_else(invalid_sector)?;
            components.push(ProductComponentAxis {
                index,
                stride: dimension,
            });
            dimension = dimension
                .checked_mul(local_dimension)
                .ok_or(PauliError::Overflow {
                    context: "sizing Product component shape",
                })?;
        }
        components.reverse();
        Ok(Self {
            components: components.into_boxed_slice(),
            dimension,
        })
    }

    fn indices(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.components.iter().map(|component| component.index)
    }

    fn len(&self) -> usize {
        self.components.len()
    }

    fn write_component(&self, component: usize, dimensions: &[usize], values: &mut [usize]) {
        debug_assert!(component < self.dimension);
        for component_axis in &self.components {
            values[component_axis.index] =
                (component / component_axis.stride) % dimensions[component_axis.index];
        }
    }
}

struct ProductFactorPublicIndex {
    axis_count: usize,
    prefix_offsets: Box<[usize]>,
    completion_counts: Box<[usize]>,
}

impl ProductFactorPublicIndex {
    fn prefix_offset(&self, rank: usize, axis: usize) -> usize {
        self.prefix_offsets[rank * self.axis_count + axis]
    }

    fn completion_count(&self, rank: usize, prefix_len: usize) -> usize {
        self.completion_counts[rank * (self.axis_count + 1) + prefix_len]
    }
}

#[derive(Clone, Copy)]
enum ProductAxisRole {
    Factor { factor: usize, local_axis: usize },
    Spectator { spectator: usize },
}

struct ProductPublicIndexer {
    axis_roles: Box<[ProductAxisRole]>,
    spectator_suffix_dimensions: Box<[usize]>,
    factors: Box<[ProductFactorPublicIndex]>,
    flat_component_strides: Option<ProductFlatComponentStrides>,
}

/// Direct mixed-radix ranks for a public layout formed from contiguous,
/// independent factor components and Cartesian spectator axes.
///
/// The general prefix-completion ranker remains necessary for interleaved
/// factor axes.  For a contiguous component layout, however, each factor's
/// own canonical rank is already its public mixed-radix digit.  Reusing it
/// avoids recomputing both global ranks for every Product segment.
struct ProductFlatComponentStrides {
    factor_strides: Box<[usize]>,
    spectator_strides: Box<[usize]>,
    dimension: usize,
}

impl ProductPublicIndexer {
    fn public_rank(
        &self,
        factor_ranks: &[usize],
        factor_values: &[ProductFactorKeys],
        spectator_values: &[usize],
        prefix_lengths: &mut [usize],
    ) -> Result<usize, PauliError> {
        if factor_ranks.len() != self.factors.len()
            || factor_values.len() != self.factors.len()
            || prefix_lengths.len() != self.factors.len()
        {
            return Err(invalid_sector());
        }
        if let Some(strides) = self.flat_component_strides.as_ref() {
            let mut rank = 0usize;
            for (factor, &factor_rank) in factor_ranks.iter().enumerate() {
                debug_assert!(factor_rank < self.factors[factor].completion_count(factor_rank, 0));
                rank += factor_rank * strides.factor_strides[factor];
            }
            for (spectator, &value) in spectator_values.iter().enumerate() {
                rank += value * strides.spectator_strides[spectator];
            }
            debug_assert!(rank < strides.dimension);
            return Ok(rank);
        }
        prefix_lengths.fill(0);
        let mut rank = 0usize;
        for (axis, role) in self.axis_roles.iter().enumerate() {
            let mut completion_product = self.spectator_suffix_dimensions[axis];
            match *role {
                ProductAxisRole::Factor { factor, local_axis } => {
                    for (other, factor_index) in self.factors.iter().enumerate() {
                        if other == factor {
                            continue;
                        }
                        completion_product = completion_product
                            .checked_mul(
                                factor_index
                                    .completion_count(factor_ranks[other], prefix_lengths[other]),
                            )
                            .ok_or(PauliError::Overflow {
                                context: "computing Product public rank",
                            })?;
                    }
                    let factor_index = &self.factors[factor];
                    let offset = factor_index.prefix_offset(factor_ranks[factor], local_axis);
                    rank = rank
                        .checked_add(offset.checked_mul(completion_product).ok_or(
                            PauliError::Overflow {
                                context: "computing Product public rank",
                            },
                        )?)
                        .ok_or(PauliError::Overflow {
                            context: "computing Product public rank",
                        })?;
                    debug_assert_eq!(prefix_lengths[factor], local_axis);
                    prefix_lengths[factor] = local_axis + 1;
                }
                ProductAxisRole::Spectator { spectator } => {
                    for (factor, factor_index) in self.factors.iter().enumerate() {
                        completion_product = completion_product
                            .checked_mul(
                                factor_index
                                    .completion_count(factor_ranks[factor], prefix_lengths[factor]),
                            )
                            .ok_or(PauliError::Overflow {
                                context: "computing Product public rank",
                            })?;
                    }
                    let value = spectator_values[spectator];
                    rank = rank
                        .checked_add(value.checked_mul(completion_product).ok_or(
                            PauliError::Overflow {
                                context: "computing Product public rank",
                            },
                        )?)
                        .ok_or(PauliError::Overflow {
                            context: "computing Product public rank",
                        })?;
                }
            }
        }
        Ok(rank)
    }
}

fn contiguous_product_component_strides(
    axis_roles: &[ProductAxisRole],
    factor_dimensions: &[usize],
    spectator_dimensions: &[usize],
    dimension: usize,
) -> Result<Option<ProductFlatComponentStrides>, PauliError> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Component {
        Factor(usize),
        Spectator(usize),
    }

    let mut components = Vec::<Component>::new();
    let mut factor_axis_counts = vec![0usize; factor_dimensions.len()];
    let mut factor_closed = vec![false; factor_dimensions.len()];
    let mut spectator_seen = vec![false; spectator_dimensions.len()];
    let mut previous = None;
    for role in axis_roles {
        let component = match *role {
            ProductAxisRole::Factor { factor, local_axis } => {
                if factor >= factor_dimensions.len()
                    || local_axis != factor_axis_counts[factor]
                    || factor_closed[factor]
                {
                    return Ok(None);
                }
                factor_axis_counts[factor] += 1;
                Component::Factor(factor)
            }
            ProductAxisRole::Spectator { spectator } => {
                if spectator >= spectator_dimensions.len() || spectator_seen[spectator] {
                    return Ok(None);
                }
                spectator_seen[spectator] = true;
                Component::Spectator(spectator)
            }
        };
        if previous != Some(component) {
            if let Some(Component::Factor(factor)) = previous {
                factor_closed[factor] = true;
            }
            components.push(component);
            previous = Some(component);
        }
    }
    if factor_axis_counts.contains(&0) || spectator_seen.iter().any(|&seen| !seen) {
        return Ok(None);
    }
    let mut factor_strides = vec![0usize; factor_dimensions.len()];
    let mut spectator_strides = vec![0usize; spectator_dimensions.len()];
    let mut stride = 1usize;
    for component in components.iter().rev() {
        match *component {
            Component::Factor(factor) => {
                factor_strides[factor] = stride;
                stride =
                    stride
                        .checked_mul(factor_dimensions[factor])
                        .ok_or(PauliError::Overflow {
                            context: "sizing contiguous Product public index",
                        })?;
            }
            Component::Spectator(spectator) => {
                spectator_strides[spectator] = stride;
                stride = stride.checked_mul(spectator_dimensions[spectator]).ok_or(
                    PauliError::Overflow {
                        context: "sizing contiguous Product public index",
                    },
                )?;
            }
        }
    }
    if stride != dimension {
        return Ok(None);
    }
    Ok(Some(ProductFlatComponentStrides {
        factor_strides: factor_strides.into_boxed_slice(),
        spectator_strides: spectator_strides.into_boxed_slice(),
        dimension,
    }))
}

#[derive(Default)]
struct ProductApplyCounters {
    factor_action_evaluations: AtomicUsize,
    spectator_segment_applications: AtomicUsize,
    destination_key_allocations: AtomicUsize,
    fallback_invocations: AtomicUsize,
    ranker_invocations: AtomicUsize,
    group_evaluations: AtomicUsize,
    aggregate_insertions: AtomicUsize,
}

#[derive(Default)]
struct ProductApplyCounterDeltas {
    factor_action_evaluations: usize,
    spectator_segment_applications: usize,
    aggregate_insertions: usize,
    group_evaluations: usize,
}

impl ProductApplyCounterDeltas {
    fn merge(&mut self, other: Self) {
        self.factor_action_evaluations += other.factor_action_evaluations;
        self.spectator_segment_applications += other.spectator_segment_applications;
        self.aggregate_insertions += other.aggregate_insertions;
        self.group_evaluations += other.group_evaluations;
    }
}

impl ProductApplyCounters {
    fn snapshot(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        (
            self.factor_action_evaluations.load(Ordering::Relaxed),
            self.spectator_segment_applications.load(Ordering::Relaxed),
            self.destination_key_allocations.load(Ordering::Relaxed),
            self.fallback_invocations.load(Ordering::Relaxed),
            self.ranker_invocations.load(Ordering::Relaxed),
            self.group_evaluations.load(Ordering::Relaxed),
            self.aggregate_insertions.load(Ordering::Relaxed),
        )
    }

    fn add(&self, deltas: &ProductApplyCounterDeltas) {
        self.factor_action_evaluations
            .fetch_add(deltas.factor_action_evaluations, Ordering::Relaxed);
        self.spectator_segment_applications
            .fetch_add(deltas.spectator_segment_applications, Ordering::Relaxed);
        self.aggregate_insertions
            .fetch_add(deltas.aggregate_insertions, Ordering::Relaxed);
        self.group_evaluations
            .fetch_add(deltas.group_evaluations, Ordering::Relaxed);
    }
}

fn product_schedule_key(
    action: &ProductLazyActionDescriptor,
    factor_dimensions: &[usize],
    spectator_dimensions: &[usize],
) -> ProductScheduleKey {
    let mut key = Vec::new();
    for (factor_index, maybe_action) in action.factor_actions.iter().enumerate() {
        if let Some(factor_action) = maybe_action {
            for transition in &factor_action.transitions {
                key.push(transition.as_ref().map_or((true, false, 0), |transition| {
                    (true, true, transition.destination_rank)
                }));
            }
        } else {
            for source in 0..factor_dimensions[factor_index] {
                key.push((false, true, source));
            }
        }
    }
    for (spectator_index, maybe_action) in action.spectator_actions.iter().enumerate() {
        if let Some(spectator_action) = maybe_action {
            for transition in &spectator_action.transitions {
                key.push(transition.as_ref().map_or((true, false, 0), |transition| {
                    (
                        true,
                        true,
                        usize::try_from(transition.destination).unwrap_or(usize::MAX),
                    )
                }));
            }
        } else {
            for source in 0..spectator_dimensions[spectator_index] {
                key.push((false, true, source));
            }
        }
    }
    ProductScheduleKey(key)
}

type ProductActionTarget = (Option<(usize, usize)>, Option<usize>);
type ProductFactorKeys = Box<[Box<[u64]>]>;

/// Compiled independent-factor Product execution.
///
/// Construction lowers every canonical term into factor-rank and Cartesian
/// spectator action tables. Lazy plans retain only those bounded descriptors,
/// factor rank metadata, and the spectator layout; they never retain a full
/// transition graph or the original semantic terms. Apply therefore performs
/// table lookups and one bounded aggregate per source, with leakage checked
/// only after exact destination cancellation.
pub struct FastProductMvpPlan {
    factor_dimensions: Box<[usize]>,
    factor_key_values: Box<[ProductFactorKeys]>,
    spectator_dimensions: Box<[usize]>,
    actions: Box<[ProductLazyActionDescriptor]>,
    groups: Box<[ProductScheduleGroup]>,
    public_indexer: ProductPublicIndexer,
    counters: Option<Arc<ProductApplyCounters>>,
    fermion_word_count: usize,
    dimension: usize,
    estimated_bytes: u128,
}

impl FastProductMvpPlan {
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn debug_counters(&self) -> Option<(usize, usize, usize, usize, usize, usize, usize)> {
        self.counters.as_ref().map(|counters| counters.snapshot())
    }

    fn pack_fermion_source_words(
        &self,
        descriptor: &FermionMaskDescriptor,
        factor_ranks: &[usize],
        spectator_values: &[usize],
        words: &mut [u64],
    ) -> Result<(), PauliError> {
        if words.len() != descriptor.required_occupied.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: descriptor.required_occupied.len(),
                actual: words.len(),
            });
        }
        words.fill(0);
        for (mode, &axis) in descriptor.mode_axes.iter().enumerate() {
            let occupation = match self.public_indexer.axis_roles.get(axis) {
                Some(ProductAxisRole::Factor { factor, local_axis }) => self
                    .factor_key_values
                    .get(*factor)
                    .and_then(|values| values.get(factor_ranks[*factor]))
                    .and_then(|values| values.get(*local_axis))
                    .copied()
                    .ok_or_else(invalid_sector)?,
                Some(ProductAxisRole::Spectator { spectator }) => spectator_values
                    .get(*spectator)
                    .copied()
                    .ok_or_else(invalid_sector)
                    .and_then(|value| {
                        u64::try_from(value).map_err(|_| PauliError::InvalidIndex {
                            context: "converting Product spectator value",
                        })
                    })?,
                None => {
                    return Err(PauliError::InvalidIndex {
                        context: "Product FermionMask public-axis map",
                    });
                }
            };
            if occupation > 1 {
                return Err(PauliError::InvalidIndex {
                    context: "Product FermionMask occupation",
                });
            }
            if occupation != 0 {
                words[mode / 64] |= 1_u64 << (mode % 64);
            }
        }
        Ok(())
    }

    fn has_contiguous_two_factor_kernel(&self) -> bool {
        let Some(strides) = self.public_indexer.flat_component_strides.as_ref() else {
            return false;
        };
        self.factor_dimensions.len() == 2
            && self.spectator_dimensions.is_empty()
            && strides.factor_strides.len() == 2
            && strides.factor_strides[0] == self.factor_dimensions[1]
            && strides.factor_strides[1] == 1
            && self
                .actions
                .iter()
                .all(|action| action.fermion_mask.is_none())
    }

    fn group_has_active_factor(group: &ProductScheduleGroup, factor: usize) -> bool {
        group
            .active_factors
            .components
            .iter()
            .any(|component| component.index == factor)
    }

    fn contiguous_two_factor_source_rank(
        &self,
        group: &ProductScheduleGroup,
        factor: usize,
        destination_rank: usize,
        active: bool,
    ) -> Option<usize> {
        if !active {
            return Some(destination_rank);
        }
        self.actions[group.representative].factor_actions[factor]
            .as_ref()?
            .inverse_transitions
            .get(destination_rank)?
            .as_ref()
            .map(|transition| transition.source_rank)
    }

    fn contiguous_two_factor_aggregate(
        &self,
        group: &ProductScheduleGroup,
        source: [usize; 2],
        destination: [usize; 2],
        active: [bool; 2],
    ) -> Option<Complex64> {
        let mut aggregate = Complex64::default();
        for &action_index in &group.members {
            let action = &self.actions[action_index];
            let mut value = action.coefficient;
            if active[0] {
                let transition = action.factor_actions[0]
                    .as_ref()?
                    .transitions
                    .get(source[0])?
                    .as_ref()?;
                if transition.destination_rank != destination[0] {
                    return None;
                }
                value *= transition.value;
            }
            if active[1] {
                let transition = action.factor_actions[1]
                    .as_ref()?
                    .transitions
                    .get(source[1])?
                    .as_ref()?;
                if transition.destination_rank != destination[1] {
                    return None;
                }
                value *= transition.value;
            }
            aggregate += value;
        }
        Some(aggregate)
    }

    fn apply_contiguous_two_factor_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        let zero_dimension = self.factor_dimensions[0];
        let one_dimension = self.factor_dimensions[1];
        let mut counter_deltas = ProductApplyCounterDeltas::default();
        for group in &self.groups {
            let active_zero = Self::group_has_active_factor(group, 0);
            let active_one = Self::group_has_active_factor(group, 1);
            let member_count = group.members.len();
            let group_deltas = match (active_zero, active_one) {
                (false, false) => {
                    let mut deltas = ProductApplyCounterDeltas {
                        group_evaluations: 1,
                        aggregate_insertions: member_count,
                        ..ProductApplyCounterDeltas::default()
                    };
                    let Some(aggregate) =
                        self.contiguous_two_factor_aggregate(group, [0, 0], [0, 0], [false, false])
                    else {
                        continue;
                    };
                    if aggregate.re != 0.0 || aggregate.im != 0.0 {
                        output.par_iter_mut().zip(state.par_iter()).for_each(
                            |(destination, source)| {
                                *destination += aggregate * *source;
                            },
                        );
                        deltas.spectator_segment_applications = self.dimension;
                    }
                    deltas
                }
                (true, false) => output
                    .par_chunks_mut(one_dimension)
                    .enumerate()
                    .map(|(destination_zero, destination_row)| {
                        let mut deltas = ProductApplyCounterDeltas::default();
                        let Some(source_zero) = self.contiguous_two_factor_source_rank(
                            group,
                            0,
                            destination_zero,
                            true,
                        ) else {
                            return deltas;
                        };
                        deltas.group_evaluations = 1;
                        deltas.factor_action_evaluations = 1;
                        deltas.aggregate_insertions = member_count;
                        let Some(aggregate) = self.contiguous_two_factor_aggregate(
                            group,
                            [source_zero, 0],
                            [destination_zero, 0],
                            [true, false],
                        ) else {
                            return deltas;
                        };
                        if aggregate.re != 0.0 || aggregate.im != 0.0 {
                            let source_start = source_zero * one_dimension;
                            for (destination, source) in destination_row
                                .iter_mut()
                                .zip(&state[source_start..source_start + one_dimension])
                            {
                                *destination += aggregate * *source;
                            }
                            deltas.spectator_segment_applications = one_dimension;
                        }
                        deltas
                    })
                    .reduce(ProductApplyCounterDeltas::default, |mut total, deltas| {
                        total.merge(deltas);
                        total
                    }),
                (false, true) => {
                    let mut values = Vec::with_capacity(one_dimension);
                    let mut deltas = ProductApplyCounterDeltas::default();
                    for destination_one in 0..one_dimension {
                        let Some(source_one) =
                            self.contiguous_two_factor_source_rank(group, 1, destination_one, true)
                        else {
                            values.push(None);
                            continue;
                        };
                        deltas.group_evaluations += 1;
                        deltas.factor_action_evaluations += 1;
                        deltas.aggregate_insertions += member_count;
                        let aggregate = self.contiguous_two_factor_aggregate(
                            group,
                            [0, source_one],
                            [0, destination_one],
                            [false, true],
                        );
                        values.push(aggregate.map(|value| (source_one, value)));
                    }
                    if values.iter().any(|value| {
                        value.is_some_and(|(_, value)| value.re != 0.0 || value.im != 0.0)
                    }) {
                        output.par_chunks_mut(one_dimension).enumerate().for_each(
                            |(factor_zero, destination_row)| {
                                let source_start = factor_zero * one_dimension;
                                for (destination_one, value) in values.iter().enumerate() {
                                    let Some((source_one, aggregate)) = value else {
                                        continue;
                                    };
                                    if aggregate.re != 0.0 || aggregate.im != 0.0 {
                                        destination_row[destination_one] +=
                                            *aggregate * state[source_start + *source_one];
                                    }
                                }
                            },
                        );
                        deltas.spectator_segment_applications = self.dimension;
                    }
                    deltas
                }
                (true, true) => output
                    .par_chunks_mut(one_dimension)
                    .enumerate()
                    .map(|(destination_zero, destination_row)| {
                        let mut deltas = ProductApplyCounterDeltas::default();
                        let Some(source_zero) = self.contiguous_two_factor_source_rank(
                            group,
                            0,
                            destination_zero,
                            true,
                        ) else {
                            return deltas;
                        };
                        for (destination_one, destination) in destination_row.iter_mut().enumerate()
                        {
                            let Some(source_one) = self.contiguous_two_factor_source_rank(
                                group,
                                1,
                                destination_one,
                                true,
                            ) else {
                                continue;
                            };
                            deltas.group_evaluations += 1;
                            deltas.factor_action_evaluations += 2;
                            deltas.aggregate_insertions += member_count;
                            let Some(aggregate) = self.contiguous_two_factor_aggregate(
                                group,
                                [source_zero, source_one],
                                [destination_zero, destination_one],
                                [true, true],
                            ) else {
                                continue;
                            };
                            if aggregate.re != 0.0 || aggregate.im != 0.0 {
                                *destination +=
                                    aggregate * state[source_zero * one_dimension + source_one];
                                deltas.spectator_segment_applications += 1;
                            }
                        }
                        deltas
                    })
                    .reduce(ProductApplyCounterDeltas::default, |mut total, deltas| {
                        total.merge(deltas);
                        total
                    }),
            };
            counter_deltas.merge(group_deltas);
        }
        if let Some(counters) = self.counters.as_ref() {
            counters.add(&counter_deltas);
        }
        debug_assert_eq!(zero_dimension * one_dimension, self.dimension);
        Ok(())
    }

    fn apply_destination_gather_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        let factor_count = self.factor_dimensions.len();
        let spectator_count = self.spectator_dimensions.len();
        let mut destination_factor_ranks = vec![0_usize; factor_count];
        let mut source_factor_ranks = vec![0_usize; factor_count];
        let mut destination_spectator_values = vec![0_usize; spectator_count];
        let mut source_spectator_values = vec![0_usize; spectator_count];
        let mut source_prefix_lengths = vec![0_usize; factor_count];
        let mut destination_prefix_lengths = vec![0_usize; factor_count];
        let mut fermion_words = vec![0_u64; self.fermion_word_count];

        for group in &self.groups {
            for active_factor_component in 0..group.active_factors.dimension {
                group.active_factors.write_component(
                    active_factor_component,
                    &self.factor_dimensions,
                    &mut destination_factor_ranks,
                );
                source_factor_ranks.copy_from_slice(&destination_factor_ranks);
                let representative = &self.actions[group.representative];
                let mut inverse_valid = true;
                for factor in group.active_factors.indices() {
                    let action = representative.factor_actions[factor]
                        .as_ref()
                        .expect("active Product factor is present");
                    let destination_rank = destination_factor_ranks[factor];
                    let Some(inverse) = action.inverse_transitions[destination_rank].as_ref()
                    else {
                        inverse_valid = false;
                        break;
                    };
                    source_factor_ranks[factor] = inverse.source_rank;
                }
                if !inverse_valid {
                    continue;
                }
                for active_spectator_component in 0..group.active_spectators.dimension {
                    group.active_spectators.write_component(
                        active_spectator_component,
                        &self.spectator_dimensions,
                        &mut destination_spectator_values,
                    );
                    source_spectator_values.copy_from_slice(&destination_spectator_values);
                    let mut inverse_valid = true;
                    for spectator in group.active_spectators.indices() {
                        let action = representative.spectator_actions[spectator]
                            .as_ref()
                            .expect("active Product spectator is present");
                        let destination_digit = destination_spectator_values[spectator];
                        let Some(inverse) = action.inverse_transitions[destination_digit].as_ref()
                        else {
                            inverse_valid = false;
                            break;
                        };
                        source_spectator_values[spectator] = inverse.source_rank;
                    }
                    if !inverse_valid {
                        continue;
                    }
                    if let Some(counters) = self.counters.as_ref() {
                        counters.group_evaluations.fetch_add(1, Ordering::Relaxed);
                        counters
                            .factor_action_evaluations
                            .fetch_add(group.active_factors.len(), Ordering::Relaxed);
                    }
                    let fermion_mask = group
                        .members
                        .iter()
                        .find_map(|&index| self.actions[index].fermion_mask.as_ref());
                    if let Some(mask) = fermion_mask {
                        self.pack_fermion_source_words(
                            mask,
                            &source_factor_ranks,
                            &source_spectator_values,
                            &mut fermion_words,
                        )?;
                    }
                    let mut aggregate = Complex64::default();
                    let mut members_valid = true;
                    for &action_index in &group.members {
                        let action = &self.actions[action_index];
                        let mut value = action.coefficient;
                        for factor in group.active_factors.indices() {
                            let factor_action = action.factor_actions[factor]
                                .as_ref()
                                .expect("grouped Product factor is present");
                            let Some(transition) =
                                factor_action.transitions[source_factor_ranks[factor]].as_ref()
                            else {
                                members_valid = false;
                                break;
                            };
                            if transition.destination_rank != destination_factor_ranks[factor] {
                                members_valid = false;
                                break;
                            }
                            value *= transition.value;
                        }
                        if !members_valid {
                            break;
                        }
                        for spectator in group.active_spectators.indices() {
                            let spectator_action = action.spectator_actions[spectator]
                                .as_ref()
                                .expect("grouped Product spectator is present");
                            let source_digit = source_spectator_values[spectator];
                            let Some(transition) =
                                spectator_action.transitions[source_digit].as_ref()
                            else {
                                members_valid = false;
                                break;
                            };
                            if usize::try_from(transition.destination).map_err(|_| {
                                PauliError::InvalidIndex {
                                    context: "converting Product spectator destination",
                                }
                            })? != destination_spectator_values[spectator]
                            {
                                members_valid = false;
                                break;
                            }
                            value *= transition.value;
                        }
                        if !members_valid {
                            break;
                        }
                        if let Some(mask) = action.fermion_mask.as_ref() {
                            let Some(masked) = mask.amplitude_from_words(&fermion_words, value)
                            else {
                                continue;
                            };
                            value = masked;
                        }
                        aggregate += value;
                        if let Some(counters) = self.counters.as_ref() {
                            counters
                                .aggregate_insertions
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    if !members_valid || (aggregate.re == 0.0 && aggregate.im == 0.0) {
                        continue;
                    }
                    for inactive_factor_component in 0..group.inactive_factors.dimension {
                        group.inactive_factors.write_component(
                            inactive_factor_component,
                            &self.factor_dimensions,
                            &mut destination_factor_ranks,
                        );
                        for factor in group.inactive_factors.indices() {
                            source_factor_ranks[factor] = destination_factor_ranks[factor];
                        }
                        for inactive_spectator_component in 0..group.inactive_spectators.dimension {
                            group.inactive_spectators.write_component(
                                inactive_spectator_component,
                                &self.spectator_dimensions,
                                &mut destination_spectator_values,
                            );
                            for spectator in group.inactive_spectators.indices() {
                                source_spectator_values[spectator] =
                                    destination_spectator_values[spectator];
                            }
                            for &spectator in &group.trailing_identity_spectators {
                                destination_spectator_values[spectator] = 0;
                                source_spectator_values[spectator] = 0;
                            }
                            let source_row = self.public_indexer.public_rank(
                                &source_factor_ranks,
                                &self.factor_key_values,
                                &source_spectator_values,
                                &mut source_prefix_lengths,
                            )?;
                            let destination_row = self.public_indexer.public_rank(
                                &destination_factor_ranks,
                                &self.factor_key_values,
                                &destination_spectator_values,
                                &mut destination_prefix_lengths,
                            )?;
                            let source_end = source_row
                                .checked_add(group.trailing_identity_dimension)
                                .ok_or(PauliError::Overflow {
                                    context: "applying Product identity segment",
                                })?;
                            let destination_end = destination_row
                                .checked_add(group.trailing_identity_dimension)
                                .ok_or(PauliError::Overflow {
                                    context: "applying Product identity segment",
                                })?;
                            if source_end > self.dimension || destination_end > self.dimension {
                                return Err(invalid_sector());
                            }
                            for offset in 0..group.trailing_identity_dimension {
                                output[destination_row + offset] +=
                                    aggregate * state[source_row + offset];
                            }
                            if let Some(counters) = self.counters.as_ref() {
                                counters.spectator_segment_applications.fetch_add(
                                    group.trailing_identity_dimension,
                                    Ordering::Relaxed,
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        _max_bytes: u128,
    ) -> Result<(), PauliError> {
        if state.len() != self.dimension || output.len() != self.dimension {
            return Err(invalid_sector());
        }
        output.fill(Complex64::default());
        if self.has_contiguous_two_factor_kernel() {
            self.apply_contiguous_two_factor_into(state, output)
        } else {
            self.apply_destination_gather_into(state, output)
        }
    }
}

fn apply_product_local_ops(
    values: &mut [u64],
    dimensions: &[u64],
    ops: &[ProductLocalOp],
    qudit_dimension: u64,
) -> Result<Option<Complex64>, PauliError> {
    let mut value = Complex64::new(1.0, 0.0);
    for op in ops {
        match *op {
            ProductLocalOp::Fermion { axis, create } => {
                let occupation = values[axis];
                if occupation > 1 || (create && occupation != 0) || (!create && occupation == 0) {
                    return Ok(None);
                }
                values[axis] = u64::from(create);
            }
            ProductLocalOp::Boson {
                axis,
                creation,
                annihilation,
            } => {
                let mut occupation = values[axis];
                for _ in 0..annihilation {
                    if occupation == 0 {
                        return Ok(None);
                    }
                    value *= Complex64::new((occupation as f64).sqrt(), 0.0);
                    occupation -= 1;
                }
                for _ in 0..creation {
                    let next = occupation.checked_add(1).ok_or(PauliError::Overflow {
                        context: "applying Product boson action",
                    })?;
                    if next >= dimensions[axis] {
                        return Ok(None);
                    }
                    value *= Complex64::new((next as f64).sqrt(), 0.0);
                    occupation = next;
                }
                values[axis] = occupation;
            }
            ProductLocalOp::Pauli { axis, code } => {
                let bit = values[axis];
                if bit > 1 {
                    return Err(invalid_sector());
                }
                match code {
                    0 => {}
                    1 => values[axis] = 1 - bit,
                    2 => {
                        values[axis] = 1 - bit;
                        value *= if bit == 0 {
                            Complex64::new(0.0, 1.0)
                        } else {
                            Complex64::new(0.0, -1.0)
                        };
                    }
                    3 => {
                        if bit == 1 {
                            value = -value;
                        }
                    }
                    _ => return Err(invalid_sector()),
                }
            }
            ProductLocalOp::Qudit {
                axis,
                shift,
                phase_power,
            } => {
                if qudit_dimension == 0 || values[axis] >= qudit_dimension {
                    return Err(invalid_sector());
                }
                let digit = values[axis];
                let exponent =
                    (u128::from(phase_power) * u128::from(digit)) % u128::from(qudit_dimension);
                let angle = 2.0 * std::f64::consts::PI * exponent as f64 / qudit_dimension as f64;
                value *= Complex64::from_polar(1.0, angle);
                values[axis] = (digit + u64::from(shift)) % qudit_dimension;
            }
        }
    }
    Ok(Some(value))
}

fn product_action_axis(
    position: usize,
    factor_axis_lookup: &[(usize, usize)],
    spectator_axis_lookup: &[usize],
) -> Result<ProductActionTarget, PauliError> {
    if let Some(&(factor, axis)) = factor_axis_lookup.get(position) {
        if factor != usize::MAX {
            return Ok((Some((factor, axis)), None));
        }
    }
    let spectator = spectator_axis_lookup
        .get(position)
        .copied()
        .filter(|&index| index != usize::MAX)
        .ok_or(PauliError::InvalidSector {
            context: "Product action axis is not in a factor or spectator",
        })?;
    Ok((None, Some(spectator)))
}

fn invert_product_rank_transitions(
    transitions: &[Option<ProductRankTransition>],
) -> Result<Box<[Option<ProductInverseTransition>]>, PauliError> {
    let mut inverse = vec![None; transitions.len()];
    for (source_rank, transition) in transitions.iter().enumerate() {
        let Some(transition) = transition else {
            continue;
        };
        let Some(slot) = inverse.get_mut(transition.destination_rank) else {
            return Err(PauliError::InvalidSector {
                context: "Product factor transition destination is outside its factor",
            });
        };
        if slot.is_some() {
            return Err(PauliError::InvalidSector {
                context: "Product factor transition is not injective",
            });
        }
        *slot = Some(ProductInverseTransition { source_rank });
    }
    Ok(inverse.into_boxed_slice())
}

fn invert_product_digit_transitions(
    transitions: &[Option<ProductDigitTransition>],
) -> Result<Box<[Option<ProductInverseTransition>]>, PauliError> {
    let mut inverse = vec![None; transitions.len()];
    for (source_digit, transition) in transitions.iter().enumerate() {
        let Some(transition) = transition else {
            continue;
        };
        let destination_digit =
            usize::try_from(transition.destination).map_err(|_| PauliError::Overflow {
                context: "converting Product spectator destination",
            })?;
        let Some(slot) = inverse.get_mut(destination_digit) else {
            return Err(PauliError::InvalidSector {
                context: "Product spectator transition destination is outside its factor",
            });
        };
        if slot.is_some() {
            return Err(PauliError::InvalidSector {
                context: "Product spectator transition is not injective",
            });
        }
        *slot = Some(ProductInverseTransition {
            source_rank: source_digit,
        });
    }
    Ok(inverse.into_boxed_slice())
}

#[allow(clippy::too_many_arguments)]
fn build_product_action_descriptor(
    term: &ChargeTransitionTerm,
    factor_positions: &[Vec<usize>],
    spectator_positions: &[usize],
    local_dimensions: &[u64],
    fermion_positions: &[usize],
    boson_positions: &[usize],
    qubit_positions: &[usize],
    qudit_positions: &[usize],
    qudit_dimension: u64,
    factor_key_values: &[Vec<Vec<u64>>],
    factor_key_ranks: &[BTreeMap<Vec<u64>, usize>],
    factor_dimensions: &[usize],
    spectator_dimensions: &[usize],
) -> Result<ProductLazyActionDescriptor, PauliError> {
    let axis_count = local_dimensions.len();
    let mut factor_axis_lookup = vec![(usize::MAX, usize::MAX); axis_count];
    for (factor, positions) in factor_positions.iter().enumerate() {
        for (axis, &position) in positions.iter().enumerate() {
            factor_axis_lookup[position] = (factor, axis);
        }
    }
    let mut spectator_axis_lookup = vec![usize::MAX; axis_count];
    for (index, &position) in spectator_positions.iter().enumerate() {
        spectator_axis_lookup[position] = index;
    }
    let mut factor_ops = vec![Vec::<ProductLocalOp>::new(); factor_positions.len()];
    let mut spectator_ops = vec![Vec::<ProductLocalOp>::new(); spectator_positions.len()];
    let mut push_op = |position: usize, mut op: ProductLocalOp| -> Result<(), PauliError> {
        let (factor, spectator) =
            product_action_axis(position, &factor_axis_lookup, &spectator_axis_lookup)?;
        if let Some((factor, local_axis)) = factor {
            match &mut op {
                ProductLocalOp::Fermion { axis, .. }
                | ProductLocalOp::Boson { axis, .. }
                | ProductLocalOp::Pauli { axis, .. }
                | ProductLocalOp::Qudit { axis, .. } => *axis = local_axis,
            }
            factor_ops[factor].push(op);
        } else if let Some(spectator) = spectator {
            match &mut op {
                ProductLocalOp::Fermion { axis, .. }
                | ProductLocalOp::Boson { axis, .. }
                | ProductLocalOp::Pauli { axis, .. }
                | ProductLocalOp::Qudit { axis, .. } => *axis = 0,
            }
            spectator_ops[spectator].push(op);
        }
        Ok(())
    };
    for &mode in term.fermion_annihilation.iter().rev() {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        let position = *fermion_positions.get(mode).ok_or_else(invalid_sector)?;
        push_op(
            position,
            ProductLocalOp::Fermion {
                axis: 0,
                create: false,
            },
        )?;
    }
    for &mode in term.fermion_creation.iter().rev() {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        let position = *fermion_positions.get(mode).ok_or_else(invalid_sector)?;
        push_op(
            position,
            ProductLocalOp::Fermion {
                axis: 0,
                create: true,
            },
        )?;
    }
    for &(mode, creation, annihilation) in &term.boson_blocks {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        let position = *boson_positions.get(mode).ok_or_else(invalid_sector)?;
        push_op(
            position,
            ProductLocalOp::Boson {
                axis: 0,
                creation,
                annihilation,
            },
        )?;
    }
    for (&code, &position) in term.qubit_codes.iter().zip(qubit_positions) {
        if code != 0 {
            push_op(position, ProductLocalOp::Pauli { axis: 0, code })?;
        }
    }
    if term.mapped_present {
        for (&code, &position) in term.mapped_codes.iter().zip(fermion_positions) {
            if code != 0 {
                push_op(position, ProductLocalOp::Pauli { axis: 0, code })?;
            }
        }
    }
    if term.qudit_present {
        for &(site, shift, phase_power) in &term.qudit_triples {
            let site = usize::try_from(site).map_err(|_| invalid_sector())?;
            let position = *qudit_positions.get(site).ok_or_else(invalid_sector)?;
            push_op(
                position,
                ProductLocalOp::Qudit {
                    axis: 0,
                    shift,
                    phase_power,
                },
            )?;
        }
    }
    let mut factor_actions = Vec::with_capacity(factor_positions.len());
    for (factor, positions) in factor_positions.iter().enumerate() {
        if factor_ops[factor].is_empty() {
            factor_actions.push(None);
            continue;
        }
        let selected_dimension = factor_dimensions[factor];
        let mut transitions = Vec::with_capacity(selected_dimension);
        let cartesian_dimensions = positions
            .iter()
            .map(|&position| local_dimensions[position])
            .collect::<Vec<_>>();
        for key in factor_key_values[factor].iter().take(selected_dimension) {
            let mut values = key.clone();
            let Some(value) = apply_product_local_ops(
                &mut values,
                &cartesian_dimensions,
                &factor_ops[factor],
                qudit_dimension,
            )?
            else {
                transitions.push(None);
                continue;
            };
            let Some(&destination_rank) = factor_key_ranks[factor].get(&values) else {
                transitions.push(None);
                continue;
            };
            transitions.push(Some(ProductRankTransition {
                destination_rank,
                value,
            }));
        }
        let inverse_transitions = invert_product_rank_transitions(&transitions)?;
        factor_actions.push(Some(ProductFactorAction {
            transitions: transitions.into_boxed_slice(),
            inverse_transitions,
        }));
    }
    let mut spectator_actions = Vec::with_capacity(spectator_positions.len());
    for (index, &position) in spectator_positions.iter().enumerate() {
        if spectator_ops[index].is_empty() {
            spectator_actions.push(None);
            continue;
        }
        let dimension = spectator_dimensions[index];
        let dimensions = [local_dimensions[position]];
        let mut transitions = Vec::with_capacity(dimension);
        for digit in 0..dimension {
            let mut values = vec![digit as u64];
            let Some(value) = apply_product_local_ops(
                &mut values,
                &dimensions,
                &spectator_ops[index],
                qudit_dimension,
            )?
            else {
                transitions.push(None);
                continue;
            };
            transitions.push(Some(ProductDigitTransition {
                destination: values[0],
                value,
            }));
        }
        let inverse_transitions = invert_product_digit_transitions(&transitions)?;
        spectator_actions.push(Some(ProductSpectatorAction {
            transitions: transitions.into_boxed_slice(),
            inverse_transitions,
        }));
    }
    Ok(ProductLazyActionDescriptor {
        coefficient: term.coefficient,
        factor_actions: factor_actions.into_boxed_slice(),
        spectator_actions: spectator_actions.into_boxed_slice(),
        fermion_mask: FermionMaskDescriptor::from_creation_annihilation(
            fermion_positions,
            &term.fermion_creation,
            &term.fermion_annihilation,
        ),
    })
}

/// Compile a Product schedule for disjoint compact factors.
///
/// Factorization is established by the Python boundary from immutable charge
/// metadata. Native lowering owns the validated factor/rank metadata and
/// canonical terms during construction, then publishes flat action
/// descriptors for lazy execution. Lazy application retains no complete
/// transition graph; eager materialization uses the shared CSR builder
/// separately.
pub fn build_fast_product_mvp_plan(
    plan: &Arc<ChargeSectorPlan>,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    factor_positions: &[Vec<u64>],
    max_bytes: u128,
    lazy: bool,
) -> Result<Option<FastProductMvpPlan>, PauliError> {
    if !lazy {
        return Ok(None);
    }
    if factor_positions.is_empty()
        || factor_positions.iter().any(Vec::is_empty)
        || factor_positions.iter().flatten().any(|&position| {
            usize::try_from(position).map_or(true, |value| value >= layout.local_dimensions.len())
        })
    {
        return Ok(None);
    }
    let factor_positions = factor_positions
        .iter()
        .map(|values| positions(values, layout.local_dimensions.len(), max_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let mut occupied = FxHashSet::default();
    if factor_positions
        .iter()
        .flatten()
        .any(|position| !occupied.insert(*position))
    {
        return Ok(None);
    }
    let dimension = layout.dimension;
    let axis_count = layout.local_dimensions.len();
    let spectator_positions = (0..axis_count)
        .filter(|position| !occupied.contains(position))
        .collect::<Vec<_>>();
    let spectator_dimension =
        spectator_positions
            .iter()
            .try_fold(1usize, |value, &position| {
                value
                    .checked_mul(usize::try_from(layout.local_dimensions[position]).map_err(
                        |_| PauliError::Overflow {
                            context: "converting Product spectator dimension",
                        },
                    )?)
                    .ok_or(PauliError::Overflow {
                        context: "sizing Product spectator dimension",
                    })
            })?;
    let spectator_dimensions = spectator_positions
        .iter()
        .map(|&position| {
            usize::try_from(layout.local_dimensions[position]).map_err(|_| PauliError::Overflow {
                context: "converting Product spectator dimension",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut factor_maps = Vec::<BTreeMap<Vec<u64>, usize>>::with_capacity(factor_positions.len());
    let mut factor_dimensions = Vec::with_capacity(factor_positions.len());
    let mut factor_key_values = Vec::<Vec<Vec<u64>>>::with_capacity(factor_positions.len());
    let mut public_factor_indexes = Vec::with_capacity(factor_positions.len());
    let mut factor_metadata_bytes = 0_u128;
    for positions in &factor_positions {
        let Some(factor_plan) = plan.independent_factor_plan(positions, max_bytes)? else {
            return Ok(None);
        };
        let factor_dimension = factor_plan.dimension();
        let factor_axis_count = positions.len();
        let per_factor_entry_bytes = (factor_axis_count as u128)
            .checked_mul((std::mem::size_of::<u64>() + 2 * std::mem::size_of::<usize>()) as u128)
            .and_then(|value| value.checked_add(std::mem::size_of::<usize>() as u128))
            .ok_or(PauliError::Overflow {
                context: "estimating Product factor index",
            })?;
        factor_metadata_bytes = factor_metadata_bytes
            .checked_add(
                (factor_dimension as u128)
                    .checked_mul(per_factor_entry_bytes)
                    .ok_or(PauliError::Overflow {
                        context: "estimating Product factor index",
                    })?,
            )
            .ok_or(PauliError::Overflow {
                context: "estimating Product factor index",
            })?;
        if factor_metadata_bytes > max_bytes {
            return Ok(None);
        }
        let mut factor_map = BTreeMap::new();
        let mut keys = Vec::with_capacity(factor_dimension);
        let mut prefix_offsets =
            Vec::with_capacity(factor_dimension.checked_mul(factor_axis_count).ok_or(
                PauliError::Overflow {
                    context: "sizing Product factor prefixes",
                },
            )?);
        let mut completion_counts =
            Vec::with_capacity(factor_dimension.checked_mul(factor_axis_count + 1).ok_or(
                PauliError::Overflow {
                    context: "sizing Product factor completions",
                },
            )?);
        let mut values = vec![0_u64; factor_axis_count];
        let mut remaining = vec![0_i128; factor_plan.constraint_count()];
        let mut candidate_remaining = vec![0_i128; factor_plan.constraint_count()];
        let mut prefix = vec![0_usize; factor_axis_count];
        let mut completions = vec![0_usize; factor_axis_count + 1];
        for rank in 0..factor_dimension {
            factor_plan.unrank_into_with_scratch(
                u64::try_from(rank).map_err(|_| PauliError::Overflow {
                    context: "indexing Product factor source",
                })?,
                &mut values,
                &mut remaining,
                &mut candidate_remaining,
            )?;
            factor_plan.prefix_metadata_into(
                &values,
                &mut completions,
                &mut prefix,
                &mut remaining,
                &mut candidate_remaining,
            )?;
            factor_map.insert(values.clone(), rank);
            keys.push(values.clone());
            prefix_offsets.extend_from_slice(&prefix);
            completion_counts.extend_from_slice(&completions);
        }
        factor_maps.push(factor_map);
        factor_dimensions.push(factor_dimension);
        factor_key_values.push(keys);
        public_factor_indexes.push(ProductFactorPublicIndex {
            axis_count: factor_axis_count,
            prefix_offsets: prefix_offsets.into_boxed_slice(),
            completion_counts: completion_counts.into_boxed_slice(),
        });
    }
    if factor_dimensions.len() > 16 {
        return Ok(None);
    }
    let factor_component_dimension =
        factor_dimensions
            .iter()
            .try_fold(1usize, |value, &factor| {
                value.checked_mul(factor).ok_or(PauliError::Overflow {
                    context: "sizing Product component dimension",
                })
            })?;
    let component_dimension = factor_component_dimension
        .checked_mul(spectator_dimension)
        .ok_or(PauliError::Overflow {
            context: "sizing Product component dimension",
        })?;
    if component_dimension != dimension {
        return Ok(None);
    }
    let mut axis_roles = vec![None; axis_count];
    for (factor, positions) in factor_positions.iter().enumerate() {
        for (local_axis, &position) in positions.iter().enumerate() {
            axis_roles[position] = Some(ProductAxisRole::Factor { factor, local_axis });
        }
    }
    for (spectator, &position) in spectator_positions.iter().enumerate() {
        axis_roles[position] = Some(ProductAxisRole::Spectator { spectator });
    }
    let axis_roles = axis_roles
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(invalid_sector)?;
    let mut spectator_suffix_dimensions = vec![1_usize; axis_count];
    let mut suffix_dimension = 1_usize;
    for axis in (0..axis_count).rev() {
        spectator_suffix_dimensions[axis] = suffix_dimension;
        if let ProductAxisRole::Spectator { spectator } = axis_roles[axis] {
            suffix_dimension = suffix_dimension
                .checked_mul(spectator_dimensions[spectator])
                .ok_or(PauliError::Overflow {
                    context: "sizing Product spectator suffix",
                })?;
        }
    }
    if suffix_dimension != spectator_dimension {
        return Err(invalid_sector());
    }
    let flat_component_strides = contiguous_product_component_strides(
        &axis_roles,
        &factor_dimensions,
        &spectator_dimensions,
        dimension,
    )?;
    let public_indexer = ProductPublicIndexer {
        axis_roles: axis_roles.into_boxed_slice(),
        spectator_suffix_dimensions: spectator_suffix_dimensions.into_boxed_slice(),
        factors: public_factor_indexes.into_boxed_slice(),
        flat_component_strides,
    };
    let mut actions = Vec::new();
    if lazy {
        for term in terms {
            actions.push(build_product_action_descriptor(
                term,
                &factor_positions,
                &spectator_positions,
                &layout.local_dimensions,
                &layout.fermion_positions,
                &layout.boson_positions,
                &layout.qubit_positions,
                &layout.qudit_positions,
                layout.qudit_dimension,
                &factor_key_values,
                &factor_maps,
                &factor_dimensions,
                &spectator_dimensions,
            )?);
        }
    }
    let mut group_map = BTreeMap::<ProductScheduleKey, Vec<usize>>::new();
    for (action_index, action) in actions.iter().enumerate() {
        group_map
            .entry(product_schedule_key(
                action,
                &factor_dimensions,
                &spectator_dimensions,
            ))
            .or_default()
            .push(action_index);
    }
    let groups = group_map
        .into_values()
        .map(|members| {
            let representative = members[0];
            let mut active_factors = vec![false; factor_dimensions.len()];
            let mut active_spectators = vec![false; spectator_dimensions.len()];
            for &member in &members {
                let action = &actions[member];
                for (index, factor_action) in action.factor_actions.iter().enumerate() {
                    active_factors[index] |= factor_action.is_some();
                }
                for (index, spectator_action) in action.spectator_actions.iter().enumerate() {
                    active_spectators[index] |= spectator_action.is_some();
                }
                if let Some(descriptor) = action.fermion_mask.as_ref() {
                    for (mode, &axis) in descriptor.mode_axes.iter().enumerate() {
                        let word = mode / 64;
                        let bit = 1_u64 << (mode % 64);
                        if descriptor.required_occupied[word] & bit == 0
                            && descriptor.required_empty[word] & bit == 0
                            && descriptor.flip_mask[word] & bit == 0
                            && descriptor.source_parity_mask[word] & bit == 0
                        {
                            continue;
                        }
                        match public_indexer.axis_roles[axis] {
                            ProductAxisRole::Factor { factor, .. } => active_factors[factor] = true,
                            ProductAxisRole::Spectator { spectator } => {
                                active_spectators[spectator] = true
                            }
                        }
                    }
                }
            }
            let active_factor_indices = active_factors
                .iter()
                .enumerate()
                .filter_map(|(index, &active)| active.then_some(index))
                .collect::<Vec<_>>();
            let inactive_factor_indices = active_factors
                .iter()
                .enumerate()
                .filter_map(|(index, &active)| (!active).then_some(index))
                .collect::<Vec<_>>();
            let active_spectator_indices = active_spectators
                .iter()
                .enumerate()
                .filter_map(|(index, &active)| active.then_some(index))
                .collect::<Vec<_>>();
            let mut trailing_identity_dimension = 1usize;
            let mut trailing_identity_spectators = vec![false; spectator_dimensions.len()];
            for role in public_indexer.axis_roles.iter().rev() {
                let ProductAxisRole::Spectator { spectator } = *role else {
                    break;
                };
                if active_spectators[spectator] {
                    break;
                }
                trailing_identity_spectators[spectator] = true;
                trailing_identity_dimension = trailing_identity_dimension
                    .checked_mul(spectator_dimensions[spectator])
                    .ok_or(PauliError::Overflow {
                        context: "sizing Product identity suffix",
                    })?;
            }
            let inactive_spectator_indices = active_spectators
                .iter()
                .enumerate()
                .filter_map(|(index, &active)| {
                    (!active && !trailing_identity_spectators[index]).then_some(index)
                })
                .collect::<Vec<_>>();
            let trailing_identity_spectator_indices = trailing_identity_spectators
                .iter()
                .enumerate()
                .filter_map(|(index, &trailing)| trailing.then_some(index))
                .collect::<Vec<_>>();
            Ok(ProductScheduleGroup {
                representative,
                members: members.into_boxed_slice(),
                active_factors: ProductComponentShape::new(
                    active_factor_indices,
                    &factor_dimensions,
                )?,
                inactive_factors: ProductComponentShape::new(
                    inactive_factor_indices,
                    &factor_dimensions,
                )?,
                active_spectators: ProductComponentShape::new(
                    active_spectator_indices,
                    &spectator_dimensions,
                )?,
                inactive_spectators: ProductComponentShape::new(
                    inactive_spectator_indices,
                    &spectator_dimensions,
                )?,
                trailing_identity_spectators: trailing_identity_spectator_indices
                    .into_boxed_slice(),
                trailing_identity_dimension,
            })
        })
        .collect::<Result<Vec<_>, PauliError>>()?;
    let action_bytes = actions.iter().try_fold(0_u128, |total, action| {
        let factor_bytes = action
            .factor_actions
            .iter()
            .map(|maybe| {
                let transition_bytes = maybe
                    .as_ref()
                    .map(|value| {
                        (value.transitions.len() as u128)
                            .checked_mul(
                                std::mem::size_of::<Option<ProductRankTransition>>() as u128
                            )
                            .and_then(|bytes| {
                                bytes.checked_add(
                                    (value.inverse_transitions.len()
                                        * std::mem::size_of::<Option<ProductInverseTransition>>())
                                        as u128,
                                )
                            })
                            .and_then(|bytes| {
                                bytes
                                    .checked_add(std::mem::size_of::<ProductFactorAction>() as u128)
                            })
                            .unwrap_or(u128::MAX)
                    })
                    .unwrap_or(0);
                (std::mem::size_of::<Option<ProductFactorAction>>() as u128)
                    .checked_add(transition_bytes)
                    .ok_or(PauliError::Overflow {
                        context: "estimating Product factor actions",
                    })
            })
            .try_fold(0_u128, |sum, value| {
                let value = value?;
                sum.checked_add(value).ok_or(PauliError::Overflow {
                    context: "estimating Product factor actions",
                })
            })?;
        let spectator_bytes = action
            .spectator_actions
            .iter()
            .map(|maybe| {
                let transition_bytes = maybe
                    .as_ref()
                    .map(|value| {
                        (value.transitions.len() as u128)
                            .checked_mul(
                                std::mem::size_of::<Option<ProductDigitTransition>>() as u128
                            )
                            .and_then(|bytes| {
                                bytes.checked_add(
                                    (value.inverse_transitions.len()
                                        * std::mem::size_of::<Option<ProductInverseTransition>>())
                                        as u128,
                                )
                            })
                            .and_then(|bytes| {
                                bytes.checked_add(
                                    std::mem::size_of::<ProductSpectatorAction>() as u128
                                )
                            })
                            .unwrap_or(u128::MAX)
                    })
                    .unwrap_or(0);
                (std::mem::size_of::<Option<ProductSpectatorAction>>() as u128)
                    .checked_add(transition_bytes)
                    .ok_or(PauliError::Overflow {
                        context: "estimating Product spectator actions",
                    })
            })
            .try_fold(0_u128, |sum, value| {
                let value = value?;
                sum.checked_add(value).ok_or(PauliError::Overflow {
                    context: "estimating Product spectator actions",
                })
            })?;
        let mask_bytes = action
            .fermion_mask
            .as_ref()
            .map(|mask| {
                let words = mask.required_occupied.len();
                (words as u128)
                    .checked_mul(4 * std::mem::size_of::<u64>() as u128)
                    .and_then(|bytes| {
                        bytes.checked_add(
                            (mask.mode_axes.len() * std::mem::size_of::<usize>()) as u128,
                        )
                    })
                    .and_then(|bytes| {
                        bytes.checked_add(std::mem::size_of::<FermionMaskDescriptor>() as u128)
                    })
                    .unwrap_or(u128::MAX)
            })
            .unwrap_or(0);
        total
            .checked_add(std::mem::size_of::<ProductLazyActionDescriptor>() as u128)
            .and_then(|value| value.checked_add(factor_bytes))
            .and_then(|value| value.checked_add(spectator_bytes))
            .and_then(|value| value.checked_add(mask_bytes))
            .ok_or(PauliError::Overflow {
                context: "estimating Product action descriptors",
            })
    })?;
    let metadata_bytes = ((factor_dimensions.len() + spectator_dimensions.len()) as u128)
        .checked_mul(std::mem::size_of::<usize>() as u128)
        .and_then(|value| value.checked_add(factor_metadata_bytes))
        .and_then(|value| {
            value.checked_add((axis_count as u128).checked_mul(
                (std::mem::size_of::<ProductAxisRole>() + std::mem::size_of::<usize>()) as u128,
            )?)
        })
        .ok_or(PauliError::Overflow {
            context: "estimating independent-factor Product descriptors",
        })?;
    let actual_bytes = metadata_bytes
        .checked_add(action_bytes)
        .and_then(|value| {
            value
                .checked_add(
                    groups
                        .iter()
                        .map(|group| {
                            std::mem::size_of::<ProductScheduleGroup>()
                                + group.members.len() * std::mem::size_of::<usize>()
                        })
                        .sum::<usize>() as u128,
                )
                .and_then(|value| {
                    value.checked_add(
                        groups
                            .iter()
                            .map(|group| {
                                (2 * (group.active_factors.len()
                                    + group.inactive_factors.len()
                                    + group.active_spectators.len()
                                    + group.inactive_spectators.len())
                                    + group.trailing_identity_spectators.len())
                                    * std::mem::size_of::<usize>()
                            })
                            .sum::<usize>() as u128,
                    )
                })
        })
        .ok_or(PauliError::Overflow {
            context: "estimating independent-factor Product descriptors",
        })?;
    if actual_bytes > max_bytes {
        return Ok(None);
    }
    let fermion_word_count = actions
        .iter()
        .filter_map(|action| action.fermion_mask.as_ref())
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    Ok(Some(FastProductMvpPlan {
        factor_dimensions: factor_dimensions.into_boxed_slice(),
        factor_key_values: factor_key_values
            .into_iter()
            .map(|keys| {
                keys.into_iter()
                    .map(Vec::into_boxed_slice)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        spectator_dimensions: spectator_dimensions.into_boxed_slice(),
        actions: actions.into_boxed_slice(),
        groups: groups.into_boxed_slice(),
        public_indexer,
        counters: Some(Arc::new(ProductApplyCounters::default())),
        fermion_word_count,
        dimension,
        estimated_bytes: actual_bytes,
    }))
}

/// Build the cached spinful fast path when the native descriptors are eligible.
#[allow(clippy::too_many_arguments)]
pub fn build_fast_fermion_mvp_plan(
    plan: &ChargeSectorPlan,
    local_dimensions: &[u64],
    fermion_positions: &[u64],
    boson_positions: &[u64],
    qubit_positions: &[u64],
    qudit_positions: &[u64],
    terms: &[ChargeTransitionTerm],
    particles: Option<usize>,
    mode_order: Option<&[u64]>,
    max_bytes: u128,
) -> Result<Option<FastFermionMvpPlan>, PauliError> {
    let Some(particles) = particles else {
        return Ok(None);
    };
    if terms.iter().any(|term| {
        !term.qubit_codes.is_empty()
            || term.mapped_present
            // The Python descriptor keeps a fixed-width zero code row for
            // terms without a mapped-fermion Pauli.  It is semantically
            // absent unless the corresponding presence bit is set.
            || (term.mapped_present && !term.mapped_codes.is_empty())
            || !term.qudit_triples.is_empty()
            || term.qudit_present
    }) {
        return Ok(None);
    }
    let fermion_positions = positions(fermion_positions, local_dimensions.len(), max_bytes)?;
    let boson_positions = positions(boson_positions, local_dimensions.len(), max_bytes)?;
    let qubit_positions = positions(qubit_positions, local_dimensions.len(), max_bytes)?;
    let qudit_positions = positions(qudit_positions, local_dimensions.len(), max_bytes)?;
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

fn apply_fast_boson_blocks(
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

fn apply_fermion_mask_words(
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

fn apply_fast_boson_inverse(
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

/// Compile transitions directly against a reusable rank/unrank plan.
///
/// Source and destination occupation buffers are reused for every term. A
/// destination occupation vector is used as the aggregate key so the path
/// does not encode the full Cartesian Hilbert space into one integer; this is
/// essential for wide layouts whose selected sector still fits platform
/// indices. Destination aggregation still happens before sector membership is
/// checked.
fn compile_charge_row_buffers_from_prepared_plan(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    max_bytes: u128,
    chunk_capacity: usize,
) -> Result<CsrSink, PauliError> {
    let dimension = layout.dimension;
    let local_dimensions = &layout.local_dimensions;
    let qudit_dimension = layout.qudit_dimension;
    let schedules = &layout.schedules;
    let axis_count = local_dimensions.len();
    if dimension != plan.dimension() {
        return Err(invalid_sector());
    }
    if chunk_capacity == 0 {
        return Err(PauliError::InvalidStructureLength {
            expected: 1,
            actual: 0,
        });
    }
    charge_csr_memory_preflight(dimension, terms.len(), axis_count, max_bytes)?;
    let scratch_bytes = axis_count
        .checked_mul(std::mem::size_of::<u64>())
        .and_then(|value| value.checked_add(terms.len().checked_mul(64)?))
        .and_then(|value| value.checked_add(32))
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector transition workspace",
        })?;
    check_bytes(1, scratch_bytes, max_bytes)?;

    let mut source = vec![0_u64; axis_count];
    let mut destination = vec![0_u64; axis_count];
    let mut remaining = vec![0_i128; plan.constraint_count()];
    let mut candidate_remaining = vec![0_i128; plan.constraint_count()];
    let mut destinations = ChargeDestinationAccumulator::new(
        layout.full_strides.as_deref(),
        terms.len(),
        axis_count,
        Some(layout.counters.as_ref()),
    )?;
    let fermion_word_count = schedules
        .iter()
        .filter_map(|schedule| schedule.fermion.as_ref())
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    let mut fermion_words = vec![0_u64; fermion_word_count];
    let mut sink = CsrSink::with_chunk_capacity(dimension, chunk_capacity);
    for column in 0..dimension {
        plan.unrank_into_with_scratch(
            u64::try_from(column).map_err(|_| PauliError::Overflow {
                context: "indexing charge-sector transitions",
            })?,
            &mut source,
            &mut remaining,
            &mut candidate_remaining,
        )?;
        destinations.clear();
        for schedule in schedules {
            let Some(value) = apply_compiled_charge_schedule(
                schedule,
                &source,
                &mut destination,
                local_dimensions,
                qudit_dimension,
                &mut fermion_words,
            )?
            else {
                continue;
            };
            destinations.insert(&destination, value)?;
        }
        let column_index = u64::try_from(column).map_err(|_| PauliError::Overflow {
            context: "indexing charge-sector transitions",
        })?;
        destinations.drain(&mut destination, local_dimensions, |destination, value| {
            if value.re == 0.0 && value.im == 0.0 {
                return Ok(());
            }
            layout
                .counters
                .ranker_invocations
                .fetch_add(1, Ordering::Relaxed);
            let row = match plan.rank_into(destination, &mut remaining, &mut candidate_remaining) {
                Ok(row) => row,
                Err(PauliError::InvalidSector { .. }) => {
                    return Err(PauliError::InvalidSector {
                        context: "operator leaks outside the selected charge sector",
                    });
                }
                Err(error) => return Err(error),
            };
            let row = usize::try_from(row).map_err(|_| PauliError::Overflow {
                context: "converting charge-sector transition row",
            })?;
            sink.push(row, column_index, value)?;
            Ok(())
        })?;
    }
    Ok(sink)
}

/// Compile transitions directly against a reusable rank/unrank plan.
/// Compile a restricted operator directly into its authoritative CSR graph.
///
/// The native eager path consumes this representation directly, so no PyO3
/// COO round trip or Python row-pointer reconstruction is needed.  The
/// transition compiler above remains available for the explicit COO API.
pub fn compile_charge_csr_from_prepared_plan(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    max_bytes: u128,
) -> Result<ChargeCsrResult, PauliError> {
    compile_charge_csr_from_prepared_plan_with_chunk_capacity(
        plan,
        layout,
        terms,
        max_bytes,
        CSR_CHUNK_CAPACITY,
    )
}

/// Compile a charge CSR graph with an explicit diagnostic chunk capacity.
/// Production callers use [`compile_charge_csr_from_prepared_plan`]; the
/// bounded capacity is exposed only so release evidence can sweep the sink's
/// `O(E log K)` merge behavior without changing the production default.
pub fn compile_charge_csr_from_prepared_plan_with_chunk_capacity(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    max_bytes: u128,
    chunk_capacity: usize,
) -> Result<ChargeCsrResult, PauliError> {
    let dimension = layout.dimension;
    let generation_start = Instant::now();
    let sink = compile_charge_row_buffers_from_prepared_plan(
        plan,
        layout,
        terms,
        max_bytes,
        chunk_capacity,
    )?;
    let generation_nanos = generation_start.elapsed().as_nanos();
    let buffered_entries = sink.len();
    let ((indptr, columns, values, _), chunk_stats) = sink.into_csr_with_stats()?;
    let (estimated_bytes, estimated_construction_peak_bytes) = charge_csr_memory_estimate(
        dimension,
        terms.len(),
        layout.local_dimensions.len(),
        buffered_entries,
        chunk_capacity,
    )?;
    if estimated_construction_peak_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_construction_peak_bytes,
            limit: max_bytes,
        });
    }
    Ok(ChargeCsrResult {
        dimension,
        indptr,
        columns,
        values,
        estimated_bytes,
        estimated_construction_peak_bytes,
        schedule_count: terms.len(),
        generation_nanos,
        chunk_count: chunk_stats.chunk_count,
        sorted_entries: chunk_stats.sorted_entries,
        merged_entries: chunk_stats.merged_entries,
        heap_operations: chunk_stats.heap_operations,
        final_array_allocation_entries: chunk_stats.final_array_allocation_entries,
        chunk_metadata_bytes: chunk_stats.chunk_metadata_bytes,
        sort_nanos: chunk_stats.sort_nanos,
        merge_nanos: chunk_stats.merge_nanos,
        final_array_allocation_nanos: chunk_stats.final_array_allocation_nanos,
    })
}

/// Apply a structured operator directly against a reusable charge-sector plan.
///
/// This is the explicitly lazy counterpart to the native CSR compiler. It keeps only the input/output
/// vectors and one source-column destination aggregate in memory; it does not
/// retain the full restricted transition graph. Destination terms are
/// aggregated before sector membership is checked, preserving the same exact
/// cancellation semantics as eager compilation.
pub fn apply_charge_mvp_from_prepared_plan(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    termwise_conserved: bool,
    fast_plan: Option<&FastChargeMvpPlan>,
    max_bytes: u128,
) -> Result<Vec<Complex64>, PauliError> {
    let dimension = layout.dimension;
    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP output",
        })?;
    if output_bytes as u128 > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: output_bytes as u128,
            limit: max_bytes,
        });
    }
    let mut output = vec![Complex64::new(0.0, 0.0); dimension];
    apply_charge_mvp_from_prepared_plan_into(
        plan,
        layout,
        terms,
        state,
        &mut output,
        termwise_conserved,
        fast_plan,
        max_bytes,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn apply_charge_mvp_from_prepared_plan_into(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    output: &mut [Complex64],
    termwise_conserved: bool,
    fast_plan: Option<&FastChargeMvpPlan>,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let dimension = layout.dimension;
    let local_dimensions = &layout.local_dimensions;
    let qudit_dimension = layout.qudit_dimension;
    let schedules = &layout.schedules;
    if dimension != plan.dimension() || state.len() != dimension || output.len() != dimension {
        return Err(invalid_sector());
    }

    // The cached spinful kernel has no execution-time major allocation. Run it
    // before the generic workspace preflight so apply_into(max_bytes=0) remains
    // valid for a caller-owned output buffer.
    if let Some(fast_plan) = fast_plan {
        fast_plan.apply_into(state, output, max_bytes)?;
        return Ok(());
    }

    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP output",
        })?;
    if output_bytes as u128 > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: output_bytes as u128,
            limit: max_bytes,
        });
    }
    let scratch_limit = max_bytes - output_bytes as u128;
    let axis_count = local_dimensions.len();
    let destination_entry_bytes = if layout.full_strides.is_some() {
        std::mem::size_of::<(usize, Complex64)>()
    } else {
        64usize
            .checked_add(axis_count.checked_mul(std::mem::size_of::<u64>()).ok_or(
                PauliError::Overflow {
                    context: "estimating charge MVP destination storage",
                },
            )?)
            .ok_or(PauliError::Overflow {
                context: "estimating charge MVP destination storage",
            })?
    };
    let scratch_bytes = axis_count
        .checked_mul(std::mem::size_of::<u64>())
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(terms.len().checked_mul(destination_entry_bytes)?))
        .and_then(|value| value.checked_add(32))
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP workspace",
        })?;
    check_bytes(1, scratch_bytes, scratch_limit)?;

    output.fill(Complex64::new(0.0, 0.0));
    if !termwise_conserved {
        layout
            .counters
            .fallback_invocations
            .fetch_add(1, Ordering::Relaxed);
    }
    let mut source = vec![0_u64; axis_count];
    let mut destination = vec![0_u64; axis_count];
    let mut remaining = vec![0_i128; plan.constraint_count()];
    let mut candidate_remaining = vec![0_i128; plan.constraint_count()];
    let mut destinations = ChargeDestinationAccumulator::new(
        layout.full_strides.as_deref(),
        terms.len(),
        axis_count,
        Some(layout.counters.as_ref()),
    )?;
    let fermion_word_count = schedules
        .iter()
        .filter_map(|schedule| schedule.fermion.as_ref())
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    let mut fermion_words = vec![0_u64; fermion_word_count];

    for (column, state_value) in state.iter().enumerate() {
        plan.unrank_into_with_scratch(
            u64::try_from(column).map_err(|_| PauliError::Overflow {
                context: "indexing charge-sector MVP source",
            })?,
            &mut source,
            &mut remaining,
            &mut candidate_remaining,
        )?;
        destinations.clear();
        if termwise_conserved {
            for schedule in schedules {
                let Some(value) = apply_compiled_charge_schedule(
                    schedule,
                    &source,
                    &mut destination,
                    local_dimensions,
                    qudit_dimension,
                    &mut fermion_words,
                )?
                else {
                    continue;
                };
                let row = if destination == source {
                    u64::try_from(column).map_err(|_| PauliError::Overflow {
                        context: "indexing charge-sector MVP destination",
                    })?
                } else {
                    layout
                        .counters
                        .ranker_invocations
                        .fetch_add(1, Ordering::Relaxed);
                    plan.rank_into(&destination, &mut remaining, &mut candidate_remaining)?
                };
                let row = usize::try_from(row).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector MVP destination",
                })?;
                output[row] += value * *state_value;
            }
        } else {
            for schedule in schedules {
                let Some(value) = apply_compiled_charge_schedule(
                    schedule,
                    &source,
                    &mut destination,
                    local_dimensions,
                    qudit_dimension,
                    &mut fermion_words,
                )?
                else {
                    continue;
                };
                destinations.insert(&destination, value)?;
            }
        }
        destinations.drain(&mut destination, local_dimensions, |destination, value| {
            if value.re == 0.0 && value.im == 0.0 {
                return Ok(());
            }
            layout
                .counters
                .ranker_invocations
                .fetch_add(1, Ordering::Relaxed);
            let row = match plan.rank_into(destination, &mut remaining, &mut candidate_remaining) {
                Ok(row) => usize::try_from(row).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector MVP destination",
                })?,
                Err(PauliError::InvalidSector { .. }) => {
                    return Err(PauliError::InvalidSector {
                        context: "operator leaks outside the selected charge sector",
                    });
                }
                Err(error) => return Err(error),
            };
            output[row] += value * *state_value;
            Ok(())
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ChargeDestinationAccumulator;
    use crate::{allocation_count, Complex64};

    #[test]
    fn wide_destination_accumulator_reuses_fixed_width_storage() {
        let mut accumulator = ChargeDestinationAccumulator::new(None, 3, 4, None).unwrap();
        let first_capacity = accumulator.wide.as_ref().unwrap().values.capacity();
        accumulator
            .insert(&[1, 0, 0, 0], Complex64::new(1.0, 0.0))
            .unwrap();
        accumulator
            .insert(&[1, 0, 0, 0], Complex64::new(-1.0, 0.0))
            .unwrap();
        accumulator
            .insert(&[0, 1, 0, 0], Complex64::new(2.0, 0.0))
            .unwrap();
        let mut destination = [0_u64; 4];
        let mut entries = Vec::new();
        accumulator
            .drain(
                &mut destination,
                &[4, 4, 4, 4],
                |destination: &[u64], value| {
                    entries.push((destination.to_vec(), value));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(first_capacity, 12);
        assert_eq!(
            accumulator.wide.as_ref().unwrap().values.capacity(),
            first_capacity
        );
        assert_eq!(
            entries,
            vec![
                (vec![1, 0, 0, 0], Complex64::new(0.0, 0.0)),
                (vec![0, 1, 0, 0], Complex64::new(2.0, 0.0))
            ]
        );
    }

    #[test]
    fn scalar_and_wide_accumulator_cycles_allocate_nothing() {
        let mut scalar = ChargeDestinationAccumulator::new(Some(&[1, 2]), 3, 2, None).unwrap();
        let mut wide = ChargeDestinationAccumulator::new(None, 3, 4, None).unwrap();
        let mut scalar_destination = [0_u64; 2];
        let mut wide_destination = [0_u64; 4];
        let (_, allocations) = allocation_count(|| {
            for _ in 0..64 {
                scalar.clear();
                scalar.insert(&[1, 0], Complex64::new(1.0, 0.0)).unwrap();
                scalar.insert(&[0, 1], Complex64::new(-1.0, 0.0)).unwrap();
                scalar
                    .drain(&mut scalar_destination, &[2, 2], |_, _| Ok(()))
                    .unwrap();

                wide.clear();
                wide.insert(&[1, 0, 0, 0], Complex64::new(1.0, 0.0))
                    .unwrap();
                wide.insert(&[0, 1, 0, 0], Complex64::new(-1.0, 0.0))
                    .unwrap();
                wide.drain(&mut wide_destination, &[2, 2, 2, 2], |_, _| Ok(()))
                    .unwrap();
            }
        });
        assert_eq!(allocations, 0);
    }
}
