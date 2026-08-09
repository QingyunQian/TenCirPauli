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
    let mut apply_stats = ChargeApplyStats::default();
    let mut destinations =
        ChargeDestinationAccumulator::new(layout.full_strides.as_deref(), terms.len(), axis_count)?;
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
            apply_stats.merge(destinations.insert(&destination, value)?);
        }
        let column_index = u64::try_from(column).map_err(|_| PauliError::Overflow {
            context: "indexing charge-sector transitions",
        })?;
        destinations.drain(&mut destination, local_dimensions, |destination, value| {
            if value.re == 0.0 && value.im == 0.0 {
                return Ok(());
            }
            apply_stats.ranker_invocations += 1;
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
    layout.counters.publish(apply_stats);
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
use super::*;
