/// Apply a structured operator directly against a reusable charge-sector plan.
///
/// This is the explicitly lazy counterpart to the native CSR compiler. It keeps only the input/output
/// vectors and one source-column destination aggregate in memory; it does not
/// retain the full restricted transition graph. Destination terms are
/// aggregated before sector membership is checked, preserving the same exact
/// cancellation semantics as eager compilation.
fn apply_charge_mvp_from_prepared_plan(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    termwise_conserved: bool,
    kernel: &ChargeLazyKernel,
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
        kernel,
        max_bytes,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn apply_charge_mvp_from_prepared_plan_into(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    state: &[Complex64],
    output: &mut [Complex64],
    termwise_conserved: bool,
    kernel: &ChargeLazyKernel,
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
    if let Some(result) = kernel.apply_specialized_into(state, output, max_bytes) {
        result?;
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
    let mut apply_stats = ChargeApplyStats {
        fallback_invocations: usize::from(!termwise_conserved),
        ..ChargeApplyStats::default()
    };
    let mut source = vec![0_u64; axis_count];
    let mut destination = vec![0_u64; axis_count];
    let mut remaining = vec![0_i128; plan.constraint_count()];
    let mut candidate_remaining = vec![0_i128; plan.constraint_count()];
    let mut destinations =
        ChargeDestinationAccumulator::new(layout.full_strides.as_deref(), terms.len(), axis_count)?;
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
                    apply_stats.ranker_invocations += 1;
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
                apply_stats.merge(destinations.insert(&destination, value)?);
            }
        }
        destinations.drain(&mut destination, local_dimensions, |destination, value| {
            if value.re == 0.0 && value.im == 0.0 {
                return Ok(());
            }
            apply_stats.ranker_invocations += 1;
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
    layout.counters.publish(apply_stats);
    Ok(())
}
/// One immutable owner for general charge-sector execution.
///
/// The selector is resolved during construction; public calls perform one enum
/// match before entering either the selected specialized kernel or Generic.
pub struct ChargeBuildContext<'a> {
    sector: &'a ChargeSectorPlan,
    layout: &'a PreparedChargeTransitionPlanLayout,
    terms: &'a [ChargeTransitionTerm],
    max_bytes: u128,
}

impl<'a> ChargeBuildContext<'a> {
    pub fn new(
        sector: &'a ChargeSectorPlan,
        layout: &'a PreparedChargeTransitionPlanLayout,
        terms: &'a [ChargeTransitionTerm],
        max_bytes: u128,
    ) -> Self {
        Self {
            sector,
            layout,
            terms,
            max_bytes,
        }
    }

    pub(crate) fn sector(&self) -> &ChargeSectorPlan {
        self.sector
    }

    pub(crate) fn local_dimensions(&self) -> &[u64] {
        &self.layout.local_dimensions
    }

    pub(crate) fn fermion_positions(&self) -> &[usize] {
        &self.layout.fermion_positions
    }

    pub(crate) fn boson_positions(&self) -> &[usize] {
        &self.layout.boson_positions
    }

    pub(crate) fn qubit_positions(&self) -> &[usize] {
        &self.layout.qubit_positions
    }

    pub(crate) fn qudit_positions(&self) -> &[usize] {
        &self.layout.qudit_positions
    }

    pub(crate) fn qudit_dimension(&self) -> u64 {
        self.layout.qudit_dimension
    }

    pub(crate) fn terms(&self) -> &[ChargeTransitionTerm] {
        self.terms
    }

    pub(crate) fn max_bytes(&self) -> u128 {
        self.max_bytes
    }
}

/// Native-boundary classification data for the general charge selector.
///
/// This borrows only Python-normalized hints. The core selector remains the
/// authority on whether a representation is actually eligible.
pub struct ChargeKernelHints<'a> {
    pub fermion_particles: Option<usize>,
    pub fermion_mode_order: Option<&'a [u64]>,
    pub qubit_particles: Option<usize>,
    pub qubit_positions: Option<&'a [u64]>,
    pub qudit_weights: Option<&'a [i128]>,
    pub qudit_target: Option<i128>,
    pub qudit_positions: Option<&'a [u64]>,
    pub boson_weights: Option<&'a [i128]>,
    pub boson_target: Option<i128>,
    pub boson_positions: Option<&'a [u64]>,
    pub product_positions: Option<&'a [Vec<u64>]>,
}

fn select_charge_lazy_kernel(
    context: &ChargeBuildContext<'_>,
    termwise_conserved: bool,
    hints: &ChargeKernelHints<'_>,
) -> Result<ChargeLazyKernel, PauliError> {
    if termwise_conserved {
        if let Some(plan) =
            build_fast_fermion_mvp_plan(context, hints.fermion_particles, hints.fermion_mode_order)?
        {
            return Ok(ChargeLazyKernel::Fermion(plan));
        }
    }

    if hints.qubit_particles.is_some() {
        if let Some(plan) =
            build_fast_qubit_mvp_plan(context, hints.qubit_particles, hints.qubit_positions)?
        {
            return Ok(ChargeLazyKernel::Qubit(plan));
        }
    } else if hints.qudit_weights.is_some() {
        if let Some(plan) = build_fast_qudit_mvp_plan(
            context,
            hints.qudit_weights,
            hints.qudit_target,
            hints.qudit_positions,
        )? {
            return Ok(ChargeLazyKernel::Qudit(plan));
        }
    } else if hints.boson_weights.is_some() {
        if let Some(plan) = build_fast_boson_mvp_plan(
            context,
            hints.boson_weights,
            hints.boson_target,
            hints.boson_positions,
        )? {
            return Ok(ChargeLazyKernel::Boson(plan));
        }
    }

    if let Some(factor_positions) = hints.product_positions {
        if let Some(plan) = build_fast_product_mvp_plan(context, factor_positions, true)? {
            return Ok(ChargeLazyKernel::Product(plan));
        }
    }
    Ok(ChargeLazyKernel::Generic)
}

/// Build the single owner of lazy general-charge execution and resolve its
/// immutable native kernel exactly once.
pub fn prepare_charge_mvp_plan(
    sector: Arc<ChargeSectorPlan>,
    layout: PreparedChargeTransitionPlanLayout,
    terms: Box<[ChargeTransitionTerm]>,
    termwise_conserved: bool,
    max_bytes: u128,
    hints: ChargeKernelHints<'_>,
) -> Result<PreparedChargeMvpPlan, PauliError> {
    let context = ChargeBuildContext::new(&sector, &layout, &terms, max_bytes);
    let kernel = select_charge_lazy_kernel(&context, termwise_conserved, &hints)?;
    Ok(PreparedChargeMvpPlan::new(
        sector,
        layout,
        terms,
        termwise_conserved,
        kernel,
    ))
}

pub struct PreparedChargeMvpPlan {
    sector: Arc<ChargeSectorPlan>,
    layout: PreparedChargeTransitionPlanLayout,
    terms: Box<[ChargeTransitionTerm]>,
    termwise_conserved: bool,
    kernel: ChargeLazyKernel,
}

impl PreparedChargeMvpPlan {
    pub fn new(
        sector: Arc<ChargeSectorPlan>,
        layout: PreparedChargeTransitionPlanLayout,
        terms: Box<[ChargeTransitionTerm]>,
        termwise_conserved: bool,
        kernel: ChargeLazyKernel,
    ) -> Self {
        Self {
            sector,
            layout,
            terms,
            termwise_conserved,
            kernel,
        }
    }

    pub fn dimension(&self) -> usize {
        self.sector.dimension()
    }

    pub fn estimated_bytes(&self) -> Result<u128, PauliError> {
        self.sector
            .estimated_bytes()
            .checked_add(estimate_charge_transition_terms_bytes(&self.terms)?)
            .and_then(|value| value.checked_add(self.layout.estimated_bytes()))
            .and_then(|value| value.checked_add(self.kernel.estimated_bytes()))
            .ok_or(PauliError::Overflow {
                context: "estimating prepared charge MVP plan",
            })
    }

    pub fn execution_family(&self) -> &'static str {
        match &self.kernel {
            ChargeLazyKernel::Generic => "generic",
            _ => "product",
        }
    }

    pub fn native_variant(&self) -> &'static str {
        self.kernel.native_variant()
    }

    pub fn schedule_count(&self) -> usize {
        self.terms.len()
    }

    pub fn debug_counters(&self) -> Option<Vec<usize>> {
        self.kernel.debug_counters()
    }

    pub fn generic_debug_counters(&self) -> [usize; 8] {
        let (
            scalar_insertions,
            wide_insertions,
            capacity_growths,
            owned_keys,
            fallbacks,
            rankers,
            validations,
            destination_allocations,
        ) = self.layout.debug_counters();
        [
            scalar_insertions,
            wide_insertions,
            capacity_growths,
            owned_keys,
            fallbacks,
            rankers,
            validations,
            destination_allocations,
        ]
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        apply_charge_mvp_from_prepared_plan(
            &self.sector,
            &self.layout,
            &self.terms,
            state,
            self.termwise_conserved,
            &self.kernel,
            max_bytes,
        )
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        apply_charge_mvp_from_prepared_plan_into(
            &self.sector,
            &self.layout,
            &self.terms,
            state,
            output,
            self.termwise_conserved,
            &self.kernel,
            max_bytes,
        )
    }

    pub fn compile_eager(&self, max_bytes: u128) -> Result<ChargeCsrResult, PauliError> {
        compile_charge_csr_from_prepared_plan(&self.sector, &self.layout, &self.terms, max_bytes)
    }

    pub fn compile_eager_with_chunk_capacity(
        &self,
        max_bytes: u128,
        chunk_capacity: usize,
    ) -> Result<ChargeCsrResult, PauliError> {
        compile_charge_csr_from_prepared_plan_with_chunk_capacity(
            &self.sector,
            &self.layout,
            &self.terms,
            max_bytes,
            chunk_capacity,
        )
    }
}
use super::*;
