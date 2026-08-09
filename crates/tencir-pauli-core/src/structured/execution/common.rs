/// One local finite-basis operation in a canonical structured term.
/// `kind=0` is Pauli, `kind=1` is a boson block, and `kind=2` is direct Weyl.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StructuredOperation {
    pub axis: usize,
    pub kind: u8,
    pub p: u32,
    pub q: u32,
}

#[derive(Clone)]
pub(super) struct CompiledStructuredOperation {
    pub(super) axis: usize,
    pub(super) kind: u8,
    pub(super) p: usize,
    pub(super) q: usize,
    pub(super) stride: usize,
    pub(super) dimension: usize,
    pub(super) local_table: Vec<Complex64>,
}

#[derive(Clone)]
pub(super) struct CompiledNetAction {
    pub(super) axis: usize,
    pub(super) stride: usize,
    pub(super) dimension: usize,
    pub(super) forward_destinations: Box<[usize]>,
    pub(super) forward_amplitudes: Box<[Complex64]>,
    pub(super) inverse_sources: Box<[usize]>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CompiledScheduleKey {
    /// Destination digits for each non-Fermion active axis.  Amplitudes are
    /// intentionally absent: source-dependent phases and ladder factors are
    /// evaluated once per grouped source configuration below.
    net_axes: Vec<(usize, Vec<usize>)>,
    fermion_axes: Vec<usize>,
    fermion_flip: Vec<u64>,
    fallback_shape: Vec<(usize, u8, usize, usize)>,
}

pub(super) struct StructuredScheduleGroup {
    pub(super) members: Box<[usize]>,
    pub(super) suffix_stride: usize,
}

#[derive(Clone, Copy)]
pub(super) struct ContiguousFermionPrefix {
    pub(super) suffix_dimension: usize,
}

#[derive(Default)]
pub(super) struct StructuredApplyCounters {
    schedule_group_evaluations: AtomicUsize,
    segment_applications: AtomicUsize,
    fermion_mask_applications: AtomicUsize,
    nonfermion_fallback_invocations: AtomicUsize,
}

#[derive(Default)]
pub(super) struct StructuredApplyStats {
    pub(super) schedule_group_evaluations: usize,
    pub(super) segment_applications: usize,
    pub(super) fermion_mask_applications: usize,
    pub(super) nonfermion_fallback_invocations: usize,
}

impl StructuredApplyCounters {
    pub(super) fn snapshot(&self) -> (usize, usize, usize, usize) {
        (
            self.schedule_group_evaluations.load(Ordering::Relaxed),
            self.segment_applications.load(Ordering::Relaxed),
            self.fermion_mask_applications.load(Ordering::Relaxed),
            self.nonfermion_fallback_invocations.load(Ordering::Relaxed),
        )
    }

    pub(super) fn publish(&self, stats: StructuredApplyStats) {
        self.schedule_group_evaluations
            .fetch_add(stats.schedule_group_evaluations, Ordering::Relaxed);
        self.segment_applications
            .fetch_add(stats.segment_applications, Ordering::Relaxed);
        self.fermion_mask_applications
            .fetch_add(stats.fermion_mask_applications, Ordering::Relaxed);
        self.nonfermion_fallback_invocations
            .fetch_add(stats.nonfermion_fallback_invocations, Ordering::Relaxed);
    }
}

pub(super) fn compile_net_actions(
    term: &[StructuredOperation],
    local_dimensions: &[usize],
    strides: &[usize],
) -> Result<Option<Box<[CompiledNetAction]>>, PauliError> {
    let nonfermion_term = term
        .iter()
        .copied()
        .filter(|operation| operation.kind != 3)
        .collect::<Vec<_>>();
    let key = structured_net_key(&nonfermion_term, local_dimensions)?;
    let mut actions = Vec::with_capacity(key.len());
    for (axis, rows) in key {
        let dimension = *local_dimensions.get(axis).ok_or(PauliError::InvalidIndex {
            context: "structured net-action axis",
        })?;
        if rows.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: rows.len(),
            });
        }
        let stride = *strides.get(axis).ok_or(PauliError::InvalidIndex {
            context: "structured net-action stride",
        })?;
        let mut forward_destinations = Vec::with_capacity(dimension);
        let mut forward_amplitudes = Vec::with_capacity(dimension);
        let mut inverse_sources = vec![usize::MAX; dimension];
        for (source, &(destination, real, imag)) in rows.iter().enumerate() {
            let amplitude = Complex64::new(f64::from_bits(real), f64::from_bits(imag));
            forward_destinations.push(if is_exact_zero(amplitude) {
                usize::MAX
            } else {
                destination
            });
            forward_amplitudes.push(amplitude);
            if !is_exact_zero(amplitude) {
                if destination >= dimension || inverse_sources[destination] != usize::MAX {
                    return Err(PauliError::InvalidStructureLength {
                        expected: dimension,
                        actual: destination,
                    });
                }
                inverse_sources[destination] = source;
            }
        }
        actions.push(CompiledNetAction {
            axis,
            stride,
            dimension,
            forward_destinations: forward_destinations.into_boxed_slice(),
            forward_amplitudes: forward_amplitudes.into_boxed_slice(),
            inverse_sources: inverse_sources.into_boxed_slice(),
        });
    }
    Ok(Some(actions.into_boxed_slice()))
}

pub(super) fn compiled_schedule_key(
    term: &[CompiledStructuredOperation],
    actions: Option<&[CompiledNetAction]>,
    fermion: Option<&FermionMaskDescriptor>,
) -> CompiledScheduleKey {
    let net_axes = actions
        .map(|actions| {
            actions
                .iter()
                .map(|action| (action.axis, action.forward_destinations.to_vec()))
                .collect()
        })
        .unwrap_or_default();
    let (fermion_axes, fermion_flip) = fermion
        .map(|descriptor| (descriptor.mode_axes.to_vec(), descriptor.flip_mask.to_vec()))
        .unwrap_or_default();
    let fallback_shape = if actions.is_none() {
        term.iter()
            .map(|operation| (operation.axis, operation.kind, operation.p, operation.q))
            .collect()
    } else {
        Vec::new()
    };
    CompiledScheduleKey {
        net_axes,
        fermion_axes,
        fermion_flip,
        fallback_shape,
    }
}

pub(super) fn schedule_suffix_stride(
    active_axes: impl Iterator<Item = usize>,
    strides: &[usize],
    dimension: usize,
) -> usize {
    active_axes
        .max()
        .and_then(|axis| strides.get(axis).copied())
        .unwrap_or(dimension)
}

pub(super) fn compile_schedule_groups(
    terms: &[Vec<CompiledStructuredOperation>],
    net_actions: &[Option<Box<[CompiledNetAction]>>],
    fermion_masks: &[Option<FermionMaskDescriptor>],
    strides: &[usize],
    dimension: usize,
) -> Result<Box<[StructuredScheduleGroup]>, PauliError> {
    if terms.len() != net_actions.len() || terms.len() != fermion_masks.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: terms.len(),
            actual: net_actions.len().max(fermion_masks.len()),
        });
    }
    let mut grouped = BTreeMap::<CompiledScheduleKey, Vec<usize>>::new();
    for index in 0..terms.len() {
        let key = compiled_schedule_key(
            &terms[index],
            net_actions[index].as_deref(),
            fermion_masks[index].as_ref(),
        );
        grouped.entry(key).or_default().push(index);
    }
    grouped
        .into_values()
        .map(|members| {
            let representative = members[0];
            let mut active_axes = BTreeSet::new();
            if let Some(actions) = net_actions[representative].as_deref() {
                active_axes.extend(actions.iter().map(|action| action.axis));
            } else {
                active_axes.extend(terms[representative].iter().map(|operation| operation.axis));
            }
            if let Some(descriptor) = fermion_masks[representative].as_ref() {
                // The descriptor's parity mask spans lower modes, so the full
                // mode-axis map is the conservative active shape.  This still
                // leaves trailing Cartesian spectator blocks reusable, which is
                // the dominant Fermion×Boson layout.
                active_axes.extend(descriptor.mode_axes.iter().copied());
            }
            let suffix_stride = schedule_suffix_stride(active_axes.into_iter(), strides, dimension);
            Ok(StructuredScheduleGroup {
                members: members.into_boxed_slice(),
                suffix_stride,
            })
        })
        .collect::<Result<Vec<_>, PauliError>>()
        .map(Vec::into_boxed_slice)
}

pub(super) fn apply_compiled_net_forward(
    actions: &[CompiledNetAction],
    source: usize,
    coefficient: Complex64,
) -> Result<(usize, Complex64, bool), PauliError> {
    let mut destination = source;
    let mut amplitude = coefficient;
    for action in actions {
        let digit = (source / action.stride) % action.dimension;
        let new_digit = action.forward_destinations[digit];
        if new_digit == usize::MAX {
            return Ok((source, amplitude, false));
        }
        let old_digit = digit;
        if new_digit >= old_digit {
            destination = destination
                .checked_add((new_digit - old_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        } else {
            destination = destination
                .checked_sub((old_digit - new_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        }
        amplitude *= action.forward_amplitudes[digit];
    }
    Ok((destination, amplitude, true))
}

pub(super) fn apply_compiled_net_inverse(
    actions: &[CompiledNetAction],
    destination: usize,
    coefficient: Complex64,
) -> Result<(usize, Complex64, bool), PauliError> {
    let mut source = destination;
    for action in actions.iter().rev() {
        let digit = (source / action.stride) % action.dimension;
        let old_digit = action.inverse_sources[digit];
        if old_digit == usize::MAX {
            return Ok((destination, coefficient, false));
        }
        if old_digit >= digit {
            source = source
                .checked_add((old_digit - digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net source",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net source",
                })?;
        } else {
            source = source
                .checked_sub((digit - old_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net source",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net source",
                })?;
        }
    }
    // Re-evaluate amplitudes in the canonical forward action order. The
    // source derivation above must run backwards, but multiplying the local
    // factors backwards would change the last bit of complex128 results for
    // multi-axis schedules relative to dense/source-major execution.
    let mut amplitude = coefficient;
    let mut check_destination = source;
    for action in actions {
        let digit = (check_destination / action.stride) % action.dimension;
        let new_digit = action.forward_destinations[digit];
        if new_digit == usize::MAX {
            return Ok((destination, amplitude, false));
        }
        amplitude *= action.forward_amplitudes[digit];
        if new_digit >= digit {
            check_destination = check_destination
                .checked_add((new_digit - digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        } else {
            check_destination = check_destination
                .checked_sub((digit - new_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        }
    }
    Ok((source, amplitude, true))
}

pub(super) fn compile_fermion_mask_descriptor(
    term: &[CompiledStructuredOperation],
    fermion_positions: &[usize],
) -> Option<FermionMaskDescriptor> {
    let fermion_term = term
        .iter()
        .filter(|operation| operation.kind == 3)
        .collect::<Vec<_>>();
    if fermion_term.is_empty() {
        return None;
    }
    let mode_count = if fermion_positions.is_empty() {
        fermion_term.iter().map(|operation| operation.p + 1).max()?
    } else {
        fermion_positions.len()
    };
    let mode_axes = if fermion_positions.is_empty() {
        (0..mode_count).collect::<Vec<_>>()
    } else {
        fermion_positions.to_vec()
    };
    let steps = fermion_term
        .iter()
        .map(|operation| {
            (
                if fermion_positions.is_empty() {
                    operation.axis
                } else {
                    operation.p
                },
                operation.q == 1,
            )
        })
        .collect::<Vec<_>>();
    FermionMaskDescriptor::from_steps(&mode_axes, &steps)
}

/// Compact reusable matrix-free plan for finite structured operators.
///
/// The plan retains one canonical local-operation list and coefficient per
/// symbolic term. It deliberately does not retain the expanded transition
/// table or a sparse matrix.
pub struct StructuredMvpPlan {
    pub(super) dimension: usize,
    pub(super) strides: Vec<usize>,
    pub(super) terms: Vec<Vec<CompiledStructuredOperation>>,
    pub(super) net_actions: Vec<Option<Box<[CompiledNetAction]>>>,
    pub(super) fermion_masks: Vec<Option<FermionMaskDescriptor>>,
    pub(super) groups: Box<[StructuredScheduleGroup]>,
    pub(super) coefficients: Vec<Complex64>,
    pub(super) source_term_count: usize,
    pub(super) estimated_bytes: u128,
    pub(super) execution_family: StructuredExecutionFamily,
    pub(super) inner: StructuredPlanInner,
    pub(super) counters: Arc<StructuredApplyCounters>,
    pub(super) fermion_schedule_count: usize,
    pub(super) fermion_mask_schedule_count: usize,
    pub(super) descriptor_ineligible_schedule_count: usize,
    pub(super) contiguous_fermion_source_mode_count: Option<usize>,
    pub(super) contiguous_fermion_prefix: Option<ContiguousFermionPrefix>,
}

/// Batched native phase measurements for the shared FermionMask descriptor.
///
/// This diagnostic is intentionally separate from normal MVP execution. It
/// times each phase over a complete source/descriptor batch, avoiding a clock
/// read inside individual Fermion contributions.
pub struct FermionMaskPhaseProfile {
    pub descriptor_count: usize,
    pub sample_count: usize,
    pub packed_source_unit_count: usize,
    pub schedule_member_evaluations: usize,
    pub packed_extraction_nanos: u128,
    pub parity_nanos: u128,
    pub index_update_nanos: u128,
    pub input_output_nanos: u128,
}

pub(super) enum StructuredPlanInner {
    Pauli(MvpPlan),
    Compiled,
}

pub(super) struct StructuredCsrResult {
    pub(super) dimension: usize,
    pub(super) indptr: Vec<u64>,
    pub(super) columns: Vec<u64>,
    pub(super) values: Vec<Complex64>,
    pub(super) estimated_bytes: u128,
    pub(super) estimated_construction_peak_bytes: u128,
    pub(super) schedule_count: usize,
}

/// Reusable finite structured MVP plan backed by a duplicate-aggregated CSR graph.
pub struct StructuredEagerMvpPlan {
    pub(super) graph: CsrGraph<u64>,
    pub(super) estimated_bytes: u128,
    pub(super) estimated_construction_peak_bytes: u128,
    pub(super) source_term_count: usize,
    pub(super) schedule_count: usize,
    pub(super) execution_family: StructuredExecutionFamily,
    pub(super) native_variant: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StructuredExecutionFamily {
    Pauli,
    Cartesian,
    Generic,
}

impl StructuredExecutionFamily {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Pauli => "pauli",
            Self::Cartesian => "cartesian",
            Self::Generic => "generic",
        }
    }
}

pub(super) fn classify_structured_execution(
    local_dimensions: &[usize],
    terms: &[Vec<CompiledStructuredOperation>],
    fermion_masks: &[Option<FermionMaskDescriptor>],
    net_actions: &[Option<Box<[CompiledNetAction]>>],
) -> StructuredExecutionFamily {
    if local_dimensions.iter().all(|&dimension| dimension == 2)
        && terms.iter().flatten().all(|operation| operation.kind == 0)
    {
        StructuredExecutionFamily::Pauli
    } else if net_actions.iter().any(Option::is_none)
        || terms.iter().enumerate().any(|(index, term)| {
            term.iter().any(|operation| operation.kind == 3)
                && fermion_masks.get(index).and_then(Option::as_ref).is_none()
                && net_actions.get(index).and_then(Option::as_deref).is_none()
        })
    {
        StructuredExecutionFamily::Generic
    } else {
        StructuredExecutionFamily::Cartesian
    }
}

pub(super) fn compile_packed_pauli_operator(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
) -> Result<Option<PauliOperator>, PauliError> {
    if !local_dimensions.iter().all(|&dimension| dimension == 2)
        || terms.iter().flatten().any(|operation| operation.kind != 0)
    {
        return Ok(None);
    }
    let mut structures = Vec::with_capacity(terms.len());
    let mut packed_coefficients = Vec::with_capacity(coefficients.len());
    for (term, &coefficient) in terms.iter().zip(coefficients) {
        let mut codes = vec![0_u8; local_dimensions.len()];
        let mut phase = Complex64::new(1.0, 0.0);
        for operation in term {
            let axis = operation.axis;
            let current = codes.get(axis).copied().ok_or(PauliError::InvalidIndex {
                context: "packed Structured Pauli axis",
            })?;
            let (code, local_phase) = pauli_code_product(operation.p as u8, current);
            codes[axis] = code;
            phase *= local_phase;
        }
        structures.push(codes);
        packed_coefficients.push(coefficient * phase);
    }
    let operator =
        PauliOperator::from_terms(local_dimensions.len(), &structures, &packed_coefficients)?;
    Ok(Some(operator))
}

pub(super) fn compile_packed_pauli_plan(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
) -> Result<Option<MvpPlan>, PauliError> {
    compile_packed_pauli_operator(local_dimensions, terms, coefficients)?
        .as_ref()
        .map(MvpPlan::from_operator)
        .transpose()
}
use super::*;
