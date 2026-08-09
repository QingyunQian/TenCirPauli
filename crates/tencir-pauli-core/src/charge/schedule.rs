/// Owned charge-sector layout validated once for repeated native execution.
pub struct PreparedChargeTransitionPlanLayout {
    pub(super) dimension: usize,
    pub(super) local_dimensions: Vec<u64>,
    pub(super) fermion_positions: Vec<usize>,
    pub(super) boson_positions: Vec<usize>,
    pub(super) qubit_positions: Vec<usize>,
    pub(super) qudit_positions: Vec<usize>,
    pub(super) qudit_dimension: u64,
    pub(super) schedules: Vec<CompiledChargeSchedule>,
    pub(super) full_strides: Option<Box<[usize]>>,
    estimated_bytes: u128,
    pub(super) counters: Arc<ChargeApplyCounters>,
}

#[derive(Clone)]
pub(super) struct CompiledChargeSchedule {
    pub(super) coefficient: Complex64,
    pub(super) zero: bool,
    pub(super) fermion: Option<FermionMaskDescriptor>,
    pub(super) boson_blocks: Box<[(usize, u32, u32)]>,
    pub(super) qubit_ops: Box<[(usize, u8)]>,
    pub(super) mapped_ops: Box<[(usize, u8)]>,
    pub(super) qudit_ops: Box<[(usize, u32, u32)]>,
}

#[derive(Default)]
pub(super) struct ChargeApplyCounters {
    scalar_insertions: AtomicUsize,
    wide_insertions: AtomicUsize,
    accumulator_capacity_growths: AtomicUsize,
    owned_destination_key_creations: AtomicUsize,
    fallback_invocations: AtomicUsize,
    ranker_invocations: AtomicUsize,
    construction_descriptor_validations: AtomicUsize,
    destination_key_allocations: AtomicUsize,
}

#[derive(Default)]
pub(super) struct ChargeApplyStats {
    pub(super) scalar_insertions: usize,
    pub(super) wide_insertions: usize,
    pub(super) accumulator_capacity_growths: usize,
    pub(super) owned_destination_key_creations: usize,
    pub(super) fallback_invocations: usize,
    pub(super) ranker_invocations: usize,
    pub(super) destination_key_allocations: usize,
}

impl ChargeApplyStats {
    pub(super) fn merge(&mut self, other: Self) {
        self.scalar_insertions += other.scalar_insertions;
        self.wide_insertions += other.wide_insertions;
        self.accumulator_capacity_growths += other.accumulator_capacity_growths;
        self.owned_destination_key_creations += other.owned_destination_key_creations;
        self.fallback_invocations += other.fallback_invocations;
        self.ranker_invocations += other.ranker_invocations;
        self.destination_key_allocations += other.destination_key_allocations;
    }
}

impl ChargeApplyCounters {
    pub(super) fn snapshot(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
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

    pub(super) fn publish(&self, stats: ChargeApplyStats) {
        self.scalar_insertions
            .fetch_add(stats.scalar_insertions, Ordering::Relaxed);
        self.wide_insertions
            .fetch_add(stats.wide_insertions, Ordering::Relaxed);
        self.accumulator_capacity_growths
            .fetch_add(stats.accumulator_capacity_growths, Ordering::Relaxed);
        self.owned_destination_key_creations
            .fetch_add(stats.owned_destination_key_creations, Ordering::Relaxed);
        self.fallback_invocations
            .fetch_add(stats.fallback_invocations, Ordering::Relaxed);
        self.ranker_invocations
            .fetch_add(stats.ranker_invocations, Ordering::Relaxed);
        self.destination_key_allocations
            .fetch_add(stats.destination_key_allocations, Ordering::Relaxed);
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

pub(super) struct WideDestinationAccumulator {
    entries: Vec<ScalarAggregateEntry>,
    pub(super) values: Vec<u64>,
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

pub(super) struct ChargeDestinationAccumulator<'a> {
    scalar: Option<ReusableScalarAccumulator>,
    pub(super) wide: Option<WideDestinationAccumulator>,
    strides: Option<&'a [usize]>,
}

impl<'a> ChargeDestinationAccumulator<'a> {
    pub(super) fn new(
        strides: Option<&'a [usize]>,
        capacity: usize,
        width: usize,
    ) -> Result<Self, PauliError> {
        Ok(Self {
            scalar: strides.map(|_| ReusableScalarAccumulator::with_capacity(capacity)),
            wide: strides
                .is_none()
                .then(|| WideDestinationAccumulator::with_capacity(capacity, width))
                .transpose()?,
            strides,
        })
    }

    pub(super) fn clear(&mut self) {
        if let Some(scalar) = self.scalar.as_mut() {
            scalar.clear();
        }
        if let Some(wide) = self.wide.as_mut() {
            wide.clear();
        }
    }

    pub(super) fn insert(
        &mut self,
        destination: &[u64],
        value: Complex64,
    ) -> Result<ChargeApplyStats, PauliError> {
        if let (Some(scalar), Some(strides)) = (self.scalar.as_mut(), self.strides) {
            let key = encode_full_radix(destination, strides)?;
            let is_new = !scalar.entries.iter().any(|entry| entry.key == key);
            let mut stats = ChargeApplyStats {
                scalar_insertions: 1,
                ..ChargeApplyStats::default()
            };
            if is_new && scalar.entries.len() == scalar.entries.capacity() {
                stats.accumulator_capacity_growths = 1;
                stats.destination_key_allocations = 1;
            }
            scalar.insert(key, value)?;
            Ok(stats)
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
            let mut stats = ChargeApplyStats {
                wide_insertions: 1,
                ..ChargeApplyStats::default()
            };
            if is_new
                && (wide.entries.len() == wide.entries.capacity()
                    || wide.values.capacity() < required_values)
            {
                stats.accumulator_capacity_growths = 1;
                stats.owned_destination_key_creations = 1;
                stats.destination_key_allocations = 1;
            }
            wide.insert(destination, value)?;
            Ok(stats)
        } else {
            Err(invalid_sector())
        }
    }

    pub(super) fn drain<F>(
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

pub(super) fn invalid_sector() -> PauliError {
    PauliError::InvalidSector {
        context: "invalid charge-sector transition input",
    }
}

pub(super) fn check_bytes(
    entries: usize,
    bytes_per_entry: usize,
    limit: u128,
) -> Result<(), PauliError> {
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

pub(super) fn charge_csr_memory_estimate(
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

pub(super) fn charge_csr_memory_preflight(
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

pub(super) fn positions(
    values: &[u64],
    axis_count: usize,
    limit: u128,
) -> Result<Vec<usize>, PauliError> {
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

pub(super) fn apply_compiled_charge_schedule(
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
use super::*;
