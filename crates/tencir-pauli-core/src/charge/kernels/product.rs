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
    destination_key_allocations: usize,
    fallback_invocations: usize,
    ranker_invocations: usize,
    aggregate_insertions: usize,
    group_evaluations: usize,
}

impl ProductApplyCounterDeltas {
    fn merge(&mut self, other: Self) {
        self.factor_action_evaluations += other.factor_action_evaluations;
        self.spectator_segment_applications += other.spectator_segment_applications;
        self.destination_key_allocations += other.destination_key_allocations;
        self.fallback_invocations += other.fallback_invocations;
        self.ranker_invocations += other.ranker_invocations;
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
        self.destination_key_allocations
            .fetch_add(deltas.destination_key_allocations, Ordering::Relaxed);
        self.fallback_invocations
            .fetch_add(deltas.fallback_invocations, Ordering::Relaxed);
        self.ranker_invocations
            .fetch_add(deltas.ranker_invocations, Ordering::Relaxed);
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
        deltas: &mut ProductApplyCounterDeltas,
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
                    deltas.group_evaluations += 1;
                    deltas.factor_action_evaluations += group.active_factors.len();
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
                        deltas.aggregate_insertions += 1;
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
                            deltas.spectator_segment_applications +=
                                group.trailing_identity_dimension;
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
            let mut deltas = ProductApplyCounterDeltas::default();
            self.apply_destination_gather_into(state, output, &mut deltas)?;
            if let Some(counters) = self.counters.as_ref() {
                counters.add(&deltas);
            }
            Ok(())
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

/// Construction-only owner for the independent-factor Product plan phases.
///
/// All fields here are scratch or construction metadata. `finish` is the only
/// phase that publishes a `FastProductMvpPlan`.
struct ProductPlanBuilder<'context, 'input, 'positions> {
    context: &'context ChargeBuildContext<'input>,
    factor_positions: &'positions [Vec<u64>],
    lazy: bool,
}

struct ProductComponents {
    factor_positions: Vec<Vec<usize>>,
    spectator_positions: Vec<usize>,
    spectator_dimensions: Vec<usize>,
    spectator_dimension: usize,
    dimension: usize,
    axis_count: usize,
}

/// Factor-rank construction data. Rank maps are build scratch used while
/// compiling actions; only the keys and public index metadata survive in the
/// published plan.
struct ProductFactorIndexes {
    factor_maps: Vec<BTreeMap<Vec<u64>, usize>>,
    factor_dimensions: Vec<usize>,
    factor_key_values: Vec<Vec<Vec<u64>>>,
    public_factor_indexes: Vec<ProductFactorPublicIndex>,
    metadata_bytes: u128,
}

struct ProductFactorIndex {
    rank_map: BTreeMap<Vec<u64>, usize>,
    dimension: usize,
    keys: Vec<Vec<u64>>,
    public_index: ProductFactorPublicIndex,
}

impl<'context, 'input, 'positions> ProductPlanBuilder<'context, 'input, 'positions> {
    fn new(
        context: &'context ChargeBuildContext<'input>,
        factor_positions: &'positions [Vec<u64>],
        lazy: bool,
    ) -> Self {
        Self {
            context,
            factor_positions,
            lazy,
        }
    }

    fn discover_components(&self) -> Result<Option<ProductComponents>, PauliError> {
        let local_dimensions = self.context.local_dimensions();
        if self.factor_positions.is_empty()
            || self.factor_positions.iter().any(Vec::is_empty)
            || self.factor_positions.iter().flatten().any(|&position| {
                usize::try_from(position).map_or(true, |value| value >= local_dimensions.len())
            })
        {
            return Ok(None);
        }
        let factor_positions = self
            .factor_positions
            .iter()
            .map(|values| positions(values, local_dimensions.len(), self.context.max_bytes()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut occupied = FxHashSet::default();
        if factor_positions
            .iter()
            .flatten()
            .any(|position| !occupied.insert(*position))
        {
            return Ok(None);
        }
        let axis_count = local_dimensions.len();
        let spectator_positions = (0..axis_count)
            .filter(|position| !occupied.contains(position))
            .collect::<Vec<_>>();
        let spectator_dimensions = spectator_positions
            .iter()
            .map(|&position| {
                usize::try_from(local_dimensions[position]).map_err(|_| PauliError::Overflow {
                    context: "converting Product spectator dimension",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let spectator_dimension =
            spectator_dimensions
                .iter()
                .try_fold(1usize, |value, &dimension| {
                    value.checked_mul(dimension).ok_or(PauliError::Overflow {
                        context: "sizing Product spectator dimension",
                    })
                })?;
        Ok(Some(ProductComponents {
            factor_positions,
            spectator_positions,
            spectator_dimensions,
            spectator_dimension,
            dimension: self.context.sector().dimension(),
            axis_count,
        }))
    }

    fn build(self) -> Result<Option<FastProductMvpPlan>, PauliError> {
        let Some(components) = self.discover_components()? else {
            return Ok(None);
        };
        if !self.lazy {
            return Ok(None);
        }
        let Some(mut indexes) = self.build_factor_indexes(&components)? else {
            return Ok(None);
        };
        let Some(public_indexer) = self.build_public_indexer(&components, &mut indexes)? else {
            return Ok(None);
        };
        let actions = self.compile_actions(
            &components.factor_positions,
            &components.spectator_positions,
            &indexes.factor_key_values,
            &indexes.factor_maps,
            &indexes.factor_dimensions,
            &components.spectator_dimensions,
        )?;
        let groups = self.group_actions(
            &actions,
            &public_indexer,
            &indexes.factor_dimensions,
            &components.spectator_dimensions,
        )?;
        self.finish(components, indexes, public_indexer, actions, groups)
    }

    fn build_factor_indexes(
        &self,
        components: &ProductComponents,
    ) -> Result<Option<ProductFactorIndexes>, PauliError> {
        if components.factor_positions.len() > 16 {
            return Ok(None);
        }
        let mut factor_maps = Vec::with_capacity(components.factor_positions.len());
        let mut factor_dimensions = Vec::with_capacity(components.factor_positions.len());
        let mut factor_key_values = Vec::with_capacity(components.factor_positions.len());
        let mut public_factor_indexes = Vec::with_capacity(components.factor_positions.len());
        let mut metadata_bytes = 0_u128;
        for positions in &components.factor_positions {
            let Some(ProductFactorIndex {
                rank_map,
                dimension,
                keys,
                public_index,
            }) = self.build_factor_index(positions, &mut metadata_bytes)?
            else {
                return Ok(None);
            };
            factor_maps.push(rank_map);
            factor_dimensions.push(dimension);
            factor_key_values.push(keys);
            public_factor_indexes.push(public_index);
        }
        Ok(Some(ProductFactorIndexes {
            factor_maps,
            factor_dimensions,
            factor_key_values,
            public_factor_indexes,
            metadata_bytes,
        }))
    }

    fn build_factor_index(
        &self,
        positions: &[usize],
        metadata_bytes: &mut u128,
    ) -> Result<Option<ProductFactorIndex>, PauliError> {
        let max_bytes = self.context.max_bytes();
        let Some(factor_plan) = self
            .context
            .sector()
            .independent_factor_plan(positions, max_bytes)?
        else {
            return Ok(None);
        };
        let factor_dimension = factor_plan.dimension();
        let factor_axis_count = positions.len();
        let per_entry = (factor_axis_count as u128)
            .checked_mul((std::mem::size_of::<u64>() + 2 * std::mem::size_of::<usize>()) as u128)
            .and_then(|value| value.checked_add(std::mem::size_of::<usize>() as u128))
            .ok_or(PauliError::Overflow {
                context: "estimating Product factor index",
            })?;
        *metadata_bytes = metadata_bytes
            .checked_add((factor_dimension as u128).checked_mul(per_entry).ok_or(
                PauliError::Overflow {
                    context: "estimating Product factor index",
                },
            )?)
            .ok_or(PauliError::Overflow {
                context: "estimating Product factor index",
            })?;
        if *metadata_bytes > max_bytes {
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
        Ok(Some(ProductFactorIndex {
            rank_map: factor_map,
            dimension: factor_dimension,
            keys,
            public_index: ProductFactorPublicIndex {
                axis_count: factor_axis_count,
                prefix_offsets: prefix_offsets.into_boxed_slice(),
                completion_counts: completion_counts.into_boxed_slice(),
            },
        }))
    }

    fn build_public_indexer(
        &self,
        components: &ProductComponents,
        indexes: &mut ProductFactorIndexes,
    ) -> Result<Option<ProductPublicIndexer>, PauliError> {
        let factor_component_dimension =
            indexes
                .factor_dimensions
                .iter()
                .try_fold(1usize, |value, &factor| {
                    value.checked_mul(factor).ok_or(PauliError::Overflow {
                        context: "sizing Product component dimension",
                    })
                })?;
        let component_dimension = factor_component_dimension
            .checked_mul(components.spectator_dimension)
            .ok_or(PauliError::Overflow {
                context: "sizing Product component dimension",
            })?;
        if component_dimension != components.dimension {
            return Ok(None);
        }
        let mut axis_roles = vec![None; components.axis_count];
        for (factor, positions) in components.factor_positions.iter().enumerate() {
            for (local_axis, &position) in positions.iter().enumerate() {
                axis_roles[position] = Some(ProductAxisRole::Factor { factor, local_axis });
            }
        }
        for (spectator, &position) in components.spectator_positions.iter().enumerate() {
            axis_roles[position] = Some(ProductAxisRole::Spectator { spectator });
        }
        let axis_roles = axis_roles
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(invalid_sector)?;
        let mut spectator_suffix_dimensions = vec![1_usize; components.axis_count];
        let mut suffix_dimension = 1usize;
        for axis in (0..components.axis_count).rev() {
            spectator_suffix_dimensions[axis] = suffix_dimension;
            if let ProductAxisRole::Spectator { spectator } = axis_roles[axis] {
                suffix_dimension = suffix_dimension
                    .checked_mul(components.spectator_dimensions[spectator])
                    .ok_or(PauliError::Overflow {
                        context: "sizing Product spectator suffix",
                    })?;
            }
        }
        if suffix_dimension != components.spectator_dimension {
            return Err(invalid_sector());
        }
        let flat_component_strides = contiguous_product_component_strides(
            &axis_roles,
            &indexes.factor_dimensions,
            &components.spectator_dimensions,
            components.dimension,
        )?;
        Ok(Some(ProductPublicIndexer {
            axis_roles: axis_roles.into_boxed_slice(),
            spectator_suffix_dimensions: spectator_suffix_dimensions.into_boxed_slice(),
            factors: std::mem::take(&mut indexes.public_factor_indexes).into_boxed_slice(),
            flat_component_strides,
        }))
    }

    fn compile_actions(
        &self,
        factor_positions: &[Vec<usize>],
        spectator_positions: &[usize],
        factor_key_values: &[Vec<Vec<u64>>],
        factor_maps: &[BTreeMap<Vec<u64>, usize>],
        factor_dimensions: &[usize],
        spectator_dimensions: &[usize],
    ) -> Result<Vec<ProductLazyActionDescriptor>, PauliError> {
        if !self.lazy {
            return Ok(Vec::new());
        }
        let mut actions = Vec::with_capacity(self.context.terms().len());
        for term in self.context.terms() {
            actions.push(build_product_action_descriptor(
                term,
                factor_positions,
                spectator_positions,
                self.context.local_dimensions(),
                self.context.fermion_positions(),
                self.context.boson_positions(),
                self.context.qubit_positions(),
                self.context.qudit_positions(),
                self.context.qudit_dimension(),
                factor_key_values,
                factor_maps,
                factor_dimensions,
                spectator_dimensions,
            )?);
        }
        Ok(actions)
    }

    fn group_actions(
        &self,
        actions: &[ProductLazyActionDescriptor],
        public_indexer: &ProductPublicIndexer,
        factor_dimensions: &[usize],
        spectator_dimensions: &[usize],
    ) -> Result<Vec<ProductScheduleGroup>, PauliError> {
        let mut group_map = BTreeMap::<ProductScheduleKey, Vec<usize>>::new();
        for (action_index, action) in actions.iter().enumerate() {
            group_map
                .entry(product_schedule_key(
                    action,
                    factor_dimensions,
                    spectator_dimensions,
                ))
                .or_default()
                .push(action_index);
        }
        group_map
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
                                ProductAxisRole::Factor { factor, .. } => {
                                    active_factors[factor] = true
                                }
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
                        factor_dimensions,
                    )?,
                    inactive_factors: ProductComponentShape::new(
                        inactive_factor_indices,
                        factor_dimensions,
                    )?,
                    active_spectators: ProductComponentShape::new(
                        active_spectator_indices,
                        spectator_dimensions,
                    )?,
                    inactive_spectators: ProductComponentShape::new(
                        inactive_spectator_indices,
                        spectator_dimensions,
                    )?,
                    trailing_identity_spectators: trailing_identity_spectator_indices
                        .into_boxed_slice(),
                    trailing_identity_dimension,
                })
            })
            .collect()
    }
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
    context: &ChargeBuildContext<'_>,
    factor_positions: &[Vec<u64>],
    lazy: bool,
) -> Result<Option<FastProductMvpPlan>, PauliError> {
    ProductPlanBuilder::new(context, factor_positions, lazy).build()
}

impl ProductPlanBuilder<'_, '_, '_> {
    fn finish(
        &self,
        components: ProductComponents,
        indexes: ProductFactorIndexes,
        public_indexer: ProductPublicIndexer,
        actions: Vec<ProductLazyActionDescriptor>,
        groups: Vec<ProductScheduleGroup>,
    ) -> Result<Option<FastProductMvpPlan>, PauliError> {
        let ProductComponents {
            spectator_dimensions,
            dimension,
            axis_count,
            ..
        } = components;
        let ProductFactorIndexes {
            factor_dimensions,
            factor_key_values,
            metadata_bytes: factor_metadata_bytes,
            ..
        } = indexes;
        let max_bytes = self.context.max_bytes();
        let action_bytes = product_action_bytes(&actions)?;
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
}
use super::*;

fn product_action_bytes(actions: &[ProductLazyActionDescriptor]) -> Result<u128, PauliError> {
    actions.iter().try_fold(0_u128, |total, action| {
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
    })
}
