//! Exact additive-charge sector dynamic programming and basis indexing.

use rustc_hash::FxHashMap;

use crate::PauliError;

/// A reusable suffix-count plan for one or more additive charge constraints.
pub struct ChargeSectorPlan {
    local_dimensions: Vec<usize>,
    contributions: Vec<Vec<Vec<i128>>>,
    target: Vec<i128>,
    suffix_counts: Vec<FxHashMap<Vec<i128>, u128>>,
    factorized_layout: Option<FactorizedLayout>,
    dimension: usize,
    estimated_bytes: u128,
}

const CHARGE_CONTRIBUTION_BYTES: u128 = 16;

/// A compact, execution-private partition proved from additive equations.
///
/// The exact suffix-count plan above remains the public rank/unrank authority.
/// This sidecar records only the factorization proof; it deliberately contains
/// neither selected occupations nor a permutation of the selected sector.
#[derive(Clone, Debug)]
pub(crate) struct FactorizedLayout {
    components: Box<[FactorComponent]>,
    axis_components: Box<[usize]>,
    estimated_bytes: u128,
}

/// Caller-owned mutable state for direct public-order factorized traversal.
pub(crate) struct FactorizedScratch {
    local_values: Vec<Vec<u64>>,
    prefix_lengths: Vec<usize>,
    completions: Vec<u128>,
}

#[derive(Clone, Debug)]
pub(crate) struct FactorComponent {
    axes: Box<[usize]>,
    kind: FactorComponentKind,
    index: ComponentIndex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FactorComponentKind {
    Cartesian,
    FixedPopulation { population: usize },
    FiniteAdditive,
}

/// Compact selected-space index for one proved independent component.
///
/// It owns only component-local DP state.  In particular, it never stores a
/// selected-state address book or a public/component permutation.
#[derive(Clone, Debug)]
pub(crate) enum ComponentIndex {
    Cartesian(CartesianIndex),
    FixedPopulation(FixedPopulationIndex),
    FiniteAdditive(FiniteAdditiveIndex),
}

#[derive(Clone, Debug)]
pub(crate) struct CartesianIndex {
    dimensions: Box<[usize]>,
    strides: Box<[usize]>,
    dimension: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FixedPopulationIndex {
    sites: usize,
    population: usize,
    dimension: usize,
    /// Small Pascal-triangle slice used by hot component rank/unrank calls.
    /// This is combinatorial metadata, not a selected-state map or basis
    /// permutation.  Large sectors retain the bounded direct formula.
    combinations: Option<Box<[u128]>>,
    combination_width: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FiniteAdditiveIndex {
    dimensions: Box<[usize]>,
    contributions: Box<[Box<[[i64; 4]]>]>,
    target: [i64; 4],
    rows: usize,
    suffix_counts: FiniteAdditiveSuffixCounts,
    dimension: usize,
}

/// Exact component-local completion tables. A one-row component keeps only
/// the scalar remainder; coupled components retain their inline row vector.
/// This is a data-layout choice, never a charge-family selector.
#[derive(Clone, Debug)]
enum FiniteAdditiveSuffixCounts {
    Scalar(Box<[FxHashMap<i64, u128>]>),
    Wide(Box<[FxHashMap<[i64; 4], u128>]>),
}

impl FactorizedLayout {
    pub(crate) fn components(&self) -> &[FactorComponent] {
        &self.components
    }

    pub(crate) fn axis_components(&self) -> &[usize] {
        &self.axis_components
    }

    /// A single multi-row finite-additive component has no independent
    /// component product to execute.  Its compact index remains useful for
    /// public rank/unrank, but charge execution should retain the exact
    /// General suffix-count traversal instead of claiming a factorized
    /// executor.
    pub(crate) fn is_single_coupled_component(&self) -> bool {
        matches!(
            self.components.as_ref(),
            [FactorComponent {
                index: ComponentIndex::FiniteAdditive(FiniteAdditiveIndex { rows: 2..=4, .. }),
                ..
            }]
        )
    }

    /// Return the one-component binary fixed-population index when physical
    /// axis order is also the component order.  This is a layout fact, not a
    /// public charge-family classification.
    pub(crate) fn contiguous_fixed_population(&self) -> Option<&FixedPopulationIndex> {
        let [component] = &*self.components else {
            return None;
        };
        if !component
            .axes
            .iter()
            .copied()
            .eq(0..self.axis_components.len())
        {
            return None;
        }
        match component.index() {
            ComponentIndex::FixedPopulation(index) => Some(index),
            ComponentIndex::Cartesian(_) | ComponentIndex::FiniteAdditive(_) => None,
        }
    }

    /// Return two contiguous binary fixed-population components in public
    /// component order. This permits a mixed-radix gather loop without a
    /// global selected-state permutation.
    pub(crate) fn contiguous_fixed_population_pair(
        &self,
    ) -> Option<(&FixedPopulationIndex, &FixedPopulationIndex)> {
        let [left, right] = &*self.components else {
            return None;
        };
        if !left.axes.iter().copied().eq(0..left.axes.len())
            || !right
                .axes
                .iter()
                .copied()
                .eq(left.axes.len()..self.axis_components.len())
        {
            return None;
        }
        match (left.index(), right.index()) {
            (ComponentIndex::FixedPopulation(left), ComponentIndex::FixedPopulation(right)) => {
                Some((left, right))
            }
            _ => None,
        }
    }

    fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub(crate) fn scratch(&self) -> FactorizedScratch {
        FactorizedScratch {
            local_values: self
                .components
                .iter()
                .map(|component| vec![0_u64; component.axes.len()])
                .collect(),
            prefix_lengths: vec![0; self.components.len()],
            completions: vec![0; self.components.len()],
        }
    }

    /// Return the direct mixed-radix strides for contiguous independent
    /// components.  These are component metadata, never a selected-state
    /// permutation; a factorized executor uses them to avoid global ranking
    /// after each local transition.
    pub(crate) fn contiguous_component_strides(&self) -> Option<Vec<usize>> {
        let mut expected_axis = 0usize;
        for component in &self.components {
            if component
                .axes
                .iter()
                .copied()
                .ne(expected_axis..expected_axis + component.axes.len())
            {
                return None;
            }
            expected_axis += component.axes.len();
        }
        if expected_axis != self.axis_components.len() {
            return None;
        }
        let mut strides = vec![1_usize; self.components.len()];
        let mut stride = 1usize;
        for component in (0..self.components.len()).rev() {
            strides[component] = stride;
            stride = stride.checked_mul(self.components[component].index.dimension())?;
        }
        Some(strides)
    }

    /// Decode one retained component rank into its local occupations.
    pub(crate) fn unrank_component_into(
        &self,
        component: usize,
        rank: usize,
        values: &mut [u64],
    ) -> Result<(), PauliError> {
        let component = self
            .components
            .get(component)
            .ok_or(PauliError::InvalidIndex {
                context: "factorized component",
            })?;
        component.index.unrank_into(
            u64::try_from(rank).map_err(|_| PauliError::Overflow {
                context: "converting factorized component rank",
            })?,
            values,
        )
    }

    /// Encode one component-local occupation without consulting other
    /// components or materializing a public-sector address book.
    pub(crate) fn rank_component(
        &self,
        component: usize,
        values: &[u64],
    ) -> Result<usize, PauliError> {
        let component = self
            .components
            .get(component)
            .ok_or(PauliError::InvalidIndex {
                context: "factorized component",
            })?;
        usize::try_from(component.index.rank(values)?).map_err(|_| PauliError::Overflow {
            context: "converting factorized component rank",
        })
    }

    pub(crate) fn rank(&self, values: &[u64]) -> Result<u64, PauliError> {
        if values.len() != self.axis_components.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.axis_components.len(),
                actual: values.len(),
            });
        }
        let mut local_values = self
            .components
            .iter()
            .map(|component| {
                component
                    .axes
                    .iter()
                    .map(|&axis| values[axis])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (component, values) in self.components.iter().zip(&local_values) {
            let _ = component.index.rank(values)?;
        }
        let mut prefix_lengths = vec![0_usize; self.components.len()];
        let mut completions = self
            .components
            .iter()
            .map(|component| component.index.completion(&[]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut total = completions.iter().try_fold(1_u128, |value, &count| {
            value.checked_mul(count).ok_or(PauliError::Overflow {
                context: "counting factorized sector dimension",
            })
        })?;
        let mut rank = 0_u128;
        for axis in 0..values.len() {
            let component_index = self.axis_components[axis];
            let component = &self.components[component_index];
            let local_axis = component
                .axes
                .iter()
                .position(|&value| value == axis)
                .ok_or(PauliError::InvalidSector {
                    context: "factorized axis mapping",
                })?;
            debug_assert_eq!(local_axis, prefix_lengths[component_index]);
            let selected = local_values[component_index][local_axis];
            let current = completions[component_index];
            let other = total
                .checked_div(current)
                .ok_or(PauliError::InvalidSector {
                    context: "factorized zero completion",
                })?;
            for candidate in 0..selected {
                local_values[component_index][local_axis] = candidate;
                let count = component
                    .index
                    .completion(&local_values[component_index][..=local_axis])?;
                rank = rank
                    .checked_add(count.checked_mul(other).ok_or(PauliError::Overflow {
                        context: "ranking factorized sector",
                    })?)
                    .ok_or(PauliError::Overflow {
                        context: "ranking factorized sector",
                    })?;
            }
            local_values[component_index][local_axis] = selected;
            let next = component
                .index
                .completion(&local_values[component_index][..=local_axis])?;
            if next == 0 {
                return Err(PauliError::InvalidSector {
                    context: "factorized occupation",
                });
            }
            completions[component_index] = next;
            total = other.checked_mul(next).ok_or(PauliError::Overflow {
                context: "counting factorized completion",
            })?;
            prefix_lengths[component_index] += 1;
        }
        u64::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting factorized rank",
        })
    }

    pub(crate) fn unrank_into(&self, rank: u64, values: &mut [u64]) -> Result<(), PauliError> {
        if values.len() != self.axis_components.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.axis_components.len(),
                actual: values.len(),
            });
        }
        let mut local_values = self
            .components
            .iter()
            .map(|component| vec![0_u64; component.axes.len()])
            .collect::<Vec<_>>();
        let mut completions = self
            .components
            .iter()
            .map(|component| component.index.completion(&[]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut total = completions.iter().try_fold(1_u128, |value, &count| {
            value.checked_mul(count).ok_or(PauliError::Overflow {
                context: "counting factorized sector dimension",
            })
        })?;
        let mut rank = u128::from(rank);
        if rank >= total {
            return Err(PauliError::InvalidIndex {
                context: "factorized rank",
            });
        }
        for (axis, output) in values.iter_mut().enumerate() {
            let component_index = self.axis_components[axis];
            let component = &self.components[component_index];
            let local_axis = component
                .axes
                .iter()
                .position(|&value| value == axis)
                .ok_or(PauliError::InvalidSector {
                    context: "factorized axis mapping",
                })?;
            let other = total.checked_div(completions[component_index]).ok_or(
                PauliError::InvalidSector {
                    context: "factorized zero completion",
                },
            )?;
            let mut selected = None;
            for candidate in 0..component_local_dimension(component.index(), local_axis) {
                local_values[component_index][local_axis] = candidate as u64;
                let count = component
                    .index
                    .completion(&local_values[component_index][..=local_axis])?;
                let block = count.checked_mul(other).ok_or(PauliError::Overflow {
                    context: "unranking factorized sector",
                })?;
                if rank < block {
                    selected = Some((candidate, count));
                    break;
                }
                rank -= block;
            }
            let Some((selected, next)) = selected else {
                return Err(PauliError::InvalidSector {
                    context: "factorized rank",
                });
            };
            *output = selected as u64;
            completions[component_index] = next;
            total = other.checked_mul(next).ok_or(PauliError::Overflow {
                context: "counting factorized completion",
            })?;
        }
        for (component, values) in self.components.iter().zip(&local_values) {
            let rank = component.index.rank(values)?;
            let mut round_trip = vec![0_u64; values.len()];
            component.index.unrank_into(rank, &mut round_trip)?;
            debug_assert_eq!(&round_trip, values);
        }
        Ok(())
    }

    pub(crate) fn rank_with_scratch(
        &self,
        values: &[u64],
        scratch: &mut FactorizedScratch,
    ) -> Result<u64, PauliError> {
        if values.len() != self.axis_components.len()
            || scratch.local_values.len() != self.components.len()
        {
            return Err(PauliError::InvalidStructureLength {
                expected: self.axis_components.len(),
                actual: values.len(),
            });
        }
        for (component, local) in self.components.iter().zip(&mut scratch.local_values) {
            for (slot, &axis) in local.iter_mut().zip(&component.axes) {
                *slot = values[axis];
            }
        }
        if let Some(strides) = self.contiguous_component_strides() {
            let mut rank = 0_u128;
            for ((component, local), &stride) in self
                .components
                .iter()
                .zip(&scratch.local_values)
                .zip(&strides)
            {
                let local_rank = component.index.rank(local)?;
                rank = rank
                    .checked_add(u128::from(local_rank).checked_mul(stride as u128).ok_or(
                        PauliError::Overflow {
                            context: "ranking contiguous factorized sector",
                        },
                    )?)
                    .ok_or(PauliError::Overflow {
                        context: "ranking contiguous factorized sector",
                    })?;
            }
            return u64::try_from(rank).map_err(|_| PauliError::Overflow {
                context: "converting factorized rank",
            });
        }
        scratch.prefix_lengths.fill(0);
        for (slot, component) in scratch.completions.iter_mut().zip(&self.components) {
            *slot = component.index.completion(&[])?;
        }
        let mut total = scratch
            .completions
            .iter()
            .try_fold(1_u128, |value, &count| {
                value.checked_mul(count).ok_or(PauliError::Overflow {
                    context: "counting factorized sector dimension",
                })
            })?;
        let mut rank = 0_u128;
        for axis in 0..values.len() {
            let component_index = self.axis_components[axis];
            let component = &self.components[component_index];
            let local_axis = scratch.prefix_lengths[component_index];
            debug_assert_eq!(component.axes[local_axis], axis);
            let selected = scratch.local_values[component_index][local_axis];
            let current = scratch.completions[component_index];
            let other = total
                .checked_div(current)
                .ok_or(PauliError::InvalidSector {
                    context: "factorized zero completion",
                })?;
            for candidate in 0..selected {
                scratch.local_values[component_index][local_axis] = candidate;
                let count = component
                    .index
                    .completion(&scratch.local_values[component_index][..=local_axis])?;
                rank = rank
                    .checked_add(count.checked_mul(other).ok_or(PauliError::Overflow {
                        context: "ranking factorized sector",
                    })?)
                    .ok_or(PauliError::Overflow {
                        context: "ranking factorized sector",
                    })?;
            }
            scratch.local_values[component_index][local_axis] = selected;
            let next = component
                .index
                .completion(&scratch.local_values[component_index][..=local_axis])?;
            if next == 0 {
                return Err(PauliError::InvalidSector {
                    context: "factorized occupation",
                });
            }
            scratch.completions[component_index] = next;
            total = other.checked_mul(next).ok_or(PauliError::Overflow {
                context: "counting factorized completion",
            })?;
            scratch.prefix_lengths[component_index] += 1;
        }
        u64::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting factorized rank",
        })
    }

    pub(crate) fn unrank_with_scratch(
        &self,
        rank: u64,
        values: &mut [u64],
        scratch: &mut FactorizedScratch,
    ) -> Result<(), PauliError> {
        if values.len() != self.axis_components.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.axis_components.len(),
                actual: values.len(),
            });
        }
        if let Some(strides) = self.contiguous_component_strides() {
            let rank = usize::try_from(rank).map_err(|_| PauliError::InvalidIndex {
                context: "factorized rank",
            })?;
            let dimension = self
                .components
                .iter()
                .try_fold(1usize, |total, component| {
                    total
                        .checked_mul(component.index.dimension())
                        .ok_or(PauliError::Overflow {
                            context: "sizing contiguous factorized sector",
                        })
                })?;
            if rank >= dimension {
                return Err(PauliError::InvalidIndex {
                    context: "factorized rank",
                });
            }
            for ((component, local), &stride) in self
                .components
                .iter()
                .zip(&mut scratch.local_values)
                .zip(&strides)
            {
                let local_rank = (rank / stride) % component.index.dimension();
                component.index.unrank_into(local_rank as u64, local)?;
                for (&axis, &value) in component.axes.iter().zip(local.iter()) {
                    values[axis] = value;
                }
            }
            return Ok(());
        }
        scratch.prefix_lengths.fill(0);
        for (slot, component) in scratch.completions.iter_mut().zip(&self.components) {
            *slot = component.index.completion(&[])?;
        }
        let mut total = scratch
            .completions
            .iter()
            .try_fold(1_u128, |value, &count| {
                value.checked_mul(count).ok_or(PauliError::Overflow {
                    context: "counting factorized sector dimension",
                })
            })?;
        let mut rank = u128::from(rank);
        if rank >= total {
            return Err(PauliError::InvalidIndex {
                context: "factorized rank",
            });
        }
        for (axis, output) in values.iter_mut().enumerate() {
            let component_index = self.axis_components[axis];
            let component = &self.components[component_index];
            let local_axis = scratch.prefix_lengths[component_index];
            let other = total
                .checked_div(scratch.completions[component_index])
                .ok_or(PauliError::InvalidSector {
                    context: "factorized zero completion",
                })?;
            let mut selected = None;
            for candidate in 0..component_local_dimension(component.index(), local_axis) {
                scratch.local_values[component_index][local_axis] = candidate as u64;
                let count = component
                    .index
                    .completion(&scratch.local_values[component_index][..=local_axis])?;
                let block = count.checked_mul(other).ok_or(PauliError::Overflow {
                    context: "unranking factorized sector",
                })?;
                if rank < block {
                    selected = Some((candidate, count));
                    break;
                }
                rank -= block;
            }
            let Some((selected, next)) = selected else {
                return Err(PauliError::InvalidSector {
                    context: "factorized rank",
                });
            };
            *output = selected as u64;
            scratch.completions[component_index] = next;
            scratch.prefix_lengths[component_index] += 1;
            total = other.checked_mul(next).ok_or(PauliError::Overflow {
                context: "counting factorized completion",
            })?;
        }
        Ok(())
    }

    fn prefix_has_completion(
        &self,
        values: &[u64],
        physical_prefix_len: usize,
        scratch: &mut FactorizedScratch,
    ) -> Result<bool, PauliError> {
        if values.len() != self.axis_components.len() || physical_prefix_len > values.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.axis_components.len(),
                actual: values.len(),
            });
        }
        let mut total = 1_u128;
        for (component_index, component) in self.components.iter().enumerate() {
            let prefix_len = component
                .axes
                .iter()
                .take_while(|&&axis| axis < physical_prefix_len)
                .count();
            for (slot, &axis) in scratch.local_values[component_index]
                .iter_mut()
                .zip(&component.axes)
                .take(prefix_len)
            {
                *slot = values[axis];
            }
            let count = component
                .index
                .completion(&scratch.local_values[component_index][..prefix_len])?;
            if count == 0 {
                return Ok(false);
            }
            total = total.checked_mul(count).ok_or(PauliError::Overflow {
                context: "counting factorized public prefix",
            })?;
        }
        Ok(total != 0)
    }

    /// Advance one valid physical mixed-radix state without a global unrank.
    pub(crate) fn advance_public_cursor(
        &self,
        values: &mut [u64],
        scratch: &mut FactorizedScratch,
    ) -> Result<bool, PauliError> {
        if values.len() != self.axis_components.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.axis_components.len(),
                actual: values.len(),
            });
        }
        if let [component] = &*self.components {
            if let ComponentIndex::FixedPopulation(index) = component.index() {
                if component.axes.iter().copied().eq(0..values.len()) {
                    for pivot in (0..values.len()).rev() {
                        let occupied_prefix =
                            values[..pivot].iter().filter(|&&value| value == 1).count();
                        if values[pivot] != 0 || occupied_prefix >= index.population {
                            continue;
                        }
                        values[pivot] = 1;
                        let remaining = index.population - occupied_prefix - 1;
                        values[pivot + 1..].fill(0);
                        let suffix_start = values.len() - remaining;
                        for value in values[suffix_start..].iter_mut() {
                            *value = 1;
                        }
                        return Ok(true);
                    }
                    return Ok(false);
                }
            }
        }
        for pivot in (0..values.len()).rev() {
            let component = self.axis_components[pivot];
            let local_axis = self.components[component]
                .axes
                .iter()
                .position(|&axis| axis == pivot)
                .ok_or(PauliError::InvalidSector {
                    context: "factorized cursor axis mapping",
                })?;
            let dimension =
                component_local_dimension(self.components[component].index(), local_axis);
            let original =
                usize::try_from(values[pivot]).map_err(|_| PauliError::InvalidIndex {
                    context: "factorized cursor occupation",
                })?;
            for candidate in original + 1..dimension {
                values[pivot] = candidate as u64;
                if !self.prefix_has_completion(values, pivot + 1, scratch)? {
                    continue;
                }
                for axis in pivot + 1..values.len() {
                    let component = self.axis_components[axis];
                    let local_axis = self.components[component]
                        .axes
                        .iter()
                        .position(|&value| value == axis)
                        .ok_or(PauliError::InvalidSector {
                            context: "factorized cursor axis mapping",
                        })?;
                    let dimension =
                        component_local_dimension(self.components[component].index(), local_axis);
                    let mut selected = None;
                    for value in 0..dimension {
                        values[axis] = value as u64;
                        if self.prefix_has_completion(values, axis + 1, scratch)? {
                            selected = Some(value);
                            break;
                        }
                    }
                    if selected.is_none() {
                        return Err(PauliError::InvalidSector {
                            context: "factorized cursor completion",
                        });
                    }
                }
                return Ok(true);
            }
            values[pivot] = original as u64;
        }
        Ok(false)
    }
}

fn component_local_dimension(index: &ComponentIndex, axis: usize) -> usize {
    match index {
        ComponentIndex::Cartesian(index) => index.dimensions[axis],
        ComponentIndex::FixedPopulation(_) => 2,
        ComponentIndex::FiniteAdditive(index) => index.dimensions[axis],
    }
}

impl FactorComponent {
    pub(crate) fn axes(&self) -> &[usize] {
        &self.axes
    }

    pub(crate) fn kind(&self) -> FactorComponentKind {
        self.kind
    }

    pub(crate) fn index(&self) -> &ComponentIndex {
        &self.index
    }
}

impl ComponentIndex {
    pub(crate) fn dimension(&self) -> usize {
        match self {
            Self::Cartesian(index) => index.dimension,
            Self::FixedPopulation(index) => index.dimension,
            Self::FiniteAdditive(index) => index.dimension,
        }
    }

    fn estimated_bytes(&self) -> u128 {
        match self {
            Self::Cartesian(index) => {
                ((index.dimensions.len() + index.strides.len()) * std::mem::size_of::<usize>())
                    as u128
            }
            Self::FixedPopulation(index) => {
                (std::mem::size_of::<FixedPopulationIndex>() as u128)
                    + index.combinations.as_ref().map_or(0, |values| {
                        (values.len() * std::mem::size_of::<u128>()) as u128
                    })
            }
            Self::FiniteAdditive(index) => index.estimated_bytes().unwrap_or(u128::MAX),
        }
    }

    pub(crate) fn rank(&self, values: &[u64]) -> Result<u64, PauliError> {
        match self {
            Self::Cartesian(index) => index.rank(values),
            Self::FixedPopulation(index) => index.rank(values),
            Self::FiniteAdditive(index) => index.rank(values),
        }
    }

    pub(crate) fn unrank_into(&self, rank: u64, values: &mut [u64]) -> Result<(), PauliError> {
        match self {
            Self::Cartesian(index) => index.unrank_into(rank, values),
            Self::FixedPopulation(index) => index.unrank_into(rank, values),
            Self::FiniteAdditive(index) => index.unrank_into(rank, values),
        }
    }

    fn completion(&self, prefix: &[u64]) -> Result<u128, PauliError> {
        match self {
            Self::Cartesian(index) => index.completion(prefix),
            Self::FixedPopulation(index) => index.completion(prefix),
            Self::FiniteAdditive(index) => index.completion(prefix),
        }
    }
}

impl CartesianIndex {
    fn new(dimensions: &[usize]) -> Option<Self> {
        let mut strides = vec![1; dimensions.len()];
        let mut dimension = 1usize;
        for axis in (0..dimensions.len()).rev() {
            strides[axis] = dimension;
            dimension = dimension.checked_mul(*dimensions.get(axis)?)?;
        }
        Some(Self {
            dimensions: dimensions.to_vec().into_boxed_slice(),
            strides: strides.into_boxed_slice(),
            dimension,
        })
    }

    fn rank(&self, values: &[u64]) -> Result<u64, PauliError> {
        if values.len() != self.dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimensions.len(),
                actual: values.len(),
            });
        }
        let mut rank = 0usize;
        for ((&value, &dimension), &stride) in
            values.iter().zip(&self.dimensions).zip(&self.strides)
        {
            let value = usize::try_from(value).map_err(|_| PauliError::InvalidIndex {
                context: "component occupation",
            })?;
            if value >= dimension {
                return Err(PauliError::InvalidIndex {
                    context: "component occupation",
                });
            }
            rank = rank
                .checked_add(value.checked_mul(stride).ok_or(PauliError::Overflow {
                    context: "ranking Cartesian component",
                })?)
                .ok_or(PauliError::Overflow {
                    context: "ranking Cartesian component",
                })?;
        }
        u64::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting component rank",
        })
    }

    fn unrank_into(&self, rank: u64, values: &mut [u64]) -> Result<(), PauliError> {
        let mut rank = usize::try_from(rank).map_err(|_| PauliError::InvalidIndex {
            context: "component rank",
        })?;
        if rank >= self.dimension || values.len() != self.dimensions.len() {
            return Err(PauliError::InvalidIndex {
                context: "component rank",
            });
        }
        for ((value, &dimension), &stride) in
            values.iter_mut().zip(&self.dimensions).zip(&self.strides)
        {
            *value = ((rank / stride) % dimension) as u64;
            rank %= stride.checked_mul(dimension).ok_or(PauliError::Overflow {
                context: "unranking Cartesian component",
            })?;
        }
        Ok(())
    }

    fn completion(&self, prefix: &[u64]) -> Result<u128, PauliError> {
        if prefix.len() > self.dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimensions.len(),
                actual: prefix.len(),
            });
        }
        let mut count = 1_u128;
        for (axis, &dimension) in self.dimensions.iter().enumerate() {
            if axis < prefix.len() && prefix[axis] >= dimension as u64 {
                return Err(PauliError::InvalidIndex {
                    context: "Cartesian component prefix",
                });
            }
            if axis >= prefix.len() {
                count = count
                    .checked_mul(dimension as u128)
                    .ok_or(PauliError::Overflow {
                        context: "counting Cartesian completion",
                    })?;
            }
        }
        Ok(count)
    }
}

impl FixedPopulationIndex {
    fn new(sites: usize, population: usize, max_bytes: u128) -> Option<Self> {
        let dimension = usize::try_from(binomial_count(sites, population)?).ok()?;
        let combination_width = population.checked_add(1)?;
        // The compact Pascal table accelerates direct rank/unrank without
        // retaining any selected states. Its small internal cap is independent
        // of the public best-effort memory budget.
        let cache_budget = max_bytes.max(1024 * 1024 * 1024);
        let combinations = Self::combination_cache(sites, combination_width, cache_budget);
        Some(Self {
            sites,
            population,
            dimension,
            combinations,
            combination_width,
        })
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn sites(&self) -> usize {
        self.sites
    }

    fn combination_cache(sites: usize, width: usize, max_bytes: u128) -> Option<Box<[u128]>> {
        const COMBINATION_CACHE_LIMIT: u128 = 1024 * 1024;
        let entries = sites.checked_add(1)?.checked_mul(width)?;
        let bytes = (entries as u128).checked_mul(std::mem::size_of::<u128>() as u128)?;
        if bytes > COMBINATION_CACHE_LIMIT || bytes > max_bytes {
            return None;
        }
        let mut values = vec![0_u128; entries];
        for site in 0..=sites {
            values[site * width] = 1;
            for population in 1..width.min(site.saturating_add(1)) {
                values[site * width + population] = values[(site - 1) * width + population - 1]
                    .checked_add(values[(site - 1) * width + population])?;
            }
            if site < width {
                values[site * width + site] = 1;
            }
        }
        Some(values.into_boxed_slice())
    }

    #[inline]
    fn combination(&self, sites: usize, population: usize) -> Option<u128> {
        if population > sites {
            return Some(0);
        }
        self.combinations
            .as_ref()
            .and_then(|values| {
                (population < self.combination_width)
                    .then(|| values[sites * self.combination_width + population])
            })
            .or_else(|| binomial_count(sites, population))
    }

    #[inline]
    pub(crate) fn rank_contiguous_binary_word(&self, word: usize) -> Result<usize, PauliError> {
        if self.sites > usize::BITS as usize
            || (self.sites < usize::BITS as usize && word >> self.sites != 0)
        {
            return Err(PauliError::InvalidIndex {
                context: "fixed-population binary occupation",
            });
        }
        let mut needed = self.population;
        let mut rank = 0_u128;
        for axis in 0..self.sites {
            let bit = 1_usize << (self.sites - axis - 1);
            if word & bit == 0 {
                continue;
            }
            let remaining = self.sites - axis - 1;
            rank =
                rank.checked_add(self.combination(remaining, needed).ok_or(
                    PauliError::Overflow {
                        context: "ranking fixed-population binary occupation",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "ranking fixed-population binary occupation",
                })?;
            needed = needed.checked_sub(1).ok_or(PauliError::InvalidSector {
                context: "fixed-population binary occupation",
            })?;
        }
        if needed != 0 {
            return Err(PauliError::InvalidSector {
                context: "fixed-population binary occupation",
            });
        }
        usize::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting fixed-population binary rank",
        })
    }

    #[inline]
    pub(crate) fn unrank_contiguous_binary_word(&self, rank: usize) -> Result<usize, PauliError> {
        if self.sites > usize::BITS as usize || rank >= self.dimension {
            return Err(PauliError::InvalidIndex {
                context: "fixed-population binary rank",
            });
        }
        let mut word = 0_usize;
        let mut remaining_rank = rank as u128;
        let mut needed = self.population;
        for axis in 0..self.sites {
            let zero_count =
                self.combination(self.sites - axis - 1, needed)
                    .ok_or(PauliError::Overflow {
                        context: "unranking fixed-population binary occupation",
                    })?;
            if remaining_rank < zero_count {
                continue;
            }
            remaining_rank -= zero_count;
            word |= 1_usize << (self.sites - axis - 1);
            needed = needed.checked_sub(1).ok_or(PauliError::InvalidSector {
                context: "fixed-population binary rank",
            })?;
        }
        Ok(word)
    }

    fn rank(&self, values: &[u64]) -> Result<u64, PauliError> {
        if values.len() != self.sites {
            return Err(PauliError::InvalidStructureLength {
                expected: self.sites,
                actual: values.len(),
            });
        }
        let mut needed = self.population;
        let mut rank = 0_u128;
        for (axis, &value) in values.iter().enumerate() {
            if value > 1 {
                return Err(PauliError::InvalidIndex {
                    context: "fixed-population occupation",
                });
            }
            let remaining = self.sites - axis - 1;
            if value == 1 {
                rank = rank
                    .checked_add(self.combination(remaining, needed).ok_or(
                        PauliError::Overflow {
                            context: "ranking fixed-population component",
                        },
                    )?)
                    .ok_or(PauliError::Overflow {
                        context: "ranking fixed-population component",
                    })?;
                needed = needed.checked_sub(1).ok_or(PauliError::InvalidSector {
                    context: "fixed-population occupation",
                })?;
            }
        }
        if needed != 0 {
            return Err(PauliError::InvalidSector {
                context: "fixed-population occupation",
            });
        }
        u64::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting component rank",
        })
    }

    fn unrank_into(&self, rank: u64, values: &mut [u64]) -> Result<(), PauliError> {
        if values.len() != self.sites
            || usize::try_from(rank).map_or(true, |value| value >= self.dimension)
        {
            return Err(PauliError::InvalidIndex {
                context: "fixed-population rank",
            });
        }
        let mut rank = u128::from(rank);
        let mut needed = self.population;
        for (axis, value) in values.iter_mut().enumerate().take(self.sites) {
            let zero_count =
                self.combination(self.sites - axis - 1, needed)
                    .ok_or(PauliError::Overflow {
                        context: "unranking fixed-population component",
                    })?;
            if rank < zero_count {
                *value = 0;
            } else {
                *value = 1;
                rank -= zero_count;
                needed = needed.checked_sub(1).ok_or(PauliError::InvalidSector {
                    context: "fixed-population rank",
                })?;
            }
        }
        Ok(())
    }

    fn completion(&self, prefix: &[u64]) -> Result<u128, PauliError> {
        if prefix.len() > self.sites || prefix.iter().any(|&value| value > 1) {
            return Err(PauliError::InvalidIndex {
                context: "fixed-population prefix",
            });
        }
        let occupied = prefix.iter().filter(|&&value| value == 1).count();
        let Some(needed) = self.population.checked_sub(occupied) else {
            return Ok(0);
        };
        self.combination(self.sites - prefix.len(), needed)
            .ok_or(PauliError::Overflow {
                context: "counting fixed-population completion",
            })
    }
}

impl FiniteAdditiveIndex {
    fn new(
        dimensions: &[usize],
        contributions: Vec<Vec<[i64; 4]>>,
        target: [i64; 4],
        rows: usize,
        max_bytes: u128,
    ) -> Option<Self> {
        if rows == 0 || rows > 4 {
            return None;
        }
        let (suffix_counts, dimension) = if rows == 1 {
            let mut counts = (0..=dimensions.len())
                .map(|_| FxHashMap::default())
                .collect::<Vec<FxHashMap<i64, u128>>>();
            let mut retained_bytes = 0_u128;
            counts[dimensions.len()].insert(0_i64, 1);
            for axis in (0..dimensions.len()).rev() {
                let mut table = FxHashMap::default();
                for contribution in contributions.get(axis)? {
                    for (&remainder, &count) in &counts[axis + 1] {
                        let key = contribution[0].checked_add(remainder)?;
                        let entry = table.entry(key).or_insert(0_u128);
                        *entry = (*entry).checked_add(count)?;
                    }
                }
                retained_bytes =
                    retained_bytes.checked_add((table.len() as u128).checked_mul(32)?)?;
                if retained_bytes > max_bytes {
                    return None;
                }
                counts[axis] = table;
            }
            let dimension = usize::try_from(*counts[0].get(&target[0])?).ok()?;
            (
                FiniteAdditiveSuffixCounts::Scalar(counts.into_boxed_slice()),
                dimension,
            )
        } else {
            let zero = [0_i64; 4];
            let mut counts = (0..=dimensions.len())
                .map(|_| FxHashMap::default())
                .collect::<Vec<FxHashMap<[i64; 4], u128>>>();
            let mut retained_bytes = 0_u128;
            counts[dimensions.len()].insert(zero, 1);
            for axis in (0..dimensions.len()).rev() {
                let mut table = FxHashMap::default();
                for contribution in contributions.get(axis)? {
                    for (remainder, &count) in &counts[axis + 1] {
                        let mut key = [0_i64; 4];
                        for row in 0..rows {
                            key[row] = contribution[row].checked_add(remainder[row])?;
                        }
                        let entry = table.entry(key).or_insert(0_u128);
                        *entry = (*entry).checked_add(count)?;
                    }
                }
                retained_bytes =
                    retained_bytes.checked_add((table.len() as u128).checked_mul(64)?)?;
                if retained_bytes > max_bytes {
                    return None;
                }
                counts[axis] = table;
            }
            let dimension = usize::try_from(*counts[0].get(&target)?).ok()?;
            (
                FiniteAdditiveSuffixCounts::Wide(counts.into_boxed_slice()),
                dimension,
            )
        };
        Some(Self {
            dimensions: dimensions.to_vec().into_boxed_slice(),
            contributions: contributions
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            target,
            rows,
            suffix_counts,
            dimension,
        })
    }

    fn suffix_count(&self, position: usize, remainder: &[i64; 4]) -> u128 {
        match &self.suffix_counts {
            FiniteAdditiveSuffixCounts::Scalar(counts) => {
                counts[position].get(&remainder[0]).copied().unwrap_or(0)
            }
            FiniteAdditiveSuffixCounts::Wide(counts) => {
                counts[position].get(remainder).copied().unwrap_or(0)
            }
        }
    }

    fn estimated_bytes(&self) -> Option<u128> {
        let dimensions_bytes =
            (self.dimensions.len() as u128).checked_mul(std::mem::size_of::<usize>() as u128)?;
        let contribution_entries = self
            .contributions
            .iter()
            .try_fold(0_u128, |total, levels| {
                total.checked_add(levels.len() as u128)
            })?;
        let contributions_bytes =
            contribution_entries.checked_mul(std::mem::size_of::<[i64; 4]>() as u128)?;
        let suffix_bytes = match &self.suffix_counts {
            FiniteAdditiveSuffixCounts::Scalar(counts) => {
                counts.iter().try_fold(0_u128, |total, table| {
                    total.checked_add((table.len() as u128).checked_mul(32)?)
                })?
            }
            FiniteAdditiveSuffixCounts::Wide(counts) => {
                counts.iter().try_fold(0_u128, |total, table| {
                    total.checked_add((table.len() as u128).checked_mul(64)?)
                })?
            }
        };
        (std::mem::size_of::<FiniteAdditiveIndex>() as u128)
            .checked_add(dimensions_bytes)?
            .checked_add(contributions_bytes)?
            .checked_add(suffix_bytes)
    }

    #[cfg(test)]
    fn uses_scalar_remainders(&self) -> bool {
        matches!(self.suffix_counts, FiniteAdditiveSuffixCounts::Scalar(_))
    }

    fn rank(&self, values: &[u64]) -> Result<u64, PauliError> {
        if values.len() != self.dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimensions.len(),
                actual: values.len(),
            });
        }
        let mut remainder = self.target;
        let mut rank = 0_u128;
        for (axis, &value) in values.iter().enumerate() {
            let value = usize::try_from(value).map_err(|_| PauliError::InvalidIndex {
                context: "finite-additive occupation",
            })?;
            if value >= self.dimensions[axis] {
                return Err(PauliError::InvalidIndex {
                    context: "finite-additive occupation",
                });
            }
            for candidate in 0..value {
                let next = subtract_component(
                    &remainder,
                    &self.contributions[axis][candidate],
                    self.rows,
                )?;
                rank = rank.checked_add(self.suffix_count(axis + 1, &next)).ok_or(
                    PauliError::Overflow {
                        context: "ranking finite-additive component",
                    },
                )?;
            }
            remainder =
                subtract_component(&remainder, &self.contributions[axis][value], self.rows)?;
        }
        if remainder[..self.rows].iter().any(|&value| value != 0) {
            return Err(PauliError::InvalidSector {
                context: "finite-additive occupation",
            });
        }
        u64::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting component rank",
        })
    }

    fn unrank_into(&self, rank: u64, values: &mut [u64]) -> Result<(), PauliError> {
        if values.len() != self.dimensions.len()
            || usize::try_from(rank).map_or(true, |value| value >= self.dimension)
        {
            return Err(PauliError::InvalidIndex {
                context: "finite-additive rank",
            });
        }
        let mut rank = u128::from(rank);
        let mut remainder = self.target;
        for (axis, value) in values.iter_mut().enumerate().take(self.dimensions.len()) {
            let mut selected = None;
            for candidate in 0..self.dimensions[axis] {
                let next = subtract_component(
                    &remainder,
                    &self.contributions[axis][candidate],
                    self.rows,
                )?;
                let count = self.suffix_count(axis + 1, &next);
                if rank < count {
                    selected = Some((candidate, next));
                    break;
                }
                rank -= count;
            }
            let Some((candidate, next)) = selected else {
                return Err(PauliError::InvalidSector {
                    context: "finite-additive rank",
                });
            };
            *value = candidate as u64;
            remainder = next;
        }
        Ok(())
    }

    fn completion(&self, prefix: &[u64]) -> Result<u128, PauliError> {
        if prefix.len() > self.dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimensions.len(),
                actual: prefix.len(),
            });
        }
        let mut remainder = self.target;
        for (axis, &value) in prefix.iter().enumerate() {
            let value = usize::try_from(value).map_err(|_| PauliError::InvalidIndex {
                context: "finite-additive prefix",
            })?;
            let contribution =
                self.contributions[axis]
                    .get(value)
                    .ok_or(PauliError::InvalidIndex {
                        context: "finite-additive prefix",
                    })?;
            remainder = subtract_component(&remainder, contribution, self.rows)?;
        }
        Ok(self.suffix_count(prefix.len(), &remainder))
    }
}

fn subtract_component(
    left: &[i64; 4],
    right: &[i64; 4],
    rows: usize,
) -> Result<[i64; 4], PauliError> {
    let mut output = [0_i64; 4];
    for row in 0..rows {
        output[row] = left[row]
            .checked_sub(right[row])
            .ok_or(PauliError::Overflow {
                context: "computing component remainder",
            })?;
    }
    Ok(output)
}

fn binomial_count(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut value = 1_u128;
    for step in 0..k {
        value = value.checked_mul((n - step) as u128)? / (step + 1) as u128;
    }
    Some(value)
}

impl ChargeSectorPlan {
    /// Return the finite local dimensions used by the mixed-radix basis.
    pub fn local_dimensions(&self) -> &[usize] {
        &self.local_dimensions
    }

    /// Return the number of selected basis states.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return the number of simultaneous charge constraints.
    pub fn constraint_count(&self) -> usize {
        self.target.len()
    }

    /// Return the best-effort plan-size estimate.
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    /// Return the compact factor proof when small checked analysis found one.
    ///
    /// It is intentionally crate-private: factor components are execution
    /// coordinates, never a public charge-sector abstraction.
    pub(crate) fn factorized_layout(&self) -> Option<&FactorizedLayout> {
        self.factorized_layout.as_ref()
    }

    /// Return the factor proof only when it also defines a factorized
    /// execution.  A single irreducible multi-row finite-additive component
    /// deliberately uses the original General suffix-count executor.
    pub(crate) fn factorized_execution_layout(&self) -> Option<&FactorizedLayout> {
        self.factorized_layout
            .as_ref()
            .filter(|layout| !layout.is_single_coupled_component())
    }

    /// Return the lexicographic rank of one selected occupation.
    pub fn rank(&self, occupations: &[u64]) -> Result<u64, PauliError> {
        if occupations.len() != self.local_dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.local_dimensions.len(),
                actual: occupations.len(),
            });
        }
        if let Some(layout) = &self.factorized_layout {
            return layout.rank(occupations);
        }
        let mut remaining = vec![0_i128; self.target.len()];
        let mut candidate_remaining = vec![0_i128; self.target.len()];
        self.rank_into(occupations, &mut remaining, &mut candidate_remaining)
    }

    /// Rank a validated-length occupation using caller-owned scratch buffers.
    pub fn rank_into<'a>(
        &self,
        occupations: &[u64],
        mut remaining: &'a mut [i128],
        mut candidate_remaining: &'a mut [i128],
    ) -> Result<u64, PauliError> {
        if occupations.len() != self.local_dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.local_dimensions.len(),
                actual: occupations.len(),
            });
        }
        if remaining.len() != self.target.len() || candidate_remaining.len() != self.target.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.target.len(),
                actual: remaining.len(),
            });
        }
        remaining.copy_from_slice(&self.target);
        let mut rank = 0_u128;
        for (position, &raw_value) in occupations.iter().enumerate() {
            let value = usize::try_from(raw_value).map_err(|_| PauliError::InvalidIndex {
                context: "occupation is outside the finite sector layout",
            })?;
            if value >= self.local_dimensions[position] {
                return Err(PauliError::InvalidIndex {
                    context: "occupation is outside the finite sector layout",
                });
            }
            for candidate in 0..value {
                subtract_contribution_into(
                    remaining,
                    &self.contributions[position][candidate],
                    candidate_remaining,
                )?;
                rank = rank
                    .checked_add(
                        self.suffix_counts[position + 1]
                            .get(&*candidate_remaining)
                            .copied()
                            .unwrap_or(0),
                    )
                    .ok_or(PauliError::Overflow {
                        context: "computing charge-sector rank",
                    })?;
            }
            subtract_contribution_into(
                remaining,
                &self.contributions[position][value],
                candidate_remaining,
            )?;
            std::mem::swap(&mut remaining, &mut candidate_remaining);
        }
        if remaining.iter().any(|&value| value != 0) {
            return Err(PauliError::InvalidSector {
                context: "occupation does not satisfy every charge constraint",
            });
        }
        u64::try_from(rank).map_err(|_| PauliError::Overflow {
            context: "converting charge-sector rank",
        })
    }

    /// Return one selected occupation by lexicographic rank.
    pub fn unrank(&self, index: u64) -> Result<Vec<u64>, PauliError> {
        if u128::from(index) >= self.dimension as u128 {
            return Err(PauliError::InvalidIndex {
                context: "sector index is out of range",
            });
        }
        let mut values = vec![0_u64; self.local_dimensions.len()];
        if let Some(layout) = &self.factorized_layout {
            layout.unrank_into(index, &mut values)?;
            return Ok(values);
        }
        let mut remaining = vec![0_i128; self.target.len()];
        let mut candidate_remaining = vec![0_i128; self.target.len()];
        self.unrank_into_with_scratch(
            index,
            &mut values,
            &mut remaining,
            &mut candidate_remaining,
        )?;
        Ok(values)
    }

    /// Unrank using caller-owned occupation and remainder scratch buffers.
    pub fn unrank_into_with_scratch(
        &self,
        index: u64,
        values: &mut [u64],
        remaining: &mut [i128],
        candidate_remaining: &mut [i128],
    ) -> Result<(), PauliError> {
        if values.len() != self.local_dimensions.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.local_dimensions.len(),
                actual: values.len(),
            });
        }
        if remaining.len() != self.target.len() || candidate_remaining.len() != self.target.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.target.len(),
                actual: remaining.len(),
            });
        }
        remaining.copy_from_slice(&self.target);
        let mut index = u128::from(index);
        for (position, &dimension) in self.local_dimensions.iter().enumerate() {
            let mut selected = None;
            for candidate in 0..dimension {
                subtract_contribution_into(
                    remaining,
                    &self.contributions[position][candidate],
                    candidate_remaining,
                )?;
                let count = self.suffix_counts[position + 1]
                    .get(&*candidate_remaining)
                    .copied()
                    .unwrap_or(0);
                if index < count {
                    selected = Some(candidate);
                    break;
                }
                index -= count;
            }
            let Some(candidate) = selected else {
                return Err(PauliError::InvalidSector {
                    context: "charge-sector rank/unrank plan is inconsistent",
                });
            };
            values[position] = u64::try_from(candidate).map_err(|_| PauliError::Overflow {
                context: "converting charge-sector occupation",
            })?;
            remaining.copy_from_slice(candidate_remaining);
        }
        Ok(())
    }

    /// Materialize the selected occupations in one flat row-major buffer.
    pub fn basis_states(&self, max_bytes: u128) -> Result<Vec<u64>, PauliError> {
        let axis_count = self.local_dimensions.len();
        let requested = (self.dimension as u128)
            .checked_mul(axis_count.max(1) as u128)
            .and_then(|value| value.checked_mul(8))
            .ok_or(PauliError::Overflow {
                context: "estimating charge-sector basis states",
            })?;
        if requested > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }
        let output_len = self
            .dimension
            .checked_mul(axis_count)
            .ok_or(PauliError::Overflow {
                context: "allocating charge-sector basis states",
            })?;
        let mut output = Vec::with_capacity(output_len);
        let mut values = vec![0_u64; axis_count];
        let mut remaining = vec![0_i128; self.target.len()];
        let mut candidate_remaining = vec![0_i128; self.target.len()];
        for index in 0..self.dimension {
            self.unrank_into_with_scratch(
                index as u64,
                &mut values,
                &mut remaining,
                &mut candidate_remaining,
            )?;
            output.extend_from_slice(&values);
        }
        Ok(output)
    }
}

/// Build a checked suffix-count dynamic-programming plan.
pub fn build_charge_sector_plan(
    local_dimensions: Vec<usize>,
    contributions: Vec<Vec<Vec<i128>>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> Result<ChargeSectorPlan, PauliError> {
    if contributions.len() != local_dimensions.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: local_dimensions.len(),
            actual: contributions.len(),
        });
    }
    if target.is_empty() {
        return Err(PauliError::InvalidSector {
            context: "charge-sector requires at least one charge constraint",
        });
    }
    let contribution_entries = local_dimensions
        .iter()
        .try_fold(0_u128, |total, &dimension| {
            total
                .checked_add(dimension as u128)
                .ok_or(PauliError::Overflow {
                    context: "estimating charge-sector contributions",
                })
        })?;
    let contribution_bytes = contribution_entries
        .checked_mul(target.len() as u128)
        .and_then(|value| value.checked_mul(CHARGE_CONTRIBUTION_BYTES))
        .and_then(|value| value.checked_add((local_dimensions.len() as u128).checked_mul(64)?))
        .and_then(|value| value.checked_add((target.len() as u128).checked_mul(16)?))
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector contributions",
        })?;
    if contribution_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: contribution_bytes,
            limit: max_bytes,
        });
    }
    for (dimension, table) in local_dimensions.iter().zip(&contributions) {
        if table.len() != *dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: *dimension,
                actual: table.len(),
            });
        }
        if table
            .iter()
            .any(|contribution| contribution.len() != target.len())
        {
            return Err(PauliError::InvalidStructureLength {
                expected: target.len(),
                actual: table
                    .iter()
                    .find(|contribution| contribution.len() != target.len())
                    .map_or(0, Vec::len),
            });
        }
    }

    let zero = vec![0_i128; target.len()];
    // Map iteration order is never observable: candidate traversal defines the
    // public lexicographic order, while lookup is the DP hot path.
    let mut suffix_counts = (0..=local_dimensions.len())
        .map(|_| FxHashMap::default())
        .collect::<Vec<FxHashMap<Vec<i128>, u128>>>();
    suffix_counts[local_dimensions.len()].insert(zero, 1);
    let entry_bytes = 64_u128
        .checked_add(
            24_u128
                .checked_mul(target.len() as u128)
                .ok_or(PauliError::Overflow {
                    context: "estimating charge-sector dynamic-programming plan",
                })?,
        )
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector dynamic-programming plan",
        })?;
    for position in (0..local_dimensions.len()).rev() {
        let mut table = FxHashMap::default();
        for contribution in &contributions[position] {
            for (remainder, &count) in &suffix_counts[position + 1] {
                let key = contribution
                    .iter()
                    .zip(remainder)
                    .map(|(&left, &right)| {
                        left.checked_add(right).ok_or(PauliError::Overflow {
                            context: "computing charge-sector contribution",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let entry = table.entry(key).or_insert(0_u128);
                *entry = entry.checked_add(count).ok_or(PauliError::Overflow {
                    context: "counting charge-sector basis states",
                })?;
            }
        }
        check_plan_bytes(table.len(), entry_bytes, max_bytes)?;
        suffix_counts[position] = table;
    }

    let dimension_count = suffix_counts[0].get(&target).copied().unwrap_or(0);
    if dimension_count > isize::MAX as u128 {
        return Err(PauliError::Overflow {
            context: "converting charge-sector dimension to platform index",
        });
    }
    let dimension = usize::try_from(dimension_count).map_err(|_| PauliError::Overflow {
        context: "converting charge-sector dimension",
    })?;
    let table_bytes = suffix_counts.iter().try_fold(0_u128, |total, table| {
        let table_bytes =
            (table.len() as u128)
                .checked_mul(entry_bytes)
                .ok_or(PauliError::Overflow {
                    context: "estimating charge-sector plan",
                })?;
        total.checked_add(table_bytes).ok_or(PauliError::Overflow {
            context: "estimating charge-sector plan",
        })
    })?;
    // Factor discovery is deliberately bounded and conservative.  Failure to
    // fit the small integer representation never changes the exact selected
    // set: the suffix-count plan below remains the General authority.
    let factorized_layout =
        analyze_factorized_layout(&local_dimensions, &contributions, &target, max_bytes);
    let layout_bytes = factorized_layout
        .as_ref()
        .map_or(0, FactorizedLayout::estimated_bytes);
    let estimated_bytes = contribution_bytes
        .checked_add(table_bytes)
        .and_then(|value| value.checked_add((local_dimensions.len() as u128).checked_mul(8)?))
        .and_then(|value| value.checked_add(layout_bytes))
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    Ok(ChargeSectorPlan {
        local_dimensions,
        contributions,
        target,
        suffix_counts,
        factorized_layout,
        dimension,
        estimated_bytes,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReducedChargeRow {
    coefficients: Vec<i64>,
    target: i64,
}

/// Prove a support-factor partition using only bounded integer operations.
///
/// This is not a symbolic algebra layer.  It handles the ordinary small
/// charge matrices emitted by the public API and declines everything whose
/// checked representation would need wider arithmetic.  The caller keeps the
/// exact pre-existing plan in every decline case.
fn analyze_factorized_layout(
    local_dimensions: &[usize],
    contributions: &[Vec<Vec<i128>>],
    target: &[i128],
    max_bytes: u128,
) -> Option<FactorizedLayout> {
    if target.is_empty() || target.len() > 4 || contributions.len() != local_dimensions.len() {
        return None;
    }
    let contrast_count = local_dimensions
        .iter()
        .try_fold(0usize, |total, &dimension| {
            total.checked_add(dimension.checked_sub(1)?)
        })?;
    let row_entries = target.len().checked_mul(contrast_count.checked_add(1)?)?;
    let workspace_bytes = (row_entries as u128).checked_mul(std::mem::size_of::<i64>() as u128)?;
    if workspace_bytes > max_bytes {
        return None;
    }

    let mut axis_columns = Vec::with_capacity(local_dimensions.len());
    let mut next_column = 0usize;
    for &dimension in local_dimensions {
        axis_columns.push(next_column);
        next_column = next_column.checked_add(dimension.checked_sub(1)?)?;
    }
    let mut rows = Vec::with_capacity(target.len());
    for (constraint, &requested_target) in target.iter().enumerate() {
        let mut baseline = 0_i128;
        for table in contributions {
            baseline = baseline.checked_add(*table.first()?.get(constraint)?)?;
        }
        let adjusted_target = requested_target.checked_sub(baseline)?;
        let mut coefficients = vec![0_i64; contrast_count];
        for (axis, table) in contributions.iter().enumerate() {
            let reference = *table.first()?.get(constraint)?;
            for level in 1..local_dimensions[axis] {
                let contrast = table.get(level)?.get(constraint)?.checked_sub(reference)?;
                coefficients[axis_columns[axis] + level - 1] = i64::try_from(contrast).ok()?;
            }
        }
        rows.push(ReducedChargeRow {
            coefficients,
            target: i64::try_from(adjusted_target).ok()?,
        });
    }

    let mut active = Vec::with_capacity(rows.len());
    for row in rows {
        if row.coefficients.iter().all(|&coefficient| coefficient == 0) {
            // The exact DP will produce dimension zero for this case.  Keep
            // it as an ordinary empty General plan rather than inventing an
            // alternative sector object.
            if row.target != 0 {
                return None;
            }
        } else {
            active.push(row);
        }
    }
    if active.is_empty() {
        return cartesian_layout(local_dimensions, max_bytes);
    }
    if active.len() == 1 || pairwise_disjoint_supports(&active, local_dimensions, &axis_columns) {
        return layout_from_rows(active, local_dimensions, &axis_columns, max_bytes);
    }

    reduce_charge_rows(&mut active)?;
    let mut reduced = Vec::with_capacity(active.len());
    for row in active {
        if row.coefficients.iter().all(|&coefficient| coefficient == 0) {
            if row.target != 0 {
                return None;
            }
        } else {
            reduced.push(row);
        }
    }
    if reduced.is_empty() {
        return cartesian_layout(local_dimensions, max_bytes);
    }
    layout_from_rows(reduced, local_dimensions, &axis_columns, max_bytes)
}

fn reduce_charge_rows(rows: &mut Vec<ReducedChargeRow>) -> Option<()> {
    let width = rows.first()?.coefficients.len();
    let mut pivot_row = 0usize;
    for column in 0..width {
        let Some(selected) = (pivot_row..rows.len())
            .filter(|&row| rows[row].coefficients[column] != 0)
            .max_by_key(|&row| {
                (
                    i128::from(rows[row].coefficients[column]).abs(),
                    std::cmp::Reverse(row),
                )
            })
        else {
            continue;
        };
        rows.swap(pivot_row, selected);
        primitive_normalize(&mut rows[pivot_row])?;
        let pivot = rows[pivot_row].clone();
        let mut row = 0usize;
        while row < rows.len() {
            if row == pivot_row || rows[row].coefficients[column] == 0 {
                row += 1;
                continue;
            }
            let coefficient = rows[row].coefficients[column];
            let pivot_coefficient = pivot.coefficients[column];
            let mut combined = Vec::with_capacity(width);
            for (&value, &pivot_value) in rows[row].coefficients.iter().zip(&pivot.coefficients) {
                let next = i128::from(pivot_coefficient)
                    .checked_mul(i128::from(value))?
                    .checked_sub(i128::from(coefficient).checked_mul(i128::from(pivot_value))?)?;
                combined.push(i64::try_from(next).ok()?);
            }
            let target = i128::from(pivot_coefficient)
                .checked_mul(i128::from(rows[row].target))?
                .checked_sub(i128::from(coefficient).checked_mul(i128::from(pivot.target))?)?;
            rows[row] = ReducedChargeRow {
                coefficients: combined,
                target: i64::try_from(target).ok()?,
            };
            primitive_normalize(&mut rows[row])?;
            row += 1;
        }
        pivot_row += 1;
        if pivot_row == rows.len() {
            break;
        }
    }
    rows.sort_by(|left, right| {
        let left_pivot = left
            .coefficients
            .iter()
            .position(|&coefficient| coefficient != 0)
            .unwrap_or(usize::MAX);
        let right_pivot = right
            .coefficients
            .iter()
            .position(|&coefficient| coefficient != 0)
            .unwrap_or(usize::MAX);
        left_pivot
            .cmp(&right_pivot)
            .then_with(|| left.coefficients.cmp(&right.coefficients))
            .then_with(|| left.target.cmp(&right.target))
    });
    rows.dedup_by(|left, right| {
        left.coefficients == right.coefficients && left.target == right.target
    });
    Some(())
}

fn primitive_normalize(row: &mut ReducedChargeRow) -> Option<()> {
    let mut divisor = 0_i128;
    for &coefficient in &row.coefficients {
        divisor = gcd_i128(divisor, i128::from(coefficient));
    }
    divisor = gcd_i128(divisor, i128::from(row.target));
    if divisor > 1 {
        for coefficient in &mut row.coefficients {
            *coefficient = i64::try_from(i128::from(*coefficient) / divisor).ok()?;
        }
        row.target = i64::try_from(i128::from(row.target) / divisor).ok()?;
    }
    if row
        .coefficients
        .iter()
        .find(|&&coefficient| coefficient != 0)
        .is_some_and(|&coefficient| coefficient < 0)
    {
        for coefficient in &mut row.coefficients {
            *coefficient = coefficient.checked_neg()?;
        }
        row.target = row.target.checked_neg()?;
    }
    Some(())
}

fn gcd_i128(left: i128, right: i128) -> i128 {
    let mut left = left.abs();
    let mut right = right.abs();
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn pairwise_disjoint_supports(
    rows: &[ReducedChargeRow],
    dimensions: &[usize],
    axis_columns: &[usize],
) -> bool {
    let supports = rows
        .iter()
        .map(|row| row_axis_support(row, dimensions, axis_columns))
        .collect::<Vec<_>>();
    for left in 0..supports.len() {
        for right in left + 1..supports.len() {
            if supports[left]
                .iter()
                .any(|axis| supports[right].contains(axis))
            {
                return false;
            }
        }
    }
    true
}

fn row_axis_support(
    row: &ReducedChargeRow,
    dimensions: &[usize],
    axis_columns: &[usize],
) -> Vec<usize> {
    dimensions
        .iter()
        .enumerate()
        .filter_map(|(axis, &dimension)| {
            let start = axis_columns[axis];
            row.coefficients[start..start + dimension - 1]
                .iter()
                .any(|&coefficient| coefficient != 0)
                .then_some(axis)
        })
        .collect()
}

fn layout_from_rows(
    rows: Vec<ReducedChargeRow>,
    dimensions: &[usize],
    axis_columns: &[usize],
    max_bytes: u128,
) -> Option<FactorizedLayout> {
    let supports = rows
        .iter()
        .map(|row| row_axis_support(row, dimensions, axis_columns))
        .collect::<Vec<_>>();
    let mut visited = vec![false; rows.len()];
    let mut groups = Vec::<(Vec<usize>, Vec<usize>)>::new();
    for start in 0..rows.len() {
        if visited[start] {
            continue;
        }
        let mut group_rows = vec![start];
        let mut axes = supports[start].clone();
        visited[start] = true;
        let mut cursor = 0usize;
        while cursor < group_rows.len() {
            let row = group_rows[cursor];
            for candidate in 0..rows.len() {
                if visited[candidate]
                    || !supports[row]
                        .iter()
                        .any(|axis| supports[candidate].contains(axis))
                {
                    continue;
                }
                visited[candidate] = true;
                group_rows.push(candidate);
                for axis in &supports[candidate] {
                    if !axes.contains(axis) {
                        axes.push(*axis);
                    }
                }
            }
            cursor += 1;
        }
        axes.sort_unstable();
        groups.push((axes, group_rows));
    }
    let mut components = Vec::new();
    let mut covered = vec![false; dimensions.len()];
    for (axes, group_rows) in groups {
        for &axis in &axes {
            covered[axis] = true;
        }
        let kind = if group_rows.len() == 1 {
            fixed_population_kind(&rows[group_rows[0]], &axes, dimensions, axis_columns)
                .unwrap_or(FactorComponentKind::FiniteAdditive)
        } else {
            FactorComponentKind::FiniteAdditive
        };
        let component_dimensions = axes
            .iter()
            .map(|&axis| dimensions[axis])
            .collect::<Vec<_>>();
        let index = match kind {
            FactorComponentKind::Cartesian => {
                ComponentIndex::Cartesian(CartesianIndex::new(&component_dimensions)?)
            }
            FactorComponentKind::FixedPopulation { population } => ComponentIndex::FixedPopulation(
                FixedPopulationIndex::new(axes.len(), population, max_bytes)?,
            ),
            FactorComponentKind::FiniteAdditive => {
                ComponentIndex::FiniteAdditive(finite_additive_component_index(
                    &rows,
                    &group_rows,
                    &axes,
                    dimensions,
                    axis_columns,
                    max_bytes,
                )?)
            }
        };
        components.push(FactorComponent {
            axes: axes.into_boxed_slice(),
            kind,
            index,
        });
    }
    for (axis, &is_covered) in covered.iter().enumerate() {
        if !is_covered {
            components.push(FactorComponent {
                axes: vec![axis].into_boxed_slice(),
                kind: FactorComponentKind::Cartesian,
                index: ComponentIndex::Cartesian(CartesianIndex::new(&[dimensions[axis]])?),
            });
        }
    }
    make_factorized_layout(components, dimensions.len(), max_bytes)
}

fn cartesian_layout(dimensions: &[usize], max_bytes: u128) -> Option<FactorizedLayout> {
    let components = (0..dimensions.len())
        .map(|axis| {
            Some(FactorComponent {
                axes: vec![axis].into_boxed_slice(),
                kind: FactorComponentKind::Cartesian,
                index: ComponentIndex::Cartesian(CartesianIndex::new(&[dimensions[axis]])?),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    make_factorized_layout(components, dimensions.len(), max_bytes)
}

fn fixed_population_kind(
    row: &ReducedChargeRow,
    axes: &[usize],
    dimensions: &[usize],
    axis_columns: &[usize],
) -> Option<FactorComponentKind> {
    if axes.is_empty() || axes.iter().any(|&axis| dimensions[axis] != 2) {
        return None;
    }
    let coefficient = row.coefficients[*axis_columns.get(*axes.first()?)?];
    if coefficient == 0
        || axes
            .iter()
            .any(|&axis| row.coefficients[axis_columns[axis]] != coefficient)
    {
        return None;
    }
    if row.target % coefficient != 0 {
        return None;
    }
    let population = usize::try_from(row.target / coefficient).ok()?;
    (population <= axes.len()).then_some(FactorComponentKind::FixedPopulation { population })
}

fn finite_additive_component_index(
    rows: &[ReducedChargeRow],
    group_rows: &[usize],
    axes: &[usize],
    dimensions: &[usize],
    axis_columns: &[usize],
    max_bytes: u128,
) -> Option<FiniteAdditiveIndex> {
    if group_rows.is_empty() || group_rows.len() > 4 {
        return None;
    }
    let component_dimensions = axes
        .iter()
        .map(|&axis| dimensions[axis])
        .collect::<Vec<_>>();
    let mut contributions = Vec::with_capacity(axes.len());
    for &axis in axes {
        let mut levels = Vec::with_capacity(dimensions[axis]);
        levels.push([0_i64; 4]);
        for level in 1..dimensions[axis] {
            let column = axis_columns[axis] + level - 1;
            let mut value = [0_i64; 4];
            for (row_index, &row) in group_rows.iter().enumerate() {
                value[row_index] = rows[row].coefficients[column];
            }
            levels.push(value);
        }
        contributions.push(levels);
    }
    let mut target = [0_i64; 4];
    for (row_index, &row) in group_rows.iter().enumerate() {
        target[row_index] = rows[row].target;
    }
    FiniteAdditiveIndex::new(
        &component_dimensions,
        contributions,
        target,
        group_rows.len(),
        max_bytes,
    )
}

fn make_factorized_layout(
    mut components: Vec<FactorComponent>,
    axis_count: usize,
    max_bytes: u128,
) -> Option<FactorizedLayout> {
    components.sort_by_key(|component| component.axes[0]);
    let mut axis_components = vec![usize::MAX; axis_count];
    let mut axis_entries = 0usize;
    for (component_index, component) in components.iter().enumerate() {
        for &axis in &component.axes {
            if axis >= axis_count || axis_components[axis] != usize::MAX {
                return None;
            }
            axis_components[axis] = component_index;
            axis_entries = axis_entries.checked_add(1)?;
        }
    }
    if axis_components.contains(&usize::MAX) {
        return None;
    }
    let component_index_bytes = components
        .iter()
        .map(|component| component.index.estimated_bytes())
        .try_fold(0_u128, |total, value| total.checked_add(value))?;
    let estimated_bytes = (components.len() as u128)
        .checked_mul(std::mem::size_of::<FactorComponent>() as u128)?
        .checked_add((axis_entries as u128).checked_mul(std::mem::size_of::<usize>() as u128)?)?
        .checked_add((axis_count as u128).checked_mul(std::mem::size_of::<usize>() as u128)?)?
        .checked_add(component_index_bytes)?;
    if estimated_bytes > max_bytes {
        return None;
    }
    Some(FactorizedLayout {
        components: components.into_boxed_slice(),
        axis_components: axis_components.into_boxed_slice(),
        estimated_bytes,
    })
}

/// Build a charge-sector plan from compact affine/local charge metadata.
///
/// This entry point performs the contribution-table preflight before creating
/// one row per local level.  Python therefore only passes axis metadata and
/// weights across the FFI boundary, even for very large finite boson cutoffs.
#[allow(clippy::too_many_arguments)]
pub fn build_compact_charge_sector_plan(
    local_dimensions: Vec<usize>,
    axis_kinds: Vec<u8>,
    axis_indices: Vec<usize>,
    fermion_weights: Vec<Vec<i128>>,
    boson_weights: Vec<Vec<i128>>,
    qubit_levels: Vec<Vec<(i128, i128)>>,
    qudit_weights: Vec<Vec<i128>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> Result<ChargeSectorPlan, PauliError> {
    if local_dimensions.len() != axis_kinds.len() || local_dimensions.len() != axis_indices.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: local_dimensions.len(),
            actual: axis_kinds.len(),
        });
    }
    if target.is_empty() {
        return Err(PauliError::InvalidSector {
            context: "charge-sector requires at least one charge constraint",
        });
    }
    for &dimension in &local_dimensions {
        if dimension == 0 {
            return Err(PauliError::InvalidSector {
                context: "charge-sector local dimensions must be positive",
            });
        }
    }
    let contribution_entries = local_dimensions
        .iter()
        .try_fold(0_u128, |total, &dimension| {
            total
                .checked_add(dimension as u128)
                .ok_or(PauliError::Overflow {
                    context: "estimating charge-sector contributions",
                })
        })?;
    let preflight = contribution_entries
        .checked_mul(target.len() as u128)
        .and_then(|value| value.checked_mul(CHARGE_CONTRIBUTION_BYTES))
        .and_then(|value| value.checked_add((local_dimensions.len() as u128).checked_mul(64)?))
        .and_then(|value| value.checked_add((target.len() as u128).checked_mul(16)?))
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector contributions",
        })?;
    if preflight > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: preflight,
            limit: max_bytes,
        });
    }

    let mut contributions = Vec::with_capacity(local_dimensions.len());
    for position in 0..local_dimensions.len() {
        let kind = axis_kinds[position];
        let axis_index = axis_indices[position];
        let table = (0..local_dimensions[position])
            .map(|value| {
                let value = i128::try_from(value).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector local level",
                })?;
                target
                    .iter()
                    .enumerate()
                    .map(|(constraint, _)| match kind {
                        0 => fermion_weights
                            .get(constraint)
                            .and_then(|weights| weights.get(axis_index))
                            .ok_or(PauliError::InvalidSector {
                                context: "invalid fermion charge axis",
                            })?
                            .checked_mul(value)
                            .ok_or(PauliError::Overflow {
                                context: "computing charge-sector contribution",
                            }),
                        1 => boson_weights
                            .get(constraint)
                            .and_then(|weights| weights.get(axis_index))
                            .ok_or(PauliError::InvalidSector {
                                context: "invalid boson charge axis",
                            })?
                            .checked_mul(value)
                            .ok_or(PauliError::Overflow {
                                context: "computing charge-sector contribution",
                            }),
                        2 => {
                            let levels = qubit_levels
                                .get(constraint)
                                .and_then(|values| values.get(axis_index))
                                .ok_or(PauliError::InvalidSector {
                                    context: "invalid qubit charge axis",
                                })?;
                            if value > 1 {
                                return Err(PauliError::InvalidSector {
                                    context: "qubit charge axis has a non-binary level",
                                });
                            }
                            Ok(if value == 0 { levels.0 } else { levels.1 })
                        }
                        3 => {
                            let weight = qudit_weights
                                .get(constraint)
                                .and_then(|weights| weights.get(axis_index))
                                .ok_or(PauliError::InvalidSector {
                                    context: "invalid qudit charge axis",
                                })?;
                            let dimension = local_dimensions[position];
                            if dimension < 3 {
                                return Err(PauliError::InvalidSector {
                                    context: "qudit charge axis must have dimension at least three",
                                });
                            }
                            let spectrum = if dimension % 2 == 1 {
                                i128::try_from((dimension - 1) / 2).map_err(|_| {
                                    PauliError::Overflow {
                                        context: "converting qudit spin-z spectrum",
                                    }
                                })? - value
                            } else {
                                i128::try_from(dimension - 1).map_err(|_| PauliError::Overflow {
                                    context: "converting qudit spin-z spectrum",
                                })? - value.checked_mul(2).ok_or(PauliError::Overflow {
                                    context: "computing qudit spin-z spectrum",
                                })?
                            };
                            weight.checked_mul(spectrum).ok_or(PauliError::Overflow {
                                context: "computing qudit charge contribution",
                            })
                        }
                        _ => Err(PauliError::InvalidSector {
                            context: "invalid charge-sector axis kind",
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        contributions.push(table);
    }
    build_charge_sector_plan(local_dimensions, contributions, target, max_bytes)
}

fn subtract_contribution_into(
    remaining: &[i128],
    contribution: &[i128],
    output: &mut [i128],
) -> Result<(), PauliError> {
    if output.len() != remaining.len() || output.len() != contribution.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: remaining.len(),
            actual: output.len(),
        });
    }
    for ((slot, &left), &right) in output.iter_mut().zip(remaining).zip(contribution) {
        *slot = left.checked_sub(right).ok_or(PauliError::Overflow {
            context: "computing charge-sector remainder",
        })?;
    }
    Ok(())
}

fn check_plan_bytes(
    entries: usize,
    bytes_per_entry: u128,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let requested = (entries as u128)
        .checked_mul(bytes_per_entry)
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector dynamic-programming plan",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force_basis(
        dimensions: &[usize],
        contributions: &[Vec<Vec<i128>>],
        target: &[i128],
    ) -> Vec<Vec<u64>> {
        let total = dimensions.iter().product::<usize>();
        (0..total)
            .filter_map(|encoded| {
                let mut encoded = encoded;
                let mut values = vec![0_u64; dimensions.len()];
                for axis in (0..dimensions.len()).rev() {
                    values[axis] = (encoded % dimensions[axis]) as u64;
                    encoded /= dimensions[axis];
                }
                let selected = target.iter().enumerate().all(|(constraint, &goal)| {
                    values
                        .iter()
                        .enumerate()
                        .try_fold(0_i128, |total, (axis, &value)| {
                            total.checked_add(contributions[axis][value as usize][constraint])
                        })
                        == Some(goal)
                });
                selected.then_some(values)
            })
            .collect()
    }

    fn assert_public_order_matches_bruteforce(
        dimensions: Vec<usize>,
        contributions: Vec<Vec<Vec<i128>>>,
        target: Vec<i128>,
    ) {
        let expected = brute_force_basis(&dimensions, &contributions, &target);
        let plan = build_charge_sector_plan(dimensions, contributions, target, u128::MAX)
            .expect("charge sector plan");
        assert_eq!(plan.dimension(), expected.len());
        let mut values = vec![0_u64; plan.local_dimensions().len()];
        let mut remaining = vec![0_i128; plan.constraint_count()];
        let mut candidate_remaining = vec![0_i128; plan.constraint_count()];
        for (rank, expected_values) in expected.iter().enumerate() {
            let rank = rank as u64;
            assert_eq!(
                plan.unrank(rank).expect("public unrank"),
                expected_values.clone()
            );
            plan.unrank_into_with_scratch(
                rank,
                &mut values,
                &mut remaining,
                &mut candidate_remaining,
            )
            .expect("general unrank");
            assert_eq!(values, expected_values.clone());
            assert_eq!(plan.rank(expected_values).expect("public rank"), rank);
            assert_eq!(
                plan.rank_into(expected_values, &mut remaining, &mut candidate_remaining)
                    .expect("general rank"),
                rank
            );
        }
    }

    fn rows_select_binary_state(rows: &[ReducedChargeRow], state: usize) -> bool {
        rows.iter().all(|row| {
            row.coefficients
                .iter()
                .enumerate()
                .try_fold(0_i64, |total, (axis, &coefficient)| {
                    total.checked_add(coefficient * ((state >> (3 - axis)) & 1) as i64)
                })
                == Some(row.target)
        })
    }

    #[test]
    fn rank_unrank_and_basis_follow_mixed_radix_order() {
        let plan = build_charge_sector_plan(
            vec![2, 2, 2],
            vec![
                vec![vec![0], vec![1]],
                vec![vec![0], vec![1]],
                vec![vec![0], vec![1]],
            ],
            vec![1],
            u128::MAX,
        )
        .expect("charge sector plan");
        assert_eq!(plan.dimension(), 3);
        assert_eq!(plan.unrank(0).expect("unrank"), vec![0, 0, 1]);
        assert_eq!(plan.unrank(2).expect("unrank"), vec![1, 0, 0]);
        assert_eq!(plan.rank(&[0, 1, 0]).expect("rank"), 1);
        assert_eq!(
            plan.basis_states(u128::MAX).expect("basis"),
            vec![0, 0, 1, 0, 1, 0, 1, 0, 0]
        );
    }

    #[test]
    fn reduced_equivalent_rows_prove_disjoint_fixed_population_components() {
        // N = 2 and N_up - N_down = 0 are equivalent to one occupied mode
        // in each physical half.  Raw supports overlap; integer row reduction
        // must discover the two components without looking at charge names.
        let plan = build_charge_sector_plan(
            vec![2, 2, 2, 2],
            vec![
                vec![vec![0, 0], vec![1, 1]],
                vec![vec![0, 0], vec![1, 1]],
                vec![vec![0, 0], vec![1, -1]],
                vec![vec![0, 0], vec![1, -1]],
            ],
            vec![2, 0],
            u128::MAX,
        )
        .expect("charge sector plan");
        assert_eq!(plan.dimension(), 4);
        let layout = plan.factorized_layout().expect("factor layout");
        assert_eq!(layout.axis_components(), &[0, 0, 1, 1]);
        assert_eq!(layout.components().len(), 2);
        assert_eq!(layout.components()[0].axes(), &[0, 1]);
        assert_eq!(layout.components()[1].axes(), &[2, 3]);
        assert_eq!(
            layout.components()[0].kind(),
            FactorComponentKind::FixedPopulation { population: 1 }
        );
        assert_eq!(
            layout.components()[1].kind(),
            FactorComponentKind::FixedPopulation { population: 1 }
        );
        for component in layout.components() {
            let dimension = component.index().dimension();
            let mut values = vec![0_u64; component.axes().len()];
            for rank in 0..dimension {
                component
                    .index()
                    .unrank_into(rank as u64, &mut values)
                    .expect("component unrank");
                assert_eq!(
                    component.index().rank(&values).expect("component rank"),
                    rank as u64
                );
            }
        }
        assert_eq!(plan.unrank(0).expect("first state"), vec![0, 1, 0, 1]);
        assert_eq!(plan.unrank(3).expect("last state"), vec![1, 0, 1, 0]);
    }

    #[test]
    fn factor_analysis_uses_baseline_adjusted_contributions() {
        let plan = build_charge_sector_plan(
            vec![2, 2],
            vec![vec![vec![3], vec![4]], vec![vec![3], vec![4]]],
            vec![7],
            u128::MAX,
        )
        .expect("charge sector plan");
        assert_eq!(plan.dimension(), 2);
        let layout = plan.factorized_layout().expect("factor layout");
        assert_eq!(layout.components().len(), 1);
        assert_eq!(layout.components()[0].axes(), &[0, 1]);
        assert_eq!(
            layout.components()[0].kind(),
            FactorComponentKind::FixedPopulation { population: 1 }
        );
    }

    #[test]
    fn compatible_zero_rows_are_cartesian_components() {
        let plan = build_charge_sector_plan(
            vec![2, 3],
            vec![vec![vec![5], vec![5]], vec![vec![7], vec![7], vec![7]]],
            vec![12],
            u128::MAX,
        )
        .expect("charge sector plan");
        assert_eq!(plan.dimension(), 6);
        let layout = plan.factorized_layout().expect("Cartesian layout");
        assert_eq!(layout.components().len(), 2);
        assert!(layout
            .components()
            .iter()
            .all(|component| component.kind() == FactorComponentKind::Cartesian));
    }

    #[test]
    fn finite_additive_component_preserves_local_and_public_order() {
        let plan = build_charge_sector_plan(
            vec![3, 3],
            vec![
                vec![vec![0], vec![1], vec![2]],
                vec![vec![0], vec![1], vec![2]],
            ],
            vec![2],
            u128::MAX,
        )
        .expect("finite additive sector");
        let layout = plan.factorized_layout().expect("finite layout");
        assert_eq!(layout.components().len(), 1);
        assert_eq!(
            layout.components()[0].kind(),
            FactorComponentKind::FiniteAdditive
        );
        assert_eq!(plan.dimension(), 3);
        assert_eq!(plan.unrank(0).expect("unrank"), vec![0, 2]);
        assert_eq!(plan.unrank(1).expect("unrank"), vec![1, 1]);
        assert_eq!(plan.unrank(2).expect("unrank"), vec![2, 0]);
        assert_eq!(plan.rank(&[1, 1]).expect("rank"), 1);
        let ComponentIndex::FiniteAdditive(index) = layout.components()[0].index() else {
            panic!("finite-additive index");
        };
        assert!(index.uses_scalar_remainders());
    }

    #[test]
    fn coupled_finite_additive_component_uses_inline_wide_remainders() {
        let plan = build_charge_sector_plan(
            vec![2, 2, 3],
            vec![
                vec![vec![0, 0], vec![1, 0]],
                vec![vec![0, 0], vec![0, 1]],
                vec![vec![0, 0], vec![1, 1], vec![2, 2]],
            ],
            vec![1, 1],
            u128::MAX,
        )
        .expect("coupled finite plan");
        let layout = plan.factorized_layout().expect("finite layout");
        let ComponentIndex::FiniteAdditive(index) = layout.components()[0].index() else {
            panic!("finite-additive index");
        };
        assert!(!index.uses_scalar_remainders());
    }

    #[test]
    fn reduction_preserves_all_small_binary_solution_sets_and_is_order_independent() {
        for seed in 0..64_i64 {
            let witness = (seed as usize) & 0b1111;
            let mut rows = (0..(2 + (seed as usize % 3)))
                .map(|row| {
                    let mut coefficients = (0..4)
                        .map(|column| ((seed + row as i64 * 3 + column as i64 * 5) % 5) - 2)
                        .collect::<Vec<_>>();
                    if coefficients.iter().all(|&value| value == 0) {
                        let nonzero = row % coefficients.len();
                        coefficients[nonzero] = 1;
                    }
                    let target = coefficients
                        .iter()
                        .enumerate()
                        .map(|(column, &value)| value * ((witness >> (3 - column)) & 1) as i64)
                        .sum::<i64>();
                    ReducedChargeRow {
                        coefficients,
                        target: if seed % 7 == 0 { target + 1 } else { target },
                    }
                })
                .collect::<Vec<_>>();
            let original = rows.clone();
            reduce_charge_rows(&mut rows).expect("bounded reduction");
            for state in 0..16 {
                assert_eq!(
                    rows_select_binary_state(&original, state),
                    rows_select_binary_state(&rows, state),
                    "seed={seed}, state={state}"
                );
            }
        }

        let mut rows = vec![
            ReducedChargeRow {
                coefficients: vec![1, 1, 1, 1],
                target: 2,
            },
            ReducedChargeRow {
                coefficients: vec![1, 1, -1, -1],
                target: 0,
            },
        ];
        let mut reversed = rows.clone();
        reversed.reverse();
        reduce_charge_rows(&mut rows).expect("reduction");
        reduce_charge_rows(&mut reversed).expect("reduction");
        assert_eq!(rows, reversed);
    }

    #[test]
    fn factorized_public_order_matches_general_and_bruteforce_for_mandatory_layouts() {
        // Two fixed binary components found only after reducing overlapping
        // rows, plus a three-component non-contiguous mixed-radix layout and
        // a full Cartesian constraint system.
        assert_public_order_matches_bruteforce(
            vec![2, 2, 2, 2],
            vec![
                vec![vec![0, 0], vec![1, 1]],
                vec![vec![0, 0], vec![1, 1]],
                vec![vec![0, 0], vec![1, -1]],
                vec![vec![0, 0], vec![1, -1]],
            ],
            vec![2, 0],
        );
        assert_public_order_matches_bruteforce(
            vec![2, 3, 2, 3, 2],
            vec![
                vec![vec![0, 0], vec![1, 0]],
                vec![vec![0, 0], vec![0, 1], vec![0, 2]],
                vec![vec![0, 0], vec![1, 0]],
                vec![vec![0, 0], vec![0, 1], vec![0, 2]],
                vec![vec![0, 0], vec![0, 0]],
            ],
            vec![1, 2],
        );
        assert_public_order_matches_bruteforce(
            vec![2, 3, 2],
            vec![
                vec![vec![5], vec![5]],
                vec![vec![7], vec![7], vec![7]],
                vec![vec![-2], vec![-2]],
            ],
            vec![10],
        );
    }

    #[test]
    fn contiguous_fixed_population_words_match_public_order_without_state_map() {
        let index = FixedPopulationIndex::new(5, 2, u128::MAX).expect("fixed index");
        assert!(index.combinations.is_some());
        let representative =
            FixedPopulationIndex::new(20, 10, u128::MAX).expect("representative index");
        assert!(representative
            .combinations
            .as_ref()
            .is_some_and(|values| values.len() < representative.dimension()));
        for rank in 0..index.dimension() {
            let word = index
                .unrank_contiguous_binary_word(rank)
                .expect("binary unrank");
            assert_eq!(
                index
                    .rank_contiguous_binary_word(word)
                    .expect("binary rank"),
                rank
            );
            let values = (0..index.sites())
                .map(|axis| ((word >> (index.sites() - axis - 1)) & 1) as u64)
                .collect::<Vec<_>>();
            assert_eq!(index.rank(&values).expect("component rank"), rank as u64);
        }
        let advisory_budget = FixedPopulationIndex::new(5, 2, 1).expect("fixed index");
        assert!(advisory_budget.combinations.is_some());
        assert_eq!(advisory_budget.dimension(), index.dimension());
    }

    #[test]
    fn factorized_public_cursor_matches_every_public_unrank() {
        let plan = build_charge_sector_plan(
            vec![2, 2, 2, 2],
            vec![
                vec![vec![0, 0], vec![1, 1]],
                vec![vec![0, 0], vec![1, 1]],
                vec![vec![0, 0], vec![1, -1]],
                vec![vec![0, 0], vec![1, -1]],
            ],
            vec![2, 0],
            u128::MAX,
        )
        .expect("factorized sector");
        let layout = plan.factorized_layout().expect("layout");
        let mut scratch = layout.scratch();
        let mut values = vec![0_u64; 4];
        layout
            .unrank_with_scratch(0, &mut values, &mut scratch)
            .expect("cursor start");
        for rank in 0..plan.dimension() {
            assert_eq!(values, plan.unrank(rank as u64).expect("public unrank"));
            assert_eq!(
                layout
                    .advance_public_cursor(&mut values, &mut scratch)
                    .expect("cursor advance"),
                rank + 1 < plan.dimension()
            );
        }
    }

    #[test]
    fn ineligible_row_count_retains_the_exact_general_plan() {
        let plan = build_charge_sector_plan(
            vec![2],
            vec![vec![vec![0, 0, 0, 0, 0], vec![1, 1, 1, 1, 1]]],
            vec![1, 1, 1, 1, 1],
            u128::MAX,
        )
        .expect("charge sector plan");
        assert_eq!(plan.dimension(), 1);
        assert!(plan.factorized_layout().is_none());
        assert_eq!(plan.unrank(0).expect("exact General state"), vec![1]);
    }

    #[test]
    fn factor_analysis_overflow_and_infeasible_population_retain_exact_general_results() {
        let beyond_i64 = i128::from(i64::MAX) + 1;
        let wide = build_charge_sector_plan(
            vec![2],
            vec![vec![vec![0], vec![beyond_i64]]],
            vec![beyond_i64],
            u128::MAX,
        )
        .expect("wide exact plan");
        assert_eq!(wide.dimension(), 1);
        assert!(wide.factorized_layout().is_none());
        assert_eq!(wide.unrank(0).expect("exact General state"), vec![1]);

        let infeasible = build_charge_sector_plan(
            vec![2, 2],
            vec![vec![vec![0], vec![2]], vec![vec![0], vec![2]]],
            vec![3],
            u128::MAX,
        )
        .expect("empty exact plan");
        assert_eq!(infeasible.dimension(), 0);
        assert!(infeasible.unrank(0).is_err());
    }
}
