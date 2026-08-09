use std::sync::Arc;

use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tencir_pauli_core::{
    apply_charge_csr_into, apply_charge_mvp_from_prepared_plan,
    apply_charge_mvp_from_prepared_plan_into, build_compact_charge_sector_plan,
    build_fast_boson_mvp_plan, build_fast_fermion_mvp_plan, build_fast_product_mvp_plan,
    build_fast_qubit_mvp_plan, build_fast_qudit_mvp_plan, compile_charge_csr_from_prepared_plan,
    compile_charge_csr_from_prepared_plan_with_chunk_capacity,
    estimate_charge_transition_terms_bytes, group_charge_transition_terms,
    prepare_charge_transition_plan_layout, ChargeCsrResult, ChargeSectorPlan, ChargeTransitionTerm,
    Complex64, FastChargeMvpPlan, PauliError, PreparedChargeTransitionPlanLayout,
};

use crate::convert::map_error;
use crate::operator::NativePauliOperatorHandle;
use crate::structured::NativeHybridOperatorHandle;

type ChargeCsrArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
type ChargeCooArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);

fn charge_plan_base_bytes(
    plan: &ChargeSectorPlan,
    terms: &[ChargeTransitionTerm],
) -> Result<u128, PauliError> {
    plan.estimated_bytes()
        .checked_add(estimate_charge_transition_terms_bytes(terms)?)
        .ok_or(PauliError::Overflow {
            context: "estimating native charge MVP plan",
        })
}

fn charge_remaining_budget(max_bytes: u128, base_bytes: u128) -> Result<u128, PauliError> {
    max_bytes
        .checked_sub(base_bytes)
        .ok_or(PauliError::MemoryLimit {
            requested: base_bytes,
            limit: max_bytes,
        })
}

fn charge_terms_from_pauli_handle(handle: &NativePauliOperatorHandle) -> Vec<ChargeTransitionTerm> {
    handle
        .core()
        .terms()
        .iter()
        .map(|term| ChargeTransitionTerm {
            fermion_creation: Vec::new(),
            fermion_annihilation: Vec::new(),
            boson_blocks: Vec::new(),
            qubit_codes: term.word.codes(),
            mapped_present: false,
            mapped_codes: vec![0; 0],
            qudit_present: false,
            qudit_triples: Vec::new(),
            coefficient: term.coefficient,
        })
        .collect()
}

fn charge_terms_from_hybrid_handle(
    handle: &NativeHybridOperatorHandle,
) -> Vec<ChargeTransitionTerm> {
    let batch = handle.batch();
    (0..batch.coefficients.len())
        .map(|index| ChargeTransitionTerm {
            fermion_creation: batch.fermion_creation[index].clone(),
            fermion_annihilation: batch.fermion_annihilation[index].clone(),
            boson_blocks: batch.boson_blocks[index].clone(),
            qubit_codes: batch.qubit_codes[index].clone(),
            mapped_present: batch.mapped_present[index],
            mapped_codes: batch.mapped_codes[index].clone(),
            qudit_present: batch.qudit_present[index],
            qudit_triples: batch.qudit_triples[index].clone(),
            coefficient: batch.coefficients[index],
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_native_charge_mvp_plan(
    plan: Arc<ChargeSectorPlan>,
    dimension: usize,
    local_dimensions: Vec<u64>,
    fermion_positions: Vec<u64>,
    boson_positions: Vec<u64>,
    qubit_positions: Vec<u64>,
    qudit_positions: Vec<u64>,
    qudit_dimension: u64,
    terms: Vec<ChargeTransitionTerm>,
    termwise_conserved: bool,
    max_bytes: u128,
    fast_fermion_particles: Option<usize>,
    fast_fermion_mode_order: Option<Vec<u64>>,
    fast_qubit_particles: Option<usize>,
    fast_qubit_positions: Option<Vec<u64>>,
    fast_qudit_weights: Option<Vec<i128>>,
    fast_qudit_target: Option<i128>,
    fast_qudit_positions: Option<Vec<u64>>,
    fast_boson_weights: Option<Vec<i128>>,
    fast_boson_target: Option<i128>,
    fast_boson_positions: Option<Vec<u64>>,
    fast_product_positions: Option<Vec<Vec<u64>>>,
) -> PyResult<NativeChargeMvpPlan> {
    if dimension != plan.dimension() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "charge MVP dimension does not match its sector plan",
        ));
    }
    let terms = group_charge_transition_terms(&terms).map_err(map_error)?;
    let base_bytes = charge_plan_base_bytes(&plan, &terms).map_err(map_error)?;
    let layout_budget = charge_remaining_budget(max_bytes, base_bytes).map_err(map_error)?;
    let layout = prepare_charge_transition_plan_layout(
        &plan,
        dimension,
        local_dimensions.clone(),
        &fermion_positions,
        &boson_positions,
        &qubit_positions,
        &qudit_positions,
        qudit_dimension,
        &terms,
        layout_budget,
    )
    .map_err(map_error)?;
    let fast_budget =
        charge_remaining_budget(layout_budget, layout.estimated_bytes()).map_err(map_error)?;
    let fast_plan = if termwise_conserved {
        build_fast_fermion_mvp_plan(
            &plan,
            &local_dimensions,
            &fermion_positions,
            &boson_positions,
            &qubit_positions,
            &qudit_positions,
            &terms,
            fast_fermion_particles,
            fast_fermion_mode_order.as_deref(),
            fast_budget,
        )
        .map_err(map_error)?
        .map(FastChargeMvpPlan::Fermion)
    } else {
        None
    };
    let fast_plan = if fast_plan.is_some() {
        fast_plan
    } else if fast_qubit_particles.is_some() {
        let candidate = build_fast_qubit_mvp_plan(
            &plan,
            &local_dimensions,
            &fermion_positions,
            &boson_positions,
            &qubit_positions,
            &qudit_positions,
            qudit_dimension,
            &terms,
            fast_qubit_particles,
            fast_qubit_positions.as_deref(),
            fast_budget,
        )
        .map_err(map_error)?;
        if candidate.is_some() {
            candidate.map(FastChargeMvpPlan::Qubit)
        } else if let Some(factor_positions) = fast_product_positions.as_deref() {
            build_fast_product_mvp_plan(&plan, &layout, &terms, factor_positions, fast_budget, true)
                .map_err(map_error)?
                .map(FastChargeMvpPlan::Product)
        } else {
            None
        }
    } else if fast_qudit_weights.is_some() {
        let candidate = build_fast_qudit_mvp_plan(
            &plan,
            &local_dimensions,
            &fermion_positions,
            &boson_positions,
            &qubit_positions,
            &qudit_positions,
            qudit_dimension,
            &terms,
            fast_qudit_weights.as_deref(),
            fast_qudit_target,
            fast_qudit_positions.as_deref(),
            fast_budget,
        )
        .map_err(map_error)?;
        if candidate.is_some() {
            candidate.map(FastChargeMvpPlan::Qudit)
        } else if let Some(factor_positions) = fast_product_positions.as_deref() {
            build_fast_product_mvp_plan(&plan, &layout, &terms, factor_positions, fast_budget, true)
                .map_err(map_error)?
                .map(FastChargeMvpPlan::Product)
        } else {
            None
        }
    } else if fast_boson_weights.is_some() {
        let candidate = build_fast_boson_mvp_plan(
            &plan,
            &local_dimensions,
            &fermion_positions,
            &boson_positions,
            &qubit_positions,
            &qudit_positions,
            qudit_dimension,
            &terms,
            fast_boson_weights.as_deref(),
            fast_boson_target,
            fast_boson_positions.as_deref(),
            fast_budget,
        )
        .map_err(map_error)?;
        if candidate.is_some() {
            candidate.map(FastChargeMvpPlan::Boson)
        } else if let Some(factor_positions) = fast_product_positions.as_deref() {
            build_fast_product_mvp_plan(&plan, &layout, &terms, factor_positions, fast_budget, true)
                .map_err(map_error)?
                .map(FastChargeMvpPlan::Product)
        } else {
            None
        }
    } else if let Some(factor_positions) = fast_product_positions.as_deref() {
        build_fast_product_mvp_plan(&plan, &layout, &terms, factor_positions, fast_budget, true)
            .map_err(map_error)?
            .map(FastChargeMvpPlan::Product)
    } else {
        None
    };
    Ok(NativeChargeMvpPlan {
        plan,
        layout,
        terms,
        termwise_conserved,
        fast_plan,
    })
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeSectorPlan {
    plan: Arc<ChargeSectorPlan>,
}

#[pymethods]
impl NativeChargeSectorPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    fn rank(&self, occupations: Vec<u64>) -> PyResult<u64> {
        self.plan
            .rank(&occupations)
            .map_err(crate::convert::map_error)
    }

    fn unrank(&self, index: u64) -> PyResult<Vec<u64>> {
        self.plan.unrank(index).map_err(crate::convert::map_error)
    }

    fn basis_states<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let values = py
            .allow_threads(|| self.plan.basis_states(max_bytes))
            .map_err(crate::convert::map_error)?;
        Ok(PyArray1::from_vec(py, values))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        handle,
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        termwise_conserved,
        max_bytes,
        fast_fermion_particles=None,
        fast_fermion_mode_order=None,
        fast_qubit_particles=None,
        fast_qubit_positions=None,
        fast_qudit_weights=None,
        fast_qudit_target=None,
        fast_qudit_positions=None,
        fast_boson_weights=None,
        fast_boson_target=None,
        fast_boson_positions=None,
        fast_product_positions=None
    ))]
    fn compile_mvp_pauli_handle(
        &self,
        py: Python<'_>,
        handle: &NativePauliOperatorHandle,
        dimension: usize,
        local_dimensions: Vec<u64>,
        fermion_positions: Vec<u64>,
        boson_positions: Vec<u64>,
        qubit_positions: Vec<u64>,
        qudit_positions: Vec<u64>,
        termwise_conserved: bool,
        max_bytes: u128,
        fast_fermion_particles: Option<usize>,
        fast_fermion_mode_order: Option<Vec<u64>>,
        fast_qubit_particles: Option<usize>,
        fast_qubit_positions: Option<Vec<u64>>,
        fast_qudit_weights: Option<Vec<i128>>,
        fast_qudit_target: Option<i128>,
        fast_qudit_positions: Option<Vec<u64>>,
        fast_boson_weights: Option<Vec<i128>>,
        fast_boson_target: Option<i128>,
        fast_boson_positions: Option<Vec<u64>>,
        fast_product_positions: Option<Vec<Vec<u64>>>,
    ) -> PyResult<NativeChargeMvpPlan> {
        let plan = Arc::clone(&self.plan);
        py.allow_threads(move || {
            build_native_charge_mvp_plan(
                plan,
                dimension,
                local_dimensions,
                fermion_positions,
                boson_positions,
                qubit_positions,
                qudit_positions,
                0,
                charge_terms_from_pauli_handle(handle),
                termwise_conserved,
                max_bytes,
                fast_fermion_particles,
                fast_fermion_mode_order,
                fast_qubit_particles,
                fast_qubit_positions,
                fast_qudit_weights,
                fast_qudit_target,
                fast_qudit_positions,
                fast_boson_weights,
                fast_boson_target,
                fast_boson_positions,
                fast_product_positions,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        handle,
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        qudit_dimension,
        termwise_conserved,
        max_bytes,
        fast_fermion_particles=None,
        fast_fermion_mode_order=None,
        fast_qubit_particles=None,
        fast_qubit_positions=None,
        fast_qudit_weights=None,
        fast_qudit_target=None,
        fast_qudit_positions=None,
        fast_boson_weights=None,
        fast_boson_target=None,
        fast_boson_positions=None,
        fast_product_positions=None
    ))]
    fn compile_mvp_hybrid_handle(
        &self,
        py: Python<'_>,
        handle: &NativeHybridOperatorHandle,
        dimension: usize,
        local_dimensions: Vec<u64>,
        fermion_positions: Vec<u64>,
        boson_positions: Vec<u64>,
        qubit_positions: Vec<u64>,
        qudit_positions: Vec<u64>,
        qudit_dimension: u64,
        termwise_conserved: bool,
        max_bytes: u128,
        fast_fermion_particles: Option<usize>,
        fast_fermion_mode_order: Option<Vec<u64>>,
        fast_qubit_particles: Option<usize>,
        fast_qubit_positions: Option<Vec<u64>>,
        fast_qudit_weights: Option<Vec<i128>>,
        fast_qudit_target: Option<i128>,
        fast_qudit_positions: Option<Vec<u64>>,
        fast_boson_weights: Option<Vec<i128>>,
        fast_boson_target: Option<i128>,
        fast_boson_positions: Option<Vec<u64>>,
        fast_product_positions: Option<Vec<Vec<u64>>>,
    ) -> PyResult<NativeChargeMvpPlan> {
        let plan = Arc::clone(&self.plan);
        py.allow_threads(move || {
            build_native_charge_mvp_plan(
                plan,
                dimension,
                local_dimensions,
                fermion_positions,
                boson_positions,
                qubit_positions,
                qudit_positions,
                qudit_dimension,
                charge_terms_from_hybrid_handle(handle),
                termwise_conserved,
                max_bytes,
                fast_fermion_particles,
                fast_fermion_mode_order,
                fast_qubit_particles,
                fast_qubit_positions,
                fast_qudit_weights,
                fast_qudit_target,
                fast_qudit_positions,
                fast_boson_weights,
                fast_boson_target,
                fast_boson_positions,
                fast_product_positions,
            )
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeMvpPlan {
    plan: Arc<ChargeSectorPlan>,
    layout: PreparedChargeTransitionPlanLayout,
    terms: Vec<ChargeTransitionTerm>,
    termwise_conserved: bool,
    fast_plan: Option<FastChargeMvpPlan>,
}

#[pymethods]
impl NativeChargeMvpPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan
            .estimated_bytes()
            .saturating_add(
                estimate_charge_transition_terms_bytes(&self.terms).unwrap_or(u128::MAX),
            )
            .saturating_add(self.layout.estimated_bytes())
            .saturating_add(
                self.fast_plan
                    .as_ref()
                    .map_or(0, FastChargeMvpPlan::estimated_bytes),
            )
    }

    /// Return the executor selected by native plan construction.
    ///
    /// Python must not infer `product` from constraint shapes when the native
    /// selector actually fell through to the generic executor.
    #[getter]
    fn execution_family(&self) -> &'static str {
        if self.fast_plan.is_some() {
            "product"
        } else {
            "generic"
        }
    }

    #[getter]
    fn native_variant(&self) -> &'static str {
        self.fast_plan
            .as_ref()
            .map_or("generic", FastChargeMvpPlan::native_variant)
    }

    fn debug_counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .fast_plan
            .as_ref()
            .and_then(FastChargeMvpPlan::debug_counters)
            .unwrap_or_else(|| {
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
                vec![
                    scalar_insertions,
                    wide_insertions,
                    capacity_growths,
                    owned_keys,
                    fallbacks,
                    rankers,
                    validations,
                    destination_allocations,
                ]
            });
        PyTuple::new(py, values)
    }

    #[getter]
    fn schedule_count(&self) -> usize {
        self.terms.len()
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        if state_values.len() != self.dimension() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "charge state must have shape ({},), got ({},)",
                self.dimension(),
                state_values.len()
            )));
        }
        let result = py
            .allow_threads(|| {
                apply_charge_mvp_from_prepared_plan(
                    &self.plan,
                    &self.layout,
                    &self.terms,
                    state_values,
                    self.termwise_conserved,
                    self.fast_plan.as_ref(),
                    max_bytes,
                )
            })
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, result))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_values = output.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge output must be C-contiguous")
        })?;
        let output_bytes = (self.dimension() as u128)
            .checked_mul(std::mem::size_of::<NumpyComplex128>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge MVP output size overflow")
            })?;
        let effective_max = max_bytes.saturating_add(output_bytes);
        py.allow_threads(|| {
            apply_charge_mvp_from_prepared_plan_into(
                &self.plan,
                &self.layout,
                &self.terms,
                state_values,
                output_values,
                self.termwise_conserved,
                self.fast_plan.as_ref(),
                effective_max,
            )
        })
        .map_err(map_error)
    }

    fn compile_eager<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<NativeChargeEagerMvpPlan> {
        py.allow_threads(|| {
            let result = compile_charge_csr_from_prepared_plan(
                &self.plan,
                &self.layout,
                &self.terms,
                max_bytes,
            )
            .map_err(map_error)?;
            NativeChargeEagerMvpPlan::from_csr(result, self.native_variant())
        })
    }

    #[pyo3(signature = (max_bytes, chunk_capacity))]
    fn compile_eager_with_chunk_capacity(
        &self,
        py: Python<'_>,
        max_bytes: u128,
        chunk_capacity: usize,
    ) -> PyResult<NativeChargeEagerMvpPlan> {
        py.allow_threads(|| {
            let result = compile_charge_csr_from_prepared_plan_with_chunk_capacity(
                &self.plan,
                &self.layout,
                &self.terms,
                max_bytes,
                chunk_capacity,
            )
            .map_err(map_error)?;
            NativeChargeEagerMvpPlan::from_csr(result, self.native_variant())
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeEagerMvpPlan {
    dimension: usize,
    indptr: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<Complex64>,
    estimated_bytes: u128,
    estimated_construction_peak_bytes: u128,
    schedule_count: usize,
    native_variant: &'static str,
    generation_nanos: u128,
    chunk_count: usize,
    sorted_entries: usize,
    merged_entries: usize,
    heap_operations: usize,
    final_array_allocation_entries: usize,
    chunk_metadata_bytes: u128,
    sort_nanos: u128,
    merge_nanos: u128,
    final_array_allocation_nanos: u128,
}

impl NativeChargeEagerMvpPlan {
    fn from_csr(result: ChargeCsrResult, native_variant: &'static str) -> PyResult<Self> {
        let columns = result
            .columns
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("charge CSR column overflow"))?;
        let indptr = result
            .indptr
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("charge CSR offset overflow"))?;
        Ok(Self {
            dimension: result.dimension,
            indptr,
            columns,
            values: result.values,
            estimated_bytes: result.estimated_bytes,
            estimated_construction_peak_bytes: result.estimated_construction_peak_bytes,
            schedule_count: result.schedule_count,
            native_variant,
            generation_nanos: result.generation_nanos,
            chunk_count: result.chunk_count,
            sorted_entries: result.sorted_entries,
            merged_entries: result.merged_entries,
            heap_operations: result.heap_operations,
            final_array_allocation_entries: result.final_array_allocation_entries,
            chunk_metadata_bytes: result.chunk_metadata_bytes,
            sort_nanos: result.sort_nanos,
            merge_nanos: result.merge_nanos,
            final_array_allocation_nanos: result.final_array_allocation_nanos,
        })
    }

    fn apply_values(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        parallel: bool,
    ) -> PyResult<()> {
        apply_charge_csr_into(
            &self.indptr,
            &self.columns,
            &self.values,
            state,
            output,
            parallel,
        )
        .map_err(map_error)
    }
}

#[pymethods]
impl NativeChargeEagerMvpPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[getter]
    fn transition_count(&self) -> usize {
        self.values.len()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    #[getter]
    fn estimated_construction_peak_bytes(&self) -> u128 {
        self.estimated_construction_peak_bytes
    }

    #[getter]
    fn schedule_count(&self) -> usize {
        self.schedule_count
    }

    #[getter]
    fn native_variant(&self) -> &'static str {
        self.native_variant
    }

    fn construction_profile(
        &self,
    ) -> (
        u128,
        usize,
        usize,
        usize,
        usize,
        usize,
        u128,
        u128,
        u128,
        u128,
    ) {
        (
            self.generation_nanos,
            self.chunk_count,
            self.sorted_entries,
            self.merged_entries,
            self.heap_operations,
            self.final_array_allocation_entries,
            self.chunk_metadata_bytes,
            self.sort_nanos,
            self.merge_nanos,
            self.final_array_allocation_nanos,
        )
    }

    fn csr<'py>(&self, py: Python<'py>) -> ChargeCsrArrays<'py> {
        let indptr = self.indptr.iter().map(|&value| value as u64).collect();
        let columns = self.columns.iter().map(|&value| value as u64).collect();
        (
            PyArray1::from_vec(py, indptr),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, self.values.clone()),
        )
    }

    fn coo<'py>(&self, py: Python<'py>, max_bytes: u128) -> PyResult<ChargeCooArrays<'py>> {
        let transition_bytes = (self.values.len() as u128)
            .checked_mul(32)
            .ok_or_else(|| pyo3::exceptions::PyMemoryError::new_err("charge COO size overflow"))?;
        if transition_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge COO requires approximately {transition_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut rows = Vec::with_capacity(self.values.len());
        for row in 0..self.dimension {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            rows.extend(std::iter::repeat_n(row as u64, end - start));
        }
        let columns = self.columns.iter().map(|&value| value as u64).collect();
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, self.values.clone()),
        ))
    }

    fn dense<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let entries = (self.dimension as u128)
            .checked_mul(self.dimension as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense size overflow")
            })?;
        let output_bytes = entries
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge dense matrix requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let values = vec![
            Complex64::default();
            usize::try_from(entries).map_err(|_| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense dimension overflow")
            })?
        ];
        let values = py.allow_threads(|| {
            let mut values = values;
            for row in 0..self.dimension {
                for index in self.indptr[row]..self.indptr[row + 1] {
                    values[row * self.dimension + self.columns[index]] = self.values[index];
                }
            }
            values
        });
        Ok(PyArray1::from_vec(py, values))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_bytes = (self.dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut output = vec![Complex64::default(); self.dimension];
        py.allow_threads(|| self.apply_values(state_values, &mut output, true))?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_with_parallelism<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_bytes = (self.dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut output = vec![Complex64::default(); self.dimension];
        py.allow_threads(|| self.apply_values(state_values, &mut output, parallel))?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_values = output.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge output must be C-contiguous")
        })?;
        let output_bytes = (self.dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        py.allow_threads(|| self.apply_values(state_values, output_values, true))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn charge_sector_plan_compact(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    axis_kinds: Vec<u8>,
    axis_indices: Vec<usize>,
    fermion_weights: Vec<Vec<i128>>,
    boson_weights: Vec<Vec<i128>>,
    qubit_levels: Vec<Vec<(i128, i128)>>,
    qudit_weights: Vec<Vec<i128>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> PyResult<NativeChargeSectorPlan> {
    let plan = py
        .allow_threads(|| {
            build_compact_charge_sector_plan(
                local_dimensions,
                axis_kinds,
                axis_indices,
                fermion_weights,
                boson_weights,
                qubit_levels,
                qudit_weights,
                target,
                max_bytes,
            )
        })
        .map_err(crate::convert::map_error)?;
    Ok(NativeChargeSectorPlan {
        plan: Arc::new(plan),
    })
}
