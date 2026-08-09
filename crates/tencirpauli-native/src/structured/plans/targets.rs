fn lower_hybrid_operations(
    result: &tencir_pauli_core::HybridCanonicalResult,
    axes: &[(u8, usize)],
) -> Result<Vec<Vec<StructuredOperation>>, tencir_pauli_core::PauliError> {
    let count = result.coefficients.len();
    let mut operations = Vec::with_capacity(count);
    for term_index in 0..count {
        if result.fermion_present[term_index] && result.mapped_present[term_index] {
            return Err(tencir_pauli_core::PauliError::InvalidSector {
                context: "finite compilation cannot mix raw and mapped fermion factors",
            });
        }
        let boson_blocks = &result.boson_blocks[term_index];
        let qudit_triples = &result.qudit_triples[term_index];
        let mut term_operations = Vec::new();
        if result.fermion_present[term_index] {
            let append_fermion = |mode: u32, create: u32, output: &mut Vec<StructuredOperation>| {
                let position = axes
                    .iter()
                    .position(|&(domain, index)| domain == 0 && index == mode as usize)
                    .ok_or(tencir_pauli_core::PauliError::InvalidIndex {
                        context: "raw fermion compilation axis",
                    })?;
                output.push(StructuredOperation {
                    axis: position,
                    kind: 3,
                    p: mode,
                    q: create,
                });
                Ok::<(), tencir_pauli_core::PauliError>(())
            };
            for &mode in result.fermion_annihilation[term_index].iter().rev() {
                append_fermion(mode, 0, &mut term_operations)?;
            }
            for &mode in result.fermion_creation[term_index].iter().rev() {
                append_fermion(mode, 1, &mut term_operations)?;
            }
        }
        for (position, &(domain, index)) in axes.iter().enumerate() {
            match domain {
                0 => {
                    if result.mapped_present[term_index] {
                        let code = *result.mapped_codes[term_index].get(index).ok_or(
                            tencir_pauli_core::PauliError::InvalidIndex {
                                context: "mapped fermion compilation axis",
                            },
                        )?;
                        if code != 0 {
                            term_operations.push(StructuredOperation {
                                axis: position,
                                kind: 0,
                                p: u32::from(code),
                                q: 0,
                            });
                        }
                    }
                }
                1 => {
                    if let Some((creation, annihilation)) = boson_blocks
                        .iter()
                        .find(|&&(mode, _, _)| mode as usize == index)
                        .map(|&(_, creation, annihilation)| (creation, annihilation))
                    {
                        term_operations.push(StructuredOperation {
                            axis: position,
                            kind: 1,
                            p: creation,
                            q: annihilation,
                        });
                    }
                }
                2 => {
                    let code = *result.qubit_codes[term_index].get(index).ok_or(
                        tencir_pauli_core::PauliError::InvalidIndex {
                            context: "qubit compilation axis",
                        },
                    )?;
                    if code != 0 {
                        term_operations.push(StructuredOperation {
                            axis: position,
                            kind: 0,
                            p: u32::from(code),
                            q: 0,
                        });
                    }
                }
                3 => {
                    if let Some((a, b)) = qudit_triples
                        .iter()
                        .find(|&&(site, _, _)| site as usize == index)
                        .map(|&(_, a, b)| (a, b))
                    {
                        term_operations.push(StructuredOperation {
                            axis: position,
                            kind: 2,
                            p: a,
                            q: b,
                        });
                    }
                }
                _ => {
                    return Err(tencir_pauli_core::PauliError::InvalidCode {
                        code: domain,
                        index: position,
                    })
                }
            }
        }
        operations.push(term_operations);
    }
    Ok(operations)
}

fn fermion_axis_positions(
    n_modes: usize,
    axes: &[(u8, usize)],
) -> Result<Vec<usize>, tencir_pauli_core::PauliError> {
    let mut positions = vec![usize::MAX; n_modes];
    for (position, &(domain, index)) in axes.iter().enumerate() {
        if domain == 0 {
            let slot =
                positions
                    .get_mut(index)
                    .ok_or(tencir_pauli_core::PauliError::InvalidIndex {
                        context: "fermion compilation axis",
                    })?;
            if *slot != usize::MAX {
                return Err(tencir_pauli_core::PauliError::InvalidIndex {
                    context: "duplicate fermion compilation axis",
                });
            }
            *slot = position;
        }
    }
    if positions.contains(&usize::MAX) {
        return Err(tencir_pauli_core::PauliError::InvalidStructureLength {
            expected: n_modes,
            actual: axes.iter().filter(|&&(domain, _)| domain == 0).count(),
        });
    }
    Ok(positions)
}

#[pyfunction]
pub(crate) fn structured_dense_handle<'py>(
    py: Python<'py>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<(usize, Bound<'py, PyArray1<NumpyComplex128>>)> {
    let (dimension, values) = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            let positions = fermion_axis_positions(handle.layout.n_modes, &axes)?;
            structured_dense_matrix_with_fermion_positions(
                &local_dimensions,
                &operations,
                &coefficients,
                &positions,
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok((dimension, PyArray1::from_vec(py, values)))
}

#[pyfunction]
pub(crate) fn structured_sparse_handle<'py>(
    py: Python<'py>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<StructuredSparseOutput<'py>> {
    let result = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            let positions = fermion_axis_positions(handle.layout.n_modes, &axes)?;
            structured_sparse_matrix_with_fermion_positions(
                &local_dimensions,
                &operations,
                &coefficients,
                &positions,
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok((
        result.dimension,
        PyArray1::from_vec(py, result.rows),
        PyArray1::from_vec(py, result.columns),
        PyArray1::from_vec(py, result.values),
    ))
}

/// Build and return the authoritative eager CSR arrays in one native call.
///
/// The Python facade uses this only for an explicit CSR target; COO remains a
/// separate materialization because it necessarily needs row indices.
#[pyfunction]
pub(crate) fn structured_csr_handle<'py>(
    py: Python<'py>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<NumpyStructuredCsrWithDimension<'py>> {
    let plan = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            let positions = fermion_axis_positions(handle.layout.n_modes, &axes)?;
            structured_eager_mvp_plan_with_fermion_positions(
                &local_dimensions,
                &operations,
                &coefficients,
                &positions,
                max_bytes,
            )
        })
        .map_err(map_error)?;
    let (dimension, indptr, columns, values) = plan.into_csr();
    Ok((
        dimension,
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, columns),
        PyArray1::from_vec(py, values),
    ))
}

#[pyfunction]
pub(crate) fn structured_sparse_plan_handle(
    py: Python<'_>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<StructuredMvpPlan> {
    let plan = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            let positions = fermion_axis_positions(handle.layout.n_modes, &axes)?;
            structured_mvp_plan_with_fermion_positions(
                local_dimensions,
                operations,
                coefficients,
                &positions,
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok(StructuredMvpPlan { plan })
}

#[pyfunction]
pub(crate) fn structured_eager_plan_handle(
    py: Python<'_>,
    handle: &NativeHybridOperatorHandle,
    local_dimensions: Vec<usize>,
    axes: Vec<(u8, usize)>,
    max_bytes: u128,
) -> PyResult<StructuredEagerMvpPlan> {
    let plan = py
        .allow_threads(|| {
            let operations = lower_hybrid_operations(&handle.result, &axes)?;
            let coefficients = handle.result.coefficients.clone();
            let positions = fermion_axis_positions(handle.layout.n_modes, &axes)?;
            structured_eager_mvp_plan_with_fermion_positions(
                &local_dimensions,
                &operations,
                &coefficients,
                &positions,
                max_bytes,
            )
        })
        .map_err(map_error)?;
    Ok(StructuredEagerMvpPlan { plan })
}
use super::*;
