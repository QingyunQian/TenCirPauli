/// Build a finite structured MVP plan with a precomputed CSR transition graph.
pub fn structured_eager_mvp_plan(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<StructuredEagerMvpPlan, PauliError> {
    structured_eager_mvp_plan_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Build an eager MVP plan with an explicit public-axis map for raw Fermion
/// actions.
pub fn structured_eager_mvp_plan_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredEagerMvpPlan, PauliError> {
    let (grouped_terms, grouped_coefficients) =
        group_structured_terms(terms, coefficients, local_dimensions)?;
    if let Some(operator) =
        compile_packed_pauli_operator(local_dimensions, &grouped_terms, &grouped_coefficients)?
    {
        let csr = operator.csr_matrix(max_bytes)?;
        let retained_bytes = ((csr.indptr.len() * size_of::<u64>()) as u128)
            .checked_add((csr.columns.len() * size_of::<u64>()) as u128)
            .and_then(|value| {
                value.checked_add((csr.values.len() * size_of::<Complex64>()) as u128)
            })
            .ok_or(PauliError::Overflow {
                context: "estimating packed Pauli CSR retention",
            })?;
        let indptr = csr
            .indptr
            .into_iter()
            .map(|value| {
                usize::try_from(value).map_err(|_| PauliError::Overflow {
                    context: "converting structured CSR row pointer",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(StructuredEagerMvpPlan {
            graph: CsrGraph::try_new(csr.dimension, indptr, csr.columns, csr.values)?,
            estimated_bytes: retained_bytes,
            estimated_construction_peak_bytes: retained_bytes.saturating_mul(2),
            source_term_count: terms.len(),
            schedule_count: operator.terms().len(),
            execution_family: StructuredExecutionFamily::Pauli,
            native_variant: "pauli",
        });
    }
    let csr = structured_csr_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        fermion_positions,
        max_bytes,
    )?;
    let strides = mixed_radix_strides(local_dimensions)?;
    let compiled_terms = compile_mvp_operations(
        local_dimensions,
        &grouped_terms,
        fermion_positions,
        &strides,
    )?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let net_actions = grouped_terms
        .iter()
        .map(|term| compile_net_actions(term, local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let execution_family = classify_structured_execution(
        local_dimensions,
        &compiled_terms,
        &fermion_masks,
        &net_actions,
    );
    let indptr = csr
        .indptr
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| PauliError::Overflow {
                context: "converting structured CSR row pointer",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StructuredEagerMvpPlan {
        graph: CsrGraph::try_new(csr.dimension, indptr, csr.columns, csr.values)?,
        estimated_bytes: csr.estimated_bytes,
        estimated_construction_peak_bytes: csr.estimated_construction_peak_bytes,
        source_term_count: terms.len(),
        schedule_count: csr.schedule_count,
        execution_family,
        native_variant: "compiled",
    })
}

pub struct StructuredSparseResult {
    pub dimension: usize,
    pub rows: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
}

fn structured_csr_memory_estimate(
    dimension: usize,
    term_count: usize,
    buffered_entries: usize,
) -> Result<(u128, u128), PauliError> {
    let pointer_count = dimension.checked_add(1).ok_or(PauliError::Overflow {
        context: "estimating structured CSR row pointers",
    })?;
    let pointer_bytes = (pointer_count as u128)
        .checked_mul(size_of::<u64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR row pointers",
        })?;
    let chunk_count = buffered_entries
        .checked_add(CSR_CHUNK_CAPACITY - 1)
        .and_then(|value| value.checked_div(CSR_CHUNK_CAPACITY))
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR chunk count",
        })?;
    let chunk_headers = (chunk_count as u128)
        .checked_mul(size_of::<Vec<(usize, u64, Complex64)>>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR chunk headers",
        })?;
    let entry_bytes = (buffered_entries as u128)
        .checked_mul(size_of::<(usize, u64, Complex64)>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR row buffers",
        })?;
    let final_entry_bytes = (buffered_entries as u128)
        .checked_mul((size_of::<u64>() + size_of::<Complex64>()) as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR retained graph",
        })?;
    let scratch_bytes = (term_count as u128)
        .checked_mul(64)
        .and_then(|value| value.checked_add(2 * size_of::<usize>() as u128))
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR row scratch",
        })?;
    let retained_bytes =
        pointer_bytes
            .checked_add(final_entry_bytes)
            .ok_or(PauliError::Overflow {
                context: "estimating structured CSR retained graph",
            })?;
    let construction_peak_bytes = pointer_bytes
        .checked_add(chunk_headers)
        .and_then(|value| value.checked_add(entry_bytes))
        .and_then(|value| value.checked_add(pointer_bytes))
        .and_then(|value| value.checked_add(final_entry_bytes))
        .and_then(|value| value.checked_add(scratch_bytes))
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR construction peak",
        })?;
    Ok((retained_bytes, construction_peak_bytes))
}

fn structured_csr_memory_preflight(
    dimension: usize,
    term_count: usize,
    max_bytes: u128,
) -> Result<u128, PauliError> {
    let raw_entries = dimension
        .checked_mul(term_count)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR candidate transitions",
        })?;
    let (_, construction_peak_bytes) =
        structured_csr_memory_estimate(dimension, term_count, raw_entries)?;
    if construction_peak_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: construction_peak_bytes,
            limit: max_bytes,
        });
    }
    Ok(construction_peak_bytes)
}

fn structured_csr_matrix_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredCsrResult, PauliError> {
    validate_structured_inputs(local_dimensions, terms, coefficients)?;
    let (terms, coefficients) = group_structured_terms(terms, coefficients, local_dimensions)?;
    let schedule_count = terms.len();
    let dimension = mixed_radix_dimension(local_dimensions)?;
    let strides = mixed_radix_strides(local_dimensions)?;
    let compiled_terms =
        compile_mvp_operations(local_dimensions, &terms, fermion_positions, &strides)?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let net_actions = terms
        .iter()
        .map(|term| compile_net_actions(term, local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let construction_peak_bytes =
        structured_csr_memory_preflight(dimension, terms.len(), max_bytes)?;
    let mut sink = CsrSink::new_row_ordered(dimension)?;
    let mut sources: FxHashMap<usize, Complex64> =
        FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());
    let mut row_entries = Vec::<(u64, Complex64)>::with_capacity(terms.len());
    let fermion_word_count = fermion_masks
        .iter()
        .filter_map(Option::as_ref)
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    let mut fermion_mask_scratch = vec![0_u64; fermion_word_count];
    for row in 0..dimension {
        sources.clear();
        for (term_index, ((_term, fermion_descriptor), &coefficient)) in compiled_terms
            .iter()
            .zip(&fermion_masks)
            .zip(&coefficients)
            .enumerate()
        {
            if let Some(descriptor) = fermion_descriptor.as_ref() {
                let words = descriptor.required_occupied.len();
                descriptor.pack_index(row, &strides, &mut fermion_mask_scratch[..words])?;
                for (source, &flip) in fermion_mask_scratch[..words]
                    .iter_mut()
                    .zip(&descriptor.flip_mask)
                {
                    *source ^= flip;
                }
                if let Some(amplitude) =
                    descriptor.amplitude_from_words(&fermion_mask_scratch[..words], coefficient)
                {
                    let source = descriptor.apply_flip_to_index(row, &strides)?;
                    let actions = net_actions[term_index]
                        .as_deref()
                        .expect("compiled structured net action");
                    let (source, amplitude, valid) =
                        apply_compiled_net_inverse(actions, source, amplitude)?;
                    if valid {
                        sources
                            .entry(source)
                            .and_modify(|value| *value += amplitude)
                            .or_insert(amplitude);
                    }
                }
                continue;
            }
            let actions = net_actions[term_index]
                .as_deref()
                .expect("compiled structured net action");
            let (source, amplitude, valid) = apply_compiled_net_inverse(actions, row, coefficient)?;
            if valid {
                sources
                    .entry(source)
                    .and_modify(|value| *value += amplitude)
                    .or_insert(amplitude);
            }
        }
        let mut ordered_sources = sources.drain().collect::<Vec<_>>();
        ordered_sources.sort_unstable_by_key(|&(source, _)| source);
        row_entries.clear();
        for (source, value) in ordered_sources {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            row_entries.push((
                u64::try_from(source).map_err(|_| PauliError::Overflow {
                    context: "converting structured CSR column index",
                })?,
                value,
            ));
        }
        sink.finish_row(row, &row_entries)?;
    }

    let buffered_entries = sink.len();
    let (retained_bytes, actual_peak_bytes) =
        structured_csr_memory_estimate(dimension, terms.len(), buffered_entries)?;
    let (indptr, columns, values, _) = sink.into_csr()?;
    Ok(StructuredCsrResult {
        dimension,
        indptr,
        columns,
        values,
        estimated_bytes: retained_bytes,
        estimated_construction_peak_bytes: construction_peak_bytes.max(actual_peak_bytes),
        schedule_count,
    })
}

/// Compile canonical structured terms into deterministic sparse COO arrays.
///
/// This shares the exact transition convention with the dense kernel while
/// avoiding Python basis-state and dictionary loops for larger finite spaces.
pub fn structured_sparse_matrix(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<StructuredSparseResult, PauliError> {
    structured_sparse_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Compile structured terms into sparse arrays with an explicit raw-Fermion
/// public-axis map.
pub fn structured_sparse_matrix_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredSparseResult, PauliError> {
    let csr = structured_csr_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        fermion_positions,
        max_bytes,
    )?;
    let count = csr.values.len();
    let output_bytes = (count as u128)
        .checked_mul((size_of::<u64>() * 2 + size_of::<Complex64>()) as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured COO output",
        })?;
    let peak_with_output = csr
        .estimated_construction_peak_bytes
        .checked_add(output_bytes)
        .ok_or(PauliError::Overflow {
            context: "estimating structured COO construction",
        })?;
    if peak_with_output > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: peak_with_output,
            limit: max_bytes,
        });
    }
    let mut rows = Vec::with_capacity(count);
    let mut columns = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    for row in 0..csr.dimension {
        let start = usize::try_from(csr.indptr[row]).map_err(|_| PauliError::Overflow {
            context: "converting structured COO row pointer",
        })?;
        let stop = usize::try_from(csr.indptr[row + 1]).map_err(|_| PauliError::Overflow {
            context: "converting structured COO row pointer",
        })?;
        for index in start..stop {
            rows.push(u64::try_from(row).map_err(|_| PauliError::Overflow {
                context: "converting structured COO row index",
            })?);
            columns.push(csr.columns[index]);
            values.push(csr.values[index]);
        }
    }
    check_sparse_output_bytes(rows.len(), max_bytes)?;
    Ok(StructuredSparseResult {
        dimension: csr.dimension,
        rows,
        columns,
        values,
    })
}

/// Compile canonical structured terms into a row-major mixed-radix matrix.
pub fn structured_dense_matrix(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<(usize, Vec<Complex64>), PauliError> {
    structured_dense_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Compile structured terms into dense storage with an explicit raw-Fermion
/// public-axis map.
pub fn structured_dense_matrix_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<(usize, Vec<Complex64>), PauliError> {
    validate_structured_inputs(local_dimensions, terms, coefficients)?;
    let (terms, coefficients) = group_structured_terms(terms, coefficients, local_dimensions)?;
    let dimension = mixed_radix_dimension(local_dimensions)?;
    let strides = mixed_radix_strides(local_dimensions)?;
    let compiled_terms =
        compile_mvp_operations(local_dimensions, &terms, fermion_positions, &strides)?;
    let net_actions = terms
        .iter()
        .map(|term| compile_net_actions(term, local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let entries = dimension
        .checked_mul(dimension)
        .ok_or(PauliError::Overflow {
            context: "computing structured dense matrix entries",
        })?;
    let bytes = (entries as u128)
        .checked_mul(size_of::<Complex64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured dense matrix memory",
        })?;
    if bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: bytes,
            limit: max_bytes,
        });
    }
    let mut matrix = vec![Complex64::default(); entries];
    let fermion_word_count = fermion_masks
        .iter()
        .filter_map(Option::as_ref)
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    let mut fermion_mask_scratch = vec![0_u64; fermion_word_count];
    for column in 0..dimension {
        for (term_index, ((_term, fermion_descriptor), &coefficient)) in compiled_terms
            .iter()
            .zip(&fermion_masks)
            .zip(&coefficients)
            .enumerate()
        {
            if let Some(descriptor) = fermion_descriptor.as_ref() {
                let words = descriptor.required_occupied.len();
                descriptor.pack_index(column, &strides, &mut fermion_mask_scratch[..words])?;
                if let Some(amplitude) =
                    descriptor.amplitude_from_words(&fermion_mask_scratch[..words], coefficient)
                {
                    let destination = descriptor.apply_flip_to_index(column, &strides)?;
                    let actions = net_actions[term_index]
                        .as_deref()
                        .expect("compiled structured net action");
                    let (destination, amplitude, valid) =
                        apply_compiled_net_forward(actions, destination, amplitude)?;
                    if valid {
                        let entry = &mut matrix[destination * dimension + column];
                        *entry += amplitude;
                    }
                }
                continue;
            }
            let actions = net_actions[term_index]
                .as_deref()
                .expect("compiled structured net action");
            let (destination, amplitude, valid) =
                apply_compiled_net_forward(actions, column, coefficient)?;
            if valid {
                let entry = &mut matrix[destination * dimension + column];
                *entry += amplitude;
            }
        }
    }
    Ok((dimension, matrix))
}

pub(super) fn validate_structured_inputs(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
) -> Result<(), PauliError> {
    if terms.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: terms.len(),
            actual: coefficients.len(),
        });
    }
    if local_dimensions.contains(&0) {
        return Err(PauliError::InvalidStructureLength {
            expected: 1,
            actual: 0,
        });
    }
    Ok(())
}

/// Group identical coefficient-free local action schedules once at plan
/// construction.  The operation key is canonical and ordered, so schedule
/// order never depends on hash-map iteration.
pub(super) fn group_structured_terms(
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    local_dimensions: &[usize],
) -> Result<(Vec<Vec<StructuredOperation>>, Vec<Complex64>), PauliError> {
    if terms.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: terms.len(),
            actual: coefficients.len(),
        });
    }
    let mut grouped = BTreeMap::<NetKey, (Vec<StructuredOperation>, Complex64)>::new();
    for (term, &coefficient) in terms.iter().zip(coefficients) {
        let (term, phase) = canonicalize_structured_schedule(term, local_dimensions)?;
        let key = structured_net_key(&term, local_dimensions)?;
        let coefficient = coefficient * phase;
        grouped
            .entry(key)
            .and_modify(|(_, value)| *value += coefficient)
            .or_insert((term, coefficient));
    }
    let mut grouped_terms = Vec::with_capacity(grouped.len());
    let mut grouped_coefficients = Vec::with_capacity(grouped.len());
    for (_, (term, coefficient)) in grouped {
        if !is_exact_zero(coefficient) {
            grouped_terms.push(term);
            grouped_coefficients.push(coefficient);
        }
    }
    Ok((grouped_terms, grouped_coefficients))
}

type NetAxisRow = (usize, u64, u64);
type NetKey = Vec<(usize, Vec<NetAxisRow>)>;

/// Return a deterministic local transition fingerprint for a non-Fermion
/// schedule.  The fingerprint includes destination digits and exact binary
/// floating-point amplitude bits, so distinct Boson ladder polynomials or Weyl
/// phases are never merged accidentally.  Raw Fermion schedules are excluded
/// because their sign depends on occupations outside the local axis.
pub(super) fn structured_net_key(
    term: &[StructuredOperation],
    local_dimensions: &[usize],
) -> Result<NetKey, PauliError> {
    if term.iter().any(|operation| operation.kind == 3) {
        return Ok(term
            .iter()
            .map(|operation| {
                (
                    operation.axis,
                    vec![(
                        operation.p as usize,
                        operation.kind as u64,
                        operation.q as u64,
                    )],
                )
            })
            .collect());
    }
    let mut by_axis = BTreeMap::<usize, Vec<StructuredOperation>>::new();
    for &operation in term {
        by_axis.entry(operation.axis).or_default().push(operation);
    }
    let mut key = Vec::with_capacity(by_axis.len());
    for (axis, operations) in by_axis {
        let dimension = *local_dimensions.get(axis).ok_or(PauliError::InvalidIndex {
            context: "structured schedule axis",
        })?;
        if dimension == 0 {
            return Err(PauliError::InvalidStructureLength {
                expected: 1,
                actual: 0,
            });
        }
        if operations.iter().any(|operation| operation.kind == 0) && dimension != 2 {
            return Err(PauliError::InvalidIndex {
                context: "Pauli operation requires a two-level axis",
            });
        }
        let mut rows = Vec::with_capacity(dimension);
        for source in 0..dimension {
            let mut digit = source;
            let mut amplitude = Complex64::new(1.0, 0.0);
            for operation in &operations {
                match operation.kind {
                    0 => match operation.p {
                        0 => {}
                        1 => digit = 1 - digit,
                        2 => {
                            let input = digit;
                            digit = 1 - digit;
                            amplitude *= if input == 0 {
                                Complex64::new(0.0, 1.0)
                            } else {
                                Complex64::new(0.0, -1.0)
                            };
                        }
                        3 => {
                            if digit == 1 {
                                amplitude = -amplitude;
                            }
                        }
                        code => {
                            return Err(PauliError::InvalidCode {
                                code: code as u8,
                                index: axis,
                            });
                        }
                    },
                    1 => {
                        let creation =
                            usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Boson creation count",
                            })?;
                        let annihilation =
                            usize::try_from(operation.q).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Boson annihilation count",
                            })?;
                        if digit < annihilation {
                            amplitude = Complex64::default();
                            break;
                        }
                        for offset in 0..annihilation {
                            amplitude *= ((digit - offset) as f64).sqrt();
                        }
                        let remaining = digit - annihilation;
                        let Some(new_digit) = remaining.checked_add(creation) else {
                            amplitude = Complex64::default();
                            break;
                        };
                        if new_digit >= dimension {
                            amplitude = Complex64::default();
                            break;
                        }
                        for offset in 0..creation {
                            amplitude *= ((remaining + offset + 1) as f64).sqrt();
                        }
                        digit = new_digit;
                    }
                    2 => {
                        let shift =
                            usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Weyl shift",
                            })? % dimension;
                        let phase_power =
                            usize::try_from(operation.q).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Weyl phase",
                            })?;
                        let angle = 2.0 * std::f64::consts::PI * (phase_power * digit) as f64
                            / dimension as f64;
                        amplitude *= Complex64::from_polar(1.0, angle);
                        digit = (digit + shift) % dimension;
                    }
                    kind => {
                        return Err(PauliError::InvalidCode {
                            code: kind,
                            index: axis,
                        });
                    }
                }
            }
            if is_exact_zero(amplitude) {
                rows.push((0, 0, 0));
            } else {
                rows.push((digit, amplitude.re.to_bits(), amplitude.im.to_bits()));
            }
        }
        key.push((axis, rows));
    }
    Ok(key)
}

/// Canonicalize coefficient-free net schedules for operations whose domains
/// commute.  Pauli products and Weyl shifts are folded locally, while raw
/// Fermion sequences remain in their original order because their parity is
/// state dependent.  The returned phase is the exact scalar generated by the
/// local composition.
fn canonicalize_structured_schedule(
    term: &[StructuredOperation],
    local_dimensions: &[usize],
) -> Result<(Vec<StructuredOperation>, Complex64), PauliError> {
    if term.iter().any(|operation| operation.kind == 3) {
        return Ok((term.to_vec(), Complex64::new(1.0, 0.0)));
    }
    let mut by_axis = BTreeMap::<usize, Vec<StructuredOperation>>::new();
    for &operation in term {
        if operation.axis >= local_dimensions.len() {
            return Err(PauliError::InvalidIndex {
                context: "structured schedule axis",
            });
        }
        by_axis.entry(operation.axis).or_default().push(operation);
    }
    let mut phase = Complex64::new(1.0, 0.0);
    let mut canonical = Vec::with_capacity(term.len());
    for (_axis, operations) in by_axis {
        let mut local = Vec::<StructuredOperation>::with_capacity(operations.len());
        for operation in operations {
            if let Some(previous) = local.last_mut() {
                if previous.kind == 0 && operation.kind == 0 {
                    let (code, local_phase) =
                        pauli_code_product(operation.p as u8, previous.p as u8);
                    previous.p = u32::from(code);
                    phase *= local_phase;
                    continue;
                }
                if previous.kind == 2 && operation.kind == 2 {
                    let dimension = local_dimensions[operation.axis];
                    if dimension == 0 {
                        return Err(PauliError::InvalidStructureLength {
                            expected: 1,
                            actual: 0,
                        });
                    }
                    let shift = (previous.p as usize + operation.p as usize) % dimension;
                    let phase_power = (previous.q as usize + operation.q as usize) % dimension;
                    let cross =
                        (operation.q as usize * (previous.p as usize % dimension)) % dimension;
                    let angle = 2.0 * std::f64::consts::PI * cross as f64 / dimension as f64;
                    phase *= Complex64::from_polar(1.0, angle);
                    previous.p = u32::try_from(shift).map_err(|_| PauliError::Overflow {
                        context: "canonicalizing Weyl shift",
                    })?;
                    previous.q = u32::try_from(phase_power).map_err(|_| PauliError::Overflow {
                        context: "canonicalizing Weyl phase",
                    })?;
                    continue;
                }
            }
            if operation.kind == 0 && operation.p == 0 {
                continue;
            }
            local.push(operation);
        }
        local.retain(|operation| !(operation.kind == 0 && operation.p == 0));
        canonical.extend(local);
    }
    Ok((canonical, phase))
}

pub(super) fn mixed_radix_dimension(local_dimensions: &[usize]) -> Result<usize, PauliError> {
    local_dimensions.iter().try_fold(1usize, |value, &factor| {
        value.checked_mul(factor).ok_or(PauliError::Overflow {
            context: "computing mixed-radix basis dimension",
        })
    })
}

fn check_sparse_output_bytes(count: usize, max_bytes: u128) -> Result<(), PauliError> {
    let requested = (count as u128)
        .checked_mul(32)
        .ok_or(PauliError::Overflow {
            context: "estimating structured sparse output",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}
use super::*;
