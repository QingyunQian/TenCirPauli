#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeBosonOperatorHandle {
    pub(super) n_modes: usize,
    pub(super) blocks: Vec<Vec<(u32, u32, u32)>>,
    pub(super) coefficients: Vec<Complex64>,
}

impl NativeBosonOperatorHandle {
    pub(in crate::structured::implementation) fn from_result(
        n_modes: usize,
        result: BosonCanonicalResult,
    ) -> Self {
        Self {
            n_modes,
            blocks: result.0,
            coefficients: result.1,
        }
    }

    pub(super) fn to_hybrid_result(&self) -> NativeHybridOperatorHandle {
        let mut result = empty_hybrid_result(self.coefficients.len());
        for index in 0..self.coefficients.len() {
            result.fermion_present.push(false);
            result.fermion_creation.push(Vec::new());
            result.fermion_annihilation.push(Vec::new());
            result.boson_present.push(true);
            result.boson_blocks.push(self.blocks[index].clone());
            result.qubit_codes.push(Vec::new());
            result.mapped_present.push(false);
            result.mapped_codes.push(Vec::new());
            result.qudit_present.push(false);
            result.qudit_triples.push(Vec::new());
            result.coefficients.push(self.coefficients[index]);
        }
        NativeHybridOperatorHandle::from_result(
            HybridLayout {
                n_modes: 0,
                n_bosons: self.n_modes,
                nqubits: 0,
                n_qudit_sites: 0,
                qudit_dimension: 0,
            },
            result,
        )
    }

    pub(super) fn content_hash_inner(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.n_modes.hash(&mut hasher);
        self.blocks.hash(&mut hasher);
        for coefficient in &self.coefficients {
            hash_complex(*coefficient, &mut hasher);
        }
        hasher.finish()
    }
}

pub(in crate::structured::implementation) fn empty_hybrid_result(
    capacity: usize,
) -> tencir_pauli_core::HybridCanonicalResult {
    tencir_pauli_core::HybridCanonicalResult {
        fermion_present: Vec::with_capacity(capacity),
        fermion_creation: Vec::with_capacity(capacity),
        fermion_annihilation: Vec::with_capacity(capacity),
        boson_present: Vec::with_capacity(capacity),
        boson_blocks: Vec::with_capacity(capacity),
        qubit_codes: Vec::with_capacity(capacity),
        mapped_present: Vec::with_capacity(capacity),
        mapped_codes: Vec::with_capacity(capacity),
        qudit_present: Vec::with_capacity(capacity),
        qudit_triples: Vec::with_capacity(capacity),
        coefficients: Vec::with_capacity(capacity),
    }
}

fn check_native_output(
    term_count: usize,
    bytes_per_term: usize,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), tencir_pauli_core::PauliError> {
    let requested = (term_count.max(1) as u128)
        .checked_mul(bytes_per_term as u128)
        .ok_or(tencir_pauli_core::PauliError::Overflow { context })?;
    if requested > max_bytes {
        return Err(tencir_pauli_core::PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

fn coefficient_close(left: Complex64, right: Complex64, tolerance: f64) -> bool {
    (left.re - right.re).abs() <= tolerance && (left.im - right.im).abs() <= tolerance
}

pub(in crate::structured::implementation) fn same_coefficient_maps<K: Ord>(
    left: &BTreeMap<K, Complex64>,
    right: &BTreeMap<K, Complex64>,
    tolerance: f64,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|(key, value)| {
            right
                .get(key)
                .is_some_and(|other| coefficient_close(*value, *other, tolerance))
        })
}

pub(in crate::structured::implementation) fn same_structured_coefficients(
    left_creation: &[Vec<u32>],
    left_annihilation: &[Vec<u32>],
    left_coefficients: &[Complex64],
    right_creation: &[Vec<u32>],
    right_annihilation: &[Vec<u32>],
    right_coefficients: &[Complex64],
    tolerance: f64,
) -> bool {
    if left_creation.len() != right_creation.len()
        || left_annihilation.len() != right_annihilation.len()
        || left_coefficients.len() != right_coefficients.len()
    {
        return false;
    }
    left_creation
        .iter()
        .zip(left_annihilation)
        .zip(left_coefficients)
        .zip(
            right_creation
                .iter()
                .zip(right_annihilation)
                .zip(right_coefficients),
        )
        .all(
            |(((left_c, left_a), left_value), ((right_c, right_a), right_value))| {
                left_c == right_c
                    && left_a == right_a
                    && coefficient_close(*left_value, *right_value, tolerance)
            },
        )
}

pub(in crate::structured::implementation) fn boson_adjoint(
    input: &NativeBosonOperatorHandle,
) -> NativeBosonOperatorHandle {
    NativeBosonOperatorHandle {
        n_modes: input.n_modes,
        blocks: input
            .blocks
            .iter()
            .map(|word| {
                word.iter()
                    .map(|&(mode, creation, annihilation)| (mode, annihilation, creation))
                    .collect()
            })
            .collect(),
        coefficients: input
            .coefficients
            .iter()
            .map(|value| value.conj())
            .collect(),
    }
}

pub(in crate::structured::implementation) fn same_hybrid_coefficients(
    left: &NativeHybridOperatorHandle,
    right: &NativeHybridOperatorHandle,
    tolerance: f64,
) -> bool {
    let left_map = (0..left.term_count())
        .map(|index| (left.key_at(index), left.result.coefficients[index]))
        .collect::<BTreeMap<_, _>>();
    let right_map = (0..right.term_count())
        .map(|index| (right.key_at(index), right.result.coefficients[index]))
        .collect::<BTreeMap<_, _>>();
    same_coefficient_maps(&left_map, &right_map, tolerance)
}

pub(in crate::structured::implementation) fn merge_fermion_handles(
    left: &NativeFermionOperatorHandle,
    right: &NativeFermionOperatorHandle,
    max_bytes: u128,
) -> Result<NativeFermionOperatorHandle, tencir_pauli_core::PauliError> {
    check_native_output(
        left.term_count() + right.term_count(),
        left.n_modes.div_ceil(64) * 16 + 32,
        max_bytes,
        "estimating fermion operator addition",
    )?;
    let capacity = left.term_count() + right.term_count();
    let mut creation = Vec::with_capacity(capacity);
    let mut annihilation = Vec::with_capacity(capacity);
    let mut coefficients = Vec::with_capacity(capacity);
    let mut index = 0;
    let mut other_index = 0;
    while index < left.term_count() && other_index < right.term_count() {
        let left_key = (&left.creation[index], &left.annihilation[index]);
        let right_key = (
            &right.creation[other_index],
            &right.annihilation[other_index],
        );
        match left_key.cmp(&right_key) {
            Ordering::Less => {
                creation.push(left.creation[index].clone());
                annihilation.push(left.annihilation[index].clone());
                coefficients.push(left.coefficients[index]);
                index += 1;
            }
            Ordering::Greater => {
                creation.push(right.creation[other_index].clone());
                annihilation.push(right.annihilation[other_index].clone());
                coefficients.push(right.coefficients[other_index]);
                other_index += 1;
            }
            Ordering::Equal => {
                let value = left.coefficients[index] + right.coefficients[other_index];
                if value.re != 0.0 || value.im != 0.0 {
                    creation.push(left.creation[index].clone());
                    annihilation.push(left.annihilation[index].clone());
                    coefficients.push(value);
                }
                index += 1;
                other_index += 1;
            }
        }
    }
    while index < left.term_count() {
        creation.push(left.creation[index].clone());
        annihilation.push(left.annihilation[index].clone());
        coefficients.push(left.coefficients[index]);
        index += 1;
    }
    while other_index < right.term_count() {
        creation.push(right.creation[other_index].clone());
        annihilation.push(right.annihilation[other_index].clone());
        coefficients.push(right.coefficients[other_index]);
        other_index += 1;
    }
    Ok(NativeFermionOperatorHandle::from_result(
        left.n_modes,
        (creation, annihilation, coefficients),
    ))
}

pub(in crate::structured::implementation) fn scale_fermion_handle(
    input: &NativeFermionOperatorHandle,
    scalar: Complex64,
) -> Result<NativeFermionOperatorHandle, tencir_pauli_core::PauliError> {
    let mut creation = Vec::with_capacity(input.term_count());
    let mut annihilation = Vec::with_capacity(input.term_count());
    let mut coefficients = Vec::with_capacity(input.term_count());
    if scalar.re == 0.0 && scalar.im == 0.0 {
        return Ok(NativeFermionOperatorHandle::from_result(
            input.n_modes,
            (creation, annihilation, coefficients),
        ));
    }
    for index in 0..input.term_count() {
        creation.push(input.creation[index].clone());
        annihilation.push(input.annihilation[index].clone());
        coefficients.push(input.coefficients[index] * scalar);
    }
    Ok(NativeFermionOperatorHandle::from_result(
        input.n_modes,
        (creation, annihilation, coefficients),
    ))
}

pub(in crate::structured::implementation) fn merge_boson_handles(
    left: &NativeBosonOperatorHandle,
    right: &NativeBosonOperatorHandle,
    max_bytes: u128,
) -> Result<NativeBosonOperatorHandle, tencir_pauli_core::PauliError> {
    check_native_output(
        left.term_count() + right.term_count(),
        left.n_modes.div_ceil(64) * 16 + 32,
        max_bytes,
        "estimating boson operator addition",
    )?;
    let mut blocks = Vec::with_capacity(left.term_count() + right.term_count());
    let mut coefficients = Vec::with_capacity(left.term_count() + right.term_count());
    let mut index = 0;
    let mut other_index = 0;
    while index < left.term_count() && other_index < right.term_count() {
        match left.blocks[index].cmp(&right.blocks[other_index]) {
            Ordering::Less => {
                blocks.push(left.blocks[index].clone());
                coefficients.push(left.coefficients[index]);
                index += 1;
            }
            Ordering::Greater => {
                blocks.push(right.blocks[other_index].clone());
                coefficients.push(right.coefficients[other_index]);
                other_index += 1;
            }
            Ordering::Equal => {
                let value = left.coefficients[index] + right.coefficients[other_index];
                if value.re != 0.0 || value.im != 0.0 {
                    blocks.push(left.blocks[index].clone());
                    coefficients.push(value);
                }
                index += 1;
                other_index += 1;
            }
        }
    }
    while index < left.term_count() {
        blocks.push(left.blocks[index].clone());
        coefficients.push(left.coefficients[index]);
        index += 1;
    }
    while other_index < right.term_count() {
        blocks.push(right.blocks[other_index].clone());
        coefficients.push(right.coefficients[other_index]);
        other_index += 1;
    }
    Ok(NativeBosonOperatorHandle {
        n_modes: left.n_modes,
        blocks,
        coefficients,
    })
}

pub(in crate::structured::implementation) fn scale_boson_handle(
    input: &NativeBosonOperatorHandle,
    scalar: Complex64,
) -> Result<NativeBosonOperatorHandle, tencir_pauli_core::PauliError> {
    let mut blocks = Vec::with_capacity(input.term_count());
    let mut coefficients = Vec::with_capacity(input.term_count());
    if scalar.re == 0.0 && scalar.im == 0.0 {
        return Ok(NativeBosonOperatorHandle {
            n_modes: input.n_modes,
            blocks,
            coefficients,
        });
    }
    for index in 0..input.term_count() {
        blocks.push(input.blocks[index].clone());
        coefficients.push(input.coefficients[index] * scalar);
    }
    Ok(NativeBosonOperatorHandle {
        n_modes: input.n_modes,
        blocks,
        coefficients,
    })
}

fn append_hybrid_term(
    result: &mut tencir_pauli_core::HybridCanonicalResult,
    source: &NativeHybridOperatorHandle,
    index: usize,
    coefficient: Complex64,
) {
    result
        .fermion_present
        .push(source.result.fermion_present[index]);
    result
        .fermion_creation
        .push(source.result.fermion_creation[index].clone());
    result
        .fermion_annihilation
        .push(source.result.fermion_annihilation[index].clone());
    result
        .boson_present
        .push(source.result.boson_present[index]);
    result
        .boson_blocks
        .push(source.result.boson_blocks[index].clone());
    result
        .qubit_codes
        .push(source.result.qubit_codes[index].clone());
    result
        .mapped_present
        .push(source.result.mapped_present[index]);
    result
        .mapped_codes
        .push(source.result.mapped_codes[index].clone());
    result
        .qudit_present
        .push(source.result.qudit_present[index]);
    result
        .qudit_triples
        .push(source.result.qudit_triples[index].clone());
    result.coefficients.push(coefficient);
}

pub(in crate::structured::implementation) fn merge_hybrid_handles(
    left: &NativeHybridOperatorHandle,
    right: &NativeHybridOperatorHandle,
    max_bytes: u128,
) -> Result<NativeHybridOperatorHandle, tencir_pauli_core::PauliError> {
    check_native_output(
        left.term_count() + right.term_count(),
        128,
        max_bytes,
        "estimating hybrid operator addition",
    )?;
    let mut result = empty_hybrid_result(left.term_count() + right.term_count());
    let mut index = 0;
    let mut other_index = 0;
    while index < left.term_count() && other_index < right.term_count() {
        match left.key_at(index).cmp(&right.key_at(other_index)) {
            Ordering::Less => {
                append_hybrid_term(&mut result, left, index, left.result.coefficients[index]);
                index += 1;
            }
            Ordering::Greater => {
                append_hybrid_term(
                    &mut result,
                    right,
                    other_index,
                    right.result.coefficients[other_index],
                );
                other_index += 1;
            }
            Ordering::Equal => {
                let value =
                    left.result.coefficients[index] + right.result.coefficients[other_index];
                if value.re != 0.0 || value.im != 0.0 {
                    append_hybrid_term(&mut result, left, index, value);
                }
                index += 1;
                other_index += 1;
            }
        }
    }
    while index < left.term_count() {
        append_hybrid_term(&mut result, left, index, left.result.coefficients[index]);
        index += 1;
    }
    while other_index < right.term_count() {
        append_hybrid_term(
            &mut result,
            right,
            other_index,
            right.result.coefficients[other_index],
        );
        other_index += 1;
    }
    Ok(NativeHybridOperatorHandle::from_result(left.layout, result))
}

pub(in crate::structured::implementation) fn scale_hybrid_handle(
    input: &NativeHybridOperatorHandle,
    scalar: Complex64,
) -> Result<NativeHybridOperatorHandle, tencir_pauli_core::PauliError> {
    let mut result = empty_hybrid_result(input.term_count());
    if scalar.re == 0.0 && scalar.im == 0.0 {
        return Ok(NativeHybridOperatorHandle::from_result(
            input.layout,
            result,
        ));
    }
    for index in 0..input.term_count() {
        append_hybrid_term(
            &mut result,
            input,
            index,
            input.result.coefficients[index] * scalar,
        );
    }
    Ok(NativeHybridOperatorHandle::from_result(
        input.layout,
        result,
    ))
}

pub(in crate::structured::implementation) fn hybrid_to_pauli(
    layout: &HybridLayout,
    result: &tencir_pauli_core::HybridCanonicalResult,
    axes: &[(u8, usize)],
    max_bytes: u128,
) -> Result<tencir_pauli_core::PauliOperator, tencir_pauli_core::PauliError> {
    if layout.n_bosons != 0 || layout.n_qudit_sites != 0 {
        return Err(tencir_pauli_core::PauliError::InvalidSector {
            context: "hybrid-to-Pauli projection requires a pure fermion/qubit space",
        });
    }
    check_native_output(
        result.coefficients.len(),
        axes.len().div_ceil(64) * 16 + 16,
        max_bytes,
        "estimating hybrid-to-Pauli projection",
    )?;
    let mut structures = Vec::with_capacity(result.coefficients.len());
    for term_index in 0..result.coefficients.len() {
        if result.fermion_present[term_index] {
            return Err(tencir_pauli_core::PauliError::InvalidSector {
                context: "hybrid-to-Pauli projection received raw fermion terms",
            });
        }
        let mut codes = Vec::with_capacity(axes.len());
        for &(domain, index) in axes {
            let code = match domain {
                0 => {
                    if index >= layout.n_modes {
                        return Err(tencir_pauli_core::PauliError::InvalidIndex {
                            context: "hybrid fermion axis",
                        });
                    }
                    if result.mapped_present[term_index] {
                        result.mapped_codes[term_index][index]
                    } else {
                        0
                    }
                }
                1 => {
                    if index >= layout.nqubits {
                        return Err(tencir_pauli_core::PauliError::InvalidIndex {
                            context: "hybrid qubit axis",
                        });
                    }
                    result.qubit_codes[term_index][index]
                }
                _ => {
                    return Err(tencir_pauli_core::PauliError::InvalidSector {
                        context: "hybrid-to-Pauli projection received a non-qubit axis",
                    })
                }
            };
            codes.push(code);
        }
        structures.push(codes);
    }
    tencir_pauli_core::PauliOperator::from_terms(axes.len(), &structures, &result.coefficients)
}

pub(in crate::structured::implementation) type HybridNativeKey = (
    Option<(Vec<u32>, Vec<u32>)>,
    Option<Vec<(u32, u32, u32)>>,
    Vec<u8>,
    Option<Vec<u8>>,
    Option<Vec<(u32, u32, u32)>>,
);
use super::*;
