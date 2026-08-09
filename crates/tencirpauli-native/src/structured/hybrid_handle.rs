#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeHybridOperatorHandle {
    pub(super) layout: HybridLayout,
    pub(super) result: tencir_pauli_core::HybridCanonicalResult,
}

impl NativeHybridOperatorHandle {
    pub(crate) fn from_result(
        layout: HybridLayout,
        mut result: tencir_pauli_core::HybridCanonicalResult,
    ) -> Self {
        for (present, codes) in result
            .mapped_present
            .iter()
            .copied()
            .zip(&mut result.mapped_codes)
        {
            if !present {
                *codes = vec![0; layout.n_modes];
            }
        }
        Self { layout, result }
    }

    pub(crate) fn batch(&self) -> HybridBatch<'_> {
        HybridBatch {
            fermion_present: &self.result.fermion_present,
            fermion_creation: &self.result.fermion_creation,
            fermion_annihilation: &self.result.fermion_annihilation,
            boson_present: &self.result.boson_present,
            boson_blocks: &self.result.boson_blocks,
            qubit_codes: &self.result.qubit_codes,
            mapped_present: &self.result.mapped_present,
            mapped_codes: &self.result.mapped_codes,
            qudit_present: &self.result.qudit_present,
            qudit_triples: &self.result.qudit_triples,
            coefficients: &self.result.coefficients,
        }
    }

    pub(crate) fn layout(&self) -> HybridLayout {
        self.layout
    }

    pub(super) fn key_at(&self, index: usize) -> HybridNativeKey {
        (
            self.result.fermion_present[index].then(|| {
                (
                    self.result.fermion_creation[index].clone(),
                    self.result.fermion_annihilation[index].clone(),
                )
            }),
            self.result.boson_present[index].then(|| self.result.boson_blocks[index].clone()),
            self.result.qubit_codes[index].clone(),
            self.result.mapped_present[index].then(|| self.result.mapped_codes[index].clone()),
            self.result.qudit_present[index].then(|| self.result.qudit_triples[index].clone()),
        )
    }

    fn flat_parts(&self) -> HybridFlatParts {
        let (fermion_creation, fermion_creation_offsets) =
            flatten_words(&self.result.fermion_creation);
        let (fermion_annihilation, fermion_annihilation_offsets) =
            flatten_words(&self.result.fermion_annihilation);
        let (boson_blocks, boson_offsets) = flatten_blocks(&self.result.boson_blocks);
        let (qudit_triples, qudit_offsets) = flatten_blocks(&self.result.qudit_triples);
        let mut flags = Vec::with_capacity(self.term_count() * 4);
        for index in 0..self.term_count() {
            flags.extend([
                u8::from(self.result.fermion_present[index]),
                u8::from(self.result.boson_present[index]),
                u8::from(self.result.mapped_present[index]),
                u8::from(self.result.qudit_present[index]),
            ]);
        }
        let mut qubit_codes = Vec::new();
        let mut mapped_codes = Vec::new();
        for index in 0..self.term_count() {
            qubit_codes.extend_from_slice(&self.result.qubit_codes[index]);
            mapped_codes.extend_from_slice(&self.result.mapped_codes[index]);
        }
        (
            flags,
            fermion_creation,
            fermion_creation_offsets,
            fermion_annihilation,
            fermion_annihilation_offsets,
            boson_blocks,
            boson_offsets,
            qubit_codes,
            mapped_codes,
            qudit_triples,
            qudit_offsets,
            self.result.coefficients.clone(),
        )
    }

    fn direct_weyl_flat_parts(
        &self,
        max_bytes: u128,
    ) -> Result<DirectWeylFlatParts, tencir_pauli_core::PauliError> {
        if self.layout.n_modes != 0 || self.layout.n_bosons != 0 || self.layout.nqubits != 0 {
            return Err(tencir_pauli_core::PauliError::InvalidStructureLength {
                expected: self.layout.n_qudit_sites,
                actual: 0,
            });
        }
        let flat_len = self
            .term_count()
            .checked_mul(self.layout.n_qudit_sites)
            .ok_or(tencir_pauli_core::PauliError::Overflow {
                context: "estimating direct Weyl backend plan",
            })?;
        let requested = (flat_len as u128)
            .checked_mul(8)
            .and_then(|value| value.checked_add((self.term_count() as u128) * 16))
            .ok_or(tencir_pauli_core::PauliError::Overflow {
                context: "estimating direct Weyl backend plan",
            })?;
        if requested > max_bytes {
            return Err(tencir_pauli_core::PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }
        let mut a_exponents = vec![0_u32; flat_len];
        let mut b_exponents = vec![0_u32; flat_len];
        for index in 0..self.term_count() {
            for &(site, a, b) in &self.result.qudit_triples[index] {
                let site = usize::try_from(site).map_err(|_| {
                    tencir_pauli_core::PauliError::InvalidIndex {
                        context: "direct Weyl backend plan site",
                    }
                })?;
                if site >= self.layout.n_qudit_sites {
                    return Err(tencir_pauli_core::PauliError::InvalidIndex {
                        context: "direct Weyl backend plan site",
                    });
                }
                let offset = index
                    .checked_mul(self.layout.n_qudit_sites)
                    .and_then(|value| value.checked_add(site))
                    .ok_or(tencir_pauli_core::PauliError::Overflow {
                        context: "estimating direct Weyl backend plan",
                    })?;
                a_exponents[offset] = a;
                b_exponents[offset] = b;
            }
        }
        Ok((a_exponents, b_exponents, self.result.coefficients.clone()))
    }

    fn has_raw_fermions_inner(&self) -> bool {
        self.result.fermion_present.iter().any(|&present| present)
    }

    fn has_mapped_fermions_inner(&self) -> bool {
        self.result.mapped_present.iter().any(|&present| present)
    }

    fn has_mixed_fermion_roles_inner(&self) -> bool {
        self.result
            .fermion_present
            .iter()
            .zip(&self.result.mapped_present)
            .any(|(&raw, &mapped)| raw && mapped)
    }

    fn content_hash_inner(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.layout.n_modes.hash(&mut hasher);
        self.layout.n_bosons.hash(&mut hasher);
        self.layout.nqubits.hash(&mut hasher);
        self.layout.n_qudit_sites.hash(&mut hasher);
        self.layout.qudit_dimension.hash(&mut hasher);
        self.result.fermion_present.hash(&mut hasher);
        self.result.fermion_creation.hash(&mut hasher);
        self.result.fermion_annihilation.hash(&mut hasher);
        self.result.boson_present.hash(&mut hasher);
        self.result.boson_blocks.hash(&mut hasher);
        self.result.qubit_codes.hash(&mut hasher);
        self.result.mapped_present.hash(&mut hasher);
        self.result.mapped_codes.hash(&mut hasher);
        self.result.qudit_present.hash(&mut hasher);
        self.result.qudit_triples.hash(&mut hasher);
        for coefficient in &self.result.coefficients {
            hash_complex(*coefficient, &mut hasher);
        }
        hasher.finish()
    }

    fn adjoint_result(&self) -> Result<Self, String> {
        if self.layout.n_qudit_sites != 0 && self.layout.qudit_dimension < 3 {
            return Err("invalid qudit dimension".to_owned());
        }
        let dimension = self.layout.qudit_dimension;
        let mut result = self.result.clone();
        let mut coefficients = Vec::with_capacity(self.term_count());
        for index in 0..self.term_count() {
            if self.result.fermion_present[index] && self.result.mapped_present[index] {
                return Err(
                    "cannot take the adjoint of a term containing both raw and mapped fermion factors"
                        .to_owned(),
                );
            }
            if self.result.fermion_present[index] {
                result.fermion_creation[index] = self.result.fermion_annihilation[index]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
                result.fermion_annihilation[index] = self.result.fermion_creation[index]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
            }
            if self.result.boson_present[index] {
                for block in &mut result.boson_blocks[index] {
                    std::mem::swap(&mut block.1, &mut block.2);
                }
            }
            let mut phase = Complex64::new(1.0, 0.0);
            if self.result.qudit_present[index] {
                let exponent: u128 = self.result.qudit_triples[index]
                    .iter()
                    .map(|&(_, a, b)| u128::from(a) * u128::from(b))
                    .sum::<u128>()
                    % dimension as u128;
                for triple in &mut result.qudit_triples[index] {
                    triple.1 = (dimension as u32 - triple.1) % dimension as u32;
                    triple.2 = (dimension as u32 - triple.2) % dimension as u32;
                }
                let angle = 2.0 * std::f64::consts::PI * exponent as f64 / dimension as f64;
                phase = Complex64::from_polar(1.0, angle);
            }
            coefficients.push(self.result.coefficients[index].conj() * phase);
        }
        result.coefficients = coefficients;
        Ok(Self::from_result(self.layout, result))
    }
}

#[pymethods]
impl NativeHybridOperatorHandle {
    #[getter]
    pub(super) fn term_count(&self) -> usize {
        self.result.coefficients.len()
    }

    fn has_raw_fermions(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.has_raw_fermions_inner())
    }

    fn has_mapped_fermions(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.has_mapped_fermions_inner())
    }

    fn has_mixed_fermion_roles(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.has_mixed_fermion_roles_inner())
    }

    fn add(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(&self.layout, &other.layout, "hybrid handle layouts differ")?;
        run_binary_handle(py, self, other, max_bytes, merge_hybrid_handles)
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        run_unary_handle(py, || {
            scale_hybrid_handle(self, Complex64::new(scalar_re, scalar_im))
        })
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(&self.layout, &other.layout, "hybrid handle layouts differ")?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            multiply_hybrid_terms(left.layout, left.batch(), right.batch(), budget)
                .map(|result| Self::from_result(left.layout, result))
        })
    }

    fn commutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(&self.layout, &other.layout, "hybrid handle layouts differ")?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            binary_hybrid_terms(left.layout, left.batch(), right.batch(), budget, 1, -1)
                .map(|result| Self::from_result(left.layout, result))
        })
    }

    fn anticommutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(&self.layout, &other.layout, "hybrid handle layouts differ")?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            binary_hybrid_terms(left.layout, left.batch(), right.batch(), budget, 1, 1)
                .map(|result| Self::from_result(left.layout, result))
        })
    }

    fn jordan_wigner(&self, py: Python<'_>, max_bytes: usize) -> PyResult<Self> {
        if self.has_mixed_fermion_roles_inner() {
            return Err(PyValueError::new_err(
                "cannot map a term containing both raw and mapped fermion factors",
            ));
        }
        let result = py
            .allow_threads(|| {
                jordan_wigner_hybrid_terms_trusted(self.layout, self.batch(), max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.layout, result))
    }

    fn adjoint(&self, py: Python<'_>) -> PyResult<Self> {
        py.allow_threads(|| self.adjoint_result())
            .map_err(PyValueError::new_err)
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> PyResult<bool> {
        py.allow_threads(|| {
            let adjoint = self.adjoint_result()?;
            Ok::<bool, String>(same_hybrid_coefficients(self, &adjoint, tolerance))
        })
        .map_err(PyValueError::new_err)
    }

    fn content_eq(&self, py: Python<'_>, other: &Self) -> bool {
        py.allow_threads(|| {
            self.layout == other.layout
                && self.result.fermion_present == other.result.fermion_present
                && self.result.fermion_creation == other.result.fermion_creation
                && self.result.fermion_annihilation == other.result.fermion_annihilation
                && self.result.boson_present == other.result.boson_present
                && self.result.boson_blocks == other.result.boson_blocks
                && self.result.qubit_codes == other.result.qubit_codes
                && self.result.mapped_present == other.result.mapped_present
                && self.result.mapped_codes == other.result.mapped_codes
                && self.result.qudit_present == other.result.qudit_present
                && self.result.qudit_triples == other.result.qudit_triples
                && self.result.coefficients == other.result.coefficients
        })
    }

    fn content_hash(&self, py: Python<'_>) -> u64 {
        py.allow_threads(|| self.content_hash_inner())
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyHybridOutput<'py> {
        let parts = py.allow_threads(|| self.flat_parts());
        (
            self.term_count(),
            self.layout.nqubits,
            (
                PyArray1::from_vec(py, parts.0),
                PyArray1::from_vec(py, parts.1),
                PyArray1::from_vec(py, parts.2),
                PyArray1::from_vec(py, parts.3),
                PyArray1::from_vec(py, parts.4),
                PyArray1::from_vec(py, parts.5),
                PyArray1::from_vec(py, parts.6),
            ),
            (
                PyArray1::from_vec(py, parts.7),
                PyArray1::from_vec(py, parts.8),
                PyArray1::from_vec(py, parts.9),
                PyArray1::from_vec(py, parts.10),
                PyArray1::from_vec(py, parts.11),
            ),
        )
    }

    fn direct_weyl_flat<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<NumpyDirectWeylFlat<'py>> {
        let (a_exponents, b_exponents, coefficients) = py
            .allow_threads(|| self.direct_weyl_flat_parts(max_bytes))
            .map_err(map_error)?;
        Ok((
            PyArray1::from_vec(py, a_exponents),
            PyArray1::from_vec(py, b_exponents),
            PyArray1::from_vec(py, coefficients),
        ))
    }

    fn to_pauli(
        &self,
        py: Python<'_>,
        axes: Vec<(u8, usize)>,
        max_bytes: usize,
    ) -> PyResult<NativePauliOperatorHandle> {
        let operator = py
            .allow_threads(|| hybrid_to_pauli(&self.layout, &self.result, &axes, max_bytes as u128))
            .map_err(map_error)?;
        Ok(NativePauliOperatorHandle::from_operator(operator))
    }

    fn to_fermion(&self, py: Python<'_>) -> PyResult<NativeFermionOperatorHandle> {
        if self.layout.n_bosons != 0 || self.layout.nqubits != 0 || self.layout.n_qudit_sites != 0 {
            return Err(PyValueError::new_err(
                "hybrid handle is not a pure fermion layout",
            ));
        }
        py.allow_threads(|| {
            if self.result.boson_present.iter().any(|&value| value)
                || self.result.mapped_present.iter().any(|&value| value)
                || self.result.qudit_present.iter().any(|&value| value)
                || self
                    .result
                    .qubit_codes
                    .iter()
                    .any(|codes| !codes.is_empty())
            {
                return Err("hybrid handle contains non-fermion factors");
            }
            Ok(NativeFermionOperatorHandle::from_result(
                self.layout.n_modes,
                (
                    self.result.fermion_creation.clone(),
                    self.result.fermion_annihilation.clone(),
                    self.result.coefficients.clone(),
                ),
            ))
        })
        .map_err(PyValueError::new_err)
    }

    fn to_boson(&self, py: Python<'_>) -> PyResult<NativeBosonOperatorHandle> {
        if self.layout.n_modes != 0 || self.layout.nqubits != 0 || self.layout.n_qudit_sites != 0 {
            return Err(PyValueError::new_err(
                "hybrid handle is not a pure boson layout",
            ));
        }
        py.allow_threads(|| {
            if self.result.fermion_present.iter().any(|&value| value)
                || self.result.mapped_present.iter().any(|&value| value)
                || self.result.qudit_present.iter().any(|&value| value)
                || self
                    .result
                    .qubit_codes
                    .iter()
                    .any(|codes| !codes.is_empty())
            {
                return Err("hybrid handle contains non-boson factors");
            }
            Ok(NativeBosonOperatorHandle::from_result(
                self.layout.n_bosons,
                (
                    self.result.boson_blocks.clone(),
                    self.result.coefficients.clone(),
                ),
            ))
        })
        .map_err(PyValueError::new_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn embed(
        &self,
        py: Python<'_>,
        target_n_modes: usize,
        target_n_bosons: usize,
        target_nqubits: usize,
        target_n_qudit_sites: usize,
        target_qudit_dimension: usize,
        fermion_map: Vec<usize>,
        boson_map: Vec<usize>,
        qubit_map: Vec<usize>,
        qudit_map: Vec<usize>,
        max_bytes: usize,
    ) -> PyResult<Self> {
        let target_layout = HybridLayout {
            n_modes: target_n_modes,
            n_bosons: target_n_bosons,
            nqubits: target_nqubits,
            n_qudit_sites: target_n_qudit_sites,
            qudit_dimension: target_qudit_dimension,
        };
        let result = py
            .allow_threads(|| {
                embed_hybrid_terms(
                    self.layout,
                    target_layout,
                    self.batch(),
                    &fermion_map,
                    &boson_map,
                    &qubit_map,
                    &qudit_map,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        Ok(Self::from_result(target_layout, result))
    }

    fn analyze_charge(
        &self,
        py: Python<'_>,
        fermion_weights: Vec<f64>,
        boson_weights: Vec<f64>,
        qubit_levels: Vec<(f64, f64)>,
        qudit_weights: Vec<f64>,
        max_bytes: usize,
    ) -> PyResult<(bool, usize)> {
        py.allow_threads(|| {
            analyze_hybrid_charge(
                self.layout,
                self.batch(),
                &fermion_weights,
                &boson_weights,
                &qubit_levels,
                &qudit_weights,
                max_bytes as u128,
            )
        })
        .map_err(map_error)
    }

    fn termwise_conserves_charge(
        &self,
        py: Python<'_>,
        fermion_weights: Vec<f64>,
        boson_weights: Vec<f64>,
        qubit_levels: Vec<(f64, f64)>,
        qudit_weights: Vec<f64>,
    ) -> PyResult<bool> {
        py.allow_threads(|| {
            hybrid_terms_conserve_charge(
                self.layout,
                self.batch(),
                &fermion_weights,
                &boson_weights,
                &qubit_levels,
                &qudit_weights,
            )
        })
        .map_err(map_error)
    }
}
use super::*;
