#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeFermionOperatorHandle {
    pub(super) n_modes: usize,
    pub(super) creation: Vec<Vec<u32>>,
    pub(super) annihilation: Vec<Vec<u32>>,
    pub(super) coefficients: Vec<Complex64>,
}

impl NativeFermionOperatorHandle {
    pub(crate) fn from_result(n_modes: usize, result: FermionCanonicalResult) -> Self {
        Self {
            n_modes,
            creation: result.0,
            annihilation: result.1,
            coefficients: result.2,
        }
    }

    pub(super) fn to_hybrid_result(&self) -> NativeHybridOperatorHandle {
        let mut result = empty_hybrid_result(self.coefficients.len());
        for index in 0..self.coefficients.len() {
            result.fermion_present.push(true);
            result.fermion_creation.push(self.creation[index].clone());
            result
                .fermion_annihilation
                .push(self.annihilation[index].clone());
            result.boson_present.push(false);
            result.boson_blocks.push(Vec::new());
            result.qubit_codes.push(Vec::new());
            result.mapped_present.push(false);
            result.mapped_codes.push(Vec::new());
            result.qudit_present.push(false);
            result.qudit_triples.push(Vec::new());
            result.coefficients.push(self.coefficients[index]);
        }
        NativeHybridOperatorHandle::from_result(
            HybridLayout {
                n_modes: self.n_modes,
                n_bosons: 0,
                nqubits: 0,
                n_qudit_sites: 0,
                qudit_dimension: 0,
            },
            result,
        )
    }

    fn content_hash_inner(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.n_modes.hash(&mut hasher);
        self.creation.hash(&mut hasher);
        self.annihilation.hash(&mut hasher);
        for coefficient in &self.coefficients {
            hash_complex(*coefficient, &mut hasher);
        }
        hasher.finish()
    }
}

#[pymethods]
impl NativeFermionOperatorHandle {
    #[getter]
    fn n_modes(&self) -> usize {
        self.n_modes
    }

    #[getter]
    pub(super) fn term_count(&self) -> usize {
        self.coefficients.len()
    }

    fn add(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(
            &self.n_modes,
            &other.n_modes,
            "fermion handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, merge_fermion_handles)
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        run_unary_handle(py, || {
            scale_fermion_handle(self, Complex64::new(scalar_re, scalar_im))
        })
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(
            &self.n_modes,
            &other.n_modes,
            "fermion handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            multiply_fermion_terms(
                left.n_modes,
                FermionBatch {
                    creation: &left.creation,
                    annihilation: &left.annihilation,
                    coefficients: &left.coefficients,
                },
                FermionBatch {
                    creation: &right.creation,
                    annihilation: &right.annihilation,
                    coefficients: &right.coefficients,
                },
                budget,
            )
            .map(|result| Self::from_result(left.n_modes, result))
        })
    }

    fn commutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(
            &self.n_modes,
            &other.n_modes,
            "fermion handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            binary_fermion_terms(
                left.n_modes,
                FermionBatch {
                    creation: &left.creation,
                    annihilation: &left.annihilation,
                    coefficients: &left.coefficients,
                },
                FermionBatch {
                    creation: &right.creation,
                    annihilation: &right.annihilation,
                    coefficients: &right.coefficients,
                },
                budget,
                -1,
            )
            .map(|result| Self::from_result(left.n_modes, result))
        })
    }

    fn anticommutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(
            &self.n_modes,
            &other.n_modes,
            "fermion handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            binary_fermion_terms(
                left.n_modes,
                FermionBatch {
                    creation: &left.creation,
                    annihilation: &left.annihilation,
                    coefficients: &left.coefficients,
                },
                FermionBatch {
                    creation: &right.creation,
                    annihilation: &right.annihilation,
                    coefficients: &right.coefficients,
                },
                budget,
                1,
            )
            .map(|result| Self::from_result(left.n_modes, result))
        })
    }

    fn adjoint(&self, py: Python<'_>, max_bytes: usize) -> PyResult<Self> {
        let factors = self
            .annihilation
            .iter()
            .zip(&self.creation)
            .map(|(annihilation, creation)| {
                annihilation
                    .iter()
                    .rev()
                    .map(|&mode| (mode as usize, 0))
                    .chain(creation.iter().rev().map(|&mode| (mode as usize, 1)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coefficients = self
            .coefficients
            .iter()
            .map(|value| value.conj())
            .collect::<Vec<_>>();
        let result = py
            .allow_threads(|| {
                canonicalize_fermion_terms(self.n_modes, &factors, &coefficients, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(Self::from_result(self.n_modes, result))
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyFermionOutput<'py> {
        let (creation, annihilation, coefficients) = py.allow_threads(|| {
            (
                flatten_words(&self.creation),
                flatten_words(&self.annihilation),
                self.coefficients.clone(),
            )
        });
        (
            self.term_count(),
            PyArray1::from_vec(py, creation.0),
            PyArray1::from_vec(py, creation.1),
            PyArray1::from_vec(py, annihilation.0),
            PyArray1::from_vec(py, annihilation.1),
            PyArray1::from_vec(py, coefficients),
        )
    }

    fn jordan_wigner(
        &self,
        py: Python<'_>,
        max_bytes: usize,
    ) -> PyResult<NativePauliOperatorHandle> {
        let (structures, coefficients) = py
            .allow_threads(|| {
                jordan_wigner_terms(
                    self.n_modes,
                    &self.creation,
                    &self.annihilation,
                    &self.coefficients,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        let operator =
            tencir_pauli_core::PauliOperator::from_terms(self.n_modes, &structures, &coefficients)
                .map_err(map_error)?;
        Ok(NativePauliOperatorHandle::from_operator(operator))
    }

    fn to_majorana(
        &self,
        py: Python<'_>,
        max_bytes: usize,
    ) -> PyResult<NativeMajoranaOperatorHandle> {
        let result = py
            .allow_threads(|| {
                fermion_to_majorana_terms(
                    self.n_modes,
                    &self.creation,
                    &self.annihilation,
                    &self.coefficients,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        Ok(NativeMajoranaOperatorHandle::from_result(
            self.n_modes,
            result,
        ))
    }

    fn to_hybrid(&self, py: Python<'_>) -> NativeHybridOperatorHandle {
        py.allow_threads(|| self.to_hybrid_result())
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> bool {
        py.allow_threads(|| {
            let factors = self
                .annihilation
                .iter()
                .zip(&self.creation)
                .map(|(annihilation, creation)| {
                    annihilation
                        .iter()
                        .rev()
                        .map(|&mode| (mode as usize, 0))
                        .chain(creation.iter().rev().map(|&mode| (mode as usize, 1)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let coefficients = self
                .coefficients
                .iter()
                .map(|value| value.conj())
                .collect::<Vec<_>>();
            let Ok((creation, annihilation, adjoint_coefficients)) = canonicalize_fermion_terms(
                self.n_modes,
                &factors,
                &coefficients,
                usize::MAX as u128,
            ) else {
                return false;
            };
            same_structured_coefficients(
                &self.creation,
                &self.annihilation,
                &self.coefficients,
                &creation,
                &annihilation,
                &adjoint_coefficients,
                tolerance,
            )
        })
    }

    fn content_eq(&self, py: Python<'_>, other: &Self) -> bool {
        py.allow_threads(|| {
            self.n_modes == other.n_modes
                && self.creation == other.creation
                && self.annihilation == other.annihilation
                && self.coefficients == other.coefficients
        })
    }

    fn content_hash(&self, py: Python<'_>) -> u64 {
        py.allow_threads(|| self.content_hash_inner())
    }
}
use super::*;
