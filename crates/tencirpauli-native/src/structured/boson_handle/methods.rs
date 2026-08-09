#[pymethods]
impl NativeBosonOperatorHandle {
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
            "boson handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, merge_boson_handles)
    }

    fn scale(&self, py: Python<'_>, scalar_re: f64, scalar_im: f64) -> PyResult<Self> {
        run_unary_handle(py, || {
            scale_boson_handle(self, Complex64::new(scalar_re, scalar_im))
        })
    }

    fn multiply(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(
            &self.n_modes,
            &other.n_modes,
            "boson handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            multiply_boson_terms(
                left.n_modes,
                &left.blocks,
                &left.coefficients,
                &right.blocks,
                &right.coefficients,
                budget,
            )
            .map(|result| Self::from_result(left.n_modes, result))
        })
    }

    fn commutator(&self, py: Python<'_>, other: &Self, max_bytes: usize) -> PyResult<Self> {
        require_same_layout(
            &self.n_modes,
            &other.n_modes,
            "boson handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            binary_boson_terms(
                left.n_modes,
                &left.blocks,
                &left.coefficients,
                &right.blocks,
                &right.coefficients,
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
            "boson handle mode counts differ",
        )?;
        run_binary_handle(py, self, other, max_bytes, |left, right, budget| {
            binary_boson_terms(
                left.n_modes,
                &left.blocks,
                &left.coefficients,
                &right.blocks,
                &right.coefficients,
                budget,
                1,
            )
            .map(|result| Self::from_result(left.n_modes, result))
        })
    }

    fn adjoint(&self, py: Python<'_>) -> Self {
        py.allow_threads(|| boson_adjoint(self))
    }

    fn is_hermitian(&self, py: Python<'_>, tolerance: f64) -> bool {
        py.allow_threads(|| {
            let adjoint = boson_adjoint(self);
            let left = self
                .blocks
                .iter()
                .zip(&self.coefficients)
                .map(|(key, value)| (key.clone(), *value))
                .collect::<BTreeMap<_, _>>();
            let right = adjoint
                .blocks
                .iter()
                .zip(&adjoint.coefficients)
                .map(|(key, value)| (key.clone(), *value))
                .collect::<BTreeMap<_, _>>();
            same_coefficient_maps(&left, &right, tolerance)
        })
    }

    fn materialize<'py>(&self, py: Python<'py>) -> NumpyBosonOutput<'py> {
        let (blocks, coefficients) =
            py.allow_threads(|| (flatten_blocks(&self.blocks), self.coefficients.clone()));
        (
            self.term_count(),
            PyArray1::from_vec(py, blocks.0),
            PyArray1::from_vec(py, blocks.1),
            PyArray1::from_vec(py, coefficients),
        )
    }

    fn to_hybrid(&self, py: Python<'_>) -> NativeHybridOperatorHandle {
        py.allow_threads(|| self.to_hybrid_result())
    }

    fn content_eq(&self, py: Python<'_>, other: &Self) -> bool {
        py.allow_threads(|| {
            self.n_modes == other.n_modes
                && self.blocks == other.blocks
                && self.coefficients == other.coefficients
        })
    }

    fn content_hash(&self, py: Python<'_>) -> u64 {
        py.allow_threads(|| self.content_hash_inner())
    }
}
use super::*;
