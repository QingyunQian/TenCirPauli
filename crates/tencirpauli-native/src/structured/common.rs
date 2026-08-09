use super::*;

pub(super) fn require_same_layout<K: Eq>(
    left: &K,
    right: &K,
    message: &'static str,
) -> PyResult<()> {
    if left == right {
        Ok(())
    } else {
        Err(PyValueError::new_err(message))
    }
}

pub(super) fn run_binary_handle<T: Send + Sync>(
    py: Python<'_>,
    left: &T,
    right: &T,
    max_bytes: usize,
    operation: impl FnOnce(&T, &T, u128) -> Result<T, PauliError> + Send,
) -> PyResult<T> {
    py.allow_threads(|| operation(left, right, max_bytes as u128))
        .map_err(map_error)
}

pub(super) fn run_unary_handle<T: Send>(
    py: Python<'_>,
    operation: impl FnOnce() -> Result<T, PauliError> + Send,
) -> PyResult<T> {
    py.allow_threads(operation).map_err(map_error)
}

pub(super) type NumpyFermionOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(super) type NumpyBosonOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(super) type NumpyStructuredCsrOutput<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(super) type NumpyStructuredCsrWithDimension<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(super) type HybridInput = (
    Vec<bool>,
    Vec<Vec<u32>>,
    Vec<Vec<u32>>,
    Vec<bool>,
    Vec<Vec<(u32, u32, u32)>>,
    Vec<Vec<u8>>,
    Vec<bool>,
    Vec<Vec<u8>>,
    Vec<bool>,
    Vec<Vec<(u32, u32, u32)>>,
    Vec<f64>,
    Vec<f64>,
);
pub(super) type HybridRawInput = (
    Vec<Vec<(usize, u8)>>,
    Vec<Vec<(usize, u8)>>,
    Vec<Vec<u8>>,
    Vec<bool>,
    Vec<Vec<(u32, u32, u32)>>,
    Vec<f64>,
    Vec<f64>,
);
pub(super) type StructuredSparseOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(super) type DirectWeylFlatParts = (Vec<u32>, Vec<u32>, Vec<Complex64>);
pub(super) type NumpyDirectWeylFlat<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(super) type FermionMaskPhaseProfileOutput =
    Option<(usize, usize, usize, usize, u128, u128, u128, u128)>;
pub(super) type HybridFlatParts = (
    Vec<u8>,
    Vec<u32>,
    Vec<usize>,
    Vec<u32>,
    Vec<usize>,
    Vec<u32>,
    Vec<usize>,
    Vec<u8>,
    Vec<u8>,
    Vec<u32>,
    Vec<usize>,
    Vec<Complex64>,
);

pub(super) enum IntegralArray<'py> {
    Real(PyReadonlyArrayDyn<'py, f64>),
    Complex(PyReadonlyArrayDyn<'py, NumpyComplex128>),
}

pub(super) enum IntegralValues<'a> {
    Real(&'a [f64]),
    Complex(&'a [NumpyComplex128]),
}

impl IntegralArray<'_> {
    pub(super) fn values(&self, name: &str) -> PyResult<IntegralValues<'_>> {
        match self {
            Self::Real(array) => {
                Ok(IntegralValues::Real(array.as_slice().map_err(|_| {
                    PyValueError::new_err(format!("{name} must be C-contiguous"))
                })?))
            }
            Self::Complex(array) => {
                Ok(IntegralValues::Complex(array.as_slice().map_err(|_| {
                    PyValueError::new_err(format!("{name} must be C-contiguous"))
                })?))
            }
        }
    }
}

impl IntegralValues<'_> {
    pub(super) fn complex_values(&self) -> Cow<'_, [Complex64]> {
        match self {
            Self::Real(values) => Cow::Owned(
                values
                    .iter()
                    .copied()
                    .map(|value| Complex64::new(value, 0.0))
                    .collect(),
            ),
            Self::Complex(values) => Cow::Borrowed(*values),
        }
    }
}

pub(super) fn integral_array<'py>(
    value: Bound<'py, PyAny>,
    expected_shape: &[usize],
    name: &str,
) -> PyResult<IntegralArray<'py>> {
    let array = value.downcast::<PyUntypedArray>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!("{name} must be a NumPy array"))
    })?;
    if array.shape() != expected_shape {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape {:?}, got {:?}",
            expected_shape,
            array.shape()
        )));
    }
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    let f64_dtype = dtype::<f64>(array.py());
    if array.dtype().is_equiv_to(&f64_dtype) {
        let typed = array.downcast::<PyArrayDyn<f64>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!("{name} must have dtype float64"))
        })?;
        return Ok(IntegralArray::Real(typed.readonly()));
    }
    let complex_dtype = dtype::<NumpyComplex128>(array.py());
    if array.dtype().is_equiv_to(&complex_dtype) {
        let typed = array
            .downcast::<PyArrayDyn<NumpyComplex128>>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!("{name} must have dtype complex128"))
            })?;
        return Ok(IntegralArray::Complex(typed.readonly()));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must have dtype float64 or complex128"
    )))
}
pub(super) type NumpyHybridOutput<'py> = (
    usize,
    usize,
    (
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
    ),
    (
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u8>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<NumpyComplex128>>,
    ),
);

pub(super) fn flatten_words(words: &[Vec<u32>]) -> (Vec<u32>, Vec<usize>) {
    let mut payload = Vec::new();
    let mut offsets = Vec::with_capacity(words.len() + 1);
    offsets.push(0);
    for word in words {
        payload.extend_from_slice(word);
        offsets.push(payload.len());
    }
    (payload, offsets)
}

pub(super) fn flatten_blocks(words: &[Vec<(u32, u32, u32)>]) -> (Vec<u32>, Vec<usize>) {
    let mut payload = Vec::new();
    let mut offsets = Vec::with_capacity(words.len() + 1);
    offsets.push(0);
    for word in words {
        for &(mode, create, annihilate) in word {
            payload.extend([mode, create, annihilate]);
        }
        offsets.push(payload.len() / 3);
    }
    (payload, offsets)
}
