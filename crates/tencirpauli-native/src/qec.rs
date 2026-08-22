use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tencir_pauli_core::{
    pack_code_batch, BatchClassification, ErrorSetAnalysis, PackedBatchClassification,
    StabilizerCode,
};

use crate::convert::map_error;

type CheckArrayOutput<'py> = (
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray1<i8>>,
);
type LogicalArrayOutput<'py> = (
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray2<u64>>,
);
type ClassificationOutput<'py> = (
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray2<u8>>,
);
type PackedClassificationOutput<'py> = (
    Bound<'py, PyArray2<u64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray2<u8>>,
);
type OptionalBits<'py> = Option<Bound<'py, PyArray1<u8>>>;
type SingularClassificationOutput<'py> = (
    Bound<'py, PyArray1<u8>>,
    u8,
    OptionalBits<'py>,
    OptionalBits<'py>,
);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeStabilizerCodeHandle {
    code: Arc<StabilizerCode>,
}

impl NativeStabilizerCodeHandle {
    pub(crate) fn from_code(code: StabilizerCode) -> Self {
        Self {
            code: Arc::new(code),
        }
    }

    fn code(&self) -> &StabilizerCode {
        self.code.as_ref()
    }
}

#[pymethods]
impl NativeStabilizerCodeHandle {
    #[getter]
    fn nqubits(&self) -> usize {
        self.code().nqubits()
    }

    #[getter]
    fn nchecks(&self) -> usize {
        self.code().nchecks()
    }

    #[getter]
    fn rank(&self) -> usize {
        self.code().rank()
    }

    #[getter]
    fn nlogical(&self) -> usize {
        self.code().nlogical()
    }

    #[getter]
    fn nrelations(&self) -> usize {
        self.code().nrelations()
    }

    #[getter]
    fn word_count(&self) -> usize {
        self.code().word_count()
    }

    #[allow(clippy::type_complexity)]
    fn check_arrays<'py>(&self, py: Python<'py>) -> PyResult<CheckArrayOutput<'py>> {
        let (x_words, z_words, signs) = py.allow_threads(|| {
            (
                self.code().check_x_words().to_vec(),
                self.code().check_z_words().to_vec(),
                self.code().check_signs().to_vec(),
            )
        });
        Ok((
            array2_u64(py, x_words, self.nchecks(), self.word_count(), "check X")?,
            array2_u64(py, z_words, self.nchecks(), self.word_count(), "check Z")?,
            PyArray1::from_vec(py, signs),
        ))
    }

    fn check_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        let values = py.allow_threads(|| self.code().check_weights().to_vec());
        PyArray1::from_vec(py, values)
    }

    fn qubit_degrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        let values = py.allow_threads(|| self.code().qubit_degrees().to_vec());
        PyArray1::from_vec(py, values)
    }

    fn check_relations_unpacked<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let output_len = self
            .code()
            .nrelations()
            .checked_mul(self.code().nchecks())
            .ok_or_else(|| PyOverflowError::new_err("check relation shape overflow"))?;
        let values = py.allow_threads(|| {
            let mut output = vec![0_u8; output_len];
            for (row, relation) in self.code().relation_words().iter().enumerate() {
                for check in 0..self.code().nchecks() {
                    if bit_at(relation, check) {
                        output[row * self.code().nchecks() + check] = 1;
                    }
                }
            }
            output
        });
        array2_u8(
            py,
            values,
            self.code().nrelations(),
            self.code().nchecks(),
            "check relations",
        )
    }

    fn check_relations_packed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let values = py.allow_threads(|| {
            self.code()
                .relation_words()
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>()
        });
        array2_u64(
            py,
            values,
            self.code().nrelations(),
            self.code().nchecks().div_ceil(64),
            "packed check relations",
        )
    }

    #[allow(clippy::type_complexity)]
    fn logical_operator_arrays<'py>(&self, py: Python<'py>) -> PyResult<LogicalArrayOutput<'py>> {
        let (x_x, x_z, z_x, z_z) = py.allow_threads(|| self.code().logical_operator_words());
        Ok((
            array2_u64(py, x_x, self.nlogical(), self.word_count(), "logical X X")?,
            array2_u64(py, x_z, self.nlogical(), self.word_count(), "logical X Z")?,
            array2_u64(py, z_x, self.nlogical(), self.word_count(), "logical Z X")?,
            array2_u64(py, z_z, self.nlogical(), self.word_count(), "logical Z Z")?,
        ))
    }

    fn syndrome_one<'py>(
        &self,
        py: Python<'py>,
        x_words: Vec<u64>,
        z_words: Vec<u64>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let result = py
            .allow_threads(|| self.code().syndrome_one(&x_words, &z_words))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, result))
    }

    fn classify_one<'py>(
        &self,
        py: Python<'py>,
        x_words: Vec<u64>,
        z_words: Vec<u64>,
    ) -> PyResult<SingularClassificationOutput<'py>> {
        let result = py
            .allow_threads(|| self.code().classify_one(&x_words, &z_words))
            .map_err(map_error)?;
        Ok((
            PyArray1::from_vec(py, result.syndrome),
            result.kind as u8,
            optional_bits(py, result.logical_x),
            optional_bits(py, result.logical_z),
        ))
    }

    fn syndrome_many_words_unpacked<'py>(
        &self,
        py: Python<'py>,
        x_words: PyReadonlyArray2<'py, u64>,
        z_words: PyReadonlyArray2<'py, u64>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let batch = validate_word_batch(self.code(), &x_words, &z_words)?;
        let x = x_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("x_words must be C-contiguous"))?;
        let z = z_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("z_words must be C-contiguous"))?;
        let values = py
            .allow_threads(|| {
                self.code()
                    .syndrome_many_unpacked(x, z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        array2_u8(py, values, batch, self.nchecks(), "syndrome output")
    }

    fn syndrome_many_words_packed<'py>(
        &self,
        py: Python<'py>,
        x_words: PyReadonlyArray2<'py, u64>,
        z_words: PyReadonlyArray2<'py, u64>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let batch = validate_word_batch(self.code(), &x_words, &z_words)?;
        let x = x_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("x_words must be C-contiguous"))?;
        let z = z_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("z_words must be C-contiguous"))?;
        let values = py
            .allow_threads(|| {
                self.code()
                    .syndrome_many_packed(x, z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        array2_u64(
            py,
            values,
            batch,
            self.nchecks().div_ceil(64),
            "packed syndrome output",
        )
    }

    fn syndrome_many_codes_unpacked<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray2<'py, u8>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let (batch, x, z) = pack_codes(py, self.code(), codes)?;
        let values = py
            .allow_threads(|| {
                self.code()
                    .syndrome_many_unpacked(&x, &z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        array2_u8(py, values, batch, self.nchecks(), "syndrome output")
    }

    fn syndrome_many_codes_packed<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray2<'py, u8>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let (batch, x, z) = pack_codes(py, self.code(), codes)?;
        let values = py
            .allow_threads(|| {
                self.code()
                    .syndrome_many_packed(&x, &z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        array2_u64(
            py,
            values,
            batch,
            self.nchecks().div_ceil(64),
            "packed syndrome output",
        )
    }

    #[allow(clippy::type_complexity)]
    fn classify_many_words_unpacked<'py>(
        &self,
        py: Python<'py>,
        x_words: PyReadonlyArray2<'py, u64>,
        z_words: PyReadonlyArray2<'py, u64>,
        max_bytes: usize,
    ) -> PyResult<ClassificationOutput<'py>> {
        let batch = validate_word_batch(self.code(), &x_words, &z_words)?;
        let x = x_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("x_words must be C-contiguous"))?;
        let z = z_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("z_words must be C-contiguous"))?;
        let result = py
            .allow_threads(|| self.code().classify_many(x, z, batch, max_bytes as u128))
            .map_err(map_error)?;
        classification_output(py, result, batch, self.nchecks(), self.nlogical())
    }

    #[allow(clippy::type_complexity)]
    fn classify_many_words_packed<'py>(
        &self,
        py: Python<'py>,
        x_words: PyReadonlyArray2<'py, u64>,
        z_words: PyReadonlyArray2<'py, u64>,
        max_bytes: usize,
    ) -> PyResult<PackedClassificationOutput<'py>> {
        let batch = validate_word_batch(self.code(), &x_words, &z_words)?;
        let x = x_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("x_words must be C-contiguous"))?;
        let z = z_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("z_words must be C-contiguous"))?;
        let result = py
            .allow_threads(|| {
                self.code()
                    .classify_many_packed(x, z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        packed_classification_output(py, result, batch, self.nchecks(), self.nlogical())
    }

    #[allow(clippy::type_complexity)]
    fn classify_many_codes_unpacked<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray2<'py, u8>,
        max_bytes: usize,
    ) -> PyResult<ClassificationOutput<'py>> {
        let (batch, x, z) = pack_codes(py, self.code(), codes)?;
        let result = py
            .allow_threads(|| self.code().classify_many(&x, &z, batch, max_bytes as u128))
            .map_err(map_error)?;
        classification_output(py, result, batch, self.nchecks(), self.nlogical())
    }

    #[allow(clippy::type_complexity)]
    fn classify_many_codes_packed<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray2<'py, u8>,
        max_bytes: usize,
    ) -> PyResult<PackedClassificationOutput<'py>> {
        let (batch, x, z) = pack_codes(py, self.code(), codes)?;
        let result = py
            .allow_threads(|| {
                self.code()
                    .classify_many_packed(&x, &z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        packed_classification_output(py, result, batch, self.nchecks(), self.nlogical())
    }

    fn error_set_words<'py>(
        &self,
        py: Python<'py>,
        x_words: PyReadonlyArray2<'_, u64>,
        z_words: PyReadonlyArray2<'_, u64>,
        max_bytes: usize,
    ) -> PyResult<ErrorSetOutput<'py>> {
        let batch = validate_word_batch(self.code(), &x_words, &z_words)?;
        let x = x_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("x_words must be C-contiguous"))?;
        let z = z_words
            .as_slice()
            .map_err(|_| PyValueError::new_err("z_words must be C-contiguous"))?;
        let result = py
            .allow_threads(|| {
                self.code()
                    .analyze_error_set(x, z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(error_set_output(py, result))
    }

    fn error_set_codes<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray2<'_, u8>,
        max_bytes: usize,
    ) -> PyResult<ErrorSetOutput<'py>> {
        let (batch, x, z) = pack_codes(py, self.code(), codes)?;
        let result = py
            .allow_threads(|| {
                self.code()
                    .analyze_error_set(&x, &z, batch, max_bytes as u128)
            })
            .map_err(map_error)?;
        Ok(error_set_output(py, result))
    }

    #[allow(clippy::type_complexity)]
    fn corrections_words_unpacked<'py>(
        &self,
        py: Python<'py>,
        error_x: PyReadonlyArray2<'py, u64>,
        error_z: PyReadonlyArray2<'py, u64>,
        correction_x: PyReadonlyArray2<'py, u64>,
        correction_z: PyReadonlyArray2<'py, u64>,
        max_bytes: usize,
    ) -> PyResult<ClassificationOutput<'py>> {
        let batch = validate_pair_batches(
            self.code(),
            &error_x,
            &error_z,
            &correction_x,
            &correction_z,
        )?;
        let errors_x = error_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("error_x must be C-contiguous"))?;
        let errors_z = error_z
            .as_slice()
            .map_err(|_| PyValueError::new_err("error_z must be C-contiguous"))?;
        let corrections_x = correction_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("correction_x must be C-contiguous"))?;
        let corrections_z = correction_z
            .as_slice()
            .map_err(|_| PyValueError::new_err("correction_z must be C-contiguous"))?;
        let result = py
            .allow_threads(|| {
                self.code().analyze_corrections_unpacked(
                    (errors_x, errors_z),
                    (corrections_x, corrections_z),
                    batch,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        classification_output(py, result, batch, self.nchecks(), self.nlogical())
    }

    #[allow(clippy::type_complexity)]
    fn corrections_words_packed<'py>(
        &self,
        py: Python<'py>,
        error_x: PyReadonlyArray2<'py, u64>,
        error_z: PyReadonlyArray2<'py, u64>,
        correction_x: PyReadonlyArray2<'py, u64>,
        correction_z: PyReadonlyArray2<'py, u64>,
        max_bytes: usize,
    ) -> PyResult<PackedClassificationOutput<'py>> {
        let batch = validate_pair_batches(
            self.code(),
            &error_x,
            &error_z,
            &correction_x,
            &correction_z,
        )?;
        let errors_x = error_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("error_x must be C-contiguous"))?;
        let errors_z = error_z
            .as_slice()
            .map_err(|_| PyValueError::new_err("error_z must be C-contiguous"))?;
        let corrections_x = correction_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("correction_x must be C-contiguous"))?;
        let corrections_z = correction_z
            .as_slice()
            .map_err(|_| PyValueError::new_err("correction_z must be C-contiguous"))?;
        let result = py
            .allow_threads(|| {
                self.code().analyze_corrections_packed(
                    (errors_x, errors_z),
                    (corrections_x, corrections_z),
                    batch,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        packed_classification_output(py, result, batch, self.nchecks(), self.nlogical())
    }

    #[allow(clippy::type_complexity)]
    fn corrections_codes_unpacked<'py>(
        &self,
        py: Python<'py>,
        error_codes: PyReadonlyArray2<'py, u8>,
        correction_codes: PyReadonlyArray2<'py, u8>,
        max_bytes: usize,
    ) -> PyResult<ClassificationOutput<'py>> {
        let (error_batch, errors_x, errors_z) = pack_codes(py, self.code(), error_codes)?;
        let (correction_batch, corrections_x, corrections_z) =
            pack_codes(py, self.code(), correction_codes)?;
        if error_batch != correction_batch {
            return Err(PyValueError::new_err(
                "errors and corrections must have equal batch sizes",
            ));
        }
        let result = py
            .allow_threads(|| {
                self.code().analyze_corrections_unpacked(
                    (&errors_x, &errors_z),
                    (&corrections_x, &corrections_z),
                    error_batch,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        classification_output(py, result, error_batch, self.nchecks(), self.nlogical())
    }

    #[allow(clippy::type_complexity)]
    fn corrections_codes_packed<'py>(
        &self,
        py: Python<'py>,
        error_codes: PyReadonlyArray2<'py, u8>,
        correction_codes: PyReadonlyArray2<'py, u8>,
        max_bytes: usize,
    ) -> PyResult<PackedClassificationOutput<'py>> {
        let (error_batch, errors_x, errors_z) = pack_codes(py, self.code(), error_codes)?;
        let (correction_batch, corrections_x, corrections_z) =
            pack_codes(py, self.code(), correction_codes)?;
        if error_batch != correction_batch {
            return Err(PyValueError::new_err(
                "errors and corrections must have equal batch sizes",
            ));
        }
        let result = py
            .allow_threads(|| {
                self.code().analyze_corrections_packed(
                    (&errors_x, &errors_z),
                    (&corrections_x, &corrections_z),
                    error_batch,
                    max_bytes as u128,
                )
            })
            .map_err(map_error)?;
        packed_classification_output(py, result, error_batch, self.nchecks(), self.nlogical())
    }
}

type ErrorSetOutput<'py> = (
    bool,
    Option<(usize, usize)>,
    OptionalBits<'py>,
    OptionalBits<'py>,
);

fn error_set_output<'py>(py: Python<'py>, result: ErrorSetAnalysis) -> ErrorSetOutput<'py> {
    (
        result.is_correctable,
        result.conflicting_pair,
        optional_bits(py, result.logical_x),
        optional_bits(py, result.logical_z),
    )
}

fn optional_bits<'py>(py: Python<'py>, values: Option<Vec<u8>>) -> OptionalBits<'py> {
    values.map(|values| PyArray1::from_vec(py, values))
}

fn classification_output<'py>(
    py: Python<'py>,
    result: BatchClassification,
    batch: usize,
    nchecks: usize,
    nlogical: usize,
) -> PyResult<ClassificationOutput<'py>> {
    let zero_count = result.zero_syndrome_indices.len();
    let BatchClassification {
        syndromes,
        kinds,
        zero_syndrome_indices,
        logical_x,
        logical_z,
        ..
    } = result;
    let syndromes = array2_u8(py, syndromes, batch, nchecks, "classification syndromes")?;
    Ok((
        syndromes,
        PyArray1::from_vec(py, kinds),
        PyArray1::from_vec(py, zero_syndrome_indices),
        array2_u8(
            py,
            logical_x,
            zero_count,
            nlogical,
            "classification logical X",
        )?,
        array2_u8(
            py,
            logical_z,
            zero_count,
            nlogical,
            "classification logical Z",
        )?,
    ))
}

fn packed_classification_output<'py>(
    py: Python<'py>,
    result: PackedBatchClassification,
    batch: usize,
    nchecks: usize,
    nlogical: usize,
) -> PyResult<PackedClassificationOutput<'py>> {
    let syndrome_words = nchecks.div_ceil(64);
    let PackedBatchClassification {
        syndromes,
        kinds,
        zero_syndrome_indices,
        logical_x,
        logical_z,
    } = result;
    let zero_count = zero_syndrome_indices.len();
    Ok((
        array2_u64(
            py,
            syndromes,
            batch,
            syndrome_words,
            "packed classification syndromes",
        )?,
        PyArray1::from_vec(py, kinds),
        PyArray1::from_vec(py, zero_syndrome_indices),
        array2_u8(
            py,
            logical_x,
            zero_count,
            nlogical,
            "classification logical X",
        )?,
        array2_u8(
            py,
            logical_z,
            zero_count,
            nlogical,
            "classification logical Z",
        )?,
    ))
}

fn validate_word_batch(
    code: &StabilizerCode,
    x_words: &PyReadonlyArray2<'_, u64>,
    z_words: &PyReadonlyArray2<'_, u64>,
) -> PyResult<usize> {
    if x_words.shape() != z_words.shape() {
        return Err(PyValueError::new_err(
            "x_words and z_words must have identical shapes",
        ));
    }
    let shape = x_words.shape();
    if shape.len() != 2 || shape[1] != code.word_count() {
        return Err(PyValueError::new_err(format!(
            "packed Pauli batches must have shape (batch, {})",
            code.word_count()
        )));
    }
    Ok(shape[0])
}

fn validate_pair_batches(
    code: &StabilizerCode,
    error_x: &PyReadonlyArray2<'_, u64>,
    error_z: &PyReadonlyArray2<'_, u64>,
    correction_x: &PyReadonlyArray2<'_, u64>,
    correction_z: &PyReadonlyArray2<'_, u64>,
) -> PyResult<usize> {
    let error_batch = validate_word_batch(code, error_x, error_z)?;
    let correction_batch = validate_word_batch(code, correction_x, correction_z)?;
    if error_batch != correction_batch {
        return Err(PyValueError::new_err(
            "errors and corrections must have equal batch sizes",
        ));
    }
    Ok(error_batch)
}

fn pack_codes<'py>(
    py: Python<'py>,
    code: &StabilizerCode,
    codes: PyReadonlyArray2<'py, u8>,
) -> PyResult<(usize, Vec<u64>, Vec<u64>)> {
    let shape = codes.shape();
    if shape.len() != 2 || shape[1] != code.nqubits() {
        return Err(PyValueError::new_err(format!(
            "Pauli-code batches must have shape (batch, {})",
            code.nqubits()
        )));
    }
    let values = codes
        .as_slice()
        .map_err(|_| PyValueError::new_err("Pauli-code batches must be C-contiguous"))?;
    let batch = shape[0];
    py.allow_threads(|| pack_code_batch(code.nqubits(), values, batch))
        .map(|(x, z)| (batch, x, z))
        .map_err(map_error)
}

fn bit_at(row: &[u64], index: usize) -> bool {
    row[index / 64] & (1_u64 << (index % 64)) != 0
}

fn array2_u8<'py>(
    py: Python<'py>,
    values: Vec<u8>,
    rows: usize,
    columns: usize,
    context: &str,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| PyOverflowError::new_err(format!("{context} shape overflow")))?;
    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{context} returned {} values, expected {expected}",
            values.len()
        )));
    }
    PyArray1::from_vec(py, values)
        .reshape((rows, columns))
        .map_err(|_| PyOverflowError::new_err(format!("{context} shape overflow")))
}

fn array2_u64<'py>(
    py: Python<'py>,
    values: Vec<u64>,
    rows: usize,
    columns: usize,
    context: &str,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| PyOverflowError::new_err(format!("{context} shape overflow")))?;
    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{context} returned {} values, expected {expected}",
            values.len()
        )));
    }
    PyArray1::from_vec(py, values)
        .reshape((rows, columns))
        .map_err(|_| PyOverflowError::new_err(format!("{context} shape overflow")))
}

#[pyfunction]
#[pyo3(signature = (nqubits, x_words, z_words, signs, max_bytes))]
pub(crate) fn stabilizer_code_from_packed(
    py: Python<'_>,
    nqubits: usize,
    x_words: PyReadonlyArray2<'_, u64>,
    z_words: PyReadonlyArray2<'_, u64>,
    signs: PyReadonlyArray1<'_, i8>,
    max_bytes: usize,
) -> PyResult<NativeStabilizerCodeHandle> {
    if x_words.shape() != z_words.shape() || x_words.shape()[0] != signs.len() {
        return Err(PyValueError::new_err(
            "packed stabilizer arrays have incompatible shapes",
        ));
    }
    let x = x_words
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_words must be C-contiguous"))?;
    let z = z_words
        .as_slice()
        .map_err(|_| PyValueError::new_err("z_words must be C-contiguous"))?;
    let signs = signs
        .as_slice()
        .map_err(|_| PyValueError::new_err("signs must be C-contiguous"))?;
    let code = py
        .allow_threads(|| StabilizerCode::from_packed(nqubits, x, z, signs, max_bytes as u128))
        .map_err(map_error)?;
    Ok(NativeStabilizerCodeHandle::from_code(code))
}

#[pyfunction]
#[pyo3(signature = (x, z, signs, max_bytes))]
pub(crate) fn stabilizer_code_from_symplectic(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, u8>,
    z: PyReadonlyArray2<'_, u8>,
    signs: PyReadonlyArray1<'_, i8>,
    max_bytes: usize,
) -> PyResult<NativeStabilizerCodeHandle> {
    if x.shape() != z.shape() || x.shape()[0] != signs.len() {
        return Err(PyValueError::new_err(
            "symplectic stabilizer arrays have incompatible shapes",
        ));
    }
    let shape = x.shape();
    let nqubits = shape[1];
    let x_values = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let z_values = z
        .as_slice()
        .map_err(|_| PyValueError::new_err("z must be C-contiguous"))?;
    let signs = signs
        .as_slice()
        .map_err(|_| PyValueError::new_err("signs must be C-contiguous"))?;
    let code = py
        .allow_threads(|| {
            StabilizerCode::from_binary(nqubits, x_values, z_values, signs, max_bytes as u128)
        })
        .map_err(map_error)?;
    Ok(NativeStabilizerCodeHandle::from_code(code))
}

#[pyfunction]
#[pyo3(signature = (x_checks, z_checks, x_signs, z_signs, max_bytes))]
pub(crate) fn stabilizer_code_from_css(
    py: Python<'_>,
    x_checks: PyReadonlyArray2<'_, u8>,
    z_checks: PyReadonlyArray2<'_, u8>,
    x_signs: PyReadonlyArray1<'_, i8>,
    z_signs: PyReadonlyArray1<'_, i8>,
    max_bytes: usize,
) -> PyResult<NativeStabilizerCodeHandle> {
    if x_checks.shape().len() != 2 || z_checks.shape().len() != 2 {
        return Err(PyValueError::new_err("CSS checks must be two-dimensional"));
    }
    if x_checks.shape()[1] != z_checks.shape()[1]
        || x_checks.shape()[0] != x_signs.len()
        || z_checks.shape()[0] != z_signs.len()
    {
        return Err(PyValueError::new_err(
            "CSS checks and signs have incompatible shapes",
        ));
    }
    let nqubits = x_checks.shape()[1];
    let x_values = x_checks
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_checks must be C-contiguous"))?;
    let z_values = z_checks
        .as_slice()
        .map_err(|_| PyValueError::new_err("z_checks must be C-contiguous"))?;
    let x_sign_values = x_signs
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_signs must be C-contiguous"))?;
    let z_sign_values = z_signs
        .as_slice()
        .map_err(|_| PyValueError::new_err("z_signs must be C-contiguous"))?;
    let code = py
        .allow_threads(|| {
            StabilizerCode::from_css_binary(
                nqubits,
                x_values,
                z_values,
                x_sign_values,
                z_sign_values,
                max_bytes as u128,
            )
        })
        .map_err(map_error)?;
    Ok(NativeStabilizerCodeHandle::from_code(code))
}
