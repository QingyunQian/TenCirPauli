use numpy::{
    Complex64 as NumpyComplex128, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use tencir_pauli_core::{SnapshotProtocol, Snapshots, StabilizerState};

use crate::convert::map_error;
use crate::operator::NativePauliOperatorHandle;
use crate::propagation::NativeGateTape;

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeSnapshotsHandle {
    pub(crate) snapshots: Snapshots,
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeStabilizerStateHandle {
    pub(crate) state: StabilizerState,
}

fn protocol(value: u8) -> PyResult<SnapshotProtocol> {
    match value {
        0 => Ok(SnapshotProtocol::Pauli),
        1 => Ok(SnapshotProtocol::Clifford),
        _ => Err(PyValueError::new_err(
            "protocol code must be 0 (pauli) or 1 (clifford)",
        )),
    }
}

fn checked_max_bytes(value: usize) -> u128 {
    value as u128
}

#[pymethods]
impl NativeSnapshotsHandle {
    #[getter]
    fn protocol(&self) -> &'static str {
        self.snapshots.protocol().as_str()
    }

    #[getter]
    fn nqubits(&self) -> usize {
        self.snapshots.nqubits()
    }

    #[getter]
    fn nsettings(&self) -> usize {
        self.snapshots.nsettings()
    }

    #[getter]
    fn repeats(&self) -> usize {
        self.snapshots.repeats()
    }

    #[getter]
    fn shots(&self) -> PyResult<usize> {
        self.snapshots.shots().map_err(map_error)
    }

    #[allow(clippy::type_complexity)]
    fn pauli_arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray3<u8>>)> {
        let Some((bases, outcomes)) = self.snapshots.pauli_arrays() else {
            return Err(PyValueError::new_err("snapshot protocol is not pauli"));
        };
        let base_array = PyArray1::from_slice(py, bases)
            .reshape((self.nsettings(), self.nqubits()))
            .map_err(|_| PyValueError::new_err("Pauli basis shape overflow"))?;
        let outcome_array = PyArray1::from_slice(py, outcomes)
            .reshape((self.nsettings(), self.repeats(), self.nqubits()))
            .map_err(|_| PyValueError::new_err("Pauli outcome shape overflow"))?;
        Ok((base_array, outcome_array))
    }

    #[allow(clippy::type_complexity)]
    fn clifford_arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<u64>>,
        Bound<'py, PyArray2<u64>>,
        Bound<'py, PyArray2<i8>>,
        Bound<'py, PyArray3<u8>>,
        usize,
        u8,
    )> {
        let Some((frame_x, frame_z, signs, outcomes)) = self.snapshots.clifford_arrays() else {
            return Err(PyValueError::new_err("snapshot protocol is not clifford"));
        };
        let word_count = self.nqubits().div_ceil(64);
        let rows = self
            .nsettings()
            .checked_mul(self.nqubits())
            .ok_or_else(|| PyOverflowError::new_err("Clifford frame row count overflow"))?;
        let x_array = PyArray1::from_slice(py, frame_x)
            .reshape((rows, word_count))
            .map_err(|_| PyValueError::new_err("Clifford X frame shape overflow"))?;
        let z_array = PyArray1::from_slice(py, frame_z)
            .reshape((rows, word_count))
            .map_err(|_| PyValueError::new_err("Clifford Z frame shape overflow"))?;
        let sign_array = PyArray1::from_slice(py, signs)
            .reshape((self.nsettings(), self.nqubits()))
            .map_err(|_| PyValueError::new_err("Clifford sign shape overflow"))?;
        let outcome_array = PyArray1::from_slice(py, outcomes)
            .reshape((self.nsettings(), self.repeats(), self.nqubits()))
            .map_err(|_| PyValueError::new_err("Clifford outcome shape overflow"))?;
        Ok((
            x_array,
            z_array,
            sign_array,
            outcome_array,
            self.nqubits(),
            1,
        ))
    }

    fn expectation(
        &self,
        py: Python<'_>,
        observable: &NativePauliOperatorHandle,
        blocks: usize,
        return_blocks: bool,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.snapshots
                .expectation(observable.core(), blocks, return_blocks)
                .map_err(map_error)
        })
    }

    fn estimate_many(
        &self,
        py: Python<'_>,
        observables: Vec<Py<NativePauliOperatorHandle>>,
        blocks: usize,
        return_blocks: bool,
    ) -> PyResult<Vec<f64>> {
        let operators = observables
            .iter()
            .map(|observable| observable.borrow(py).core_arc())
            .collect::<Vec<_>>();
        py.allow_threads(|| {
            let operator_refs = operators.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            self.snapshots
                .estimate_many_refs(&operator_refs, blocks, return_blocks)
                .map_err(map_error)
        })
    }

    fn energy(
        &self,
        py: Python<'_>,
        hamiltonian: &NativePauliOperatorHandle,
        blocks: usize,
        return_blocks: bool,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.snapshots
                .energy(hamiltonian.core(), blocks, return_blocks)
                .map_err(map_error)
        })
    }

    fn rdm<'py>(
        &self,
        py: Python<'py>,
        keep: Vec<usize>,
        max_bytes: usize,
    ) -> PyResult<(usize, Bound<'py, PyArray1<NumpyComplex128>>)> {
        let (qubits, values) = py
            .allow_threads(|| self.snapshots.rdm(&keep, checked_max_bytes(max_bytes)))
            .map_err(map_error)?;
        Ok((qubits, PyArray1::from_vec(py, values)))
    }

    fn renyi_entropy(
        &self,
        py: Python<'_>,
        keep: Vec<usize>,
        blocks: usize,
        method: u8,
        max_bytes: usize,
    ) -> PyResult<f64> {
        py.allow_threads(|| {
            self.snapshots
                .renyi_entropy(&keep, blocks, method, checked_max_bytes(max_bytes))
                .map_err(map_error)
        })
    }

    fn energy_variance(
        &self,
        py: Python<'_>,
        hamiltonian: &NativePauliOperatorHandle,
        blocks: usize,
        max_bytes: usize,
    ) -> PyResult<f64> {
        py.allow_threads(|| {
            self.snapshots
                .energy_variance(hamiltonian.core(), blocks, checked_max_bytes(max_bytes))
                .map_err(map_error)
        })
    }

    fn fidelity(
        &self,
        py: Python<'_>,
        target: &NativeStabilizerStateHandle,
        blocks: usize,
        return_blocks: bool,
    ) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            self.snapshots
                .fidelity(&target.state, blocks, return_blocks)
                .map_err(map_error)
        })
    }

    fn fidelity_statevector(
        &self,
        py: Python<'_>,
        target: PyReadonlyArray1<'_, NumpyComplex128>,
        blocks: usize,
        return_blocks: bool,
    ) -> PyResult<Vec<f64>> {
        let values = target
            .as_slice()
            .map_err(|_| PyValueError::new_err("target must be C-contiguous"))?;
        py.allow_threads(|| {
            let values = values
                .iter()
                .map(|value| tencir_pauli_core::Complex64::new(value.re, value.im))
                .collect::<Vec<_>>();
            self.snapshots
                .fidelity_statevector(&values, blocks, return_blocks)
                .map_err(map_error)
        })
    }
}

#[pymethods]
impl NativeStabilizerStateHandle {
    #[getter]
    fn nqubits(&self) -> usize {
        self.state.nqubits()
    }

    #[allow(clippy::type_complexity)]
    fn tableau<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray2<u64>>,
        Bound<'py, PyArray2<u64>>,
        Bound<'py, PyArray1<i8>>,
    ) {
        let (x, z, signs) = self.state.to_tableau();
        let word_count = self.nqubits().div_ceil(64);
        let x_array = PyArray1::from_vec(py, x)
            .reshape((self.nqubits(), word_count))
            .expect("validated tableau shape");
        let z_array = PyArray1::from_vec(py, z)
            .reshape((self.nqubits(), word_count))
            .expect("validated tableau shape");
        (x_array, z_array, PyArray1::from_vec(py, signs))
    }
}

#[pyfunction]
pub(crate) fn shadows_from_pauli_measurements(
    py: Python<'_>,
    bases: PyReadonlyArray2<'_, u8>,
    outcomes: PyReadonlyArray3<'_, u8>,
    max_bytes: usize,
) -> PyResult<NativeSnapshotsHandle> {
    let basis_shape = bases.shape();
    let outcome_shape = outcomes.shape();
    if outcome_shape[0] != basis_shape[0] || outcome_shape[2] != basis_shape[1] {
        return Err(PyValueError::new_err(
            "Pauli bases and outcomes have incompatible setting/qubit shapes",
        ));
    }
    let basis_slice = bases
        .as_slice()
        .map_err(|_| PyValueError::new_err("bases must be C-contiguous"))?;
    let outcome_slice = outcomes
        .as_slice()
        .map_err(|_| PyValueError::new_err("outcomes must be C-contiguous"))?;
    let snapshots = py
        .allow_threads(|| {
            Snapshots::from_pauli_measurements(
                basis_shape[0],
                basis_shape[1],
                outcome_shape[1],
                basis_slice,
                outcome_slice,
                checked_max_bytes(max_bytes),
            )
        })
        .map_err(map_error)?;
    Ok(NativeSnapshotsHandle { snapshots })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shadows_from_clifford_measurements(
    py: Python<'_>,
    frame_x_words: PyReadonlyArray2<'_, u64>,
    frame_z_words: PyReadonlyArray2<'_, u64>,
    frame_signs: PyReadonlyArray2<'_, i8>,
    outcomes: PyReadonlyArray3<'_, u8>,
    nqubits: usize,
    frame_version: u8,
    max_bytes: usize,
) -> PyResult<NativeSnapshotsHandle> {
    if frame_version != 1 {
        return Err(PyValueError::new_err("unsupported Clifford frame version"));
    }
    let frame_shape = frame_x_words.shape();
    let z_shape = frame_z_words.shape();
    let signs_shape = frame_signs.shape();
    let outcome_shape = outcomes.shape();
    let expected_rows = outcome_shape[0]
        .checked_mul(nqubits)
        .ok_or_else(|| PyOverflowError::new_err("Clifford frame row count overflow"))?;
    if z_shape != frame_shape
        || frame_shape[0] != expected_rows
        || signs_shape != [outcome_shape[0], nqubits]
        || outcome_shape[2] != nqubits
    {
        return Err(PyValueError::new_err(
            "Clifford frame, signs, and outcomes have incompatible shapes",
        ));
    }
    let x = frame_x_words
        .as_slice()
        .map_err(|_| PyValueError::new_err("frame_x_words must be C-contiguous"))?;
    let z = frame_z_words
        .as_slice()
        .map_err(|_| PyValueError::new_err("frame_z_words must be C-contiguous"))?;
    let signs = frame_signs
        .as_slice()
        .map_err(|_| PyValueError::new_err("frame_signs must be C-contiguous"))?;
    let outcome_values = outcomes
        .as_slice()
        .map_err(|_| PyValueError::new_err("outcomes must be C-contiguous"))?;
    let snapshots = py
        .allow_threads(|| {
            Snapshots::from_clifford_measurements(
                outcome_shape[0],
                nqubits,
                outcome_shape[1],
                x,
                z,
                signs,
                outcome_values,
                checked_max_bytes(max_bytes),
            )
        })
        .map_err(map_error)?;
    Ok(NativeSnapshotsHandle { snapshots })
}

#[pyfunction]
pub(crate) fn shadows_sample_statevector(
    py: Python<'_>,
    state: PyReadonlyArray1<'_, NumpyComplex128>,
    nsettings: usize,
    protocol_code: u8,
    repeats: usize,
    seed: u64,
    max_bytes: usize,
) -> PyResult<NativeSnapshotsHandle> {
    let values = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    let protocol = protocol(protocol_code)?;
    let snapshots = py
        .allow_threads(|| {
            let values = values
                .iter()
                .map(|value| tencir_pauli_core::Complex64::new(value.re, value.im))
                .collect::<Vec<_>>();
            Snapshots::sample_statevector(
                &values,
                nsettings,
                protocol,
                repeats,
                seed,
                checked_max_bytes(max_bytes),
            )
        })
        .map_err(map_error)?;
    Ok(NativeSnapshotsHandle { snapshots })
}

#[pyfunction]
pub(crate) fn shadows_sample_stabilizer(
    py: Python<'_>,
    state: &NativeStabilizerStateHandle,
    nsettings: usize,
    protocol_code: u8,
    repeats: usize,
    seed: u64,
    max_bytes: usize,
) -> PyResult<NativeSnapshotsHandle> {
    let protocol = protocol(protocol_code)?;
    let snapshots = py
        .allow_threads(|| {
            Snapshots::sample_stabilizer(
                &state.state,
                nsettings,
                protocol,
                repeats,
                seed,
                checked_max_bytes(max_bytes),
            )
        })
        .map_err(map_error)?;
    Ok(NativeSnapshotsHandle { snapshots })
}

#[pyfunction]
pub(crate) fn stabilizer_zero_state(
    py: Python<'_>,
    nqubits: usize,
) -> PyResult<NativeStabilizerStateHandle> {
    let state = py
        .allow_threads(|| StabilizerState::zero_state(nqubits))
        .map_err(map_error)?;
    Ok(NativeStabilizerStateHandle { state })
}

#[pyfunction]
pub(crate) fn stabilizer_from_tableau(
    py: Python<'_>,
    x_words: PyReadonlyArray2<'_, u64>,
    z_words: PyReadonlyArray2<'_, u64>,
    signs: PyReadonlyArray1<'_, i8>,
) -> PyResult<NativeStabilizerStateHandle> {
    if x_words.shape() != z_words.shape() || x_words.shape()[0] != signs.len() {
        return Err(PyValueError::new_err(
            "tableau arrays have incompatible shapes",
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
    let state = py
        .allow_threads(|| StabilizerState::from_tableau(signs.len(), x, z, signs))
        .map_err(map_error)?;
    Ok(NativeStabilizerStateHandle { state })
}

#[pyfunction]
pub(crate) fn stabilizer_from_tape(
    py: Python<'_>,
    tape: &NativeGateTape,
) -> PyResult<NativeStabilizerStateHandle> {
    let tape = tape.core_tape();
    let state = py
        .allow_threads(|| StabilizerState::from_tape(&tape))
        .map_err(map_error)?;
    Ok(NativeStabilizerStateHandle { state })
}
