//! Private structured PyO3 module tree.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use numpy::{
    dtype, Complex64 as NumpyComplex128, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArray1, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    analyze_hybrid_charge, binary_boson_terms, binary_fermion_terms, binary_hybrid_terms,
    canonicalize_boson_terms, canonicalize_fermion_integrals, canonicalize_fermion_terms,
    canonicalize_hybrid_terms, embed_hybrid_terms, fermion_to_majorana_terms, hash_complex,
    hybrid_terms_conserve_charge, jordan_wigner_hybrid_terms, jordan_wigner_hybrid_terms_trusted,
    jordan_wigner_terms, multiply_boson_terms, multiply_fermion_terms, multiply_hybrid_terms,
    structured_dense_matrix_with_fermion_positions,
    structured_eager_mvp_plan_with_fermion_positions, structured_mvp_plan_with_fermion_positions,
    structured_sparse_matrix_with_fermion_positions, BosonCanonicalResult, Complex64, FermionBatch,
    FermionCanonicalResult, FermionIntegralSource, FermionSpinBlocks, FermionSpinOrdering,
    HybridBatch, HybridLayout, HybridRawBatch, PauliError,
    StructuredEagerMvpPlan as CoreStructuredEagerMvpPlan,
    StructuredMvpPlan as CoreStructuredMvpPlan, StructuredOperation,
};

use crate::convert::{complex_coefficients, map_error};
use crate::majorana::NativeMajoranaOperatorHandle;
use crate::operator::NativePauliOperatorHandle;

#[path = "boson_handle.rs"]
mod boson_handle;
#[path = "common.rs"]
mod common;
#[path = "fermion_handle.rs"]
mod fermion_handle;
#[path = "hybrid_handle.rs"]
mod hybrid_handle;
#[path = "plans.rs"]
mod plans;

pub(crate) use boson_handle::NativeBosonOperatorHandle;
use boson_handle::*;
use common::*;
pub(crate) use fermion_handle::NativeFermionOperatorHandle;
pub(crate) use hybrid_handle::NativeHybridOperatorHandle;
pub(crate) use plans::{
    structured_boson_canonicalize, structured_csr_handle, structured_dense_handle,
    structured_eager_plan_handle, structured_fermion_canonicalize,
    structured_fermion_integral_blocks, structured_fermion_integrals,
    structured_hybrid_canonicalize, structured_hybrid_jordan_wigner, structured_sparse_handle,
    structured_sparse_plan_handle, StructuredEagerMvpPlan, StructuredMvpPlan,
};
