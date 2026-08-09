//! Private additive-charge module tree.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::charge_sector::{build_charge_sector_plan, ChargeSectorPlan};
use crate::csr::{CsrSink, CSR_CHUNK_CAPACITY};
use crate::fermion_mask::FermionMaskDescriptor;
use crate::{Complex64, PauliError};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

#[cfg(test)]
#[path = "tests.rs"]
mod charge_tests;
#[path = "eager.rs"]
mod eager;
#[path = "generic.rs"]
mod generic;
#[path = "kernels/additive.rs"]
mod kernel_additive;
#[path = "kernels/common.rs"]
mod kernel_common;
#[path = "kernels/fermion.rs"]
mod kernel_fermion;
#[path = "kernels/product.rs"]
mod kernel_product;
#[path = "kernels/qubit.rs"]
mod kernel_qubit;
#[path = "kernels/mod.rs"]
mod kernels;
#[path = "schedule.rs"]
mod schedule;
#[path = "types.rs"]
mod types;

pub use eager::*;
pub use generic::*;
pub use kernel_additive::*;
use kernel_common::*;
pub use kernel_fermion::{build_fast_fermion_mvp_plan, FastFermionMvpPlan};
pub use kernel_product::*;
use kernel_qubit::*;
pub use kernel_qubit::{build_fast_qubit_mvp_plan, FastQubitMvpPlan};
pub use kernels::*;
pub use schedule::*;
pub use types::*;
