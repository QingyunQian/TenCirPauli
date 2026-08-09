//! Private additive-charge module tree.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::charge_sector::{ChargeSectorPlan, FactorComponentKind, FactorizedLayout};
use crate::csr::{CsrSink, CSR_CHUNK_CAPACITY};
use crate::fermion_mask::FermionMaskDescriptor;
use crate::{Complex64, PauliError};
use rustc_hash::FxHashSet;

#[cfg(test)]
#[path = "tests.rs"]
mod charge_tests;
#[path = "eager.rs"]
mod eager;
#[path = "generic.rs"]
mod generic;
#[path = "kernels/mod.rs"]
mod kernels;
#[path = "schedule.rs"]
mod schedule;
#[path = "types.rs"]
mod types;

pub use eager::*;
pub use generic::*;
pub use kernels::*;
pub use schedule::*;
pub use types::*;
