//! Private structured core module tree.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::csr::{CsrApplyPolicy, CsrGraph, CsrSink, CSR_CHUNK_CAPACITY};
use crate::fermion_mask::FermionMaskDescriptor;
use crate::hamiltonian::MvpPlan;
use crate::operator::PauliOperator;
use crate::{Complex64, PauliError};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

type FermionKey = (Vec<u32>, Vec<u32>);
type BosonKey = Vec<(u32, u32, u32)>;
pub type FermionCanonicalResult = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Complex64>);
pub type BosonCanonicalResult = (Vec<Vec<(u32, u32, u32)>>, Vec<Complex64>);

#[path = "boson.rs"]
mod boson;
#[path = "common.rs"]
mod common;
#[path = "execution/common.rs"]
mod execution_common;
#[path = "execution/eager.rs"]
mod execution_eager;
#[path = "execution/lazy.rs"]
mod execution_lazy;
#[path = "fermion.rs"]
mod fermion;
#[path = "hybrid.rs"]
mod hybrid;
#[path = "integrals.rs"]
mod integrals;
#[cfg(test)]
#[path = "tests.rs"]
mod structured_tests;
#[path = "types.rs"]
mod types;

pub use boson::*;
pub(crate) use common::validate_boson_blocks;
use common::*;
pub use execution_common::*;
pub use execution_eager::*;
pub use execution_lazy::*;
pub use fermion::*;
#[cfg(test)]
use hybrid::hybrid_qudit_product;
pub use hybrid::*;
pub use integrals::*;
pub use types::*;
