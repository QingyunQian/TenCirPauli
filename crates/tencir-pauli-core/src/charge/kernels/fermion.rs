#[path = "fermion/index.rs"]
mod index;
#[path = "fermion/plan.rs"]
mod plan;

use super::*;
pub use index::FastFermionMvpPlan;
use index::*;
pub use plan::build_fast_fermion_mvp_plan;
use plan::*;
