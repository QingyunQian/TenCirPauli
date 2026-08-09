#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FermionSpinOrdering {
    Interleaved,
    AlphaThenBeta,
}

/// Compact spatial-orbital molecular integral blocks.
///
/// Each two-body block is stored in chemist axes `(p, r, q, s)`, so its value
/// is the spatial integral `(p r | q s)` used by the canonical spin-orbital
/// tensor `g[p_sigma, q_tau, r_sigma, s_tau]`.
pub struct FermionSpinBlocks<'a> {
    pub n_spatial: usize,
    pub one_alpha: &'a [Complex64],
    pub one_beta: &'a [Complex64],
    pub eri_aa: &'a [Complex64],
    pub eri_ab: &'a [Complex64],
    pub eri_ba: &'a [Complex64],
    pub eri_bb: &'a [Complex64],
    pub ordering: FermionSpinOrdering,
}

/// Input representations accepted by the framework-neutral integral importer.
pub enum FermionIntegralSource<'a> {
    SpinOrbital {
        n_modes: usize,
        one_body: &'a [Complex64],
        two_body: &'a [Complex64],
    },
    SpinBlocks(FermionSpinBlocks<'a>),
}

pub struct FermionBatch<'a> {
    pub creation: &'a [Vec<u32>],
    pub annihilation: &'a [Vec<u32>],
    pub coefficients: &'a [Complex64],
}

/// A flattened batch of canonical mixed-domain terms.
///
/// Presence flags distinguish an absent domain factor from its identity word.
/// This keeps the Python-facing representation compact while allowing one
/// coarse-grained Rust multiplication for a complete hybrid operator.
pub struct HybridBatch<'a> {
    pub fermion_present: &'a [bool],
    pub fermion_creation: &'a [Vec<u32>],
    pub fermion_annihilation: &'a [Vec<u32>],
    pub boson_present: &'a [bool],
    pub boson_blocks: &'a [Vec<(u32, u32, u32)>],
    pub qubit_codes: &'a [Vec<u8>],
    pub mapped_present: &'a [bool],
    pub mapped_codes: &'a [Vec<u8>],
    pub qudit_present: &'a [bool],
    pub qudit_triples: &'a [Vec<(u32, u32, u32)>],
    pub coefficients: &'a [Complex64],
}

pub struct HybridRawBatch<'a> {
    pub fermion_factors: &'a [Vec<(usize, u8)>],
    pub boson_factors: &'a [Vec<(usize, u8)>],
    pub qubit_codes: &'a [Vec<u8>],
    pub qudit_present: &'a [bool],
    pub qudit_triples: &'a [Vec<(u32, u32, u32)>],
    pub coefficients: &'a [Complex64],
}

/// Result of a batched mixed-domain symbolic multiplication.
#[derive(Clone)]
pub struct HybridCanonicalResult {
    pub fermion_present: Vec<bool>,
    pub fermion_creation: Vec<Vec<u32>>,
    pub fermion_annihilation: Vec<Vec<u32>>,
    pub boson_present: Vec<bool>,
    pub boson_blocks: Vec<Vec<(u32, u32, u32)>>,
    pub qubit_codes: Vec<Vec<u8>>,
    pub mapped_present: Vec<bool>,
    pub mapped_codes: Vec<Vec<u8>>,
    pub qudit_present: Vec<bool>,
    pub qudit_triples: Vec<Vec<(u32, u32, u32)>>,
    pub coefficients: Vec<Complex64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridLayout {
    pub n_modes: usize,
    pub n_bosons: usize,
    pub nqubits: usize,
    pub n_qudit_sites: usize,
    pub qudit_dimension: usize,
}

pub(super) type WeylKey = Vec<(u32, u32, u32)>;
pub(super) type HybridKey = (
    Option<FermionKey>,
    Option<BosonKey>,
    Vec<u8>,
    Option<Vec<u8>>,
    Option<WeylKey>,
);
use super::*;
