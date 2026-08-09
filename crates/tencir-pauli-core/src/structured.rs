//! Native finite-basis kernels shared by structured Python operators.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::csr::{CsrSink, CSR_CHUNK_CAPACITY};
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

/// Spin-orbital ordering used when expanding compact spatial integral blocks.
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

type WeylKey = Vec<(u32, u32, u32)>;
type HybridKey = (
    Option<FermionKey>,
    Option<BosonKey>,
    Vec<u8>,
    Option<Vec<u8>>,
    Option<WeylKey>,
);

fn validate_embedding_map(
    mapping: &[usize],
    source_count: usize,
    target_count: usize,
) -> Result<(), PauliError> {
    if mapping.len() != source_count || mapping.iter().any(|&value| value >= target_count) {
        return Err(PauliError::InvalidStructureLength {
            expected: source_count,
            actual: mapping.len(),
        });
    }
    let mut seen = FxHashMap::default();
    for &value in mapping {
        if seen.insert(value, ()).is_some() {
            return Err(PauliError::InvalidStructureLength {
                expected: source_count,
                actual: mapping.len(),
            });
        }
    }
    Ok(())
}

fn embedding_sign(values: &[u32], mapping: &[usize], descending: bool) -> f64 {
    let mut odd = false;
    for (index, &value) in values.iter().enumerate() {
        let left = mapping[value as usize];
        for &right_value in &values[index + 1..] {
            let right = mapping[right_value as usize];
            if if descending {
                left < right
            } else {
                left > right
            } {
                odd = !odd;
            }
        }
    }
    if odd {
        -1.0
    } else {
        1.0
    }
}

/// Embed a canonical mixed-domain operator into a larger ordered layout.
///
/// The source batch remains borrowed for the complete operation. Factors are
/// remapped, fermionic permutation signs are applied, and collisions are
/// aggregated before the canonical result crosses the FFI boundary.
#[allow(clippy::too_many_arguments)]
pub fn embed_hybrid_terms(
    source_layout: HybridLayout,
    target_layout: HybridLayout,
    batch: HybridBatch<'_>,
    fermion_map: &[usize],
    boson_map: &[usize],
    qubit_map: &[usize],
    qudit_map: &[usize],
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    validate_embedding_map(fermion_map, source_layout.n_modes, target_layout.n_modes)?;
    validate_embedding_map(boson_map, source_layout.n_bosons, target_layout.n_bosons)?;
    validate_embedding_map(qubit_map, source_layout.nqubits, target_layout.nqubits)?;
    validate_embedding_map(
        qudit_map,
        source_layout.n_qudit_sites,
        target_layout.n_qudit_sites,
    )?;
    if source_layout.n_qudit_sites != 0
        && source_layout.qudit_dimension != target_layout.qudit_dimension
    {
        return Err(PauliError::InvalidStructureLength {
            expected: source_layout.qudit_dimension,
            actual: target_layout.qudit_dimension,
        });
    }
    check_structured_bytes(
        batch.coefficients.len(),
        max_bytes,
        "structured embedding output",
    )?;
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..batch.coefficients.len() {
        let fermion = if batch.fermion_present[index] {
            let creation = batch.fermion_creation[index]
                .iter()
                .map(|&mode| fermion_map[mode as usize] as u32)
                .collect::<Vec<_>>();
            let annihilation = batch.fermion_annihilation[index]
                .iter()
                .map(|&mode| fermion_map[mode as usize] as u32)
                .collect::<Vec<_>>();
            let sign = embedding_sign(&batch.fermion_creation[index], fermion_map, false)
                * embedding_sign(&batch.fermion_annihilation[index], fermion_map, true);
            let mut creation = creation;
            let mut annihilation = annihilation;
            creation.sort_unstable();
            annihilation.sort_unstable_by(|left, right| right.cmp(left));
            Some((creation, annihilation, sign))
        } else {
            None
        };
        let boson = if batch.boson_present[index] {
            let mut blocks = batch.boson_blocks[index]
                .iter()
                .map(|&(mode, creation, annihilation)| {
                    (boson_map[mode as usize] as u32, creation, annihilation)
                })
                .collect::<Vec<_>>();
            blocks.sort_unstable_by_key(|block| block.0);
            Some(blocks)
        } else {
            None
        };
        let mut qubit = vec![0_u8; target_layout.nqubits];
        for (source, &code) in batch.qubit_codes[index].iter().enumerate() {
            qubit[qubit_map[source]] = code;
        }
        let mapped = if batch.mapped_present[index] {
            let mut codes = vec![0_u8; target_layout.n_modes];
            for (source, &code) in batch.mapped_codes[index].iter().enumerate() {
                codes[fermion_map[source]] = code;
            }
            Some(codes)
        } else {
            None
        };
        let qudit = if batch.qudit_present[index] {
            let mut triples = batch.qudit_triples[index]
                .iter()
                .map(|&(site, a, b)| (qudit_map[site as usize] as u32, a, b))
                .collect::<Vec<_>>();
            triples.sort_unstable_by_key(|triple| triple.0);
            Some(triples)
        } else {
            None
        };
        let fermion_sign = fermion.as_ref().map_or(1.0, |(_, _, sign)| *sign);
        let fermion_key = fermion.map(|(creation, annihilation, _)| (creation, annihilation));
        push_aggregate(
            &mut aggregate,
            (fermion_key, boson, qubit, mapped, qudit),
            batch.coefficients[index] * fermion_sign,
            &mut total_values,
            max_bytes,
        )?;
    }
    finish_hybrid_aggregate(aggregate)
}

/// Analyze a diagonal additive charge directly on canonical hybrid storage.
pub fn analyze_hybrid_charge(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    fermion_weights: &[f64],
    boson_weights: &[f64],
    qubit_levels: &[(f64, f64)],
    qudit_weights: &[f64],
    max_bytes: u128,
) -> Result<(bool, usize), PauliError> {
    if fermion_weights.len() != layout.n_modes
        || boson_weights.len() != layout.n_bosons
        || qubit_levels.len() != layout.nqubits
        || qudit_weights.len() != layout.n_qudit_sites
    {
        return Err(PauliError::InvalidStructureLength {
            expected: layout.n_modes + layout.n_bosons + layout.nqubits + layout.n_qudit_sites,
            actual: fermion_weights.len()
                + boson_weights.len()
                + qubit_levels.len()
                + qudit_weights.len(),
        });
    }
    if batch.mapped_present.iter().any(|&present| present) {
        return Err(PauliError::InvalidStructureLength {
            expected: 0,
            actual: 1,
        });
    }
    check_structured_bytes(
        batch.coefficients.len(),
        max_bytes,
        "estimating additive-charge analysis",
    )?;
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut nonconserving_qudit: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..batch.coefficients.len() {
        let fermion_key = if batch.fermion_present[index] {
            Some((
                batch.fermion_creation[index].clone(),
                batch.fermion_annihilation[index].clone(),
            ))
        } else {
            None
        };
        let boson_key = if batch.boson_present[index] {
            Some(batch.boson_blocks[index].clone())
        } else {
            None
        };
        let qubit_key = batch.qubit_codes[index].clone();
        let qudit_key = if batch.qudit_present[index] {
            Some(batch.qudit_triples[index].clone())
        } else {
            None
        };
        let base_key = (
            fermion_key.clone(),
            boson_key.clone(),
            qubit_key.clone(),
            None,
            qudit_key.clone(),
        );
        let mut raw_delta = 0.0;
        if batch.fermion_present[index] {
            raw_delta += batch.fermion_creation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
            raw_delta -= batch.fermion_annihilation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
        }
        if batch.boson_present[index] {
            raw_delta += batch.boson_blocks[index]
                .iter()
                .map(|&(mode, creation, annihilation)| {
                    boson_weights[mode as usize] * (creation as f64 - annihilation as f64)
                })
                .sum::<f64>();
        }
        if batch.qudit_present[index]
            && batch.qudit_triples[index]
                .iter()
                .any(|&(site, a, _)| a != 0 && qudit_weights[site as usize] != 0.0)
        {
            push_aggregate(
                &mut nonconserving_qudit,
                base_key.clone(),
                batch.coefficients[index],
                &mut total_values,
                max_bytes,
            )?;
        }
        if raw_delta != 0.0 {
            push_aggregate(
                &mut aggregate,
                base_key.clone(),
                batch.coefficients[index] * -raw_delta,
                &mut total_values,
                max_bytes,
            )?;
        }
        for (qubit, &code) in qubit_key.iter().enumerate() {
            if code != 1 && code != 2 {
                continue;
            }
            let difference = qubit_levels[qubit].0 - qubit_levels[qubit].1;
            if difference == 0.0 {
                continue;
            }
            let mut changed = qubit_key.clone();
            changed[qubit] = if code == 1 { 2 } else { 1 };
            let key = (
                fermion_key.clone(),
                boson_key.clone(),
                changed,
                None,
                qudit_key.clone(),
            );
            let scale = if code == 1 { -difference } else { difference };
            push_aggregate(
                &mut aggregate,
                key,
                batch.coefficients[index] * Complex64::new(0.0, scale),
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    let nonzero = aggregate
        .into_values()
        .chain(nonconserving_qudit.into_values())
        .filter_map(|mut values| {
            values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
            let value = deterministic_sum(values);
            (!is_exact_zero(value)).then_some(())
        })
        .count();
    Ok((nonzero == 0, nonzero))
}

/// Return whether every canonical hybrid term preserves a diagonal charge.
pub fn hybrid_terms_conserve_charge(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    fermion_weights: &[f64],
    boson_weights: &[f64],
    qubit_levels: &[(f64, f64)],
    qudit_weights: &[f64],
) -> Result<bool, PauliError> {
    if fermion_weights.len() != layout.n_modes
        || boson_weights.len() != layout.n_bosons
        || qubit_levels.len() != layout.nqubits
        || qudit_weights.len() != layout.n_qudit_sites
    {
        return Err(PauliError::InvalidStructureLength {
            expected: layout.n_modes + layout.n_bosons + layout.nqubits + layout.n_qudit_sites,
            actual: fermion_weights.len()
                + boson_weights.len()
                + qubit_levels.len()
                + qudit_weights.len(),
        });
    }
    for index in 0..batch.coefficients.len() {
        if batch.mapped_present[index] {
            return Ok(false);
        }
        if batch.qudit_present[index]
            && batch.qudit_triples[index]
                .iter()
                .any(|&(site, a, _)| a != 0 && qudit_weights[site as usize] != 0.0)
        {
            return Ok(false);
        }
        let mut delta = 0.0;
        if batch.fermion_present[index] {
            delta += batch.fermion_creation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
            delta -= batch.fermion_annihilation[index]
                .iter()
                .map(|&mode| fermion_weights[mode as usize])
                .sum::<f64>();
        }
        if batch.boson_present[index] {
            delta += batch.boson_blocks[index]
                .iter()
                .map(|&(mode, creation, annihilation)| {
                    boson_weights[mode as usize] * (creation as f64 - annihilation as f64)
                })
                .sum::<f64>();
        }
        if delta != 0.0
            || batch.qubit_codes[index]
                .iter()
                .enumerate()
                .any(|(qubit, &code)| {
                    (code == 1 || code == 2) && qubit_levels[qubit].0 != qubit_levels[qubit].1
                })
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Multiply complete canonical hybrid operators in one coarse-grained call.
pub fn multiply_hybrid_terms(
    layout: HybridLayout,
    left: HybridBatch<'_>,
    right: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    binary_hybrid_terms(layout, left, right, max_bytes, 1, 0)
}

/// Compute a commutator or anticommutator of canonical hybrid operators in a
/// single aggregate. `reverse_sign` is `-1` for a commutator and `1` for an
/// anticommutator; `forward_sign` is kept explicit so the same kernel also
/// serves the ordinary product.
pub fn binary_hybrid_terms(
    layout: HybridLayout,
    left: HybridBatch<'_>,
    right: HybridBatch<'_>,
    max_bytes: u128,
    forward_sign: i8,
    reverse_sign: i8,
) -> Result<HybridCanonicalResult, PauliError> {
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating hybrid product expansion",
        })?;
    let directional_count = if reverse_sign == 0 {
        pair_count
    } else {
        pair_count.checked_mul(2).ok_or(PauliError::Overflow {
            context: "estimating hybrid binary expansion",
        })?
    };
    check_structured_bytes(directional_count, max_bytes, "hybrid product expansion")?;

    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            accumulate_hybrid_product(
                layout,
                &left,
                left_index,
                &right,
                right_index,
                forward_sign,
                &mut aggregate,
                &mut total_values,
                max_bytes,
            )?;
            if reverse_sign != 0 {
                accumulate_hybrid_product(
                    layout,
                    &right,
                    right_index,
                    &left,
                    left_index,
                    reverse_sign,
                    &mut aggregate,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_hybrid_aggregate(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_hybrid_product(
    layout: HybridLayout,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
    direction_sign: i8,
    aggregate: &mut FxHashMap<HybridKey, Vec<Complex64>>,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let fermion_products =
        hybrid_fermion_products(left, left_index, right, right_index, max_bytes)?;
    let boson_products = hybrid_boson_products(left, left_index, right, right_index, max_bytes)?;
    let (qubit_codes, qubit_phase) = multiply_pauli_codes(
        &left.qubit_codes[left_index],
        &right.qubit_codes[right_index],
    );
    let (mapped_codes, mapped_phase) =
        hybrid_mapped_product(layout.n_modes, left, left_index, right, right_index);
    let (qudit_triples, qudit_phase) =
        hybrid_qudit_product(layout.qudit_dimension, left, left_index, right, right_index);
    let scalar = left.coefficients[left_index]
        * right.coefficients[right_index]
        * qubit_phase
        * mapped_phase
        * qudit_phase
        * f64::from(direction_sign);
    for (fermion, fermion_integer) in &fermion_products {
        for (boson, boson_integer) in &boson_products {
            let value = scalar
                * checked_integer_to_f64(*fermion_integer, "fermion product expansion")?
                * checked_integer_to_f64(*boson_integer, "boson product expansion")?;
            push_aggregate(
                aggregate,
                (
                    fermion.clone(),
                    boson.clone(),
                    qubit_codes.clone(),
                    mapped_codes.clone(),
                    qudit_triples.clone(),
                ),
                value,
                total_values,
                max_bytes,
            )?;
        }
    }
    Ok(())
}

/// Canonicalize raw mixed-domain product specifications in one batch.
pub fn canonicalize_hybrid_terms(
    layout: HybridLayout,
    batch: HybridRawBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    let count = batch.coefficients.len();
    if batch.fermion_factors.len() != count
        || batch.boson_factors.len() != count
        || batch.qubit_codes.len() != count
        || batch.qudit_present.len() != count
        || batch.qudit_triples.len() != count
    {
        return Err(PauliError::InvalidStructureLength {
            expected: count,
            actual: batch.fermion_factors.len(),
        });
    }
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..count {
        if batch.qubit_codes[index].len() != layout.nqubits {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.nqubits,
                actual: batch.qubit_codes[index].len(),
            });
        }
        if batch.qubit_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        let fermion_products = if batch.fermion_factors[index].is_empty() {
            vec![(None, 1)]
        } else {
            check_structured_bytes_u128(
                batch.fermion_factors[index].len() as u128,
                max_bytes,
                "hybrid fermion raw expansion",
            )?;
            let sequence = raw_sequence(
                &batch.fermion_factors[index],
                layout.n_modes,
                index,
                "fermion mode",
            )?;
            fermion_rewrite(sequence, max_bytes)?
                .into_iter()
                .map(|(key, value)| (Some(key), value))
                .collect()
        };
        let boson_products = if batch.boson_factors[index].is_empty() {
            vec![(None, 1)]
        } else {
            check_structured_bytes_u128(
                batch.boson_factors[index].len() as u128,
                max_bytes,
                "hybrid boson raw expansion",
            )?;
            let sequence = raw_sequence(
                &batch.boson_factors[index],
                layout.n_bosons,
                index,
                "boson mode",
            )?;
            boson_rewrite(sequence, max_bytes)?
                .into_iter()
                .map(|(key, value)| (Some(key), value))
                .collect()
        };
        if batch.qudit_triples[index]
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
            || batch.qudit_triples[index].iter().any(|&(site, a, b)| {
                site as usize >= layout.n_qudit_sites
                    || a as usize >= layout.qudit_dimension
                    || b as usize >= layout.qudit_dimension
            })
        {
            return Err(PauliError::NonCanonicalTerms { index });
        }
        for (fermion, fermion_integer) in &fermion_products {
            for (boson, boson_integer) in &boson_products {
                push_aggregate(
                    &mut aggregate,
                    (
                        fermion.clone(),
                        boson.clone(),
                        batch.qubit_codes[index].clone(),
                        None,
                        if batch.qudit_present[index] {
                            Some(batch.qudit_triples[index].clone())
                        } else {
                            None
                        },
                    ),
                    batch.coefficients[index]
                        * checked_integer_to_f64(*fermion_integer, "fermion expansion")?
                        * checked_integer_to_f64(*boson_integer, "boson expansion")?,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_hybrid_aggregate(aggregate)
}

fn hybrid_fermion_products(
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
    max_bytes: u128,
) -> Result<Vec<(Option<FermionKey>, i128)>, PauliError> {
    if !left.fermion_present[left_index] && !right.fermion_present[right_index] {
        return Ok(vec![(None, 1)]);
    }
    let mut sequence = fermion_sequence(
        &left.fermion_creation[left_index],
        &left.fermion_annihilation[left_index],
    );
    sequence.extend(fermion_sequence(
        &right.fermion_creation[right_index],
        &right.fermion_annihilation[right_index],
    ));
    Ok(fermion_rewrite(sequence, max_bytes)?
        .into_iter()
        .map(|(key, value)| (Some(key), value))
        .collect())
}

fn hybrid_boson_products(
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
    max_bytes: u128,
) -> Result<Vec<(Option<BosonKey>, i128)>, PauliError> {
    if !left.boson_present[left_index] && !right.boson_present[right_index] {
        return Ok(vec![(None, 1)]);
    }
    boson_block_product(
        &left.boson_blocks[left_index],
        &right.boson_blocks[right_index],
        max_bytes,
    )
    .map(|values| {
        values
            .into_iter()
            .map(|(key, value)| (Some(key), value))
            .collect()
    })
}

fn hybrid_mapped_product(
    n_modes: usize,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> (Option<Vec<u8>>, Complex64) {
    if !left.mapped_present[left_index] && !right.mapped_present[right_index] {
        return (None, Complex64::new(1.0, 0.0));
    }
    let identity = vec![0_u8; n_modes];
    let left_codes = if left.mapped_present[left_index] {
        &left.mapped_codes[left_index]
    } else {
        &identity
    };
    let right_codes = if right.mapped_present[right_index] {
        &right.mapped_codes[right_index]
    } else {
        &identity
    };
    let (codes, phase) = multiply_pauli_codes(left_codes, right_codes);
    (Some(codes), phase)
}

fn hybrid_qudit_product(
    dimension: usize,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> (Option<WeylKey>, Complex64) {
    if !left.qudit_present[left_index] && !right.qudit_present[right_index] {
        return (None, Complex64::new(1.0, 0.0));
    }
    let left_triples: &[(u32, u32, u32)] = if left.qudit_present[left_index] {
        &left.qudit_triples[left_index]
    } else {
        &[]
    };
    let right_triples: &[(u32, u32, u32)] = if right.qudit_present[right_index] {
        &right.qudit_triples[right_index]
    } else {
        &[]
    };
    let mut by_site: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut phase_exponent = 0_u128;
    let mut left_by_site = BTreeMap::new();
    let mut right_by_site = BTreeMap::new();
    for &(site, a, b) in left_triples {
        left_by_site.insert(site, (a, b));
    }
    for &(site, a, b) in right_triples {
        right_by_site.insert(site, (a, b));
    }
    for site in left_by_site
        .keys()
        .chain(right_by_site.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let (a, b) = left_by_site.get(&site).copied().unwrap_or((0, 0));
        let (c, e) = right_by_site.get(&site).copied().unwrap_or((0, 0));
        phase_exponent =
            (phase_exponent + u128::from(b) * u128::from(c)) % u128::from(dimension as u64);
        let aa = (u128::from(a) + u128::from(c)) % u128::from(dimension as u64);
        let bb = (u128::from(b) + u128::from(e)) % u128::from(dimension as u64);
        if aa != 0 || bb != 0 {
            by_site.insert(site, (aa as u32, bb as u32));
        }
    }
    let triples: Vec<(u32, u32, u32)> = by_site
        .into_iter()
        .map(|(site, (a, b))| (site, a, b))
        .collect();
    let angle = 2.0 * std::f64::consts::PI * phase_exponent as f64 / dimension as f64;
    let triples = if triples.is_empty() {
        None
    } else {
        Some(triples)
    };
    (triples, Complex64::new(angle.cos(), angle.sin()))
}

fn validate_hybrid_batch(layout: HybridLayout, batch: &HybridBatch<'_>) -> Result<(), PauliError> {
    let count = batch.coefficients.len();
    let lengths = [
        batch.fermion_present.len(),
        batch.fermion_creation.len(),
        batch.fermion_annihilation.len(),
        batch.boson_present.len(),
        batch.boson_blocks.len(),
        batch.qubit_codes.len(),
        batch.mapped_present.len(),
        batch.mapped_codes.len(),
        batch.qudit_present.len(),
        batch.qudit_triples.len(),
    ];
    if lengths.iter().any(|&length| length != count) {
        return Err(PauliError::InvalidStructureLength {
            expected: count,
            actual: lengths
                .into_iter()
                .find(|&length| length != count)
                .unwrap_or(count),
        });
    }
    for index in 0..count {
        validate_coefficient(batch.coefficients[index], index)?;
        validate_fermion_arrays(
            layout.n_modes,
            std::slice::from_ref(&batch.fermion_creation[index]),
            std::slice::from_ref(&batch.fermion_annihilation[index]),
            std::slice::from_ref(&batch.coefficients[index]),
        )?;
        validate_boson_blocks(layout.n_bosons, &batch.boson_blocks[index])?;
        if batch.qubit_codes[index].len() != layout.nqubits {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.nqubits,
                actual: batch.qubit_codes[index].len(),
            });
        }
        if batch.qubit_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        if batch.mapped_codes[index].len() != layout.n_modes {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.n_modes,
                actual: batch.mapped_codes[index].len(),
            });
        }
        if batch.mapped_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        if batch.qudit_triples[index]
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
            || batch.qudit_triples[index].iter().any(|&(site, a, b)| {
                site as usize >= layout.n_qudit_sites
                    || a as usize >= layout.qudit_dimension
                    || b as usize >= layout.qudit_dimension
            })
        {
            return Err(PauliError::NonCanonicalTerms { index });
        }
    }
    if layout.n_qudit_sites != 0 && layout.qudit_dimension < 3 {
        return Err(PauliError::InvalidStructureLength {
            expected: 3,
            actual: layout.qudit_dimension,
        });
    }
    Ok(())
}

fn finish_hybrid_aggregate(
    aggregate: FxHashMap<HybridKey, Vec<Complex64>>,
) -> Result<HybridCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let capacity = entries.len();
    let mut result = HybridCanonicalResult {
        fermion_present: Vec::with_capacity(capacity),
        fermion_creation: Vec::with_capacity(capacity),
        fermion_annihilation: Vec::with_capacity(capacity),
        boson_present: Vec::with_capacity(capacity),
        boson_blocks: Vec::with_capacity(capacity),
        qubit_codes: Vec::with_capacity(capacity),
        mapped_present: Vec::with_capacity(capacity),
        mapped_codes: Vec::with_capacity(capacity),
        qudit_present: Vec::with_capacity(capacity),
        qudit_triples: Vec::with_capacity(capacity),
        coefficients: Vec::with_capacity(capacity),
    };
    for ((fermion, boson, qubit, mapped, qudit), values) in entries {
        let coefficient = deterministic_sum(values);
        if is_exact_zero(coefficient) {
            continue;
        }
        if let Some((creation, annihilation)) = fermion {
            result.fermion_present.push(true);
            result.fermion_creation.push(creation);
            result.fermion_annihilation.push(annihilation);
        } else {
            result.fermion_present.push(false);
            result.fermion_creation.push(Vec::new());
            result.fermion_annihilation.push(Vec::new());
        }
        if let Some(blocks) = boson {
            result.boson_present.push(true);
            result.boson_blocks.push(blocks);
        } else {
            result.boson_present.push(false);
            result.boson_blocks.push(Vec::new());
        }
        result.qubit_codes.push(qubit);
        if let Some(codes) = mapped {
            result.mapped_present.push(true);
            result.mapped_codes.push(codes);
        } else {
            result.mapped_present.push(false);
            result.mapped_codes.push(Vec::new());
        }
        if let Some(triples) = qudit {
            result.qudit_present.push(true);
            result.qudit_triples.push(triples);
        } else {
            result.qudit_present.push(false);
            result.qudit_triples.push(Vec::new());
        }
        result.coefficients.push(coefficient);
    }
    Ok(result)
}

/// Canonicalize a batch of raw fermion monomials with CAR rewriting.
pub fn canonicalize_fermion_terms(
    n_modes: usize,
    factors: &[Vec<(usize, u8)>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    if factors.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: factors.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for (index, (raw_factors, &coefficient)) in factors.iter().zip(coefficients).enumerate() {
        validate_coefficient(coefficient, index)?;
        check_structured_bytes_u128(
            raw_factors.len() as u128,
            max_bytes,
            "fermion raw expansion",
        )?;
        let mut sequence = Vec::with_capacity(raw_factors.len());
        for &(mode, action) in raw_factors {
            if mode >= n_modes {
                return Err(PauliError::InvalidIndex {
                    context: "fermion mode",
                });
            }
            if action > 1 {
                return Err(PauliError::InvalidCode {
                    code: action,
                    index,
                });
            }
            sequence.push((mode, action == 0));
        }
        for (key, integer) in fermion_rewrite(sequence, max_bytes)? {
            push_aggregate(
                &mut aggregate,
                key,
                coefficient * checked_integer_to_f64(integer, "fermion expansion")?,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    finish_fermion_aggregate(aggregate)
}

/// Ingest a complete or compact molecular Hamiltonian in one native batch.
///
/// The two-body coefficient is multiplied by one half exactly here. Values
/// related by Hermiticity are averaged before CAR canonicalization, which
/// removes harmless roundoff asymmetry without applying a numerical cutoff.
pub fn canonicalize_fermion_integrals(
    source: FermionIntegralSource<'_>,
    constant: Complex64,
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    let (n_modes, raw_count) = match &source {
        FermionIntegralSource::SpinOrbital {
            n_modes,
            one_body,
            two_body,
        } => {
            let n_squared = n_modes.checked_mul(*n_modes).ok_or(PauliError::Overflow {
                context: "checking molecular integral dimensions",
            })?;
            let n_fourth = n_squared
                .checked_mul(n_squared)
                .ok_or(PauliError::Overflow {
                    context: "checking molecular integral dimensions",
                })?;
            validate_integral_length(one_body.len(), n_squared)?;
            validate_integral_length(two_body.len(), n_fourth)?;
            validate_spin_orbital_integrals(one_body, two_body, *n_modes)?;
            (
                *n_modes,
                count_spin_orbital_terms(one_body, two_body, *n_modes, constant),
            )
        }
        FermionIntegralSource::SpinBlocks(blocks) => {
            let n_squared =
                blocks
                    .n_spatial
                    .checked_mul(blocks.n_spatial)
                    .ok_or(PauliError::Overflow {
                        context: "checking molecular integral dimensions",
                    })?;
            let n_fourth = n_squared
                .checked_mul(n_squared)
                .ok_or(PauliError::Overflow {
                    context: "checking molecular integral dimensions",
                })?;
            validate_integral_length(blocks.one_alpha.len(), n_squared)?;
            validate_integral_length(blocks.one_beta.len(), n_squared)?;
            for length in [
                blocks.eri_aa.len(),
                blocks.eri_ab.len(),
                blocks.eri_ba.len(),
                blocks.eri_bb.len(),
            ] {
                validate_integral_length(length, n_fourth)?;
            }
            validate_spin_block_integrals(blocks)?;
            (
                blocks
                    .n_spatial
                    .checked_mul(2)
                    .ok_or(PauliError::Overflow {
                        context: "checking molecular spin-orbital dimensions",
                    })?,
                count_spin_block_terms(blocks, constant),
            )
        }
    };
    let requested = (raw_count as u128)
        .checked_mul(768)
        .ok_or(PauliError::Overflow {
            context: "estimating molecular integral ingestion",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }

    let mut factors = Vec::with_capacity(raw_count);
    let mut coefficients = Vec::with_capacity(raw_count);
    if !is_exact_zero(constant) {
        factors.push(Vec::new());
        coefficients.push(constant);
    }
    match source {
        FermionIntegralSource::SpinOrbital {
            n_modes,
            one_body,
            two_body,
        } => append_spin_orbital_integrals(
            n_modes,
            one_body,
            two_body,
            &mut factors,
            &mut coefficients,
        ),
        FermionIntegralSource::SpinBlocks(blocks) => {
            append_spin_block_integrals(&blocks, &mut factors, &mut coefficients)
        }
    }
    canonicalize_fermion_terms(n_modes, &factors, &coefficients, max_bytes)
}

fn validate_integral_length(actual: usize, expected: usize) -> Result<(), PauliError> {
    if actual != expected {
        return Err(PauliError::InvalidStructureLength { expected, actual });
    }
    Ok(())
}

const INTEGRAL_RTOL: f64 = 1.0e-10;
const INTEGRAL_ATOL: f64 = 1.0e-12;

fn validate_spin_orbital_integrals(
    one_body: &[Complex64],
    two_body: &[Complex64],
    n_modes: usize,
) -> Result<(), PauliError> {
    for p in 0..n_modes {
        for q in 0..n_modes {
            let index = p * n_modes + q;
            let reverse = q * n_modes + p;
            validate_hermitian_pair(
                one_body[index],
                one_body[reverse],
                index,
                "one_body must be Hermitian",
            )?;
            for r in 0..n_modes {
                for s in 0..n_modes {
                    let index = spin_orbital_index(n_modes, p, q, r, s);
                    let reverse = spin_orbital_index(n_modes, r, s, p, q);
                    validate_hermitian_pair(
                        two_body[index],
                        two_body[reverse],
                        index,
                        "two_body must satisfy Hermitian pair symmetry",
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn validate_spin_block_integrals(blocks: &FermionSpinBlocks<'_>) -> Result<(), PauliError> {
    for (values, context) in [
        (blocks.one_alpha, "one_alpha must be Hermitian"),
        (blocks.one_beta, "one_beta must be Hermitian"),
    ] {
        validate_matrix_pairs(values, blocks.n_spatial, context)?;
    }
    for (values, context) in [
        (blocks.eri_aa, "eri_aa must satisfy Hermitian pair symmetry"),
        (blocks.eri_ab, "eri_ab must satisfy Hermitian pair symmetry"),
        (blocks.eri_ba, "eri_ba must satisfy Hermitian pair symmetry"),
        (blocks.eri_bb, "eri_bb must satisfy Hermitian pair symmetry"),
    ] {
        validate_chemist_pairs(values, blocks.n_spatial, context)?;
    }
    Ok(())
}

fn validate_matrix_pairs(
    values: &[Complex64],
    n: usize,
    context: &'static str,
) -> Result<(), PauliError> {
    for p in 0..n {
        for q in 0..n {
            let index = p * n + q;
            let reverse = q * n + p;
            validate_hermitian_pair(values[index], values[reverse], index, context)?;
        }
    }
    Ok(())
}

fn validate_chemist_pairs(
    values: &[Complex64],
    n_spatial: usize,
    context: &'static str,
) -> Result<(), PauliError> {
    for p in 0..n_spatial {
        for r in 0..n_spatial {
            for q in 0..n_spatial {
                for s in 0..n_spatial {
                    let index = chemist_index(n_spatial, p, r, q, s);
                    let reverse = chemist_index(n_spatial, r, p, s, q);
                    validate_hermitian_pair(values[index], values[reverse], index, context)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_hermitian_pair(
    left: Complex64,
    right: Complex64,
    index: usize,
    context: &'static str,
) -> Result<(), PauliError> {
    if !left.re.is_finite() || !left.im.is_finite() {
        return Err(PauliError::NonFiniteCoefficient { index });
    }
    if !right.re.is_finite() || !right.im.is_finite() {
        return Err(PauliError::NonFiniteCoefficient { index });
    }
    if (left - right.conj()).norm() > INTEGRAL_ATOL + INTEGRAL_RTOL * right.norm() {
        return Err(PauliError::NonHermitianIntegral { context });
    }
    Ok(())
}

fn count_spin_orbital_terms(
    one_body: &[Complex64],
    two_body: &[Complex64],
    n_modes: usize,
    constant: Complex64,
) -> usize {
    let mut count = usize::from(!is_exact_zero(constant));
    for p in 0..n_modes {
        for q in 0..n_modes {
            if !is_exact_zero(hermitian_average(
                one_body[p * n_modes + q],
                one_body[q * n_modes + p],
            )) {
                count += 1;
            }
            for r in 0..n_modes {
                for s in 0..n_modes {
                    let value = hermitian_average(
                        two_body[spin_orbital_index(n_modes, p, q, r, s)],
                        two_body[spin_orbital_index(n_modes, r, s, p, q)],
                    ) * 0.5;
                    if !is_exact_zero(value) {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

fn count_spin_block_terms(blocks: &FermionSpinBlocks<'_>, constant: Complex64) -> usize {
    let n = blocks.n_spatial;
    let mut count = usize::from(!is_exact_zero(constant));
    for one_body in [blocks.one_alpha, blocks.one_beta] {
        for p in 0..n {
            for q in 0..n {
                if !is_exact_zero(hermitian_average(one_body[p * n + q], one_body[q * n + p])) {
                    count += 1;
                }
            }
        }
    }
    for eri in [blocks.eri_aa, blocks.eri_ab, blocks.eri_ba, blocks.eri_bb] {
        for p in 0..n {
            for r in 0..n {
                for q in 0..n {
                    for s in 0..n {
                        let index = chemist_index(n, p, r, q, s);
                        let reverse = chemist_index(n, r, p, s, q);
                        if !is_exact_zero(hermitian_average(eri[index], eri[reverse]) * 0.5) {
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    count
}

fn append_spin_orbital_integrals(
    n_modes: usize,
    one_body: &[Complex64],
    two_body: &[Complex64],
    factors: &mut Vec<Vec<(usize, u8)>>,
    coefficients: &mut Vec<Complex64>,
) {
    for p in 0..n_modes {
        for q in 0..n_modes {
            let value = hermitian_average(one_body[p * n_modes + q], one_body[q * n_modes + p]);
            if !is_exact_zero(value) {
                factors.push(vec![(p, 0), (q, 1)]);
                coefficients.push(value);
            }
            for r in 0..n_modes {
                for s in 0..n_modes {
                    let value = hermitian_average(
                        two_body[spin_orbital_index(n_modes, p, q, r, s)],
                        two_body[spin_orbital_index(n_modes, r, s, p, q)],
                    ) * 0.5;
                    if !is_exact_zero(value) {
                        factors.push(vec![(p, 0), (q, 0), (s, 1), (r, 1)]);
                        coefficients.push(value);
                    }
                }
            }
        }
    }
}

fn append_spin_block_integrals(
    blocks: &FermionSpinBlocks<'_>,
    factors: &mut Vec<Vec<(usize, u8)>>,
    coefficients: &mut Vec<Complex64>,
) {
    let n = blocks.n_spatial;
    for (spin, one_body) in [(0usize, blocks.one_alpha), (1usize, blocks.one_beta)] {
        for p in 0..n {
            for q in 0..n {
                let value = hermitian_average(one_body[p * n + q], one_body[q * n + p]);
                if !is_exact_zero(value) {
                    let p_mode = spin_mode(n, p, spin, blocks.ordering);
                    let q_mode = spin_mode(n, q, spin, blocks.ordering);
                    factors.push(vec![(p_mode, 0), (q_mode, 1)]);
                    coefficients.push(value);
                }
            }
        }
    }
    for (left_spin, right_spin, eri) in [
        (0usize, 0usize, blocks.eri_aa),
        (0usize, 1usize, blocks.eri_ab),
        (1usize, 0usize, blocks.eri_ba),
        (1usize, 1usize, blocks.eri_bb),
    ] {
        for p in 0..n {
            for r in 0..n {
                for q in 0..n {
                    for s in 0..n {
                        let value = hermitian_average(
                            eri[chemist_index(n, p, r, q, s)],
                            eri[chemist_index(n, r, p, s, q)],
                        ) * 0.5;
                        if !is_exact_zero(value) {
                            let p_mode = spin_mode(n, p, left_spin, blocks.ordering);
                            let q_mode = spin_mode(n, q, right_spin, blocks.ordering);
                            let r_mode = spin_mode(n, r, left_spin, blocks.ordering);
                            let s_mode = spin_mode(n, s, right_spin, blocks.ordering);
                            factors.push(vec![(p_mode, 0), (q_mode, 0), (s_mode, 1), (r_mode, 1)]);
                            coefficients.push(value);
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn hermitian_average(left: Complex64, right: Complex64) -> Complex64 {
    (left + right.conj()) * 0.5
}

#[inline]
fn spin_orbital_index(n_modes: usize, p: usize, q: usize, r: usize, s: usize) -> usize {
    ((p * n_modes + q) * n_modes + r) * n_modes + s
}

#[inline]
fn chemist_index(n_spatial: usize, p: usize, r: usize, q: usize, s: usize) -> usize {
    ((p * n_spatial + r) * n_spatial + q) * n_spatial + s
}

#[inline]
fn spin_mode(
    n_spatial: usize,
    orbital: usize,
    spin: usize,
    ordering: FermionSpinOrdering,
) -> usize {
    match ordering {
        FermionSpinOrdering::Interleaved => 2 * orbital + spin,
        FermionSpinOrdering::AlphaThenBeta => spin * n_spatial + orbital,
    }
}

/// Multiply two canonical fermion operators in one coarse-grained call.
pub fn multiply_fermion_terms(
    n_modes: usize,
    left: FermionBatch<'_>,
    right: FermionBatch<'_>,
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    validate_fermion_arrays(n_modes, left.creation, left.annihilation, left.coefficients)?;
    validate_fermion_arrays(
        n_modes,
        right.creation,
        right.annihilation,
        right.coefficients,
    )?;
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating fermion product expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "fermion product expansion")?;
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            let mut sequence =
                fermion_sequence(&left.creation[left_index], &left.annihilation[left_index]);
            sequence.extend(fermion_sequence(
                &right.creation[right_index],
                &right.annihilation[right_index],
            ));
            for (key, integer) in fermion_rewrite(sequence, max_bytes)? {
                push_aggregate(
                    &mut aggregate,
                    key,
                    left.coefficients[left_index]
                        * right.coefficients[right_index]
                        * checked_integer_to_f64(integer, "fermion product expansion")?,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_fermion_aggregate(aggregate)
}

/// Compute a fermion commutator or anticommutator without materializing the
/// two directional products first.
pub fn binary_fermion_terms(
    n_modes: usize,
    left: FermionBatch<'_>,
    right: FermionBatch<'_>,
    max_bytes: u128,
    reverse_sign: i8,
) -> Result<FermionCanonicalResult, PauliError> {
    validate_fermion_arrays(n_modes, left.creation, left.annihilation, left.coefficients)?;
    validate_fermion_arrays(
        n_modes,
        right.creation,
        right.annihilation,
        right.coefficients,
    )?;
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .and_then(|value| value.checked_mul(if reverse_sign == 0 { 1 } else { 2 }))
        .ok_or(PauliError::Overflow {
            context: "estimating fermion binary expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "fermion product expansion")?;
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            accumulate_fermion_product(
                &left,
                left_index,
                &right,
                right_index,
                1,
                &mut aggregate,
                &mut total_values,
                max_bytes,
            )?;
            if reverse_sign != 0 {
                accumulate_fermion_product(
                    &right,
                    right_index,
                    &left,
                    left_index,
                    reverse_sign,
                    &mut aggregate,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_fermion_aggregate(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_fermion_product(
    left: &FermionBatch<'_>,
    left_index: usize,
    right: &FermionBatch<'_>,
    right_index: usize,
    direction_sign: i8,
    aggregate: &mut FxHashMap<FermionKey, Vec<Complex64>>,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let mut sequence = fermion_sequence(&left.creation[left_index], &left.annihilation[left_index]);
    sequence.extend(fermion_sequence(
        &right.creation[right_index],
        &right.annihilation[right_index],
    ));
    for (key, integer) in fermion_rewrite(sequence, max_bytes)? {
        push_aggregate(
            aggregate,
            key,
            left.coefficients[left_index]
                * right.coefficients[right_index]
                * f64::from(direction_sign)
                * checked_integer_to_f64(integer, "fermion product expansion")?,
            total_values,
            max_bytes,
        )?;
    }
    Ok(())
}

/// Map canonical fermion terms to Jordan-Wigner Pauli terms in one batch.
pub fn jordan_wigner_terms(
    n_modes: usize,
    creation: &[Vec<u32>],
    annihilation: &[Vec<u32>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<(Vec<Vec<u8>>, Vec<Complex64>), PauliError> {
    validate_fermion_arrays(n_modes, creation, annihilation, coefficients)?;
    let mut aggregate: FxHashMap<Vec<u8>, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for (index, coefficient) in coefficients.iter().copied().enumerate() {
        let current = jordan_wigner_word_expansion(
            n_modes,
            fermion_sequence(&creation[index], &annihilation[index]),
            max_bytes,
        )?;
        for (word, value) in current {
            push_pauli_aggregate(
                &mut aggregate,
                word,
                coefficient * value,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut words = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (word, contributions) in entries {
        let value = deterministic_sum(contributions);
        if !is_exact_zero(value) {
            words.push(word);
            values.push(value);
        }
    }
    Ok((words, values))
}

/// Map mixed-domain terms while retaining their non-fermionic factors.
pub fn jordan_wigner_hybrid_terms(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    validate_hybrid_batch(layout, &batch)?;
    jordan_wigner_hybrid_terms_trusted(layout, batch, max_bytes)
}

/// Map a canonical native hybrid handle without rescanning its storage.
pub fn jordan_wigner_hybrid_terms_trusted(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for index in 0..batch.coefficients.len() {
        let expansions = if batch.fermion_present[index] {
            jordan_wigner_word_expansion(
                layout.n_modes,
                fermion_sequence(
                    &batch.fermion_creation[index],
                    &batch.fermion_annihilation[index],
                ),
                max_bytes,
            )?
        } else {
            vec![(vec![0_u8; layout.n_modes], Complex64::new(1.0, 0.0))]
        };
        let base_mapped = if batch.mapped_present[index] {
            batch.mapped_codes[index].clone()
        } else {
            vec![0_u8; layout.n_modes]
        };
        for (mapped_codes, mapped_coefficient) in expansions {
            let (combined_codes, mapped_phase) = multiply_pauli_codes(&base_mapped, &mapped_codes);
            push_aggregate(
                &mut aggregate,
                (
                    None,
                    if batch.boson_present[index] {
                        Some(batch.boson_blocks[index].clone())
                    } else {
                        None
                    },
                    batch.qubit_codes[index].clone(),
                    if batch.fermion_present[index] || batch.mapped_present[index] {
                        Some(combined_codes)
                    } else {
                        None
                    },
                    if batch.qudit_present[index] {
                        Some(batch.qudit_triples[index].clone())
                    } else {
                        None
                    },
                ),
                batch.coefficients[index] * mapped_coefficient * mapped_phase,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    finish_hybrid_aggregate(aggregate)
}

fn jordan_wigner_word_expansion(
    n_modes: usize,
    sequence: Vec<(usize, bool)>,
    max_bytes: u128,
) -> Result<Vec<(Vec<u8>, Complex64)>, PauliError> {
    // This convention is also used by Python's tensor-product adapter:
    // X=1/2, Y=+i/2 for annihilation and -i/2 for creation, with Z below the
    // active mode. Keep the two adapters covered by a differential test.
    let mut current: Vec<(Vec<u8>, Complex64)> = vec![(vec![0; n_modes], Complex64::new(1.0, 0.0))];
    for (mode, is_creation) in sequence {
        let y_coefficient = if is_creation {
            Complex64::new(0.0, -0.5)
        } else {
            Complex64::new(0.0, 0.5)
        };
        let expansion_count = current.len().checked_mul(2).ok_or(PauliError::Overflow {
            context: "estimating Jordan-Wigner expansion",
        })?;
        check_structured_bytes(expansion_count, max_bytes, "Jordan-Wigner expansion")?;
        let mut next: FxHashMap<Vec<u8>, Complex64> = FxHashMap::default();
        let mut right_word = vec![3_u8; n_modes];
        right_word[mode..].fill(0);
        for (left_word, left_coefficient) in current {
            for (code, right_coefficient) in
                [(1_u8, Complex64::new(0.5, 0.0)), (2_u8, y_coefficient)]
            {
                right_word[mode] = code;
                let (word, phase) = multiply_pauli_codes(&left_word, &right_word);
                *next.entry(word).or_default() += left_coefficient * right_coefficient * phase;
            }
        }
        current = next
            .into_iter()
            .filter(|(_, value)| !is_exact_zero(*value))
            .collect();
        current.sort_by(|left, right| left.0.cmp(&right.0));
    }
    Ok(current)
}

/// Canonicalize a batch of raw boson monomials with CCR rewriting.
pub fn canonicalize_boson_terms(
    n_modes: usize,
    factors: &[Vec<(usize, u8)>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<BosonCanonicalResult, PauliError> {
    if factors.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: factors.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for (index, (raw_factors, &coefficient)) in factors.iter().zip(coefficients).enumerate() {
        validate_coefficient(coefficient, index)?;
        check_structured_bytes_u128(raw_factors.len() as u128, max_bytes, "boson raw expansion")?;
        let mut sequence = Vec::with_capacity(raw_factors.len());
        for &(mode, action) in raw_factors {
            if mode >= n_modes {
                return Err(PauliError::InvalidIndex {
                    context: "boson mode",
                });
            }
            if action > 1 {
                return Err(PauliError::InvalidCode {
                    code: action,
                    index,
                });
            }
            sequence.push((mode, action == 0));
        }
        for (key, integer) in boson_rewrite(sequence, max_bytes)? {
            push_aggregate(
                &mut aggregate,
                key,
                coefficient * checked_integer_to_f64(integer, "boson expansion")?,
                &mut total_values,
                max_bytes,
            )?;
        }
    }
    finish_boson_aggregate(aggregate)
}

/// Multiply two canonical boson operators in one coarse-grained call.
pub fn multiply_boson_terms(
    n_modes: usize,
    left_blocks: &[Vec<(u32, u32, u32)>],
    left_coefficients: &[Complex64],
    right_blocks: &[Vec<(u32, u32, u32)>],
    right_coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<BosonCanonicalResult, PauliError> {
    if left_blocks.len() != left_coefficients.len()
        || right_blocks.len() != right_coefficients.len()
    {
        return Err(PauliError::InvalidStructureLength {
            expected: left_blocks.len(),
            actual: left_coefficients.len(),
        });
    }
    for blocks in left_blocks.iter().chain(right_blocks) {
        validate_boson_blocks(n_modes, blocks)?;
    }
    let pair_count = left_coefficients
        .len()
        .checked_mul(right_coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating boson product expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "boson product expansion")?;
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left in 0..left_blocks.len() {
        for right in 0..right_blocks.len() {
            for (key, integer) in
                boson_block_product(&left_blocks[left], &right_blocks[right], max_bytes)?
            {
                push_aggregate(
                    &mut aggregate,
                    key,
                    left_coefficients[left]
                        * right_coefficients[right]
                        * checked_integer_to_f64(integer, "boson product expansion")?,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_boson_aggregate(aggregate)
}

/// Compute a boson commutator or anticommutator in one shared aggregate.
pub fn binary_boson_terms(
    n_modes: usize,
    left_blocks: &[Vec<(u32, u32, u32)>],
    left_coefficients: &[Complex64],
    right_blocks: &[Vec<(u32, u32, u32)>],
    right_coefficients: &[Complex64],
    max_bytes: u128,
    reverse_sign: i8,
) -> Result<BosonCanonicalResult, PauliError> {
    if left_blocks.len() != left_coefficients.len()
        || right_blocks.len() != right_coefficients.len()
    {
        return Err(PauliError::InvalidStructureLength {
            expected: left_blocks.len(),
            actual: left_coefficients.len(),
        });
    }
    for blocks in left_blocks.iter().chain(right_blocks) {
        validate_boson_blocks(n_modes, blocks)?;
    }
    let pair_count = left_coefficients
        .len()
        .checked_mul(right_coefficients.len())
        .and_then(|value| value.checked_mul(if reverse_sign == 0 { 1 } else { 2 }))
        .ok_or(PauliError::Overflow {
            context: "estimating boson binary expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "boson product expansion")?;
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    let mut total_values = 0usize;
    for left_index in 0..left_blocks.len() {
        for right_index in 0..right_blocks.len() {
            accumulate_boson_product(
                &left_blocks[left_index],
                left_coefficients[left_index],
                &right_blocks[right_index],
                right_coefficients[right_index],
                1,
                &mut aggregate,
                &mut total_values,
                max_bytes,
            )?;
            if reverse_sign != 0 {
                accumulate_boson_product(
                    &right_blocks[right_index],
                    right_coefficients[right_index],
                    &left_blocks[left_index],
                    left_coefficients[left_index],
                    reverse_sign,
                    &mut aggregate,
                    &mut total_values,
                    max_bytes,
                )?;
            }
        }
    }
    finish_boson_aggregate(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_boson_product(
    left: &[(u32, u32, u32)],
    left_coefficient: Complex64,
    right: &[(u32, u32, u32)],
    right_coefficient: Complex64,
    direction_sign: i8,
    aggregate: &mut FxHashMap<BosonKey, Vec<Complex64>>,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    for (key, integer) in boson_block_product(left, right, max_bytes)? {
        push_aggregate(
            aggregate,
            key,
            left_coefficient
                * right_coefficient
                * f64::from(direction_sign)
                * checked_integer_to_f64(integer, "boson product expansion")?,
            total_values,
            max_bytes,
        )?;
    }
    Ok(())
}

fn fermion_rewrite(
    sequence: Vec<(usize, bool)>,
    max_bytes: u128,
) -> Result<BTreeMap<FermionKey, i128>, PauliError> {
    if sequence.is_empty() {
        let mut result = BTreeMap::new();
        result.insert((Vec::new(), Vec::new()), 1);
        return Ok(result);
    }

    // Canonical/no-contraction inputs are common in bulk construction.  They
    // should not be rejected by a blanket 2**len estimate: the exact result
    // is one key (or zero for an adjacent nilpotent pair) and no branching is
    // needed.  The same fast path handles long inversion-only words below.
    if let Some(result) = canonical_without_contractions(&sequence) {
        return result;
    }

    fn recurse(
        sequence: &[(usize, bool)],
        max_bytes: u128,
    ) -> Result<BTreeMap<FermionKey, i128>, PauliError> {
        for index in 0..sequence.len().saturating_sub(1) {
            let (left_mode, left_creation) = sequence[index];
            let (right_mode, right_creation) = sequence[index + 1];
            if left_creation == right_creation && left_mode == right_mode {
                return Ok(BTreeMap::new());
            }
            let inversion = if left_creation && right_creation {
                left_mode > right_mode
            } else if !left_creation && !right_creation {
                left_mode < right_mode
            } else {
                !left_creation && right_creation
            };
            if !inversion {
                continue;
            }
            let mut swapped = sequence.to_vec();
            swapped.swap(index, index + 1);
            let mut result = BTreeMap::new();
            add_integer_maps(&mut result, recurse(&swapped, max_bytes)?, -1)?;
            check_structured_bytes_u128(result.len() as u128, max_bytes, "fermion expansion")?;
            if !left_creation && right_creation && left_mode == right_mode {
                let mut contracted = sequence.to_vec();
                contracted.drain(index..=index + 1);
                add_integer_maps(&mut result, recurse(&contracted, max_bytes)?, 1)?;
                check_structured_bytes_u128(result.len() as u128, max_bytes, "fermion expansion")?;
            }
            result.retain(|_, value| *value != 0);
            check_structured_bytes_u128(result.len() as u128, max_bytes, "fermion expansion")?;
            return Ok(result);
        }
        let creation = sequence
            .iter()
            .filter_map(|&(mode, is_creation)| is_creation.then_some(mode as u32))
            .collect();
        let annihilation = sequence
            .iter()
            .filter_map(|&(mode, is_creation)| (!is_creation).then_some(mode as u32))
            .collect();
        let mut result = BTreeMap::new();
        result.insert((creation, annihilation), 1);
        Ok(result)
    }

    recurse(&sequence, max_bytes)
}

/// Return the exact one-key/zero-key result for a non-branching CAR word.
///
/// `None` means that a contraction may be reachable and the general rewrite
/// must be used.  The scan is linear for the ordinary canonical case and the
/// quadratic inversion count is only used for words whose creation and
/// annihilation supports are disjoint.
fn canonical_without_contractions(
    sequence: &[(usize, bool)],
) -> Option<Result<BTreeMap<FermionKey, i128>, PauliError>> {
    let mut canonical = true;
    for pair in sequence.windows(2) {
        let (left_mode, left_creation) = pair[0];
        let (right_mode, right_creation) = pair[1];
        if left_creation == right_creation && left_mode == right_mode {
            return Some(Ok(BTreeMap::new()));
        }
        let inversion = if left_creation && right_creation {
            left_mode > right_mode
        } else if !left_creation && !right_creation {
            left_mode < right_mode
        } else {
            !left_creation && right_creation
        };
        if inversion {
            canonical = false;
            break;
        }
    }
    if canonical {
        return Some(Ok(single_canonical_fermion_key(sequence, 1)));
    }

    let creation_modes: Vec<usize> = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| is_creation.then_some(mode))
        .collect();
    let annihilation_modes: Vec<usize> = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| (!is_creation).then_some(mode))
        .collect();
    if creation_modes
        .iter()
        .any(|mode| annihilation_modes.iter().any(|other| mode == other))
    {
        return None;
    }
    if has_duplicate(&creation_modes) || has_duplicate(&annihilation_modes) {
        return Some(Ok(BTreeMap::new()));
    }

    let inversions = sequence
        .iter()
        .enumerate()
        .map(|(left_index, &left)| {
            sequence[left_index + 1..]
                .iter()
                .filter(|&&right| fermion_inversion(left, right))
                .count()
        })
        .sum::<usize>();
    let sign = if inversions & 1 == 0 { 1 } else { -1 };
    Some(Ok(single_canonical_fermion_key(sequence, sign)))
}

fn fermion_inversion(left: (usize, bool), right: (usize, bool)) -> bool {
    let (left_mode, left_creation) = left;
    let (right_mode, right_creation) = right;
    if left_creation && right_creation {
        left_mode > right_mode
    } else if !left_creation && !right_creation {
        left_mode < right_mode
    } else {
        !left_creation && right_creation
    }
}

fn has_duplicate(values: &[usize]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[index + 1..].contains(value))
}

fn single_canonical_fermion_key(
    sequence: &[(usize, bool)],
    coefficient: i128,
) -> BTreeMap<FermionKey, i128> {
    let mut creation = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| is_creation.then_some(mode as u32))
        .collect::<Vec<_>>();
    let mut annihilation = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| (!is_creation).then_some(mode as u32))
        .collect::<Vec<_>>();
    creation.sort_unstable();
    annihilation.sort_unstable_by(|left, right| right.cmp(left));
    let mut result = BTreeMap::new();
    result.insert((creation, annihilation), coefficient);
    result
}

fn boson_rewrite(
    sequence: Vec<(usize, bool)>,
    max_bytes: u128,
) -> Result<BTreeMap<BosonKey, i128>, PauliError> {
    let mut by_mode: BTreeMap<usize, Vec<bool>> = BTreeMap::new();
    for (mode, is_creation) in sequence {
        by_mode.entry(mode).or_default().push(is_creation);
    }
    let mut aggregate: BTreeMap<BosonKey, i128> = BTreeMap::new();
    aggregate.insert(Vec::new(), 1);
    for (mode, factors) in by_mode {
        let mut local: BTreeMap<(u32, u32), i128> = BTreeMap::new();
        local.insert((0, 0), 1);
        for is_creation in factors {
            let estimated = (local.len() as u128)
                .checked_mul(2)
                .ok_or(PauliError::Overflow {
                    context: "estimating boson expansion",
                })?;
            check_structured_bytes_u128(estimated, max_bytes, "boson expansion")?;
            let mut next = BTreeMap::new();
            for (&(creation, annihilation), &value) in &local {
                if is_creation {
                    let next_creation = creation.checked_add(1).ok_or(PauliError::Overflow {
                        context: "boson creation power",
                    })?;
                    add_integer(&mut next, (next_creation, annihilation), value)?;
                    if annihilation != 0 {
                        let contracted = value.checked_mul(annihilation as i128).ok_or(
                            PauliError::Overflow {
                                context: "boson contraction coefficient",
                            },
                        )?;
                        add_integer(&mut next, (creation, annihilation - 1), contracted)?;
                    }
                } else {
                    let next_annihilation =
                        annihilation.checked_add(1).ok_or(PauliError::Overflow {
                            context: "boson annihilation power",
                        })?;
                    add_integer(&mut next, (creation, next_annihilation), value)?;
                }
            }
            local = next;
        }
        let product_count = (aggregate.len() as u128)
            .checked_mul(local.len() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating boson expansion",
            })?;
        check_structured_bytes_u128(product_count, max_bytes, "boson expansion")?;
        let mut next = BTreeMap::new();
        for (prefix, prefix_value) in aggregate {
            for (&(creation, annihilation), &local_value) in &local {
                let mut blocks = prefix.clone();
                if creation != 0 || annihilation != 0 {
                    blocks.push((mode as u32, creation, annihilation));
                }
                add_integer(
                    &mut next,
                    blocks,
                    prefix_value
                        .checked_mul(local_value)
                        .ok_or(PauliError::Overflow {
                            context: "boson expansion coefficient",
                        })?,
                )?;
            }
        }
        aggregate = next;
    }
    Ok(aggregate)
}

fn boson_block_product(
    left: &[(u32, u32, u32)],
    right: &[(u32, u32, u32)],
    max_bytes: u128,
) -> Result<BTreeMap<BosonKey, i128>, PauliError> {
    let mut left_by_mode = BTreeMap::new();
    let mut right_by_mode = BTreeMap::new();
    for &(mode, creation, annihilation) in left {
        left_by_mode.insert(mode, (creation, annihilation));
    }
    for &(mode, creation, annihilation) in right {
        right_by_mode.insert(mode, (creation, annihilation));
    }
    let mut aggregate: BTreeMap<BosonKey, i128> = BTreeMap::new();
    aggregate.insert(Vec::new(), 1);
    let modes: BTreeSet<u32> = left_by_mode
        .keys()
        .chain(right_by_mode.keys())
        .copied()
        .collect();
    for mode in modes {
        let (left_creation, left_annihilation) = left_by_mode.get(&mode).copied().unwrap_or((0, 0));
        let (right_creation, right_annihilation) =
            right_by_mode.get(&mode).copied().unwrap_or((0, 0));
        let mut local = BTreeMap::new();
        let maximum = left_annihilation.min(right_creation);
        for contractions in 0..=maximum {
            let coefficient =
                contraction_coefficient(left_annihilation, right_creation, contractions)?;
            let creation = left_creation
                .checked_add(right_creation)
                .and_then(|value| value.checked_sub(contractions))
                .ok_or(PauliError::Overflow {
                    context: "boson product power",
                })?;
            let annihilation = left_annihilation
                .checked_add(right_annihilation)
                .and_then(|value| value.checked_sub(contractions))
                .ok_or(PauliError::Overflow {
                    context: "boson product power",
                })?;
            add_integer(&mut local, (creation, annihilation), coefficient)?;
        }
        let product_count = (aggregate.len() as u128)
            .checked_mul(local.len() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating boson product expansion",
            })?;
        check_structured_bytes_u128(product_count, max_bytes, "boson product expansion")?;
        let mut next = BTreeMap::new();
        for (prefix, prefix_value) in aggregate {
            for (&(creation, annihilation), &local_value) in &local {
                let mut blocks = prefix.clone();
                if creation != 0 || annihilation != 0 {
                    blocks.push((mode, creation, annihilation));
                }
                add_integer(
                    &mut next,
                    blocks,
                    prefix_value
                        .checked_mul(local_value)
                        .ok_or(PauliError::Overflow {
                            context: "boson product coefficient",
                        })?,
                )?;
            }
        }
        aggregate = next;
    }
    Ok(aggregate)
}

fn contraction_coefficient(q: u32, r: u32, k: u32) -> Result<i128, PauliError> {
    let mut binomial = 1_i128;
    for index in 0..k {
        binomial = binomial
            .checked_mul((q - index) as i128)
            .and_then(|value| value.checked_div((index + 1) as i128))
            .ok_or(PauliError::Overflow {
                context: "boson binomial coefficient",
            })?;
    }
    let mut falling = 1_i128;
    for index in 0..k {
        falling = falling
            .checked_mul((r - index) as i128)
            .ok_or(PauliError::Overflow {
                context: "boson falling factorial",
            })?;
    }
    binomial.checked_mul(falling).ok_or(PauliError::Overflow {
        context: "boson contraction coefficient",
    })
}

fn fermion_sequence(creation: &[u32], annihilation: &[u32]) -> Vec<(usize, bool)> {
    creation
        .iter()
        .map(|&mode| (mode as usize, true))
        .chain(annihilation.iter().map(|&mode| (mode as usize, false)))
        .collect()
}

fn raw_sequence(
    factors: &[(usize, u8)],
    n_modes: usize,
    index: usize,
    context: &'static str,
) -> Result<Vec<(usize, bool)>, PauliError> {
    let mut sequence = Vec::with_capacity(factors.len());
    for &(mode, action) in factors {
        if mode >= n_modes {
            return Err(PauliError::InvalidIndex { context });
        }
        if action > 1 {
            return Err(PauliError::InvalidCode {
                code: action,
                index,
            });
        }
        sequence.push((mode, action == 0));
    }
    Ok(sequence)
}

fn add_integer<K: Ord>(
    target: &mut BTreeMap<K, i128>,
    key: K,
    value: i128,
) -> Result<(), PauliError> {
    let entry = target.entry(key).or_default();
    *entry = entry.checked_add(value).ok_or(PauliError::Overflow {
        context: "structured integer coefficient",
    })?;
    Ok(())
}

fn add_integer_maps<K: Ord>(
    target: &mut BTreeMap<K, i128>,
    source: BTreeMap<K, i128>,
    factor: i128,
) -> Result<(), PauliError> {
    for (key, value) in source {
        add_integer(
            target,
            key,
            value.checked_mul(factor).ok_or(PauliError::Overflow {
                context: "structured integer coefficient",
            })?,
        )?;
    }
    Ok(())
}

fn checked_integer_to_f64(value: i128, context: &'static str) -> Result<f64, PauliError> {
    let _ = context;
    Ok(value as f64)
}

fn validate_coefficient(value: Complex64, index: usize) -> Result<(), PauliError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(PauliError::NonFiniteCoefficient { index })
    }
}

fn validate_fermion_arrays(
    n_modes: usize,
    creation: &[Vec<u32>],
    annihilation: &[Vec<u32>],
    coefficients: &[Complex64],
) -> Result<(), PauliError> {
    if creation.len() != annihilation.len() || creation.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: creation.len(),
            actual: coefficients.len(),
        });
    }
    for index in 0..creation.len() {
        if creation[index].windows(2).any(|pair| pair[0] >= pair[1])
            || annihilation[index]
                .windows(2)
                .any(|pair| pair[0] <= pair[1])
            || creation[index]
                .iter()
                .chain(&annihilation[index])
                .any(|&mode| mode as usize >= n_modes)
        {
            return Err(PauliError::NonCanonicalTerms { index });
        }
    }
    Ok(())
}

pub(crate) fn validate_boson_blocks(
    n_modes: usize,
    blocks: &[(u32, u32, u32)],
) -> Result<(), PauliError> {
    if blocks.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || blocks.iter().any(|&(mode, _, _)| mode as usize >= n_modes)
    {
        return Err(PauliError::NonCanonicalTerms { index: 0 });
    }
    Ok(())
}

fn check_structured_bytes(
    count: usize,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), PauliError> {
    check_structured_bytes_u128(count as u128, max_bytes, context)
}

fn check_structured_bytes_u128(
    count: u128,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), PauliError> {
    let requested = count
        .checked_mul(192)
        .ok_or(PauliError::Overflow { context })?;
    if requested > max_bytes {
        Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        })
    } else {
        Ok(())
    }
}

fn push_aggregate<K: Eq + std::hash::Hash>(
    aggregate: &mut FxHashMap<K, Vec<Complex64>>,
    key: K,
    value: Complex64,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let value_count = {
        let values = aggregate.entry(key).or_default();
        values.push(value);
        values.len()
    };
    *total_values = (*total_values).checked_add(1).ok_or(PauliError::Overflow {
        context: "estimating structured aggregation",
    })?;
    check_structured_bytes(
        (*total_values).max(value_count),
        max_bytes,
        "structured aggregation",
    )
}

fn push_pauli_aggregate(
    aggregate: &mut FxHashMap<Vec<u8>, Vec<Complex64>>,
    key: Vec<u8>,
    value: Complex64,
    total_values: &mut usize,
    max_bytes: u128,
) -> Result<(), PauliError> {
    push_aggregate(aggregate, key, value, total_values, max_bytes)
}

fn finish_fermion_aggregate(
    aggregate: FxHashMap<FermionKey, Vec<Complex64>>,
) -> Result<FermionCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut creation = Vec::with_capacity(entries.len());
    let mut annihilation = Vec::with_capacity(entries.len());
    let mut coefficients = Vec::with_capacity(entries.len());
    for ((left, right), values) in entries {
        let value = deterministic_sum(values);
        if !is_exact_zero(value) {
            creation.push(left);
            annihilation.push(right);
            coefficients.push(value);
        }
    }
    Ok((creation, annihilation, coefficients))
}

fn finish_boson_aggregate(
    aggregate: FxHashMap<BosonKey, Vec<Complex64>>,
) -> Result<BosonCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut blocks = Vec::with_capacity(entries.len());
    let mut coefficients = Vec::with_capacity(entries.len());
    for (key, values) in entries {
        let value = deterministic_sum(values);
        if !is_exact_zero(value) {
            blocks.push(key);
            coefficients.push(value);
        }
    }
    Ok((blocks, coefficients))
}

fn deterministic_sum(mut values: Vec<Complex64>) -> Complex64 {
    values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
    values
        .into_iter()
        .fold(Complex64::default(), |sum, value| sum + value)
}

fn is_exact_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn multiply_pauli_codes(left: &[u8], right: &[u8]) -> (Vec<u8>, Complex64) {
    let mut result = Vec::with_capacity(left.len());
    let mut phase = Complex64::new(1.0, 0.0);
    for (&left_code, &right_code) in left.iter().zip(right) {
        let (code, local_phase) = pauli_code_product(left_code, right_code);
        result.push(code);
        phase *= local_phase;
    }
    (result, phase)
}

fn pauli_code_product(left: u8, right: u8) -> (u8, Complex64) {
    match (left, right) {
        (0, code) | (code, 0) => (code, Complex64::new(1.0, 0.0)),
        (1, 1) | (2, 2) | (3, 3) => (0, Complex64::new(1.0, 0.0)),
        (1, 2) => (3, Complex64::new(0.0, 1.0)),
        (1, 3) => (2, Complex64::new(0.0, -1.0)),
        (2, 1) => (3, Complex64::new(0.0, -1.0)),
        (2, 3) => (1, Complex64::new(0.0, 1.0)),
        (3, 1) => (2, Complex64::new(0.0, 1.0)),
        (3, 2) => (1, Complex64::new(0.0, -1.0)),
        _ => (0, Complex64::new(0.0, 0.0)),
    }
}

/// One local finite-basis operation in a canonical structured term.
/// `kind=0` is Pauli, `kind=1` is a boson block, and `kind=2` is direct Weyl.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StructuredOperation {
    pub axis: usize,
    pub kind: u8,
    pub p: u32,
    pub q: u32,
}

#[derive(Clone)]
struct CompiledStructuredOperation {
    axis: usize,
    kind: u8,
    p: usize,
    q: usize,
    stride: usize,
    dimension: usize,
    local_table: Vec<Complex64>,
}

#[derive(Clone)]
struct CompiledNetAction {
    axis: usize,
    stride: usize,
    dimension: usize,
    forward_destinations: Box<[usize]>,
    forward_amplitudes: Box<[Complex64]>,
    inverse_sources: Box<[usize]>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CompiledScheduleKey {
    /// Destination digits for each non-Fermion active axis.  Amplitudes are
    /// intentionally absent: source-dependent phases and ladder factors are
    /// evaluated once per grouped source configuration below.
    net_axes: Vec<(usize, Vec<usize>)>,
    fermion_axes: Vec<usize>,
    fermion_flip: Vec<u64>,
    fallback_shape: Vec<(usize, u8, usize, usize)>,
}

struct StructuredScheduleGroup {
    members: Box<[usize]>,
    suffix_stride: usize,
    contiguous_source_flip_mask: Option<usize>,
}

#[derive(Default)]
struct StructuredApplyCounters {
    schedule_group_evaluations: AtomicUsize,
    segment_applications: AtomicUsize,
    fermion_mask_applications: AtomicUsize,
    nonfermion_fallback_invocations: AtomicUsize,
}

impl StructuredApplyCounters {
    fn snapshot(&self) -> (usize, usize, usize, usize) {
        (
            self.schedule_group_evaluations.load(Ordering::Relaxed),
            self.segment_applications.load(Ordering::Relaxed),
            self.fermion_mask_applications.load(Ordering::Relaxed),
            self.nonfermion_fallback_invocations.load(Ordering::Relaxed),
        )
    }
}

fn compile_net_actions(
    term: &[StructuredOperation],
    local_dimensions: &[usize],
    strides: &[usize],
) -> Result<Option<Box<[CompiledNetAction]>>, PauliError> {
    let nonfermion_term = term
        .iter()
        .copied()
        .filter(|operation| operation.kind != 3)
        .collect::<Vec<_>>();
    let key = structured_net_key(&nonfermion_term, local_dimensions)?;
    let mut actions = Vec::with_capacity(key.len());
    for (axis, rows) in key {
        let dimension = *local_dimensions.get(axis).ok_or(PauliError::InvalidIndex {
            context: "structured net-action axis",
        })?;
        if rows.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: rows.len(),
            });
        }
        let stride = *strides.get(axis).ok_or(PauliError::InvalidIndex {
            context: "structured net-action stride",
        })?;
        let mut forward_destinations = Vec::with_capacity(dimension);
        let mut forward_amplitudes = Vec::with_capacity(dimension);
        let mut inverse_sources = vec![usize::MAX; dimension];
        for (source, &(destination, real, imag)) in rows.iter().enumerate() {
            let amplitude = Complex64::new(f64::from_bits(real), f64::from_bits(imag));
            forward_destinations.push(if is_exact_zero(amplitude) {
                usize::MAX
            } else {
                destination
            });
            forward_amplitudes.push(amplitude);
            if !is_exact_zero(amplitude) {
                if destination >= dimension || inverse_sources[destination] != usize::MAX {
                    return Err(PauliError::InvalidStructureLength {
                        expected: dimension,
                        actual: destination,
                    });
                }
                inverse_sources[destination] = source;
            }
        }
        actions.push(CompiledNetAction {
            axis,
            stride,
            dimension,
            forward_destinations: forward_destinations.into_boxed_slice(),
            forward_amplitudes: forward_amplitudes.into_boxed_slice(),
            inverse_sources: inverse_sources.into_boxed_slice(),
        });
    }
    Ok(Some(actions.into_boxed_slice()))
}

fn compiled_schedule_key(
    term: &[CompiledStructuredOperation],
    actions: Option<&[CompiledNetAction]>,
    fermion: Option<&FermionMaskDescriptor>,
) -> CompiledScheduleKey {
    let net_axes = actions
        .map(|actions| {
            actions
                .iter()
                .map(|action| (action.axis, action.forward_destinations.to_vec()))
                .collect()
        })
        .unwrap_or_default();
    let (fermion_axes, fermion_flip) = fermion
        .map(|descriptor| (descriptor.mode_axes.to_vec(), descriptor.flip_mask.to_vec()))
        .unwrap_or_default();
    let fallback_shape = if actions.is_none() {
        term.iter()
            .map(|operation| (operation.axis, operation.kind, operation.p, operation.q))
            .collect()
    } else {
        Vec::new()
    };
    CompiledScheduleKey {
        net_axes,
        fermion_axes,
        fermion_flip,
        fallback_shape,
    }
}

fn schedule_suffix_stride(
    active_axes: impl Iterator<Item = usize>,
    strides: &[usize],
    dimension: usize,
) -> usize {
    active_axes
        .max()
        .and_then(|axis| strides.get(axis).copied())
        .unwrap_or(dimension)
}

fn compile_schedule_groups(
    terms: &[Vec<CompiledStructuredOperation>],
    net_actions: &[Option<Box<[CompiledNetAction]>>],
    fermion_masks: &[Option<FermionMaskDescriptor>],
    strides: &[usize],
    dimension: usize,
) -> Result<Box<[StructuredScheduleGroup]>, PauliError> {
    if terms.len() != net_actions.len() || terms.len() != fermion_masks.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: terms.len(),
            actual: net_actions.len().max(fermion_masks.len()),
        });
    }
    let mut grouped = BTreeMap::<CompiledScheduleKey, Vec<usize>>::new();
    for index in 0..terms.len() {
        let key = compiled_schedule_key(
            &terms[index],
            net_actions[index].as_deref(),
            fermion_masks[index].as_ref(),
        );
        grouped.entry(key).or_default().push(index);
    }
    grouped
        .into_values()
        .map(|members| {
            let representative = members[0];
            let mut active_axes = BTreeSet::new();
            if let Some(actions) = net_actions[representative].as_deref() {
                active_axes.extend(actions.iter().map(|action| action.axis));
            } else {
                active_axes.extend(terms[representative].iter().map(|operation| operation.axis));
            }
            if let Some(descriptor) = fermion_masks[representative].as_ref() {
                // The descriptor's parity mask spans lower modes, so the full
                // mode-axis map is the conservative active shape.  This still
                // leaves trailing Cartesian spectator blocks reusable, which is
                // the dominant Fermion×Boson layout.
                active_axes.extend(descriptor.mode_axes.iter().copied());
            }
            let suffix_stride = schedule_suffix_stride(active_axes.into_iter(), strides, dimension);
            Ok(StructuredScheduleGroup {
                members: members.into_boxed_slice(),
                suffix_stride,
                contiguous_source_flip_mask: None,
            })
        })
        .collect::<Result<Vec<_>, PauliError>>()
        .map(Vec::into_boxed_slice)
}

fn apply_compiled_net_forward(
    actions: &[CompiledNetAction],
    source: usize,
    coefficient: Complex64,
) -> Result<(usize, Complex64, bool), PauliError> {
    let mut destination = source;
    let mut amplitude = coefficient;
    for action in actions {
        let digit = (source / action.stride) % action.dimension;
        let new_digit = action.forward_destinations[digit];
        if new_digit == usize::MAX {
            return Ok((source, amplitude, false));
        }
        let old_digit = digit;
        if new_digit >= old_digit {
            destination = destination
                .checked_add((new_digit - old_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        } else {
            destination = destination
                .checked_sub((old_digit - new_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        }
        amplitude *= action.forward_amplitudes[digit];
    }
    Ok((destination, amplitude, true))
}

fn apply_compiled_net_inverse(
    actions: &[CompiledNetAction],
    destination: usize,
    coefficient: Complex64,
) -> Result<(usize, Complex64, bool), PauliError> {
    let mut source = destination;
    for action in actions.iter().rev() {
        let digit = (source / action.stride) % action.dimension;
        let old_digit = action.inverse_sources[digit];
        if old_digit == usize::MAX {
            return Ok((destination, coefficient, false));
        }
        if old_digit >= digit {
            source = source
                .checked_add((old_digit - digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net source",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net source",
                })?;
        } else {
            source = source
                .checked_sub((digit - old_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net source",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net source",
                })?;
        }
    }
    // Re-evaluate amplitudes in the canonical forward action order. The
    // source derivation above must run backwards, but multiplying the local
    // factors backwards would change the last bit of complex128 results for
    // multi-axis schedules relative to dense/source-major execution.
    let mut amplitude = coefficient;
    let mut check_destination = source;
    for action in actions {
        let digit = (check_destination / action.stride) % action.dimension;
        let new_digit = action.forward_destinations[digit];
        if new_digit == usize::MAX {
            return Ok((destination, amplitude, false));
        }
        amplitude *= action.forward_amplitudes[digit];
        if new_digit >= digit {
            check_destination = check_destination
                .checked_add((new_digit - digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        } else {
            check_destination = check_destination
                .checked_sub((digit - new_digit).checked_mul(action.stride).ok_or(
                    PauliError::Overflow {
                        context: "computing structured net destination",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "computing structured net destination",
                })?;
        }
    }
    Ok((source, amplitude, true))
}

fn compile_fermion_mask_descriptor(
    term: &[CompiledStructuredOperation],
    fermion_positions: &[usize],
) -> Option<FermionMaskDescriptor> {
    let fermion_term = term
        .iter()
        .filter(|operation| operation.kind == 3)
        .collect::<Vec<_>>();
    if fermion_term.is_empty() {
        return None;
    }
    let mode_count = if fermion_positions.is_empty() {
        fermion_term.iter().map(|operation| operation.p + 1).max()?
    } else {
        fermion_positions.len()
    };
    let mode_axes = if fermion_positions.is_empty() {
        (0..mode_count).collect::<Vec<_>>()
    } else {
        fermion_positions.to_vec()
    };
    let steps = fermion_term
        .iter()
        .map(|operation| {
            (
                if fermion_positions.is_empty() {
                    operation.axis
                } else {
                    operation.p
                },
                operation.q == 1,
            )
        })
        .collect::<Vec<_>>();
    FermionMaskDescriptor::from_steps(&mode_axes, &steps)
}

/// Compact reusable matrix-free plan for finite structured operators.
///
/// The plan retains one canonical local-operation list and coefficient per
/// symbolic term. It deliberately does not retain the expanded transition
/// table or a sparse matrix.
pub struct StructuredMvpPlan {
    dimension: usize,
    strides: Vec<usize>,
    terms: Vec<Vec<CompiledStructuredOperation>>,
    net_actions: Vec<Option<Box<[CompiledNetAction]>>>,
    fermion_masks: Vec<Option<FermionMaskDescriptor>>,
    groups: Box<[StructuredScheduleGroup]>,
    coefficients: Vec<Complex64>,
    source_term_count: usize,
    estimated_bytes: u128,
    execution_family: StructuredExecutionFamily,
    inner: StructuredPlanInner,
    counters: Arc<StructuredApplyCounters>,
    fermion_schedule_count: usize,
    fermion_mask_schedule_count: usize,
    descriptor_ineligible_schedule_count: usize,
    contiguous_fermion_source_mode_count: Option<usize>,
}

/// Batched native phase measurements for the shared FermionMask descriptor.
///
/// This diagnostic is intentionally separate from normal MVP execution. It
/// times each phase over a complete source/descriptor batch, avoiding a clock
/// read inside individual Fermion contributions.
pub struct FermionMaskPhaseProfile {
    pub descriptor_count: usize,
    pub sample_count: usize,
    pub packed_source_unit_count: usize,
    pub schedule_member_evaluations: usize,
    pub packed_extraction_nanos: u128,
    pub parity_nanos: u128,
    pub index_update_nanos: u128,
    pub input_output_nanos: u128,
}

enum StructuredPlanInner {
    Pauli(MvpPlan),
    Compiled,
}

struct StructuredCsrResult {
    dimension: usize,
    indptr: Vec<u64>,
    columns: Vec<u64>,
    values: Vec<Complex64>,
    estimated_bytes: u128,
    estimated_construction_peak_bytes: u128,
    schedule_count: usize,
}

/// Reusable finite structured MVP plan backed by a duplicate-aggregated CSR graph.
pub struct StructuredEagerMvpPlan {
    dimension: usize,
    indptr: Vec<u64>,
    columns: Vec<u64>,
    values: Vec<Complex64>,
    estimated_bytes: u128,
    estimated_construction_peak_bytes: u128,
    source_term_count: usize,
    schedule_count: usize,
    execution_family: StructuredExecutionFamily,
    native_variant: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructuredExecutionFamily {
    Pauli,
    Cartesian,
    Generic,
}

impl StructuredExecutionFamily {
    fn label(self) -> &'static str {
        match self {
            Self::Pauli => "pauli",
            Self::Cartesian => "cartesian",
            Self::Generic => "generic",
        }
    }
}

fn classify_structured_execution(
    local_dimensions: &[usize],
    terms: &[Vec<CompiledStructuredOperation>],
    fermion_masks: &[Option<FermionMaskDescriptor>],
    net_actions: &[Option<Box<[CompiledNetAction]>>],
) -> StructuredExecutionFamily {
    if local_dimensions.iter().all(|&dimension| dimension == 2)
        && terms.iter().flatten().all(|operation| operation.kind == 0)
    {
        StructuredExecutionFamily::Pauli
    } else if net_actions.iter().any(Option::is_none)
        || terms.iter().enumerate().any(|(index, term)| {
            term.iter().any(|operation| operation.kind == 3)
                && fermion_masks.get(index).and_then(Option::as_ref).is_none()
                && net_actions.get(index).and_then(Option::as_deref).is_none()
        })
    {
        StructuredExecutionFamily::Generic
    } else {
        StructuredExecutionFamily::Cartesian
    }
}

fn compile_packed_pauli_operator(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
) -> Result<Option<PauliOperator>, PauliError> {
    if !local_dimensions.iter().all(|&dimension| dimension == 2)
        || terms.iter().flatten().any(|operation| operation.kind != 0)
    {
        return Ok(None);
    }
    let mut structures = Vec::with_capacity(terms.len());
    let mut packed_coefficients = Vec::with_capacity(coefficients.len());
    for (term, &coefficient) in terms.iter().zip(coefficients) {
        let mut codes = vec![0_u8; local_dimensions.len()];
        let mut phase = Complex64::new(1.0, 0.0);
        for operation in term {
            let axis = operation.axis;
            let current = codes.get(axis).copied().ok_or(PauliError::InvalidIndex {
                context: "packed Structured Pauli axis",
            })?;
            let (code, local_phase) = pauli_code_product(operation.p as u8, current);
            codes[axis] = code;
            phase *= local_phase;
        }
        structures.push(codes);
        packed_coefficients.push(coefficient * phase);
    }
    let operator =
        PauliOperator::from_terms(local_dimensions.len(), &structures, &packed_coefficients)?;
    Ok(Some(operator))
}

fn compile_packed_pauli_plan(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
) -> Result<Option<MvpPlan>, PauliError> {
    compile_packed_pauli_operator(local_dimensions, terms, coefficients)?
        .as_ref()
        .map(MvpPlan::from_operator)
        .transpose()
}

impl StructuredEagerMvpPlan {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn estimated_construction_peak_bytes(&self) -> u128 {
        self.estimated_construction_peak_bytes
    }

    pub fn source_term_count(&self) -> usize {
        self.source_term_count
    }

    pub fn schedule_count(&self) -> usize {
        self.schedule_count
    }

    pub fn execution_family(&self) -> &'static str {
        self.execution_family.label()
    }

    pub fn native_variant(&self) -> &'static str {
        self.native_variant
    }

    pub fn retained_format(&self) -> &'static str {
        "csr"
    }

    pub fn transition_count(&self) -> usize {
        self.values.len()
    }

    pub fn csr(&self) -> (Vec<u64>, Vec<u64>, Vec<Complex64>) {
        (
            self.indptr.clone(),
            self.columns.clone(),
            self.values.clone(),
        )
    }

    /// Move the retained CSR arrays to a native materialization boundary.
    pub fn into_csr(self) -> (usize, Vec<u64>, Vec<u64>, Vec<Complex64>) {
        (self.dimension, self.indptr, self.columns, self.values)
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        if state.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        let output_bytes = (self.dimension as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating structured eager MVP output",
            })?;
        if output_bytes > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested: output_bytes,
                limit: max_bytes,
            });
        }
        let mut output = vec![Complex64::default(); self.dimension];
        self.apply_into(state, &mut output, max_bytes)?;
        Ok(output)
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        if state.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if output.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: output.len(),
            });
        }
        let output_bytes = (self.dimension as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating structured eager MVP output",
            })?;
        if output_bytes > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested: output_bytes,
                limit: max_bytes,
            });
        }
        output.fill(Complex64::default());
        for (row, output_value) in output.iter_mut().enumerate() {
            let start = usize::try_from(self.indptr[row]).map_err(|_| PauliError::Overflow {
                context: "converting structured eager MVP row pointer",
            })?;
            let stop = usize::try_from(self.indptr[row + 1]).map_err(|_| PauliError::Overflow {
                context: "converting structured eager MVP row pointer",
            })?;
            let mut value = Complex64::default();
            for index in start..stop {
                let column =
                    usize::try_from(self.columns[index]).map_err(|_| PauliError::Overflow {
                        context: "converting structured eager MVP column",
                    })?;
                value += self.values[index] * state[column];
            }
            *output_value = value;
        }
        Ok(())
    }
}

fn mixed_radix_strides(local_dimensions: &[usize]) -> Result<Vec<usize>, PauliError> {
    let mut strides = vec![1_usize; local_dimensions.len()];
    let mut stride = 1_usize;
    for position in (0..local_dimensions.len()).rev() {
        strides[position] = stride;
        stride = stride
            .checked_mul(local_dimensions[position])
            .ok_or(PauliError::Overflow {
                context: "computing structured mixed-radix strides",
            })?;
    }
    Ok(strides)
}

fn validate_mvp_operation(
    operation: &StructuredOperation,
    local_dimensions: &[usize],
    fermion_positions: &[usize],
) -> Result<usize, PauliError> {
    let dimension =
        local_dimensions
            .get(operation.axis)
            .copied()
            .ok_or(PauliError::InvalidIndex {
                context: "structured MVP operation axis",
            })?;
    match operation.kind {
        0 => {
            if dimension != 2 {
                return Err(PauliError::InvalidIndex {
                    context: "Pauli operation requires a two-level axis",
                });
            }
            if operation.p > 3 {
                return Err(PauliError::InvalidCode {
                    code: operation.p as u8,
                    index: operation.axis,
                });
            }
            Ok(0)
        }
        1 | 2 => Ok(dimension),
        3 => {
            if dimension != 2 {
                return Err(PauliError::InvalidIndex {
                    context: "Fermion operation requires a two-level axis",
                });
            }
            if !fermion_positions.is_empty() {
                let mode = usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                    context: "converting structured Fermion mode",
                })?;
                let position =
                    fermion_positions
                        .get(mode)
                        .copied()
                        .ok_or(PauliError::InvalidIndex {
                            context: "structured MVP Fermion mode",
                        })?;
                if position != operation.axis {
                    return Err(PauliError::InvalidIndex {
                        context: "structured MVP Fermion axis",
                    });
                }
            }
            if operation.q > 1 {
                return Err(PauliError::InvalidCode {
                    code: operation.q as u8,
                    index: operation.axis,
                });
            }
            Ok(0)
        }
        _ => Err(PauliError::InvalidCode {
            code: operation.kind,
            index: operation.axis,
        }),
    }
}

fn compile_mvp_operations(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    fermion_positions: &[usize],
    strides: &[usize],
) -> Result<Vec<Vec<CompiledStructuredOperation>>, PauliError> {
    let mut compiled_terms = Vec::with_capacity(terms.len());
    for term in terms {
        let mut compiled = Vec::with_capacity(term.len());
        for operation in term {
            let table_len = validate_mvp_operation(operation, local_dimensions, fermion_positions)?;
            let p = usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                context: "converting structured operation parameter",
            })?;
            let q = usize::try_from(operation.q).map_err(|_| PauliError::Overflow {
                context: "converting structured operation parameter",
            })?;
            let mut local_table = Vec::new();
            if table_len != 0 {
                local_table = vec![Complex64::default(); table_len];
                match operation.kind {
                    1 => {
                        let creation = p;
                        let annihilation = q;
                        for (digit, value) in local_table.iter_mut().enumerate() {
                            if digit < annihilation {
                                continue;
                            }
                            let Some(destination) = digit
                                .checked_sub(annihilation)
                                .and_then(|value| value.checked_add(creation))
                            else {
                                continue;
                            };
                            if destination >= table_len {
                                continue;
                            }
                            let mut ladder_amplitude = 1.0;
                            for offset in 0..annihilation {
                                ladder_amplitude *= ((digit - offset) as f64).sqrt();
                            }
                            let remaining = digit - annihilation;
                            for offset in 0..creation {
                                ladder_amplitude *= ((remaining + offset + 1) as f64).sqrt();
                            }
                            *value = Complex64::new(ladder_amplitude, 0.0);
                        }
                    }
                    2 => {
                        let a = p % table_len;
                        for (digit, value) in local_table.iter_mut().enumerate() {
                            let b_digit = (q as u128 * digit as u128) % table_len as u128;
                            let angle =
                                2.0 * std::f64::consts::PI * b_digit as f64 / table_len as f64;
                            *value = Complex64::from_polar(1.0, angle);
                        }
                        let _ = a;
                    }
                    _ => unreachable!("only Boson and Weyl operations have local tables"),
                }
            }
            compiled.push(CompiledStructuredOperation {
                axis: operation.axis,
                kind: operation.kind,
                p,
                q,
                stride: strides[operation.axis],
                dimension: local_dimensions[operation.axis],
                local_table,
            });
        }
        compiled_terms.push(compiled);
    }
    Ok(compiled_terms)
}

fn update_compiled_destination(
    destination: &mut usize,
    digit: usize,
    new_digit: usize,
    operation: &CompiledStructuredOperation,
) -> Result<(), PauliError> {
    if new_digit >= operation.dimension {
        return Err(PauliError::InvalidIndex {
            context: "structured operation destination",
        });
    }
    if new_digit >= digit {
        let delta =
            (new_digit - digit)
                .checked_mul(operation.stride)
                .ok_or(PauliError::Overflow {
                    context: "computing structured operation destination",
                })?;
        *destination = destination.checked_add(delta).ok_or(PauliError::Overflow {
            context: "computing structured operation destination",
        })?;
    } else {
        let delta =
            (digit - new_digit)
                .checked_mul(operation.stride)
                .ok_or(PauliError::Overflow {
                    context: "computing structured operation destination",
                })?;
        *destination = destination.checked_sub(delta).ok_or(PauliError::Overflow {
            context: "computing structured operation destination",
        })?;
    }
    Ok(())
}

fn compiled_digit(destination: usize, operation: &CompiledStructuredOperation) -> usize {
    (destination / operation.stride) % operation.dimension
}

fn apply_compiled_nonfermion_operation(
    operation: &CompiledStructuredOperation,
    destination: &mut usize,
    amplitude: &mut Complex64,
) -> Result<bool, PauliError> {
    let digit = compiled_digit(*destination, operation);
    match operation.kind {
        0 => match operation.p {
            0 => {}
            1 => update_compiled_destination(destination, digit, 1 - digit, operation)?,
            2 => {
                update_compiled_destination(destination, digit, 1 - digit, operation)?;
                *amplitude *= if digit == 0 {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(0.0, -1.0)
                };
            }
            3 => {
                if digit == 1 {
                    *amplitude = -*amplitude;
                }
            }
            _ => unreachable!("MVP Pauli operation was validated during compilation"),
        },
        1 => {
            let factor = operation.local_table[digit];
            if factor.re == 0.0 && factor.im == 0.0 {
                return Ok(false);
            }
            let creation = operation.p;
            let annihilation = operation.q;
            let new_digit = digit
                .checked_sub(annihilation)
                .and_then(|value| value.checked_add(creation))
                .ok_or(PauliError::Overflow {
                    context: "computing structured boson destination",
                })?;
            update_compiled_destination(destination, digit, new_digit, operation)?;
            *amplitude *= factor;
        }
        2 => {
            *amplitude *= operation.local_table[digit];
            let shift = operation.p % operation.dimension;
            let new_digit = if shift == 0 {
                digit
            } else if digit < operation.dimension - shift {
                digit + shift
            } else {
                digit - (operation.dimension - shift)
            };
            update_compiled_destination(destination, digit, new_digit, operation)?;
        }
        _ => unreachable!("MVP operation was validated during compilation"),
    }
    Ok(true)
}

#[cfg(test)]
fn apply_compiled_structured_operation(
    operation: &CompiledStructuredOperation,
    local_dimensions: &[usize],
    strides: &[usize],
    fermion_positions: &[usize],
    destination: &mut usize,
    amplitude: &mut Complex64,
) -> Result<bool, PauliError> {
    if operation.kind != 3 {
        return apply_compiled_nonfermion_operation(operation, destination, amplitude);
    }
    let digit = compiled_digit(*destination, operation);
    let mode = operation.p;
    let parity = if fermion_positions.is_empty() {
        (0..operation.axis)
            .map(|axis| (*destination / strides[axis]) % local_dimensions[axis])
            .sum::<usize>()
            & 1
    } else {
        fermion_positions[..mode]
            .iter()
            .map(|&axis| (*destination / strides[axis]) % local_dimensions[axis])
            .sum::<usize>()
            & 1
    };
    if parity != 0 {
        *amplitude = -*amplitude;
    }
    match operation.q {
        0 if digit == 0 => Ok(false),
        0 => {
            update_compiled_destination(destination, digit, 0, operation)?;
            Ok(true)
        }
        1 if digit != 0 => Ok(false),
        1 => {
            update_compiled_destination(destination, digit, 1, operation)?;
            Ok(true)
        }
        _ => unreachable!("MVP Fermion operation was validated during compilation"),
    }
}

impl StructuredMvpPlan {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn source_term_count(&self) -> usize {
        self.source_term_count
    }

    pub fn schedule_count(&self) -> usize {
        self.terms.len()
    }

    pub fn execution_family(&self) -> &'static str {
        self.execution_family.label()
    }

    pub fn native_variant(&self) -> &'static str {
        match &self.inner {
            StructuredPlanInner::Pauli(_) => "pauli",
            StructuredPlanInner::Compiled => "compiled",
        }
    }

    pub fn retained_format(&self) -> &'static str {
        "compiled_actions"
    }

    pub fn estimated_construction_peak_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    /// Return test/benchmark counters for the compiled Cartesian executor.
    /// The counters are deliberately diagnostic metadata; they do not affect
    /// the public numerical result.
    pub fn debug_counters(&self) -> (usize, usize, usize, usize) {
        self.counters.snapshot()
    }

    pub fn debug_metadata(&self) -> (usize, usize, usize) {
        (
            self.fermion_schedule_count,
            self.fermion_mask_schedule_count,
            self.descriptor_ineligible_schedule_count,
        )
    }

    /// Measure batched descriptor phases for a raw-Fermion Cartesian plan.
    ///
    /// The sample count is capped to keep this diagnostic bounded; it is not
    /// part of normal plan application or public execution semantics.
    pub fn fermion_mask_phase_profile(
        &self,
        requested_samples: usize,
    ) -> Result<Option<FermionMaskPhaseProfile>, PauliError> {
        let descriptors = self
            .fermion_masks
            .iter()
            .filter_map(Option::as_ref)
            .collect::<Vec<_>>();
        if descriptors.is_empty() || self.dimension == 0 {
            return Ok(None);
        }
        let sample_count = requested_samples.max(1).min(self.dimension).min(65_536);
        let descriptor_words = descriptors
            .iter()
            .map(|descriptor| descriptor.required_occupied.len())
            .collect::<Vec<_>>();
        let contiguous_source_layout = self.contiguous_fermion_source_mode_count.is_some();
        let (word_offsets, total_words, packed_source_unit_count) = if contiguous_source_layout {
            let words = descriptor_words[0];
            (
                vec![0; descriptor_words.len()],
                words
                    .checked_mul(sample_count)
                    .ok_or(PauliError::Overflow {
                        context: "sizing FermionMask profile words",
                    })?,
                sample_count,
            )
        } else {
            let mut word_offsets = Vec::with_capacity(descriptor_words.len());
            let mut total_words = 0usize;
            for &words in &descriptor_words {
                word_offsets.push(total_words);
                total_words =
                    total_words
                        .checked_add(words.checked_mul(sample_count).ok_or(
                            PauliError::Overflow {
                                context: "sizing FermionMask profile words",
                            },
                        )?)
                        .ok_or(PauliError::Overflow {
                            context: "sizing FermionMask profile words",
                        })?;
            }
            (
                word_offsets,
                total_words,
                descriptors.len().saturating_mul(sample_count),
            )
        };
        let mut packed = vec![0_u64; total_words];
        let extraction_start = Instant::now();
        if contiguous_source_layout {
            let words = descriptor_words[0];
            for source in 0..sample_count {
                let start = source * words;
                descriptors[0]
                    .pack_contiguous_binary_index(source, &mut packed[start..start + words])?;
            }
        } else {
            for (descriptor_index, descriptor) in descriptors.iter().enumerate() {
                let words = descriptor_words[descriptor_index];
                let offset = word_offsets[descriptor_index];
                for source in 0..sample_count {
                    let start = offset + source * words;
                    descriptor.pack_index(
                        source,
                        &self.strides,
                        &mut packed[start..start + words],
                    )?;
                }
            }
        }
        let packed_extraction_nanos = extraction_start.elapsed().as_nanos();

        let parity_start = Instant::now();
        for (descriptor_index, descriptor) in descriptors.iter().enumerate() {
            let words = descriptor_words[descriptor_index];
            let offset = word_offsets[descriptor_index];
            for source in 0..sample_count {
                let start = offset + source * words;
                if contiguous_source_layout {
                    black_box(
                        descriptor
                            .amplitude_from_single_word(packed[start], Complex64::new(1.0, -0.25)),
                    );
                } else {
                    black_box(descriptor.amplitude_from_words(
                        &packed[start..start + words],
                        Complex64::new(1.0, -0.25),
                    ));
                }
            }
        }
        let parity_nanos = parity_start.elapsed().as_nanos();

        let index_start = Instant::now();
        for descriptor in &descriptors {
            for source in 0..sample_count {
                if contiguous_source_layout {
                    black_box(descriptor.apply_flip_to_contiguous_binary_index(source)?);
                } else {
                    black_box(descriptor.apply_flip_to_index(source, &self.strides)?);
                }
            }
        }
        let index_update_nanos = index_start.elapsed().as_nanos();

        let state = vec![Complex64::new(0.5, -0.25); sample_count];
        let mut output = vec![Complex64::default(); sample_count];
        let input_output_start = Instant::now();
        for descriptor_index in 0..descriptors.len() {
            for (source, &input) in state.iter().enumerate() {
                let destination = (source + descriptor_index) % sample_count;
                output[destination] += input * Complex64::new(1.0, -0.25);
            }
        }
        black_box(&output);
        let input_output_nanos = input_output_start.elapsed().as_nanos();
        Ok(Some(FermionMaskPhaseProfile {
            descriptor_count: descriptors.len(),
            sample_count,
            packed_source_unit_count,
            schedule_member_evaluations: descriptors.len().saturating_mul(sample_count),
            packed_extraction_nanos,
            parity_nanos,
            index_update_nanos,
            input_output_nanos,
        }))
    }

    pub fn transition_count(&self) -> Option<usize> {
        None
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        if state.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        let output_bytes = (self.dimension as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating structured MVP output",
            })?;
        if output_bytes > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested: output_bytes,
                limit: max_bytes,
            });
        }
        let mut output = vec![Complex64::default(); self.dimension];
        self.apply_into(state, &mut output, max_bytes)?;
        Ok(output)
    }

    #[inline]
    fn contiguous_fermion_destination_value(
        &self,
        destination: usize,
        packed_sources: &[u64],
        state: &[Complex64],
    ) -> Complex64 {
        let mut value = Complex64::default();
        for group in &self.groups {
            let source = destination
                ^ group
                    .contiguous_source_flip_mask
                    .expect("contiguous Fermion group has a source flip mask");
            let mut aggregate = Complex64::default();
            for &term_index in &group.members {
                let descriptor = self.fermion_masks[term_index]
                    .as_ref()
                    .expect("contiguous Fermion member has a descriptor");
                let Some(amplitude) = descriptor.amplitude_from_single_word(
                    packed_sources[source],
                    self.coefficients[term_index],
                ) else {
                    continue;
                };
                aggregate += amplitude;
            }
            if aggregate.re != 0.0 || aggregate.im != 0.0 {
                value += aggregate * state[source];
            }
        }
        value
    }

    fn apply_contiguous_fermion_cached_gather_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        let descriptor = self.fermion_masks.iter().find_map(Option::as_ref).ok_or(
            PauliError::InvalidStructureLength {
                expected: 1,
                actual: 0,
            },
        )?;
        let mut packed_sources = vec![0_u64; self.dimension];
        for (source, packed) in packed_sources.iter_mut().enumerate() {
            descriptor.pack_contiguous_binary_index(source, std::slice::from_mut(packed))?;
        }
        let work = self.terms.len().saturating_mul(self.dimension);
        if work >= 1 << 16 {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(destination, output_value)| {
                    *output_value = self.contiguous_fermion_destination_value(
                        destination,
                        &packed_sources,
                        state,
                    );
                });
        } else {
            for (destination, output_value) in output.iter_mut().enumerate() {
                *output_value =
                    self.contiguous_fermion_destination_value(destination, &packed_sources, state);
            }
        }
        let group_evaluations = self.dimension.saturating_mul(self.groups.len());
        let fermion_mask_applications = self.dimension.saturating_mul(self.terms.len());
        self.counters
            .schedule_group_evaluations
            .fetch_add(group_evaluations, Ordering::Relaxed);
        self.counters
            .fermion_mask_applications
            .fetch_add(fermion_mask_applications, Ordering::Relaxed);
        Ok(())
    }

    /// Apply the compact plan into caller-owned output storage.
    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        _max_bytes: u128,
    ) -> Result<(), PauliError> {
        if state.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if output.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: output.len(),
            });
        }
        if let StructuredPlanInner::Pauli(plan) = &self.inner {
            return plan.apply_into(state, output, _max_bytes);
        }
        output.fill(Complex64::default());
        if self.contiguous_fermion_source_mode_count.is_some() {
            return self.apply_contiguous_fermion_cached_gather_into(state, output);
        }
        let fermion_word_count = self
            .fermion_masks
            .iter()
            .filter_map(Option::as_ref)
            .map(|descriptor| descriptor.required_occupied.len())
            .max()
            .unwrap_or(0);
        let mut fermion_mask_scratch = vec![0_u64; fermion_word_count];
        let mut member_context = CompiledScheduleApplyContext {
            strides: &self.strides,
            fermion_mask_scratch: &mut fermion_mask_scratch,
            counters: &self.counters,
        };
        for group in &self.groups {
            let suffix_stride = group.suffix_stride;
            for block_start in (0..self.dimension).step_by(suffix_stride) {
                self.counters
                    .schedule_group_evaluations
                    .fetch_add(1, Ordering::Relaxed);
                let mut destination = None;
                let mut aggregate = Complex64::default();
                for &term_index in &group.members {
                    let contribution = apply_compiled_schedule_member(
                        &self.terms[term_index],
                        self.fermion_masks[term_index].as_ref(),
                        self.net_actions[term_index].as_deref(),
                        self.coefficients[term_index],
                        block_start,
                        &mut member_context,
                    )?;
                    let Some((member_destination, member_amplitude)) = contribution else {
                        continue;
                    };
                    if let Some(expected_destination) = destination {
                        if expected_destination != member_destination {
                            // This can only occur for a non-injective fallback
                            // schedule. Keep its members semantically separate
                            // rather than applying a wrong shared segment.
                            for offset in 0..suffix_stride {
                                output[member_destination + offset] +=
                                    member_amplitude * state[block_start + offset];
                            }
                            continue;
                        }
                    } else {
                        destination = Some(member_destination);
                    }
                    aggregate += member_amplitude;
                }
                let Some(destination) = destination else {
                    continue;
                };
                if aggregate.re == 0.0 && aggregate.im == 0.0 {
                    continue;
                }
                for offset in 0..suffix_stride {
                    output[destination + offset] += aggregate * state[block_start + offset];
                }
                if suffix_stride > 1 {
                    self.counters
                        .segment_applications
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }
}

fn apply_compiled_nonfermion_fallback(
    term: &[CompiledStructuredOperation],
    destination: &mut usize,
    amplitude: &mut Complex64,
) -> Result<bool, PauliError> {
    debug_assert!(term.iter().all(|operation| operation.kind != 3));
    for operation in term.iter().filter(|operation| operation.kind != 3) {
        if !apply_compiled_nonfermion_operation(operation, destination, amplitude)? {
            return Ok(false);
        }
    }
    Ok(true)
}

struct CompiledScheduleApplyContext<'a> {
    strides: &'a [usize],
    fermion_mask_scratch: &'a mut [u64],
    counters: &'a StructuredApplyCounters,
}

fn apply_compiled_schedule_member(
    term: &[CompiledStructuredOperation],
    descriptor: Option<&FermionMaskDescriptor>,
    actions: Option<&[CompiledNetAction]>,
    coefficient: Complex64,
    source: usize,
    context: &mut CompiledScheduleApplyContext<'_>,
) -> Result<Option<(usize, Complex64)>, PauliError> {
    let (mut destination, mut amplitude) = if let Some(descriptor) = descriptor {
        context
            .counters
            .fermion_mask_applications
            .fetch_add(1, Ordering::Relaxed);
        let words = descriptor.required_occupied.len();
        descriptor.pack_index(
            source,
            context.strides,
            &mut context.fermion_mask_scratch[..words],
        )?;
        let Some(amplitude) =
            descriptor.amplitude_from_words(&context.fermion_mask_scratch[..words], coefficient)
        else {
            return Ok(None);
        };
        (
            descriptor.apply_flip_to_index(source, context.strides)?,
            amplitude,
        )
    } else {
        (source, coefficient)
    };
    let valid = if let Some(actions) = actions {
        let (new_destination, new_amplitude, valid) =
            apply_compiled_net_forward(actions, destination, amplitude)?;
        destination = new_destination;
        amplitude = new_amplitude;
        valid
    } else {
        // A supported non-Fermion local action is normally represented by a
        // compiled net action.  Keep this narrow fallback for a non-injective
        // boundary schedule, but never re-enter the legacy Fermion sign
        // interpreter: FermionMask has already handled the Fermion factor.
        context
            .counters
            .nonfermion_fallback_invocations
            .fetch_add(1, Ordering::Relaxed);
        apply_compiled_nonfermion_fallback(term, &mut destination, &mut amplitude)?
    };
    if valid {
        Ok(Some((destination, amplitude)))
    } else {
        Ok(None)
    }
}

/// Build a compact reusable matrix-free plan without enumerating transitions.
pub fn structured_mvp_plan(
    local_dimensions: Vec<usize>,
    terms: Vec<Vec<StructuredOperation>>,
    coefficients: Vec<Complex64>,
    max_bytes: u128,
) -> Result<StructuredMvpPlan, PauliError> {
    structured_mvp_plan_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Build a compact MVP plan with an explicit public-axis map for raw Fermion
/// actions. The map is indexed by Fermion mode and stores its public axis.
pub fn structured_mvp_plan_with_fermion_positions(
    local_dimensions: Vec<usize>,
    terms: Vec<Vec<StructuredOperation>>,
    coefficients: Vec<Complex64>,
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredMvpPlan, PauliError> {
    validate_structured_inputs(&local_dimensions, &terms, &coefficients)?;
    let source_term_count = terms.len();
    let (terms, coefficients) = group_structured_terms(&terms, &coefficients, &local_dimensions)?;
    let dimension = mixed_radix_dimension(&local_dimensions)?;
    let strides = mixed_radix_strides(&local_dimensions)?;
    let operation_count: usize = terms.iter().try_fold(0usize, |count, term| {
        count.checked_add(term.len()).ok_or(PauliError::Overflow {
            context: "estimating structured MVP plan",
        })
    })?;
    let table_entries = terms.iter().try_fold(0usize, |count, term| {
        term.iter().try_fold(count, |count, operation| {
            let table_len =
                validate_mvp_operation(operation, &local_dimensions, fermion_positions)?;
            count.checked_add(table_len).ok_or(PauliError::Overflow {
                context: "estimating structured MVP local tables",
            })
        })
    })?;
    let estimated_bytes = (local_dimensions.len() as u128)
        .checked_mul(size_of::<usize>() as u128)
        .and_then(|value| {
            value.checked_add((strides.len() as u128).checked_mul(size_of::<usize>() as u128)?)
        })
        .and_then(|value| {
            value.checked_add(
                (terms.len() as u128)
                    .checked_mul(size_of::<Vec<CompiledStructuredOperation>>() as u128)?,
            )
        })
        .and_then(|value| {
            value.checked_add(
                (operation_count as u128)
                    .checked_mul(size_of::<CompiledStructuredOperation>() as u128)?,
            )
        })
        .and_then(|value| {
            value.checked_add((table_entries as u128).checked_mul(size_of::<Complex64>() as u128)?)
        })
        .and_then(|value| {
            value.checked_add(
                (coefficients.len() as u128).checked_mul(size_of::<Complex64>() as u128)?,
            )
        })
        .and_then(|value| value.checked_add(64))
        .and_then(|value| {
            value.checked_add(
                (fermion_positions.len() as u128).checked_mul(size_of::<usize>() as u128)?,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating structured MVP plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    let compiled_terms =
        compile_mvp_operations(&local_dimensions, &terms, fermion_positions, &strides)?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let fermion_schedule_count = compiled_terms
        .iter()
        .filter(|term| term.iter().any(|operation| operation.kind == 3))
        .count();
    let fermion_mask_schedule_count = fermion_masks.iter().filter(|mask| mask.is_some()).count();
    let descriptor_ineligible_schedule_count =
        fermion_schedule_count.saturating_sub(fermion_mask_schedule_count);
    let net_actions = terms
        .iter()
        .map(|term| compile_net_actions(term, &local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let mut groups = compile_schedule_groups(
        &compiled_terms,
        &net_actions,
        &fermion_masks,
        &strides,
        dimension,
    )?;
    let contiguous_fermion_source_mode_count = fermion_masks
        .iter()
        .find_map(Option::as_ref)
        .and_then(|descriptor| {
            (local_dimensions
                .iter()
                .all(|&local_dimension| local_dimension == 2)
                && fermion_masks.iter().all(Option::is_some)
                && net_actions
                    .iter()
                    .all(|actions| matches!(actions.as_deref(), Some([])))
                && groups.iter().all(|group| group.suffix_stride == 1)
                && descriptor.mode_axes.len() == local_dimensions.len()
                && descriptor.required_occupied.len() == 1
                && descriptor
                    .mode_axes
                    .iter()
                    .enumerate()
                    .all(|(mode, &axis)| mode == axis))
            .then_some(descriptor.mode_axes.len())
        });
    if contiguous_fermion_source_mode_count.is_some() {
        for group in groups.iter_mut() {
            let descriptor = fermion_masks[group.members[0]]
                .as_ref()
                .expect("contiguous Fermion group has a descriptor");
            group.contiguous_source_flip_mask =
                Some(descriptor.apply_flip_to_contiguous_binary_index(0)?);
        }
    }
    let net_action_bytes = net_actions.iter().try_fold(0_u128, |total, actions| {
        let action_bytes =
            actions
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .try_fold(0_u128, |subtotal, action| {
                    let table_bytes = (action.dimension as u128)
                        .checked_mul((2 * size_of::<usize>() + size_of::<Complex64>()) as u128)
                        .ok_or(PauliError::Overflow {
                            context: "estimating structured net-action tables",
                        })?;
                    subtotal
                        .checked_add(size_of::<CompiledNetAction>() as u128)
                        .and_then(|value| value.checked_add(table_bytes))
                        .ok_or(PauliError::Overflow {
                            context: "estimating structured net-action tables",
                        })
                })?;
        total.checked_add(action_bytes).ok_or(PauliError::Overflow {
            context: "estimating structured net-action tables",
        })
    })?;
    let fermion_mask_bytes = fermion_masks.iter().try_fold(0_u128, |total, descriptor| {
        let Some(descriptor) = descriptor.as_ref() else {
            return Ok(total);
        };
        let words = descriptor.required_occupied.len();
        let payload = words
            .checked_mul(4)
            .and_then(|value| value.checked_mul(size_of::<u64>()))
            .and_then(|value| {
                descriptor
                    .mode_axes
                    .len()
                    .checked_mul(size_of::<usize>())
                    .and_then(|axes_bytes| value.checked_add(axes_bytes))
            })
            .and_then(|value| value.checked_add(size_of::<FermionMaskDescriptor>()))
            .ok_or(PauliError::Overflow {
                context: "estimating FermionMask descriptors",
            })? as u128;
        total.checked_add(payload).ok_or(PauliError::Overflow {
            context: "estimating FermionMask descriptors",
        })
    })?;
    let estimated_bytes = estimated_bytes
        .checked_add(net_action_bytes)
        .and_then(|value| value.checked_add(fermion_mask_bytes))
        .and_then(|value| {
            value.checked_add(
                groups
                    .iter()
                    .map(|group| {
                        size_of::<StructuredScheduleGroup>()
                            + group.members.len() * size_of::<usize>()
                    })
                    .sum::<usize>() as u128,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating structured MVP net actions",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    let packed_pauli = compile_packed_pauli_plan(&local_dimensions, &terms, &coefficients)?;
    let (execution_family, inner) = if let Some(plan) = packed_pauli {
        (
            StructuredExecutionFamily::Pauli,
            StructuredPlanInner::Pauli(plan),
        )
    } else {
        (
            classify_structured_execution(
                &local_dimensions,
                &compiled_terms,
                &fermion_masks,
                &net_actions,
            ),
            StructuredPlanInner::Compiled,
        )
    };
    Ok(StructuredMvpPlan {
        dimension,
        strides,
        terms: compiled_terms,
        net_actions,
        fermion_masks,
        groups,
        coefficients,
        source_term_count,
        estimated_bytes,
        execution_family,
        inner,
        counters: Arc::new(StructuredApplyCounters::default()),
        fermion_schedule_count,
        fermion_mask_schedule_count,
        descriptor_ineligible_schedule_count,
        contiguous_fermion_source_mode_count,
    })
}

/// Build a finite structured MVP plan with a precomputed CSR transition graph.
pub fn structured_eager_mvp_plan(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<StructuredEagerMvpPlan, PauliError> {
    structured_eager_mvp_plan_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Build an eager MVP plan with an explicit public-axis map for raw Fermion
/// actions.
pub fn structured_eager_mvp_plan_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredEagerMvpPlan, PauliError> {
    let (grouped_terms, grouped_coefficients) =
        group_structured_terms(terms, coefficients, local_dimensions)?;
    if let Some(operator) =
        compile_packed_pauli_operator(local_dimensions, &grouped_terms, &grouped_coefficients)?
    {
        let csr = operator.csr_matrix(max_bytes)?;
        let retained_bytes = ((csr.indptr.len() * size_of::<u64>()) as u128)
            .checked_add((csr.columns.len() * size_of::<u64>()) as u128)
            .and_then(|value| {
                value.checked_add((csr.values.len() * size_of::<Complex64>()) as u128)
            })
            .ok_or(PauliError::Overflow {
                context: "estimating packed Pauli CSR retention",
            })?;
        return Ok(StructuredEagerMvpPlan {
            dimension: csr.dimension,
            indptr: csr.indptr,
            columns: csr.columns,
            values: csr.values,
            estimated_bytes: retained_bytes,
            estimated_construction_peak_bytes: retained_bytes.saturating_mul(2),
            source_term_count: terms.len(),
            schedule_count: operator.terms().len(),
            execution_family: StructuredExecutionFamily::Pauli,
            native_variant: "pauli",
        });
    }
    let csr = structured_csr_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        fermion_positions,
        max_bytes,
    )?;
    let strides = mixed_radix_strides(local_dimensions)?;
    let compiled_terms = compile_mvp_operations(
        local_dimensions,
        &grouped_terms,
        fermion_positions,
        &strides,
    )?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let net_actions = grouped_terms
        .iter()
        .map(|term| compile_net_actions(term, local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let execution_family = classify_structured_execution(
        local_dimensions,
        &compiled_terms,
        &fermion_masks,
        &net_actions,
    );
    Ok(StructuredEagerMvpPlan {
        dimension: csr.dimension,
        indptr: csr.indptr,
        columns: csr.columns,
        values: csr.values,
        estimated_bytes: csr.estimated_bytes,
        estimated_construction_peak_bytes: csr.estimated_construction_peak_bytes,
        source_term_count: terms.len(),
        schedule_count: csr.schedule_count,
        execution_family,
        native_variant: "compiled",
    })
}

pub struct StructuredSparseResult {
    pub dimension: usize,
    pub rows: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
}

fn structured_csr_memory_estimate(
    dimension: usize,
    term_count: usize,
    buffered_entries: usize,
) -> Result<(u128, u128), PauliError> {
    let pointer_count = dimension.checked_add(1).ok_or(PauliError::Overflow {
        context: "estimating structured CSR row pointers",
    })?;
    let pointer_bytes = (pointer_count as u128)
        .checked_mul(size_of::<u64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR row pointers",
        })?;
    let chunk_count = buffered_entries
        .checked_add(CSR_CHUNK_CAPACITY - 1)
        .and_then(|value| value.checked_div(CSR_CHUNK_CAPACITY))
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR chunk count",
        })?;
    let chunk_headers = (chunk_count as u128)
        .checked_mul(size_of::<Vec<(usize, u64, Complex64)>>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR chunk headers",
        })?;
    let entry_bytes = (buffered_entries as u128)
        .checked_mul(size_of::<(usize, u64, Complex64)>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR row buffers",
        })?;
    let final_entry_bytes = (buffered_entries as u128)
        .checked_mul((size_of::<u64>() + size_of::<Complex64>()) as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR retained graph",
        })?;
    let scratch_bytes = (term_count as u128)
        .checked_mul(64)
        .and_then(|value| value.checked_add(2 * size_of::<usize>() as u128))
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR row scratch",
        })?;
    let retained_bytes =
        pointer_bytes
            .checked_add(final_entry_bytes)
            .ok_or(PauliError::Overflow {
                context: "estimating structured CSR retained graph",
            })?;
    let construction_peak_bytes = pointer_bytes
        .checked_add(chunk_headers)
        .and_then(|value| value.checked_add(entry_bytes))
        .and_then(|value| value.checked_add(pointer_bytes))
        .and_then(|value| value.checked_add(final_entry_bytes))
        .and_then(|value| value.checked_add(scratch_bytes))
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR construction peak",
        })?;
    Ok((retained_bytes, construction_peak_bytes))
}

fn structured_csr_memory_preflight(
    dimension: usize,
    term_count: usize,
    max_bytes: u128,
) -> Result<u128, PauliError> {
    let raw_entries = dimension
        .checked_mul(term_count)
        .ok_or(PauliError::Overflow {
            context: "estimating structured CSR candidate transitions",
        })?;
    let (_, construction_peak_bytes) =
        structured_csr_memory_estimate(dimension, term_count, raw_entries)?;
    if construction_peak_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: construction_peak_bytes,
            limit: max_bytes,
        });
    }
    Ok(construction_peak_bytes)
}

fn structured_csr_matrix_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredCsrResult, PauliError> {
    validate_structured_inputs(local_dimensions, terms, coefficients)?;
    let (terms, coefficients) = group_structured_terms(terms, coefficients, local_dimensions)?;
    let schedule_count = terms.len();
    let dimension = mixed_radix_dimension(local_dimensions)?;
    let strides = mixed_radix_strides(local_dimensions)?;
    let compiled_terms =
        compile_mvp_operations(local_dimensions, &terms, fermion_positions, &strides)?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let net_actions = terms
        .iter()
        .map(|term| compile_net_actions(term, local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let construction_peak_bytes =
        structured_csr_memory_preflight(dimension, terms.len(), max_bytes)?;
    let mut sink = CsrSink::new_row_ordered(dimension)?;
    let mut sources: FxHashMap<usize, Complex64> =
        FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());
    let mut row_entries = Vec::<(u64, Complex64)>::with_capacity(terms.len());
    let fermion_word_count = fermion_masks
        .iter()
        .filter_map(Option::as_ref)
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    let mut fermion_mask_scratch = vec![0_u64; fermion_word_count];
    for row in 0..dimension {
        sources.clear();
        for (term_index, ((_term, fermion_descriptor), &coefficient)) in compiled_terms
            .iter()
            .zip(&fermion_masks)
            .zip(&coefficients)
            .enumerate()
        {
            if let Some(descriptor) = fermion_descriptor.as_ref() {
                let words = descriptor.required_occupied.len();
                descriptor.pack_index(row, &strides, &mut fermion_mask_scratch[..words])?;
                for (source, &flip) in fermion_mask_scratch[..words]
                    .iter_mut()
                    .zip(&descriptor.flip_mask)
                {
                    *source ^= flip;
                }
                if let Some(amplitude) =
                    descriptor.amplitude_from_words(&fermion_mask_scratch[..words], coefficient)
                {
                    let source = descriptor.apply_flip_to_index(row, &strides)?;
                    let actions = net_actions[term_index]
                        .as_deref()
                        .expect("compiled structured net action");
                    let (source, amplitude, valid) =
                        apply_compiled_net_inverse(actions, source, amplitude)?;
                    if valid {
                        sources
                            .entry(source)
                            .and_modify(|value| *value += amplitude)
                            .or_insert(amplitude);
                    }
                }
                continue;
            }
            let actions = net_actions[term_index]
                .as_deref()
                .expect("compiled structured net action");
            let (source, amplitude, valid) = apply_compiled_net_inverse(actions, row, coefficient)?;
            if valid {
                sources
                    .entry(source)
                    .and_modify(|value| *value += amplitude)
                    .or_insert(amplitude);
            }
        }
        let mut ordered_sources = sources.drain().collect::<Vec<_>>();
        ordered_sources.sort_unstable_by_key(|&(source, _)| source);
        row_entries.clear();
        for (source, value) in ordered_sources {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            row_entries.push((
                u64::try_from(source).map_err(|_| PauliError::Overflow {
                    context: "converting structured CSR column index",
                })?,
                value,
            ));
        }
        sink.finish_row(row, &row_entries)?;
    }

    let buffered_entries = sink.len();
    let (retained_bytes, actual_peak_bytes) =
        structured_csr_memory_estimate(dimension, terms.len(), buffered_entries)?;
    let (indptr, columns, values, _) = sink.into_csr()?;
    Ok(StructuredCsrResult {
        dimension,
        indptr,
        columns,
        values,
        estimated_bytes: retained_bytes,
        estimated_construction_peak_bytes: construction_peak_bytes.max(actual_peak_bytes),
        schedule_count,
    })
}

/// Compile canonical structured terms into deterministic sparse COO arrays.
///
/// This shares the exact transition convention with the dense kernel while
/// avoiding Python basis-state and dictionary loops for larger finite spaces.
pub fn structured_sparse_matrix(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<StructuredSparseResult, PauliError> {
    structured_sparse_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Compile structured terms into sparse arrays with an explicit raw-Fermion
/// public-axis map.
pub fn structured_sparse_matrix_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredSparseResult, PauliError> {
    let csr = structured_csr_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        fermion_positions,
        max_bytes,
    )?;
    let count = csr.values.len();
    let output_bytes = (count as u128)
        .checked_mul((size_of::<u64>() * 2 + size_of::<Complex64>()) as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured COO output",
        })?;
    let peak_with_output = csr
        .estimated_construction_peak_bytes
        .checked_add(output_bytes)
        .ok_or(PauliError::Overflow {
            context: "estimating structured COO construction",
        })?;
    if peak_with_output > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: peak_with_output,
            limit: max_bytes,
        });
    }
    let mut rows = Vec::with_capacity(count);
    let mut columns = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    for row in 0..csr.dimension {
        let start = usize::try_from(csr.indptr[row]).map_err(|_| PauliError::Overflow {
            context: "converting structured COO row pointer",
        })?;
        let stop = usize::try_from(csr.indptr[row + 1]).map_err(|_| PauliError::Overflow {
            context: "converting structured COO row pointer",
        })?;
        for index in start..stop {
            rows.push(u64::try_from(row).map_err(|_| PauliError::Overflow {
                context: "converting structured COO row index",
            })?);
            columns.push(csr.columns[index]);
            values.push(csr.values[index]);
        }
    }
    check_sparse_output_bytes(rows.len(), max_bytes)?;
    Ok(StructuredSparseResult {
        dimension: csr.dimension,
        rows,
        columns,
        values,
    })
}

/// Compile canonical structured terms into a row-major mixed-radix matrix.
pub fn structured_dense_matrix(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<(usize, Vec<Complex64>), PauliError> {
    structured_dense_matrix_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Compile structured terms into dense storage with an explicit raw-Fermion
/// public-axis map.
pub fn structured_dense_matrix_with_fermion_positions(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<(usize, Vec<Complex64>), PauliError> {
    validate_structured_inputs(local_dimensions, terms, coefficients)?;
    let (terms, coefficients) = group_structured_terms(terms, coefficients, local_dimensions)?;
    let dimension = mixed_radix_dimension(local_dimensions)?;
    let strides = mixed_radix_strides(local_dimensions)?;
    let compiled_terms =
        compile_mvp_operations(local_dimensions, &terms, fermion_positions, &strides)?;
    let net_actions = terms
        .iter()
        .map(|term| compile_net_actions(term, local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let entries = dimension
        .checked_mul(dimension)
        .ok_or(PauliError::Overflow {
            context: "computing structured dense matrix entries",
        })?;
    let bytes = (entries as u128)
        .checked_mul(size_of::<Complex64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured dense matrix memory",
        })?;
    if bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: bytes,
            limit: max_bytes,
        });
    }
    let mut matrix = vec![Complex64::default(); entries];
    let fermion_word_count = fermion_masks
        .iter()
        .filter_map(Option::as_ref)
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    let mut fermion_mask_scratch = vec![0_u64; fermion_word_count];
    for column in 0..dimension {
        for (term_index, ((_term, fermion_descriptor), &coefficient)) in compiled_terms
            .iter()
            .zip(&fermion_masks)
            .zip(&coefficients)
            .enumerate()
        {
            if let Some(descriptor) = fermion_descriptor.as_ref() {
                let words = descriptor.required_occupied.len();
                descriptor.pack_index(column, &strides, &mut fermion_mask_scratch[..words])?;
                if let Some(amplitude) =
                    descriptor.amplitude_from_words(&fermion_mask_scratch[..words], coefficient)
                {
                    let destination = descriptor.apply_flip_to_index(column, &strides)?;
                    let actions = net_actions[term_index]
                        .as_deref()
                        .expect("compiled structured net action");
                    let (destination, amplitude, valid) =
                        apply_compiled_net_forward(actions, destination, amplitude)?;
                    if valid {
                        let entry = &mut matrix[destination * dimension + column];
                        *entry += amplitude;
                    }
                }
                continue;
            }
            let actions = net_actions[term_index]
                .as_deref()
                .expect("compiled structured net action");
            let (destination, amplitude, valid) =
                apply_compiled_net_forward(actions, column, coefficient)?;
            if valid {
                let entry = &mut matrix[destination * dimension + column];
                *entry += amplitude;
            }
        }
    }
    Ok((dimension, matrix))
}

fn validate_structured_inputs(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
) -> Result<(), PauliError> {
    if terms.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: terms.len(),
            actual: coefficients.len(),
        });
    }
    if local_dimensions.contains(&0) {
        return Err(PauliError::InvalidStructureLength {
            expected: 1,
            actual: 0,
        });
    }
    Ok(())
}

/// Group identical coefficient-free local action schedules once at plan
/// construction.  The operation key is canonical and ordered, so schedule
/// order never depends on hash-map iteration.
fn group_structured_terms(
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    local_dimensions: &[usize],
) -> Result<(Vec<Vec<StructuredOperation>>, Vec<Complex64>), PauliError> {
    if terms.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: terms.len(),
            actual: coefficients.len(),
        });
    }
    let mut grouped = BTreeMap::<NetKey, (Vec<StructuredOperation>, Complex64)>::new();
    for (term, &coefficient) in terms.iter().zip(coefficients) {
        let (term, phase) = canonicalize_structured_schedule(term, local_dimensions)?;
        let key = structured_net_key(&term, local_dimensions)?;
        let coefficient = coefficient * phase;
        grouped
            .entry(key)
            .and_modify(|(_, value)| *value += coefficient)
            .or_insert((term, coefficient));
    }
    let mut grouped_terms = Vec::with_capacity(grouped.len());
    let mut grouped_coefficients = Vec::with_capacity(grouped.len());
    for (_, (term, coefficient)) in grouped {
        if !is_exact_zero(coefficient) {
            grouped_terms.push(term);
            grouped_coefficients.push(coefficient);
        }
    }
    Ok((grouped_terms, grouped_coefficients))
}

type NetAxisRow = (usize, u64, u64);
type NetKey = Vec<(usize, Vec<NetAxisRow>)>;

/// Return a deterministic local transition fingerprint for a non-Fermion
/// schedule.  The fingerprint includes destination digits and exact binary
/// floating-point amplitude bits, so distinct Boson ladder polynomials or Weyl
/// phases are never merged accidentally.  Raw Fermion schedules are excluded
/// because their sign depends on occupations outside the local axis.
fn structured_net_key(
    term: &[StructuredOperation],
    local_dimensions: &[usize],
) -> Result<NetKey, PauliError> {
    if term.iter().any(|operation| operation.kind == 3) {
        return Ok(term
            .iter()
            .map(|operation| {
                (
                    operation.axis,
                    vec![(
                        operation.p as usize,
                        operation.kind as u64,
                        operation.q as u64,
                    )],
                )
            })
            .collect());
    }
    let mut by_axis = BTreeMap::<usize, Vec<StructuredOperation>>::new();
    for &operation in term {
        by_axis.entry(operation.axis).or_default().push(operation);
    }
    let mut key = Vec::with_capacity(by_axis.len());
    for (axis, operations) in by_axis {
        let dimension = *local_dimensions.get(axis).ok_or(PauliError::InvalidIndex {
            context: "structured schedule axis",
        })?;
        if dimension == 0 {
            return Err(PauliError::InvalidStructureLength {
                expected: 1,
                actual: 0,
            });
        }
        if operations.iter().any(|operation| operation.kind == 0) && dimension != 2 {
            return Err(PauliError::InvalidIndex {
                context: "Pauli operation requires a two-level axis",
            });
        }
        let mut rows = Vec::with_capacity(dimension);
        for source in 0..dimension {
            let mut digit = source;
            let mut amplitude = Complex64::new(1.0, 0.0);
            for operation in &operations {
                match operation.kind {
                    0 => match operation.p {
                        0 => {}
                        1 => digit = 1 - digit,
                        2 => {
                            let input = digit;
                            digit = 1 - digit;
                            amplitude *= if input == 0 {
                                Complex64::new(0.0, 1.0)
                            } else {
                                Complex64::new(0.0, -1.0)
                            };
                        }
                        3 => {
                            if digit == 1 {
                                amplitude = -amplitude;
                            }
                        }
                        code => {
                            return Err(PauliError::InvalidCode {
                                code: code as u8,
                                index: axis,
                            });
                        }
                    },
                    1 => {
                        let creation =
                            usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Boson creation count",
                            })?;
                        let annihilation =
                            usize::try_from(operation.q).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Boson annihilation count",
                            })?;
                        if digit < annihilation {
                            amplitude = Complex64::default();
                            break;
                        }
                        for offset in 0..annihilation {
                            amplitude *= ((digit - offset) as f64).sqrt();
                        }
                        let remaining = digit - annihilation;
                        let Some(new_digit) = remaining.checked_add(creation) else {
                            amplitude = Complex64::default();
                            break;
                        };
                        if new_digit >= dimension {
                            amplitude = Complex64::default();
                            break;
                        }
                        for offset in 0..creation {
                            amplitude *= ((remaining + offset + 1) as f64).sqrt();
                        }
                        digit = new_digit;
                    }
                    2 => {
                        let shift =
                            usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Weyl shift",
                            })? % dimension;
                        let phase_power =
                            usize::try_from(operation.q).map_err(|_| PauliError::Overflow {
                                context: "canonicalizing Weyl phase",
                            })?;
                        let angle = 2.0 * std::f64::consts::PI * (phase_power * digit) as f64
                            / dimension as f64;
                        amplitude *= Complex64::from_polar(1.0, angle);
                        digit = (digit + shift) % dimension;
                    }
                    kind => {
                        return Err(PauliError::InvalidCode {
                            code: kind,
                            index: axis,
                        });
                    }
                }
            }
            if is_exact_zero(amplitude) {
                rows.push((0, 0, 0));
            } else {
                rows.push((digit, amplitude.re.to_bits(), amplitude.im.to_bits()));
            }
        }
        key.push((axis, rows));
    }
    Ok(key)
}

/// Canonicalize coefficient-free net schedules for operations whose domains
/// commute.  Pauli products and Weyl shifts are folded locally, while raw
/// Fermion sequences remain in their original order because their parity is
/// state dependent.  The returned phase is the exact scalar generated by the
/// local composition.
fn canonicalize_structured_schedule(
    term: &[StructuredOperation],
    local_dimensions: &[usize],
) -> Result<(Vec<StructuredOperation>, Complex64), PauliError> {
    if term.iter().any(|operation| operation.kind == 3) {
        return Ok((term.to_vec(), Complex64::new(1.0, 0.0)));
    }
    let mut by_axis = BTreeMap::<usize, Vec<StructuredOperation>>::new();
    for &operation in term {
        if operation.axis >= local_dimensions.len() {
            return Err(PauliError::InvalidIndex {
                context: "structured schedule axis",
            });
        }
        by_axis.entry(operation.axis).or_default().push(operation);
    }
    let mut phase = Complex64::new(1.0, 0.0);
    let mut canonical = Vec::with_capacity(term.len());
    for (_axis, operations) in by_axis {
        let mut local = Vec::<StructuredOperation>::with_capacity(operations.len());
        for operation in operations {
            if let Some(previous) = local.last_mut() {
                if previous.kind == 0 && operation.kind == 0 {
                    let (code, local_phase) =
                        pauli_code_product(operation.p as u8, previous.p as u8);
                    previous.p = u32::from(code);
                    phase *= local_phase;
                    continue;
                }
                if previous.kind == 2 && operation.kind == 2 {
                    let dimension = local_dimensions[operation.axis];
                    if dimension == 0 {
                        return Err(PauliError::InvalidStructureLength {
                            expected: 1,
                            actual: 0,
                        });
                    }
                    let shift = (previous.p as usize + operation.p as usize) % dimension;
                    let phase_power = (previous.q as usize + operation.q as usize) % dimension;
                    let cross =
                        (operation.q as usize * (previous.p as usize % dimension)) % dimension;
                    let angle = 2.0 * std::f64::consts::PI * cross as f64 / dimension as f64;
                    phase *= Complex64::from_polar(1.0, angle);
                    previous.p = u32::try_from(shift).map_err(|_| PauliError::Overflow {
                        context: "canonicalizing Weyl shift",
                    })?;
                    previous.q = u32::try_from(phase_power).map_err(|_| PauliError::Overflow {
                        context: "canonicalizing Weyl phase",
                    })?;
                    continue;
                }
            }
            if operation.kind == 0 && operation.p == 0 {
                continue;
            }
            local.push(operation);
        }
        local.retain(|operation| !(operation.kind == 0 && operation.p == 0));
        canonical.extend(local);
    }
    Ok((canonical, phase))
}

fn mixed_radix_dimension(local_dimensions: &[usize]) -> Result<usize, PauliError> {
    local_dimensions.iter().try_fold(1usize, |value, &factor| {
        value.checked_mul(factor).ok_or(PauliError::Overflow {
            context: "computing mixed-radix basis dimension",
        })
    })
}

fn check_sparse_output_bytes(count: usize, max_bytes: u128) -> Result<(), PauliError> {
    let requested = (count as u128)
        .checked_mul(32)
        .ok_or(PauliError::Overflow {
            context: "estimating structured sparse output",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        apply_compiled_structured_operation, canonicalize_boson_terms, canonicalize_fermion_terms,
        compile_fermion_mask_descriptor, compile_mvp_operations, group_structured_terms,
        hybrid_qudit_product, jordan_wigner_terms, mixed_radix_strides, structured_dense_matrix,
        structured_mvp_plan_with_fermion_positions, structured_sparse_matrix,
        CompiledStructuredOperation, HybridBatch, StructuredOperation,
    };
    use crate::{allocation_count, Complex64};

    #[test]
    fn compiles_pauli_boson_and_weyl_operations_in_mixed_radix_order() {
        let (dimension, matrix) = structured_dense_matrix(
            &[2, 3],
            &[
                vec![StructuredOperation {
                    axis: 0,
                    kind: 0,
                    p: 1,
                    q: 0,
                }],
                vec![StructuredOperation {
                    axis: 1,
                    kind: 1,
                    p: 1,
                    q: 0,
                }],
            ],
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(dimension, 6);
        assert_eq!(matrix[3], Complex64::new(1.0, 0.0));
        assert_eq!(matrix[dimension], Complex64::new(2.0, 0.0));

        let (dimension, matrix) = structured_dense_matrix(
            &[3],
            &[vec![StructuredOperation {
                axis: 0,
                kind: 2,
                p: 1,
                q: 1,
            }]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(dimension, 3);
        assert_eq!(matrix[3], Complex64::new(1.0, 0.0));
        assert!(
            (matrix[2 * dimension + 1] - Complex64::new(-0.5, 3.0_f64.sqrt() / 2.0)).norm() < 1e-12
        );
    }

    #[test]
    fn fermion_mask_descriptor_matches_ordered_car_actions() {
        let local_dimensions = [2, 2, 2];
        let fermion_positions = [0, 1, 2];
        let strides = mixed_radix_strides(&local_dimensions).unwrap();
        let term = vec![
            StructuredOperation {
                axis: 1,
                kind: 3,
                p: 1,
                q: 0,
            },
            StructuredOperation {
                axis: 2,
                kind: 3,
                p: 2,
                q: 1,
            },
            StructuredOperation {
                axis: 0,
                kind: 3,
                p: 0,
                q: 1,
            },
        ];
        let compiled = compile_mvp_operations(
            &local_dimensions,
            std::slice::from_ref(&term),
            &fermion_positions,
            &strides,
        )
        .unwrap();
        let descriptor = compile_fermion_mask_descriptor(&compiled[0], &fermion_positions)
            .expect("raw Fermion monomial should lower to a mask");
        for source in 0..8 {
            let mut expected_destination = source;
            let mut expected_amplitude = Complex64::new(1.0, 0.0);
            let mut expected_valid = true;
            for operation in &compiled[0] {
                if !apply_compiled_structured_operation(
                    operation,
                    &local_dimensions,
                    &strides,
                    &fermion_positions,
                    &mut expected_destination,
                    &mut expected_amplitude,
                )
                .unwrap()
                {
                    expected_valid = false;
                    break;
                }
            }
            let mut source_words = vec![0_u64; descriptor.required_occupied.len()];
            descriptor
                .pack_index(source, &strides, &mut source_words)
                .unwrap();
            let lowered = descriptor
                .amplitude_from_words(&source_words, Complex64::new(1.0, 0.0))
                .map(|amplitude| {
                    (
                        descriptor.apply_flip_to_index(source, &strides).unwrap(),
                        amplitude,
                    )
                });
            match (expected_valid, lowered) {
                (false, None) => {}
                (true, Some((destination, amplitude))) => {
                    assert_eq!(destination, expected_destination);
                    assert_eq!(amplitude, expected_amplitude);
                }
                mismatch => panic!("FermionMask mismatch for source {source}: {mismatch:?}"),
            }
        }
    }

    #[test]
    fn wide_fermion_mask_descriptor_handles_mode_128() {
        let positions = (0..130).collect::<Vec<_>>();
        let term = vec![CompiledStructuredOperation {
            axis: 128,
            kind: 3,
            p: 128,
            q: 1,
            stride: 1,
            dimension: 2,
            local_table: Vec::new(),
        }];
        let descriptor = compile_fermion_mask_descriptor(&term, &positions)
            .expect("wide Fermion monomial should lower");
        let source = vec![0_u64, 0_u64, 0_u64];
        let mut destination = vec![0_u64; source.len()];
        let amplitude = descriptor
            .apply_to_words(&source, &mut destination, Complex64::new(2.0, -1.0))
            .expect("creation accepts an empty source mode");
        assert_eq!(destination[2] & (1_u64 << 0), 1);
        assert_eq!(amplitude, Complex64::new(2.0, -1.0));
        let occupied = vec![0_u64, 0_u64, 1_u64];
        assert!(descriptor
            .amplitude_from_words(&occupied, Complex64::new(1.0, 0.0))
            .is_none());
    }

    #[test]
    fn compiled_fermion_apply_has_constant_per_call_allocation() {
        let make_plan = |terms: Vec<Vec<StructuredOperation>>| {
            let coefficients = vec![Complex64::new(1.0, 0.0); terms.len()];
            structured_mvp_plan_with_fermion_positions(
                vec![2, 2, 2],
                terms,
                coefficients,
                &[0, 1, 2],
                u128::MAX,
            )
            .unwrap()
        };
        let hopping_forward = vec![
            StructuredOperation {
                axis: 0,
                kind: 3,
                p: 0,
                q: 1,
            },
            StructuredOperation {
                axis: 1,
                kind: 3,
                p: 1,
                q: 0,
            },
        ];
        let hopping_backward = vec![
            StructuredOperation {
                axis: 1,
                kind: 3,
                p: 1,
                q: 1,
            },
            StructuredOperation {
                axis: 2,
                kind: 3,
                p: 2,
                q: 0,
            },
        ];
        let single = make_plan(vec![hopping_forward.clone()]);
        let multiple = make_plan(vec![hopping_forward, hopping_backward]);
        let state = vec![Complex64::new(0.5, -0.25); 8];
        let mut output = vec![Complex64::default(); 8];
        single.apply_into(&state, &mut output, 0).unwrap();
        let (_, single_allocations) = allocation_count(|| {
            single.apply_into(&state, &mut output, 0).unwrap();
        });
        let (_, multiple_allocations) = allocation_count(|| {
            multiple.apply_into(&state, &mut output, 0).unwrap();
        });
        assert_eq!(single_allocations, multiple_allocations);
        assert_eq!(single_allocations, 1);
    }

    #[test]
    fn groups_commuting_net_schedules_with_exact_local_phases() {
        let x = StructuredOperation {
            axis: 0,
            kind: 0,
            p: 1,
            q: 0,
        };
        let y = StructuredOperation {
            axis: 0,
            kind: 0,
            p: 2,
            q: 0,
        };
        let (terms, coefficients) = group_structured_terms(
            &[vec![x, y], vec![y, x]],
            &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            &[2],
        )
        .unwrap();
        assert!(terms.is_empty());
        assert!(coefficients.is_empty());

        let shift = StructuredOperation {
            axis: 0,
            kind: 2,
            p: 1,
            q: 1,
        };
        let (terms, coefficients) =
            group_structured_terms(&[vec![shift, shift]], &[Complex64::new(1.0, 0.0)], &[3])
                .unwrap();
        assert_eq!(terms[0][0].p, 2);
        assert_eq!(terms[0][0].q, 2);
        assert!((coefficients[0] - Complex64::new(-0.5, 3.0_f64.sqrt() / 2.0)).norm() < 1e-12);

        let create = StructuredOperation {
            axis: 0,
            kind: 1,
            p: 1,
            q: 0,
        };
        let (terms, coefficients) = group_structured_terms(
            &[
                vec![create, create],
                vec![StructuredOperation { p: 2, ..create }],
            ],
            &[Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)],
            &[5],
        )
        .unwrap();
        assert!(terms.is_empty());
        assert!(coefficients.is_empty());
    }

    #[test]
    fn rejects_dense_output_before_allocation() {
        let result = structured_dense_matrix(&[2, 2], &[], &[], 15);
        assert!(matches!(result, Err(crate::PauliError::MemoryLimit { .. })));
    }

    #[test]
    fn native_car_ccr_and_jordan_wigner_kernels_match_frozen_identities() {
        let (creation, annihilation, coefficients) = canonicalize_fermion_terms(
            1,
            &[vec![(0, 1), (0, 0)]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(creation, vec![vec![], vec![0]]);
        assert_eq!(annihilation, vec![vec![], vec![0]]);
        assert_eq!(
            coefficients,
            vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]
        );

        let (blocks, coefficients) = canonicalize_boson_terms(
            1,
            &[vec![(0, 1), (0, 0)]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(blocks, vec![vec![], vec![(0, 1, 1)]]);
        assert_eq!(
            coefficients,
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]
        );

        let (words, coefficients) = jordan_wigner_terms(
            1,
            &[vec![0]],
            &[vec![0]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(words, vec![vec![0], vec![3]]);
        assert_eq!(coefficients[0], Complex64::new(0.5, 0.0));
        assert_eq!(coefficients[1], Complex64::new(-0.5, 0.0));
    }

    #[test]
    fn sparse_structured_kernel_matches_dense_mixed_radix_output() {
        let dimensions = [2, 3];
        let terms = vec![
            vec![StructuredOperation {
                axis: 0,
                kind: 0,
                p: 1,
                q: 0,
            }],
            vec![StructuredOperation {
                axis: 1,
                kind: 1,
                p: 1,
                q: 0,
            }],
        ];
        let coefficients = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let (dimension, dense) =
            structured_dense_matrix(&dimensions, &terms, &coefficients, u128::MAX).unwrap();
        let sparse =
            structured_sparse_matrix(&dimensions, &terms, &coefficients, u128::MAX).unwrap();
        let mut reconstructed = vec![Complex64::default(); dense.len()];
        for ((&row, &column), &value) in sparse.rows.iter().zip(&sparse.columns).zip(&sparse.values)
        {
            reconstructed[row as usize * dimension + column as usize] = value;
        }
        assert_eq!(reconstructed, dense);
    }

    #[test]
    fn hybrid_qudit_identity_product_is_absent_not_empty_present() {
        let left_triples = vec![vec![(0, 1, 0)]];
        let right_triples = vec![vec![(0, 2, 0)]];
        let empty_bool = vec![false];
        let empty_fermion = vec![Vec::new()];
        let empty_boson = vec![Vec::new()];
        let empty_qubit = vec![vec![0]];
        let present = vec![true];
        let left = HybridBatch {
            fermion_present: &empty_bool,
            fermion_creation: &empty_fermion,
            fermion_annihilation: &empty_fermion,
            boson_present: &empty_bool,
            boson_blocks: &empty_boson,
            qubit_codes: &empty_qubit,
            mapped_present: &empty_bool,
            mapped_codes: &empty_qubit,
            qudit_present: &present,
            qudit_triples: &left_triples,
            coefficients: &[Complex64::new(1.0, 0.0)],
        };
        let right = HybridBatch {
            fermion_present: &empty_bool,
            fermion_creation: &empty_fermion,
            fermion_annihilation: &empty_fermion,
            boson_present: &empty_bool,
            boson_blocks: &empty_boson,
            qubit_codes: &empty_qubit,
            mapped_present: &empty_bool,
            mapped_codes: &empty_qubit,
            qudit_present: &present,
            qudit_triples: &right_triples,
            coefficients: &[Complex64::new(1.0, 0.0)],
        };
        let (triples, phase) = hybrid_qudit_product(3, &left, 0, &right, 0);
        assert_eq!(triples, None);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn direct_weyl_phase_reduces_before_float_conversion() {
        let dimension = 1usize << 20;
        let operation = StructuredOperation {
            axis: 0,
            kind: 2,
            p: 0,
            q: (dimension - 1) as u32,
        };
        let compiled =
            super::compile_mvp_operations(&[dimension], &[vec![operation]], &[], &[1]).unwrap();
        let mut destination = dimension - 1;
        let mut amplitude = Complex64::new(1.0, 0.0);
        assert!(super::apply_compiled_structured_operation(
            &compiled[0][0],
            &[dimension],
            &[1],
            &[],
            &mut destination,
            &mut amplitude,
        )
        .unwrap());
        let expected = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / dimension as f64);
        assert!((amplitude - expected).norm() < 1e-15);
    }
}
