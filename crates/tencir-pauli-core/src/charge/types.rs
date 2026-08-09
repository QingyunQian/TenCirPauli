/// One canonical structured term in a finite charge-sector layout.
#[derive(Clone, Debug)]
pub struct ChargeTransitionTerm {
    pub fermion_creation: Vec<u32>,
    pub fermion_annihilation: Vec<u32>,
    pub boson_blocks: Vec<(u32, u32, u32)>,
    pub qubit_codes: Vec<u8>,
    pub mapped_present: bool,
    pub mapped_codes: Vec<u8>,
    pub qudit_present: bool,
    pub qudit_triples: Vec<(u32, u32, u32)>,
    pub coefficient: Complex64,
}

type ChargeTransitionShape = (
    Vec<u32>,
    Vec<u32>,
    Vec<(u32, u32, u32)>,
    Vec<u8>,
    bool,
    Vec<u8>,
    bool,
    Vec<(u32, u32, u32)>,
);

/// Group identical coefficient-free charge schedules once at construction.
///
/// Native apply and eager construction consume the grouped descriptors, so
/// duplicate symbolic terms cannot force repeated semantic lowering in every
/// source column. Exact-zero cancellation is removed only after aggregation.
pub fn group_charge_transition_terms(
    terms: &[ChargeTransitionTerm],
) -> Result<Vec<ChargeTransitionTerm>, PauliError> {
    let mut grouped = BTreeMap::<ChargeTransitionShape, Complex64>::new();
    for term in terms {
        let shape = (
            term.fermion_creation.clone(),
            term.fermion_annihilation.clone(),
            term.boson_blocks.clone(),
            term.qubit_codes.clone(),
            term.mapped_present,
            term.mapped_codes.clone(),
            term.qudit_present,
            term.qudit_triples.clone(),
        );
        grouped
            .entry(shape)
            .and_modify(|coefficient| *coefficient += term.coefficient)
            .or_insert(term.coefficient);
    }
    grouped
        .into_iter()
        .filter_map(|(shape, coefficient)| {
            if coefficient.re == 0.0 && coefficient.im == 0.0 {
                return None;
            }
            let (
                fermion_creation,
                fermion_annihilation,
                boson_blocks,
                qubit_codes,
                mapped_present,
                mapped_codes,
                qudit_present,
                qudit_triples,
            ) = shape;
            Some(Ok(ChargeTransitionTerm {
                fermion_creation,
                fermion_annihilation,
                boson_blocks,
                qubit_codes,
                mapped_present,
                mapped_codes,
                qudit_present,
                qudit_triples,
                coefficient,
            }))
        })
        .collect()
}

/// Estimate the logical native storage for a batch of term descriptors.
///
/// Vec headers are counted together with their logical payloads; allocator
/// capacity and Python object overhead remain outside this best-effort policy.
pub fn estimate_charge_transition_terms_bytes(
    terms: &[ChargeTransitionTerm],
) -> Result<u128, PauliError> {
    terms.iter().try_fold(0_u128, |total, term| {
        let payload = (term.fermion_creation.len() as u128)
            .checked_mul(std::mem::size_of::<u32>() as u128)
            .and_then(|value| {
                value.checked_add(
                    (term.fermion_annihilation.len() as u128)
                        .checked_mul(std::mem::size_of::<u32>() as u128)?,
                )
            })
            .and_then(|value| {
                value.checked_add(
                    (term.boson_blocks.len() as u128).checked_mul(std::mem::size_of::<(
                        u32,
                        u32,
                        u32,
                    )>()
                        as u128)?,
                )
            })
            .and_then(|value| value.checked_add(term.qubit_codes.len() as u128))
            .and_then(|value| value.checked_add(term.mapped_codes.len() as u128))
            .and_then(|value| {
                value.checked_add(
                    (term.qudit_triples.len() as u128).checked_mul(std::mem::size_of::<(
                        u32,
                        u32,
                        u32,
                    )>()
                        as u128)?,
                )
            })
            .ok_or(PauliError::Overflow {
                context: "estimating charge transition descriptors",
            })?;
        total
            .checked_add(std::mem::size_of::<ChargeTransitionTerm>() as u128)
            .and_then(|value| value.checked_add(payload))
            .ok_or(PauliError::Overflow {
                context: "estimating charge transition descriptors",
            })
    })
}

/// Deterministic destination-major CSR graph produced by native eager builds.
pub struct ChargeCsrResult {
    pub dimension: usize,
    pub indptr: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
    pub estimated_bytes: u128,
    pub estimated_construction_peak_bytes: u128,
    pub schedule_count: usize,
    pub generation_nanos: u128,
    pub chunk_count: usize,
    pub sorted_entries: usize,
    pub merged_entries: usize,
    pub heap_operations: usize,
    pub final_array_allocation_entries: usize,
    pub chunk_metadata_bytes: u128,
    pub sort_nanos: u128,
    pub merge_nanos: u128,
    pub final_array_allocation_nanos: u128,
}

/// Apply a validated destination-major CSR graph without allocating
/// state-sized worker buffers. Parallel rows have deterministic per-row
/// accumulation order because each worker owns one complete output row.
pub fn apply_charge_csr_into(
    indptr: &[usize],
    columns: &[usize],
    values: &[Complex64],
    state: &[Complex64],
    output: &mut [Complex64],
    parallel: bool,
) -> Result<(), PauliError> {
    crate::csr::apply_slices(
        output.len(),
        indptr,
        columns,
        values,
        state,
        output,
        parallel,
    )
}
use super::*;
