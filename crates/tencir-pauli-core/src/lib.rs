//! Pure Rust Pauli algebra and deterministic structural utilities.

mod charge;
mod charge_sector;
mod circuit_ir;
mod csr;
mod error;
mod fermion_mask;
mod gate;
mod grouping;
mod hamiltonian;
mod majorana;
mod mapping;
mod operator;
mod pauli_action;
mod propagation;
mod random;
mod scalar;
mod sector;
mod shadows;
mod spps;
mod structured;
mod symmetry;
mod u1_circuit;
mod word;

#[cfg(test)]
mod test_allocation_counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }

    struct TestAllocator;

    fn record_allocation() {
        ACTIVE.with(|active| {
            if active.get() {
                COUNT.with(|count| count.set(count.get().saturating_add(1)));
            }
        });
    }

    unsafe impl GlobalAlloc for TestAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            // SAFETY: this test-only allocator forwards the exact request to
            // the platform allocator without changing layout or ownership.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            // SAFETY: this is the matching zeroed-allocation forwarding path.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            record_allocation();
            // SAFETY: the allocation pointer and layout come from the caller
            // under the GlobalAlloc contract and are forwarded unchanged.
            unsafe { System.realloc(pointer, layout, size) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: deallocation likewise preserves the caller's pointer
            // and layout and delegates to the platform allocator.
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    #[global_allocator]
    static TEST_ALLOCATOR: TestAllocator = TestAllocator;

    pub(crate) fn allocation_count<R>(operation: impl FnOnce() -> R) -> (R, usize) {
        ACTIVE.with(|active| {
            assert!(!active.get(), "allocation counter cannot be nested");
            active.set(true);
        });
        COUNT.with(|count| count.set(0));
        let result = operation();
        let count = COUNT.with(Cell::get);
        ACTIVE.with(|active| active.set(false));
        (result, count)
    }
}

#[cfg(test)]
pub(crate) use test_allocation_counter::allocation_count;

pub use charge::{
    apply_charge_csr_into, charge_execution_variant, compile_charge_csr_from_prepared_plan,
    compile_charge_csr_from_prepared_plan_with_chunk_capacity,
    estimate_charge_transition_terms_bytes, group_charge_transition_terms, prepare_charge_mvp_plan,
    prepare_charge_transition_plan_layout, ChargeCsrResult, ChargeLazyKernel, ChargeTransitionTerm,
    PreparedChargeMvpPlan, PreparedChargeTransitionPlanLayout,
};
pub use charge_sector::{
    build_charge_sector_plan, build_compact_charge_sector_plan, ChargeSectorPlan,
};
pub use circuit_ir::{AngleRef, CircuitGate, CircuitProgram, CIRCUIT_SCHEMA_VERSION};
pub use csr::{CsrApplyPolicy, CsrGraph, CsrIndex};
pub use error::PauliError;
pub use gate::{
    ChannelKind, Clifford1, Clifford2, GateOperation, ParameterRef, RotationAxis, ScalarRef,
};
pub use grouping::{
    compatibility_matrix, group_words, group_words_bounded, incompatibility_edges,
    GroupingAlgorithm, GroupingMode, DEFAULT_MAX_GROUPING_ENTRIES,
};
pub use hamiltonian::{BackendMvpPlan, CooMatrix, CsrMatrix, MvpPlan, MvpStrategy};
pub use majorana::{
    binary_majorana_terms, canonicalize_majorana_terms, fermion_to_majorana_terms,
    majorana_to_fermion_terms, multiply_majorana_terms, MajoranaBatch, MajoranaCanonicalResult,
};
pub use mapping::{build_mapping_plan, MappingPlan};
pub use operator::{Canonicalization, PauliOperator, PauliTerm};
pub use propagation::{
    ProductState, PropagationBatch, PropagationBatchValueAndGradient, PropagationEngine,
    PropagationResult, PropagationStats, PropagationTape, PropagationValueAndGradient,
};
pub use scalar::{hash_complex, hash_f64, Complex64};
pub use sector::{
    PackedU1Basis, U1CooMatrix, U1CsrMatrix, U1LazyMvpPlan, U1MvpPlan, U1RestrictedOperator,
    U1Sector,
};
pub use shadows::{SnapshotProtocol, Snapshots, StabilizerState};
pub use spps::{SPPSEngine, SPPSEstimate, SPPSValueEstimate};
pub use structured::{
    analyze_hybrid_charge, binary_boson_terms, binary_fermion_terms, binary_hybrid_terms,
    canonicalize_boson_terms, canonicalize_fermion_integrals, canonicalize_fermion_terms,
    canonicalize_hybrid_terms, embed_hybrid_terms, hybrid_terms_conserve_charge,
    jordan_wigner_hybrid_terms, jordan_wigner_hybrid_terms_trusted, jordan_wigner_terms,
    multiply_boson_terms, multiply_fermion_terms, multiply_hybrid_terms, structured_dense_matrix,
    structured_dense_matrix_with_fermion_positions, structured_eager_mvp_plan,
    structured_eager_mvp_plan_with_fermion_positions, structured_mvp_plan,
    structured_mvp_plan_with_fermion_positions, structured_sparse_matrix,
    structured_sparse_matrix_with_fermion_positions, BosonCanonicalResult, FermionBatch,
    FermionCanonicalResult, FermionIntegralSource, FermionMaskPhaseProfile, FermionSpinBlocks,
    FermionSpinOrdering, HybridBatch, HybridCanonicalResult, HybridLayout, HybridRawBatch,
    StructuredEagerMvpPlan, StructuredMvpPlan, StructuredOperation, StructuredSparseResult,
};
pub use symmetry::{find_z2_symmetries, CliffordOperation, Z2SymmetryAnalysis, Z2TaperingPlan};
pub use u1_circuit::U1CircuitPlan;
pub use word::{packed_word_count, PauliPhase, PauliWord};

#[cfg(test)]
mod tests;
