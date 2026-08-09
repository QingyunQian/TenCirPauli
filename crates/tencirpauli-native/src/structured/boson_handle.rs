#[path = "boson_handle/methods.rs"]
mod methods;
#[path = "boson_handle/primary.rs"]
mod primary;

use super::*;
pub(crate) use primary::NativeBosonOperatorHandle;
pub(in crate::structured::implementation) use primary::{
    boson_adjoint, empty_hybrid_result, hybrid_to_pauli, merge_boson_handles,
    merge_fermion_handles, merge_hybrid_handles, same_coefficient_maps, same_hybrid_coefficients,
    same_structured_coefficients, scale_boson_handle, scale_fermion_handle, scale_hybrid_handle,
    HybridNativeKey,
};
