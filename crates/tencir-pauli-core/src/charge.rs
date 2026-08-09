//! Stable facade for additive-charge compilation and execution.
//!
//! The implementation is split under `charge/` while this module preserves
//! established core imports and public types.

#[path = "charge/implementation.rs"]
mod implementation;

pub use implementation::*;
