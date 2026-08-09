//! Stable facade for structured operator algebra and execution.
//!
//! Implementation is being split into focused private modules under
//! `structured/`; this facade preserves the established core import path.

#[path = "structured/implementation.rs"]
mod implementation;

pub use implementation::*;
