//! Stable registration facade for structured native bindings.
//!
//! The private implementation is split under `structured/` while this module
//! preserves registration and sibling-module imports.

#[path = "structured/implementation.rs"]
mod implementation;

pub(crate) use implementation::*;
