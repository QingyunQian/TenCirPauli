/// Native compact charge execution selected once during plan construction.
pub enum ChargeLazyKernel {
    Generic,
    /// The sector has a compact native factor proof. Shared schedules, direct
    /// public indexing, and component-local executors remain authoritative;
    /// this marker exposes the representation without resurrecting Python-side
    /// classification.
    Factorized,
}

impl ChargeLazyKernel {
    pub fn native_variant(&self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Factorized => "factorized",
        }
    }

    pub fn debug_counters(&self) -> Option<Vec<usize>> {
        match self {
            Self::Generic => None,
            Self::Factorized => None,
        }
    }

    pub fn estimated_bytes(&self) -> u128 {
        match self {
            Self::Generic => 0,
            Self::Factorized => 0,
        }
    }
}
