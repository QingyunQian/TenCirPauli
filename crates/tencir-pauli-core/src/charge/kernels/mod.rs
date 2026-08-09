/// Native compact charge execution selected once during plan construction.
pub enum ChargeLazyKernel {
    Generic,
    Fermion(FastFermionMvpPlan),
    Qubit(FastQubitMvpPlan),
    Qudit(FastAdditiveMvpPlan),
    Boson(FastAdditiveMvpPlan),
    Product(FastProductMvpPlan),
}

impl ChargeLazyKernel {
    pub fn native_variant(&self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Fermion(_) => "fermion",
            Self::Qubit(_) => "qubit",
            Self::Qudit(_) => "qudit",
            Self::Boson(_) => "boson",
            Self::Product(_) => "product",
        }
    }

    pub fn debug_counters(&self) -> Option<Vec<usize>> {
        match self {
            Self::Generic => None,
            Self::Product(plan) => plan.debug_counters().map(|counters| {
                let (
                    factor_evaluations,
                    spectator_segments,
                    destination_allocations,
                    fallbacks,
                    rankers,
                    groups,
                    insertions,
                ) = counters;
                vec![
                    factor_evaluations,
                    spectator_segments,
                    destination_allocations,
                    fallbacks,
                    rankers,
                    groups,
                    insertions,
                ]
            }),
            _ => None,
        }
    }

    pub fn estimated_bytes(&self) -> u128 {
        match self {
            Self::Generic => 0,
            Self::Fermion(plan) => plan.estimated_bytes(),
            Self::Qubit(plan) => plan.estimated_bytes(),
            Self::Qudit(plan) => plan.estimated_bytes(),
            Self::Boson(plan) => plan.estimated_bytes(),
            Self::Product(plan) => plan.estimated_bytes(),
        }
    }

    pub fn apply_specialized_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        max_bytes: u128,
    ) -> Option<Result<(), PauliError>> {
        match self {
            Self::Generic => None,
            Self::Fermion(plan) => Some(plan.apply_into(state, output, max_bytes)),
            Self::Qubit(plan) => Some(plan.apply_into(state, output, max_bytes)),
            Self::Qudit(plan) => Some(plan.apply_into(state, output, max_bytes)),
            Self::Boson(plan) => Some(plan.apply_into(state, output, max_bytes)),
            Self::Product(plan) => Some(plan.apply_into(state, output, max_bytes)),
        }
    }
}
use super::*;
