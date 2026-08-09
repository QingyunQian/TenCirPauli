#[cfg(test)]
mod tests {
    use super::super::{
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
    fn contiguous_fermion_apply_uses_no_per_call_workspace() {
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
        assert_eq!(single_allocations, 0);
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
            super::super::compile_mvp_operations(&[dimension], &[vec![operation]], &[], &[1])
                .unwrap();
        let mut destination = dimension - 1;
        let mut amplitude = Complex64::new(1.0, 0.0);
        assert!(super::super::apply_compiled_structured_operation(
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
