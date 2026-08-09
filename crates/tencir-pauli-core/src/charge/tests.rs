#[cfg(test)]
mod tests {
    use super::super::ChargeDestinationAccumulator;
    use crate::{allocation_count, Complex64};

    #[test]
    fn wide_destination_accumulator_reuses_fixed_width_storage() {
        let mut accumulator = ChargeDestinationAccumulator::new(None, 3, 4).unwrap();
        let first_capacity = accumulator.wide.as_ref().unwrap().values.capacity();
        accumulator
            .insert(&[1, 0, 0, 0], Complex64::new(1.0, 0.0))
            .unwrap();
        accumulator
            .insert(&[1, 0, 0, 0], Complex64::new(-1.0, 0.0))
            .unwrap();
        accumulator
            .insert(&[0, 1, 0, 0], Complex64::new(2.0, 0.0))
            .unwrap();
        let mut destination = [0_u64; 4];
        let mut entries = Vec::new();
        accumulator
            .drain(
                &mut destination,
                &[4, 4, 4, 4],
                |destination: &[u64], value| {
                    entries.push((destination.to_vec(), value));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(first_capacity, 12);
        assert_eq!(
            accumulator.wide.as_ref().unwrap().values.capacity(),
            first_capacity
        );
        assert_eq!(
            entries,
            vec![
                (vec![1, 0, 0, 0], Complex64::new(0.0, 0.0)),
                (vec![0, 1, 0, 0], Complex64::new(2.0, 0.0))
            ]
        );
    }

    #[test]
    fn scalar_and_wide_accumulator_cycles_allocate_nothing() {
        let mut scalar = ChargeDestinationAccumulator::new(Some(&[1, 2]), 3, 2).unwrap();
        let mut wide = ChargeDestinationAccumulator::new(None, 3, 4).unwrap();
        let mut scalar_destination = [0_u64; 2];
        let mut wide_destination = [0_u64; 4];
        let (_, allocations) = allocation_count(|| {
            for _ in 0..64 {
                scalar.clear();
                scalar.insert(&[1, 0], Complex64::new(1.0, 0.0)).unwrap();
                scalar.insert(&[0, 1], Complex64::new(-1.0, 0.0)).unwrap();
                scalar
                    .drain(&mut scalar_destination, &[2, 2], |_, _| Ok(()))
                    .unwrap();

                wide.clear();
                wide.insert(&[1, 0, 0, 0], Complex64::new(1.0, 0.0))
                    .unwrap();
                wide.insert(&[0, 1, 0, 0], Complex64::new(-1.0, 0.0))
                    .unwrap();
                wide.drain(&mut wide_destination, &[2, 2, 2, 2], |_, _| Ok(()))
                    .unwrap();
            }
        });
        assert_eq!(allocations, 0);
    }
}
