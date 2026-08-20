//! Deterministic counter-based random primitives shared by stochastic paths.

const RANDOM_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

/// The historical SPPS counter mapping.  Keep the arithmetic and arguments
/// unchanged so moving it out of `spps.rs` does not change seeded results.
pub(crate) fn counter_u64(
    seed: u64,
    mode: u64,
    first: u64,
    second: u64,
    third: u64,
    fourth: u64,
) -> u64 {
    let mut value = seed
        ^ mode.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ first.wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ second.wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ third.wrapping_mul(0xd2b7_4407_b1ce_6e93)
        ^ fourth.wrapping_mul(0x632b_be59_bd9b_4e01);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn counter_random(
    seed: u64,
    mode: u64,
    term: usize,
    replicate: usize,
    sample: usize,
    gate: usize,
) -> f64 {
    ((counter_u64(
        seed,
        mode,
        term as u64,
        replicate as u64,
        sample as u64,
        gate as u64,
    ) >> 11) as f64)
        * RANDOM_SCALE
}

#[derive(Clone, Debug)]
pub(crate) struct RandomStream {
    seed: u64,
    domain: u64,
    first: u64,
    second: u64,
    draw: u64,
}

impl RandomStream {
    pub(crate) fn new(seed: u64, domain: u64, first: usize, second: usize) -> Self {
        Self {
            seed,
            domain,
            first: first as u64,
            second: second as u64,
            draw: 0,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let value = counter_u64(
            self.seed,
            self.domain,
            self.first,
            self.second,
            self.draw,
            0,
        );
        self.draw = self.draw.wrapping_add(1);
        value
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * RANDOM_SCALE
    }

    pub(crate) fn bounded(&mut self, upper: usize) -> Result<usize, crate::error::PauliError> {
        if upper == 0 {
            return Err(crate::error::PauliError::InvalidIndex {
                context: "bounded random upper bound must be positive",
            });
        }
        let upper_u64 = u64::try_from(upper).map_err(|_| crate::error::PauliError::Overflow {
            context: "converting bounded random upper bound",
        })?;
        let threshold = upper_u64.wrapping_neg() % upper_u64;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok((value % upper_u64) as usize);
            }
        }
    }
}
