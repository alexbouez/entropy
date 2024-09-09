#![allow(non_snake_case)]

//! Entropy Gathering - Distributions - Step
//!
//! Structure for creating a new Step distribution with step start value 'a' and length 'l'. 
//! Can be called using the general purpose 'eval' function.
//!
//! Works with unsigned types u8, u16, u32, u64, usize.

use std::{ops::Add, cmp::PartialOrd, marker::Copy};
use std::io::Error;

use crate::distributions::Distribution;

/// Wrapper structure for a step distribution PDF.
/// Contains: beginning of step a, end of step b, length of distribution range N, function f.
pub struct Step<U>
{
    a: U,
    b: U,
    N: U,
    f: fn(U, U, U) -> f32
}

impl<U> Step<U>
        where U: From<u8> + Add<Output = U> + PartialOrd + Copy
{
    /// Returns a new [`Step`](crate::distributions::step::Step) structure.
    pub fn new(a: U, b: U, N: U) -> Result<Self, Error> {
        assert!(a <= b && b <= N); // a and b must define a range within [0,N]
        assert!(a >= 0_u8.into() && N > 0_u8.into()); // bounds are unsigned

        /// Step PDF function. 
        /// Parameters: beginning of step a, end of step b, value x.
        fn step<T>(a: T, b: T, x: T) -> f32 
                where T: Add<Output = T> + PartialOrd + Copy
        {
            if a <= x && x <= b {
                1.0
            } else {
                0.0
            }
        }

        Ok(Step{
            a: a, 
            b: b, 
            N: N, 
            f: step
        })
    }

    /// Getter for the step function parameters (a, b, N).
    pub fn params(&self) -> (U, U, U) {
        (self.a, self.b, self.N)
    }
}

impl<U> Distribution<U> for Step<U> 
        where U: From<u8> + PartialOrd + Copy
{
    /// Returns the evaluation of x through the step function in Self. 
    /// Option second_order is not implemented. 
    fn eval(&self, x: U, _second_order: bool) -> f32 {
        assert!(x >= 0_u8.into() && x <= self.N);
        (self.f)(self.a, self.b, x)
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn parameter_types() {
        let _step = Step::new(1, 1, 8).unwrap();
        let _step = Step::new(1.0, 1.0, 8.0).unwrap();
        let _step = Step::new(1.0 as u8, 1.0 as u8, 8.0 as u8).unwrap();
        let _step = Step::new(1.0 as u16, 1.0 as u16, 8.0 as u16).unwrap();
        let _step = Step::new(1.0 as u32, 1.0 as u32, 8.0 as u32).unwrap();
        let _step = Step::new(1.0 as u64, 1.0 as u64, 8.0 as u64).unwrap();
        let _step = Step::new(1.0 as usize, 1.0 as usize, 8.0 as usize).unwrap();
    }

    #[test]
    fn bounds() {
        let mut _step = Step::new(0, 0, 8).unwrap();
        _step = Step::new(7, 8, 8).unwrap();
        _step = Step::new(1, 1, 8).unwrap();
        _step = Step::new(0, 8, 8).unwrap();
        _step = Step::new(8, 8, 8).unwrap();
    }

    #[test]
    fn fetching_parameters() {
        let step = Step::new(1, 2, 3).unwrap();
        assert!(step.params() == (1, 2, 3));
    }

    #[test]
    fn evaluation() {
        let step = Step::new(2, 4, 8).unwrap();
        assert!(step.eval(0, false) == 0.0);
        assert!(step.eval(6, false) == 0.0 && step.eval(8, false) == 0.0);
        assert!(step.eval(2, false) == 1.0 && step.eval(4, false) == 1.0);
    }

    #[test]
    #[should_panic]
    fn bound_overflow_a() {
        let _step = Step::new(10, 1, 9).unwrap();
    }
    #[test]
    #[should_panic]
    fn bound_overflow_b() {
        let _step = Step::new(8, 10, 9).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_bound_a() {
        let _step = Step::new(-1, 4, 8).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_b() {
        let _step = Step::new(2, 1, 8).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_N() {
        let _step = Step::new(1, 4, 0).unwrap();
    }

    #[test]
    #[should_panic]
    fn evaluation_beyond_min_bound() {
        let step = Step::new(2, 4, 8).unwrap();
        step.eval(-1, false);
    }
    #[test]
    #[should_panic]
    fn evaluation_beyond_max_bound() {
        let step = Step::new(2, 4, 8).unwrap();
        step.eval(9, false);
    }
}