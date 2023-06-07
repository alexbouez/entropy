#![allow(non_snake_case)]

//! Entropy Gathering - Distributions - Cauchy
//!
//! Structure for creating a new Cauchy distribution with scale parameter 'gamma'
//! and location parameter 'x0'. Can be called using the general purpose 'eval' function.
//!
//! Works with unsigned types u8, u16, u32.


use std::{cmp::PartialOrd, marker::Copy};
use std::io::Error;
use std::f32::consts::PI;
use num::{cast::AsPrimitive, Num};

use crate::distributions::Distribution;

/// Wrapper structure for a Cauchy distribution PDF.
/// Contains: scale gamma, location x0, length of distribution range N, function f.
pub struct Cauchy<U>
{
    gamma: f32,
    x0: f32,
    N: U,
    f: fn(f32, f32, f32) -> f32
}

impl<U> Cauchy<U>
        where U: From<u8> + PartialOrd + Copy + Num + AsPrimitive<f32>
{
    /// Returns a new [`Cauchy`](crate::distributions::cauchy::Cauchy) structure.
    pub fn new(gamma: f32, x0: f32, N: U) -> Result<Self, Error> {
        assert!(gamma > 0.0);

        /// Cauchy PDF function. 
        /// Parameters: scale gamma, location x0, value x.
        fn cauchy(g: f32, x0: f32, x: f32) -> f32 
        {
            let r = 1.0 / (PI * g * (1.0 + ((x - x0)/g).powi(2)));
            if r.is_finite() && r > 0.0 {
                r
            } else {
                0.0
            }
        }

        /// DEPRECATED.
        /// Step CDF function. 
        /// Parameters: scale gamma, location x0, value x.
        fn _cauchy_cdf(g: f32, x0: f32, x: f32) -> f32 {
            let r = (1.0 / PI) * ((x - x0) / g).atan() + 0.5;
            if r.is_finite() && r >= 0.000001 && r <= 0.999999 {
                r
            } else if r < 0.000001 {
                0.0
            } else {
                1.0
            }
        }

        Ok(Cauchy{
            gamma: gamma, 
            x0: x0, 
            N: N, 
            f: cauchy
        })
    }

    /// Getter for the Cauchy distribution parameters (gamma, x0, N).
    pub fn params(&self) -> (f32, f32, U) {
        (self.gamma, self.x0, self.N)
    }
}

impl<U> Distribution<U> for Cauchy<U> 
        where U: From<u8> + PartialOrd + Num + AsPrimitive<f32>
{
    /// Returns the evaluation of x through the step function in Self. 
    /// If second_order, the distribution is centered around 0 and will wrap-around. 
    fn eval(&self, x: U, second_order: bool) -> f32 {
        let x_as_float = x.as_();
        let value = if second_order && x_as_float >= (self.N.as_() / 2.0) {
            x_as_float - self.N.as_()
        } else {
            x_as_float
        };
        let prob = (self.f)(self.gamma, self.x0, value);
        prob
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn parameter_types() {
        let _cauchy = Cauchy::new(1.0, 1.0, 8).unwrap();
        let _cauchy = Cauchy::new(1.0, 1.0, 8 as u8).unwrap();
        let _cauchy = Cauchy::new(1.0, 1.0, 8 as u16).unwrap();
        let _cauchy = Cauchy::new(1.0, 1.0, 8 as u32).unwrap();
    }

    #[test]
    fn bounds() {
        let mut _cauchy = Cauchy::new(1.0, 1.0, 8).unwrap();
        _cauchy = Cauchy::new(1.0, 0.0, 8).unwrap();
        _cauchy = Cauchy::new(1.0, 1.0, 0).unwrap();
        _cauchy = Cauchy::new(0.1, 1.0, 8).unwrap();
        _cauchy = Cauchy::new(0.1, 0.0, 0).unwrap();
    }

    #[test]
    fn fetching_parameters() {
        let cauchy = Cauchy::new(1.0, 2.0, 3).unwrap();
        assert!(cauchy.params() == (1.0, 2.0, 3));
    }

    #[test]
    fn evaluation() {
        let cauchy = Cauchy::new(1.0, 1.0, 3).unwrap();
        assert!(cauchy.eval(0, false) == 0.6065306597126334);
        assert!(cauchy.eval(1, false) == 1.0);
        assert!(cauchy.eval(2, false) == 0.6065306597126334);
    }

    #[test]
    #[should_panic]
    fn invalid_bound_sigma_0() {
        let _cauchy = Cauchy::new(0.0, 2.0, 3).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_sigma_neg() {
        let _cauchy = Cauchy::new(-1.0, 2.0, 3).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_mu() {
        let _cauchy = Cauchy::new(1.0, -2.0, 3).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_N() {
        let _cauchy = Cauchy::new(1.0, 2.0, -3).unwrap();
    }

    #[test]
    #[should_panic]
    fn evaluation_beyond_min_bound() {
        let cauchy = Cauchy::new(1.0, 2.0, 3).unwrap();
        cauchy.eval(-1, false);
    }
    #[test]
    #[should_panic]
    fn evaluation_beyond_max_bound() {
        let cauchy = Cauchy::new(1.0, 2.0, 3.0).unwrap();
        cauchy.eval(3.1, false);
    }
}