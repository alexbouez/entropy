#![allow(non_snake_case)]

//! Entropy Gathering - Distributions - Gaussian
//!
//! Structure for creating a new Gaussian distribution with standard deviation 'sigma'
//! and mean value 'mu'. Can be called using the general purpose 'eval' function.
//!
//! Works with unsigned types u8, u16, u32.

use std::{cmp::PartialOrd, marker::Copy};
use num::{cast::AsPrimitive, Num};
use std::io::Error;

use crate::distributions::Distribution;

/// Wrapper structure for a Cauchy distribution PDF.
/// Contains: standard deviation sigma, mean mu, constant s, 
/// length of distribution range N, function f.
pub struct Gauss<U>
{
    sigma: f32,
    mu: f32,
    s: f32,
    N: U,
    f: fn(f32, f32, U) -> f32
}

impl<U> Gauss<U>
        where U: From<u8> + PartialOrd + Copy + Num + AsPrimitive<f32>
{
    /// Returns a new [`Gauss`](crate::distributions::gauss::Gauss) structure.
    pub fn new(sigma: f32, mu: f32, N: U) -> Result<Self, Error> {
        assert!(sigma > 0.0 && mu >= 0.0 && N >= 0_u8.into()); 
        let s: f32 = -1.0 / (2.0 * sigma.powf(2.0));

        /// Gauss distribution PDF function. 
        /// Parameters: scale gamma, location x0, value x.
        fn gauss<T>(s: f32, mu: f32, x: T) -> f32 
                where T: Num + AsPrimitive<f32>
        {
            let r = (s * (x.as_() - mu).powf(2.0)).exp();
            if 0.0 < r && r <= 1.0 {r}
            else {0.0}
        }

        Ok(Gauss{
            sigma: sigma, 
            mu: mu, 
            s: s,
            N: N, 
            f: gauss
        })
    }

    /// Getter for the Gauss distribution parameters (sigma, mu, N).
    pub fn params(&self) -> (f32, f32, U) {
        (self.sigma, self.mu, self.N)
    }
}

impl<U> Distribution<U> for Gauss<U> 
        where U: From<u8> + PartialOrd
{
    /// Returns the evaluation of x through the step function in Self. 
    /// Option second_order is not implemented. 
    fn eval(&self, x: U, _second_order: bool) -> f32 {
        assert!(x >= 0_u8.into() && x <= self.N);
        (self.f)(self.s, self.mu, x)
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn parameter_types() {
        let _gauss = Gauss::new(1.0, 1.0, 8).unwrap();
        let _gauss = Gauss::new(1.0, 1.0, 8 as u8).unwrap();
        let _gauss = Gauss::new(1.0, 1.0, 8 as u16).unwrap();
        let _gauss = Gauss::new(1.0, 1.0, 8 as u32).unwrap();
    }

    #[test]
    fn bounds() {
        let mut _gauss = Gauss::new(1.0, 1.0, 8).unwrap();
        _gauss = Gauss::new(1.0, 0.0, 8).unwrap();
        _gauss = Gauss::new(1.0, 1.0, 0).unwrap();
        _gauss = Gauss::new(0.1, 1.0, 8).unwrap();
        _gauss = Gauss::new(0.1, 0.0, 0).unwrap();
    }

    #[test]
    fn fetching_parameters() {
        let gauss = Gauss::new(1.0, 2.0, 3).unwrap();
        assert!(gauss.params() == (1.0, 2.0, 3));
    }

    #[test]
    fn evaluation() {
        let gauss = Gauss::new(1.0, 1.0, 3).unwrap();
        assert!(gauss.eval(0, false) == 0.6065306597126334);
        assert!(gauss.eval(1, false) == 1.0);
        assert!(gauss.eval(2, false) == 0.6065306597126334);
    }

    #[test]
    #[should_panic]
    fn invalid_bound_sigma_0() {
        let _gauss = Gauss::new(0.0, 2.0, 3).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_sigma_neg() {
        let _gauss = Gauss::new(-1.0, 2.0, 3).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_mu() {
        let _gauss = Gauss::new(1.0, -2.0, 3).unwrap();
    }
    #[test]
    #[should_panic]
    fn invalid_bound_N() {
        let _gauss = Gauss::new(1.0, 2.0, -3).unwrap();
    }

    #[test]
    #[should_panic]
    fn evaluation_beyond_min_bound() {
        let gauss = Gauss::new(1.0, 2.0, 3).unwrap();
        gauss.eval(-1, false);
    }
    #[test]
    #[should_panic]
    fn evaluation_beyond_max_bound() {
        let gauss = Gauss::new(1.0, 2.0, 3.0).unwrap();
        gauss.eval(3.1, false);
    }
}