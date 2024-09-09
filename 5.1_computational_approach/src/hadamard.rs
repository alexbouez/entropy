#![allow(non_snake_case)]

//! Entropy Gathering - Hadamard
//!
//! DEPRECATED. 
//! Structure for computing the Hadamard transform of a Density. 
//! Use [`Density`](crate::density::Density) in-place instead. 

use std::io::Error;

use crate::UnsignedVec;
use crate::Distribution;
use crate::density::Density;

#[derive(Clone, Debug)]
/// DEPRECATED. 
/// Wrapper structure for the hadamard transform of a distribution. 
/// Contains: length of distribution range N, hadamard transform values F.
pub struct Hadamard<U>
{
    _N: U,
    F: Vec<f32>
}

impl<U> Hadamard<U>
    where U: From<u8> + PartialEq + Clone + Sync, 
        Density<U>: UnsignedVec<U>, f32:From<U>
{
    /// DEPRECATED. 
    /// Applies Fast Walsh-Hadamard Transform on `Density<U>` to initialize `Hadamard<U>`.
    pub fn fwht(&mut self, real: Density<U>) {
        assert!(self.F.len() % 2 == 0);
        assert!(real.get().len() % 2 == 0);

        self.F = real.get().clone();
        let mut h: usize = 1;
        while h < self.F.len() {

            let mut i:usize = 0;
            while i < self.F.len() {

                for j in i..(i + h) {
                    let (x,y) = (self.F[j], self.F[j + h]);
                    self.F[j] = x + y;
                    self.F[j + h] = x - y;
                }
 
                i += h*2;
            }

            h *= 2;
        }
    }
}

impl UnsignedVec<u16> for Hadamard<u16>
{
    fn new(N: u16) -> Result<Self, Error> {
        let n = usize::from(N) + 1;

        let mut F = vec![0.0_f32; n];
        F[0] = 1.0_f32;
        
        Ok(Hadamard{
            _N: N,
            F: F
        })  
    }

    fn initialize(&mut self, distrib: &dyn Distribution<u16>, _second_order: bool) {
        self.F.iter_mut().enumerate().for_each(|(i, y)| { 
            let a = distrib.eval(i.try_into().unwrap(), false);
            *y = if a > 0.0 { a } else { 0.0 };
        });
    }
}
