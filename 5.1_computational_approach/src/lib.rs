#![warn(missing_docs)]
#![allow(non_snake_case)]
#![feature(rustdoc_missing_doc_code_examples)]

//! Entropy Gathering 

/// Module implementing distribution functions (e.g., step, gauss, cauchy).
pub mod distributions;
/// Module implementing basic unsigned integer operations for u16 and u32 
/// (e.g., shift right/left, rotate right/left, not).
pub mod functions;
/// Module implementing the [`Density`](crate::density::Density) structure. 
pub mod density;
/// Module implementing the [`Fourier`](crate::fourier::Fourier) structure. 
pub mod fourier;
/// DEPRECATED. Module implementing the [`Fourier`](crate::fourier::Fourier) structure. 
pub mod hadamard;

use std::io::Error;
use crate::distributions::Distribution;

pub trait UnsignedVec<U> 
{
    //! Allows acces to general vectors of unsigned integers
    //! (e.g., `u16`, `u32`, `u64`).

    /// General declaration function. 
    fn new(N: U) -> Result<Self, Error> where Self: Sized;

    /// General initialization function.
    fn initialize(&mut self, distrib: &dyn Distribution<U>, second_order: bool);
}

pub trait ComplexVec<U> 
{
    //! Allows acces to general vectors of complex values (i.e., `Complex<f32>`).

    /// General initialization function.
    fn new(N: U) -> Result<Self, Error> where Self: Sized;
}