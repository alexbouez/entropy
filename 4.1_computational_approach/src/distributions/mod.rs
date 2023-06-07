#![allow(non_snake_case)]

//! Entropy Gathering - Distributions
//!
//! This module gives access to a set of common distributions (step, gaussian, cauchy).
//! These are accessible through the Distribution trait,
//! and can all be called through the public 'eval' function.

/// Module implementing the step distribution PDF. 
pub mod step;
/// Module implementing the Gauss (or normal) distribution PDF. 
pub mod gauss;
/// Module implementing the Cauchy distribution PDF. 
pub mod cauchy;

pub trait Distribution<U>
{ 
    //! Allows acces to generate purpose distribution evaluation function. 

    /// Returns the evaluation of x through the distribution. 
    /// If second_order, the distribution is centered around 0 and will wrap-around 
    /// (WIP for step, gauss). 
    fn eval(&self, x: U, second_order: bool) -> f32;
}