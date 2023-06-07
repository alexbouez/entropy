#![allow(non_snake_case)]
//! Entropy Gathering - Fourier
//!
//! Module for computing and utilizing the Fourier transform of [`Density`](crate::density::Density).

use std::io::Error;

use crate::ComplexVec;
use crate::UnsignedVec;
use crate::density::Density;

use std::sync::Arc;
use rustfft::{Fft, FftPlanner, FftDirection, num_complex::Complex};

#[derive(Clone, Debug)]
/// Wrapper structure for the Fourier transform of a [`Density`](crate::density::Density) distribution.
/// Contains: length of distribution range N, vector of `Complex<f32>` values F.
pub struct Fourier<U>
{
    N: U,
    F: Vec<Complex<f32>>
}

impl<U> Fourier<U>
    where U: From<u8> + PartialEq + Clone + Sync, 
        Density<U>: UnsignedVec<U>
{
    /// Computes the point-to-point multiplication of Self and another 
    /// [`Fourier`](crate::fourier::Fourier). Result is stored in Self. 
    pub fn convolve(&mut self, b: &Self) {
        self.F.iter_mut().enumerate().for_each(|(i, x)| { 
            *x *= b.F[i]; 
        });
    }

    /// Getter for `Complex<f32>` vec F of Self. 
    pub fn get(&self) -> Vec<Complex<f32>> {
        self.F.clone()
    }

    /// Generates a Fast-Fourier Transform planner. 
    /// Planner is reusable for same size discrete distributions.
    pub fn get_fft_planner(&self) -> Result<Arc<dyn Fft<f32>>, Error> {
        let n = self.F.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(n, FftDirection::Forward);
        Ok(fft)
    }

    /// Initializes Self with the Fast-Fourier Transform of a 
    /// [`Density`](crate::density::Density). Requires pre-computing the FFT planner. 
    pub fn fft(&mut self, real: &Density<U>, planner: &Arc<dyn Fft<f32>>) {
        assert!(self.N == real.size());
        
        let n = self.F.len();
        let mut buffer = vec![Complex::new(0.0, 0.0); n];
        buffer.iter_mut().enumerate().for_each(|(i, p)| {
            *p = Complex::new(real.get_value(i), 0.0);
        });

        planner.process(&mut buffer);
        self.F = buffer;
    }

    /// Generates an inverse Fast-Fourier Transform planner. 
    /// Planner is reusable for same size discrete distributions.
    pub fn get_fft_planner_inverse(&self) -> Result<Arc<dyn Fft<f32>>, Error> {
        let n = self.F.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(n, FftDirection::Inverse);
        Ok(fft)
    }

    /// Computes the Fast-Fourier Transform inverse of Self. 
    /// Requires pre-computing the inverse FFT planner. 
    pub fn fft_inverse(&self, planner: &Arc<dyn Fft<f32>>) -> Density<U> {
        let mut buffer = self.F.clone();
        planner.process(&mut buffer);

        let mut result = Density::<U>::new(self.N.clone()).unwrap();
        let mut result_vec = result.get();

        result_vec.iter_mut().enumerate().for_each(|(i, p)| {
            *p = buffer[i].re;
        });
        result.set(result_vec);
        result
    }
}

impl ComplexVec<u16> for Fourier<u16>
{
    fn new(N: u16) -> Result<Self, Error> {
        let n = (N as usize) + 1;
        assert!(n > 0_usize);

        let F = vec![Complex::new(1.0, 0.0); n];
        
        let new = Fourier{
            N: N,
            F: F
        };

        Ok(new)
    }
}

impl ComplexVec<u32> for Fourier<u32>
{
    fn new(N: u32) -> Result<Self, Error> {
        let n = (N as usize) + 1;
        assert!(n > 0_usize);

        let F = vec![Complex::new(1.0, 0.0); n];
        
        let new = Fourier{
            N: N,
            F: F
        };

        Ok(new)
    }
}