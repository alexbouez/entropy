#![allow(non_snake_case)]

//! Entropy Gathering - Density
//!
//! Module for creating and using a discrete probability [`Density`](crate::density::Density)
//! over range \[1,N\] with granularity N. 
//! Can be initialized using a [`Distribution`](crate::distributions::Distribution).
//!
//! Works with unsigned types `u16`, `u32`.

extern crate csv;

use csv::Writer;
use std::io::Error;
use std::marker::Sync;
use std::cmp::PartialEq;
use std::ops::{Add, Mul};
use num::{cast::AsPrimitive, Num};

use plotters::prelude::*;

use crate::ComplexVec;
use crate::UnsignedVec;
use crate::Distribution;
use crate::fourier::Fourier;

use std::sync::Arc;
use rustfft::Fft;

#[derive(Clone, Debug)]
/// Wrapper structure for a discrete probability distribution over \[1,N\].
/// Contains: length of distribution range N, vector of probabilities Y.
pub struct Density<U>
    where U: Clone
{
    N: U,
    Y: Vec<f32>
}

impl<U> Add<Density<U>> for Density<U>
    where U: From<u8> + PartialEq + Clone + Sync, Density<U>: UnsignedVec<U>
{
    type Output = Density<U>;
    fn add(self, other: Density<U>) -> Self::Output {
        assert!(self.N == other.N, "Cannot add two densities with different N values!");

        let mut result_vec = Vec::new();
        for i in 0..self.Y.len() {
            result_vec.push(self.Y[i] + other.Y[i]); 
        }

        Density { N: self.N, Y: result_vec }
    } 
}

impl<U> Mul<Density<U>> for Density<U>
    where U: From<u8> + PartialEq + Clone + Sync, Density<U>: UnsignedVec<U>
{
    type Output = Density<U>;
    fn mul(self, other: Density<U>) -> Self::Output {
        assert!(self.N == other.N, "Cannot multiply two densities with different N values!");

        let mut result_vec = Vec::new();
        for i in 0..self.Y.len() {
            result_vec.push(self.Y[i] * other.Y[i]); 
        }

        Density { N: self.N, Y: result_vec }
    }
}

impl<U> Mul<f32> for Density<U>
    where U: From<u8> + PartialEq + Clone + Sync, Density<U>: UnsignedVec<U>
{
    type Output = Density<U>;
    fn mul(self, other: f32) -> Self::Output {

        let mut result_vec = Vec::new();
        for i in 0..self.Y.len() {
            result_vec.push(self.Y[i] * other); 
        }

        Density { N: self.N, Y: result_vec }
    }
}

/// Implementation of statistical functions for [`Density`](Density).
impl<U> Density<U>
        where U: From<u8> + PartialEq + Clone + Sync, Density<U>: UnsignedVec<U>
{
    /// Returns the Shannon entropy of Self as `f32`. 
    /// Requires normalizing the distribution for accurate results. 
    /// Elements are grouped into sqrt(N) bins, values are averaged within bins to limit noise.
     pub fn entropy(&self) -> f32 {
        let num_bins = (self.Y.len() as f32).sqrt().ceil() as usize;
        let binned_probabilities = Self::bin_probabilities(&self.Y, num_bins);
    
        let mut s: f32 = 0.0;
    
        for p in &binned_probabilities {
            if *p > 100.0 * f32::MIN_POSITIVE {s -= *p * (*p).log2();}
        }
    
        s
    }

    /// Returns the min-entropy of Self as `f32`. 
    /// Requires normalizing the distribution for accurate results. 
    /// Elements are grouped into sqrt(N) bins, values are averaged within bins to limit noise.
    pub fn min_entropy(&self) -> f32 {
        let num_bins = (self.Y.len() as f32).sqrt().ceil() as usize;
        let binned_probabilities = Self::bin_probabilities(&self.Y, num_bins);
        let s = binned_probabilities.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)).log2();
        -s
    }

    /// Groups elements into bins, averages probabilities within bins and returns 1 value per bin. 
    fn bin_probabilities(probabilities: &Vec<f32>, num_bins: usize) -> Vec<f32> {
        let step = probabilities.len() / num_bins;
        let mut binned_probabilities = vec![0.0_f32; num_bins];
        for i in 0..num_bins {
            let start = i * step;
            let end = if i == num_bins - 1 { probabilities.len() } else { start + step };
            binned_probabilities[i] = probabilities[start..end].iter().sum();
        }
        binned_probabilities
    }

    /// Returns the KL divergence of Self to a uniform distribution.
    /// Requires normalizing the distribution.
    pub fn kl_divergence(&self) -> f32 {
        let uniform_probability = 1.0 / self.Y.len() as f32;
        let mut kl_div: f32 = 0.0;
        for p in &self.Y {
            if *p > f32::MIN_POSITIVE {
                kl_div += *p * (*p).log2() - *p * uniform_probability.log2();
            }
        }
        kl_div
    }
    
    /// Returns the statistical distance of Self to a uniform distribution.
    /// Requires normalizing the distribution.
    pub fn statistical_distance(&self) -> f32 {
        let uniform_probability = 1.0 / self.Y.len() as f32;
        let mut total_variation_distance: f32 = 0.0;
        for p in &self.Y {
            total_variation_distance += (p - uniform_probability).abs();
        }
        0.5 * total_variation_distance
    }

    /// Returns the mean value of Self as `f32`.
    pub fn mean(&self) -> f32 {
        let count: usize = self.Y.len();
        let mut s: f32 = 0.0;
        for p in &self.Y {
            s += *p / count as f32;
        }
        s
    }

    /// Returns the standard deviation value of Self as `f32`.
    pub fn std_div(&self) -> f32 {
        let count: usize = self.Y.len();
        let mut s: f32 = 0.0;
        let m = self.mean();
        for p in &self.Y {
            s += (*p - m).powi(2) / count as f32;
        }
        s
    }

    /// Returns a normalized distribution of Self.
    /// Computing the sum is based on the Kahan sum to limit float errors. 
    pub fn normalize(&self) -> Self {
        let mut res = self.clone();
        let S: f32 = Self::kahan_sum(&self.Y);
        if S == 0.0 {
            res.Y[0] = 1.0;
        } else if S != 1.0 {
            res.Y.iter_mut().for_each(| p |
                *p /= S
            );
        }
        res
    }

    /// Computes the Kahan sum of a vector of 'f32'.
    pub fn kahan_sum(values: &[f32]) -> f32 {
        let mut sum = 0.0;
        let mut c = 0.0;
        for &x in values {
            let y = x - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }

    /// Returns a uniform distribution of the same size as Self.
    pub fn uniform(&self) -> Self {
        let mut res = self.clone();
        res.Y.iter_mut().for_each(| p |
            *p = 1.0_f32/(self.Y.len() as f32)
        );// /(self.Y.len() as f32)
        res
    }

    /// Applies a surjective function to the indexing of the values of Self. 
    pub fn apply(&self, distrib: &impl Fn(usize) -> usize) -> Self {
        let N = self.Y.len();
        let mut res = self.clone();
        let mut buf = vec![0.0_f32; N];

        self.Y.iter().enumerate().for_each(|(i, _p)| {
            buf[distrib(i)] += self.Y[i];
        });

        res.Y = buf;
        res
    }

    /// Returns the convolution of Self with another [`Density`](crate::density::Density). 
    /// This algorithm is naive, complexity is O(N^2). 
    pub fn convolve(&self, b: &Self) -> Self {
        assert!(self.N == b.N);
        assert!(self.Y.len() == b.Y.len());

        let mut reverse = self.Y.clone();
        reverse.reverse();
        
        let n = self.Y.len(); 
        let mut conv = vec![0.0_f32; n];
        conv.iter_mut().enumerate().for_each(|(i, p)| {
            let mut S = reverse.clone();
            S.rotate_right(i + 1);
            *p = S.iter().zip(b.Y.iter()).map(|(x, y)| x * y).sum();
        });
        
        let mut result: Self = UnsignedVec::new(self.N.clone()).unwrap();
        result.Y = conv;
        result.normalize()
    }
    
    /// Returns the xor-convolution of Self with another [`Density`](crate::density::Density). 
    /// This algorithm is naive, complexity is O(N^2). 
    pub fn xor_convolve(&self, b: &Self) -> Self {
        assert!(self.N == b.N);
        assert!(self.Y.len() == b.Y.len());
        
        let n = self.Y.len();
        let mut conv = vec![0.0_f32; n];
        conv.iter_mut().enumerate().for_each(|(x, p)| {
            *p = self.Y.iter().enumerate().map(| (t, a) | *a * b.Y[x ^ t]).sum();
        });
        
        let mut result: Self = UnsignedVec::new(self.N.clone()).unwrap();
        result.Y = conv;
        result.normalize()
    }
}

impl<U> Density<U>
        where U: From<u8> + PartialEq + Clone + Sync + Add<Output=U> + std::marker::Send + 
            Num + AsPrimitive<f32>, Density<U>: UnsignedVec<U>, Fourier<U>: ComplexVec<U>
{
    /// Returns the FFT planner for distributions of size N   
    pub fn get_fft_planner(&self) -> Result<Arc<dyn Fft<f32>>, Error> {
        let fourier = Fourier::new(self.N.clone())?;
        fourier.get_fft_planner()
    }

    /// Returns the [`Fourier`](crate::fourier::Fourier) transform of Self. 
    /// Requires pre-computing the FFT planner. 
    pub fn fft(&self, planner: &Arc<dyn Fft<f32>>) -> Fourier<U> {
        let mut complex: Fourier<U> = ComplexVec::<U>::new(self.N.clone()).unwrap();
        complex.fft(self, planner);
        // complex.fourier_transform(self);
        complex
    }

    /// Returns the inverse FFT planner for distributions of size N   
    pub fn get_fft_planner_inverse(&self) -> Result<Arc<dyn Fft<f32>>, Error> {
        let fourier = Fourier::new(self.N.clone())?;
        fourier.get_fft_planner_inverse()
    }
     
    /// Returns the convolution of Self with another [`Density`](crate::density::Density). 
    /// This algorithm uses the FFT, complexity is O(N*log(N)).   
    pub fn fft_convolve(&self, b: &Self, planner: &Arc<dyn Fft<f32>>,
            planner_inv: &Arc<dyn Fft<f32>>) -> Self 
    {
        let mut fourier = self.fft(&planner);
        fourier.convolve(&b.fft(&planner));

        // let scale = 1.0/(self.Y.len() as f32);
        fourier.fft_inverse(&planner_inv)//.normalize() * scale
    }

    /// Returns the Fast Walsh-Hadamard Transform of Self. 
    /// This functions is symmetric, applying twice returns Self. 
    pub fn fwht(&self) -> Self {
        assert!(self.Y.len() % 2 == 0);
        let mut res = self.clone();
        let mut h: usize = 1;
        while h < res.Y.len() {
            let mut i: usize = 0;
            while i < res.Y.len() {

                for j in i..(i + h) {
                    let (x,y) = (res.Y[j], res.Y[j + h]);
                    res.Y[j] = x + y;
                    res.Y[j + h] = x - y;
                }
                i += h*2;
            }
            h *= 2;
        }
        res
    }
    
    /// Returns the xor-convolution of Self with another [`Density`](crate::density::Density). 
    /// This algorithm uses the FWHT, complexity is O(N*log(N)).   
    pub fn fwht_convolve(&self, b: &Self) -> Self {
        let mut left = self.fwht();
        let right = b.fwht();
        left.Y.iter_mut().enumerate().for_each(|(i, x)| { 
            *x *= right.Y[i]; 
        }); 
        left.fwht().normalize()
    }

    /// Returns the xor-convolution of Self with another [`Density`](crate::density::Density) 
    /// that is already in the hadamard domain. 
    /// This algorithm uses the FWHT, complexity is O(N*log(N)). 
    pub fn fwht_left_convolve(&self, b: &Self) -> Self {
        let mut left = self.fwht();
        left.Y.iter_mut().enumerate().for_each(|(i, x)| { 
            *x *= b.Y[i]; 
        });
        left.fwht().normalize()
    }
}

impl UnsignedVec<u16> for Density<u16>
{
    /// Returns a new [`Density`](Density) distribution for 16-bit registers.
    fn new(N: u16) -> Result<Self, Error> {
        let n = usize::from(N) + 1;

        let mut Y = vec![0.0_f32; n];
        Y[0] = 1.0_f32;
        
        Ok(Density{
            N: N,
            Y: Y
        })  
    }

    /// Initializes a 16-bit Self using a [`Distribution`](crate::distributions::Distribution).
    fn initialize(&mut self, distrib: &dyn Distribution<u16>, second_order: bool) {
        self.Y.iter_mut().enumerate().for_each(|(i, y)| { 
            let a = distrib.eval(i.try_into().unwrap(), second_order);
            *y = if a > 0.0 { a } else { 0.0 };
        });
    
        self.Y = self.normalize().Y;
    }
}

impl UnsignedVec<u32> for Density<u32>
{
    /// Returns a new [`Density`](Density) distribution for 32-bit registers.
    fn new(N: u32) -> Result<Self, Error> {
        let n = (N as usize) + 1;

        let mut Y = vec![0.0_f32; n];
        Y[0] = 1.0_f32;
        
        Ok(Density{
            N: N,
            Y: Y
        })  
    }

    /// Initializes a 32-bit Self using a [`Distribution`](crate::distributions::Distribution).
    fn initialize(&mut self, distrib: &dyn Distribution<u32>, second_order: bool) {
        self.Y.iter_mut().enumerate().for_each(|(i, y)| { 
            let a = distrib.eval(i.try_into().unwrap(), second_order);
            *y = if a > 0.0 { a } else { 0.0 };
        });
    
        self.Y = self.normalize().Y;
    }
}

/// Implementation of practical functions for [`Density`](Density).
impl<U> Density<U> 
        where U: Clone
{
    /// Multiplies in-place elements of Self with elements of other.
    pub fn multip(&mut self, other: Density<U>) {
        assert!(self.Y.len() == other.Y.len(), "Cannot multiply two densities with different N values!");
        self.Y.iter_mut().enumerate().for_each(|(i, x)| { 
            *x *= other.Y[i]; 
        });
    } 

    /// Adds elements of other to elements of Self in-place.
    pub fn addip(&mut self, other: Density<U>) {
        assert!(self.Y.len() == other.Y.len(), "Cannot add two densities with different N values!");
        self.Y.iter_mut().enumerate().for_each(|(i, x)| { 
            *x += other.Y[i]; 
        });
    } 

    /// Getter for the distrbution range length N.
    pub fn size(&self) -> U {
        self.N.clone()
    }

    /// Setter for the array of probabilities Y.
    pub fn set(&mut self, Y: Vec<f32>) {
        assert!(Y.len() == self.Y.len());
        self.Y = Y;
    }

    /// Setter for a specific probability value `Y[i]`.
    pub fn set_value(&mut self, i: usize, y: f32){
        self.Y[i] = y;
    }

    /// Getter for the array of probabilities Y.
    pub fn get(&self) -> Vec<f32> {
        self.Y.clone()
    }

    /// Getter for a specific probability value `Y[i]`.
    pub fn get_value(&self, i: usize) -> f32 {
        self.Y[i].clone()
    }

    /// Returns an iterator for the array of probabilities Y.
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.Y.iter()
    }

    /// Returns the highest probability of Y and the associated element.
    pub fn max(&self) -> (usize, f32) {
        let mut m = 0;

        for i in 0..self.Y.len() {
            if self.Y[m] < self.Y[i] { m = i; }
        }

        (m, self.Y[m])
    }

    /// Returns the lowest probability of Y and the associated element.
    pub fn min(&self) -> (usize, f32) {
        let mut m = 0;

        for i in 0..self.Y.len() {
            if self.Y[m] > self.Y[i] { m = i; }
        }

        (m, self.Y[m])
    }

    /// Saves a plot representation of Y to out.
    pub fn plot(&self, out: &str, title: &str, subtitle: &str, line: bool) 
            -> Result<(), std::io::Error> 
    {
        let (mut min, mut max) = (self.min().1 * 1.1, self.max().1 * 1.1);
        if max <= 0.0 {max = 1.0;}
        if min >= max {(min, max) = (0.0, 1.0);}

        // Check directory exists
        let path = "out/plots/";
        if !std::path::Path::new(&path).exists() {
            std::fs::create_dir(path)?;
        }

        let path_str: &str = &(path.to_owned() + out);
        let mut root = BitMapBackend::new(
                path_str, 
                (640, 480)
            ).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        // Optional subtitle argument
        let mut title_size = 40_u32;
        let mut chart_title = title.clone();
        if subtitle != "" {
            root = root.titled(title, ("sans-serif", title_size)).unwrap();
            chart_title = subtitle.clone();
            title_size = 30_u32;
        }

        // Buildind Chart
        let mut chart = ChartBuilder::on(&root)
        .caption(chart_title, ("sans-serif", title_size).into_font())
        .margin(5_u32)
        .x_label_area_size(30_u32)
        .y_label_area_size(50_u32)
        .build_cartesian_2d(0_usize..self.Y.len(), min..max).unwrap();
        
        chart.configure_mesh().draw().unwrap();

        // Line vs. scatter series
        if line {
            let line_series = LineSeries::new((0..self.Y.len()).map(|i| (i, self.Y[i])), &RED);
            chart.draw_series(line_series).unwrap();
        } else {
            let point_series = PointSeries::of_element((0..self.Y.len()).map(|i| (i, self.Y[i])), 1, &RED, 
                    &|c, s, st| {return EmptyElement::at(c) + Circle::new((0,0),s,st.filled());}
                );
            chart.draw_series(point_series).unwrap();
        }
        
        
        root.present().unwrap();
        Ok(())
    }

    /// Saves the array of probability Y to out in csv format.
    pub fn export(&self, out: &str) 
            -> Result<(), std::io::Error>
    {
        // Check directory exists
        let path = "out/csv/";
        if !std::path::Path::new(&path).exists() {
            std::fs::create_dir(path)?;
        }

        // Open file
        let mut wtr = Writer::from_path(path.to_owned() + out).unwrap();
        
        // Write to file 
        self.Y.iter().enumerate().for_each(|(i, p)| { 
            wtr.write_record(&[i.to_string(), (*p).to_string()]).unwrap()
        });
        wtr.flush()?;

        Ok(())
    }
}