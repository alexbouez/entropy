#![allow(non_snake_case)]
extern crate lazy_static;
extern crate num;

use std::io;

use rayon::prelude::*;
use std::time::Instant;
use std::sync::Mutex;

use entropy_gathering::ComplexVec;
use entropy_gathering::UnsignedVec;
use entropy_gathering::functions::u16;
use entropy_gathering::density::Density;
use entropy_gathering::fourier::Fourier;
use entropy_gathering::distributions::cauchy::Cauchy;

fn main() -> Result<(), io::Error>
{
    let execution_start = Instant::now();

    let num_threads = 3;
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Check directories exists
    let dir = "out/";
    if !std::path::Path::new(&dir).exists() {std::fs::create_dir(dir)?;}

    // Parameters
    let n = 16_u32;
    let rounds = 8_usize;

    println!("Startup ({:.2?})", execution_start.elapsed());

    // Distribution
    let sigma_params: (f32, f32) = (20.0, -10.0);
    let init_delay: u16 = 10000;

    // Data field distribution
    let data_fwht = make_data_distrib(sigma_params, n, execution_start)?;

    // let t_vec = make_timing_vec_1o(sigma_params, rounds, n, execution_start)?;
    let t_vec = make_timing_vec_2o(sigma_params, rounds, n, execution_start, init_delay)?;

    // test functions
    test_rotations(t_vec, sigma_params, n as usize, execution_start, data_fwht)?;

    println!("{:.2?}", execution_start.elapsed());
    Ok(())
}

fn make_data_distrib(params: (f32,f32), n: u32, 
    execution_start: Instant) -> Result<Density<u16>, io::Error>
{
    let N = ( 2_u64.pow(n) - 1 ) as u16;

    let mut data_distrib = Density::new(N)?;
    let data_array= [
        (11961usize, 0.953942f32),
        (12033usize, 0.013451f32),
        (12898usize, 0.004216f32),
        (13330usize, 0.003636f32),
        (13114usize, 0.002954f32),
    ];

    data_distrib.set_value(0, 0.0);
    for &(u, p) in &data_array {
        data_distrib.set_value(u, p);
    }
    data_distrib = data_distrib.normalize();

    let entropy = data_distrib.entropy();
    let min_entropy = data_distrib.min_entropy();
    let kl_div = data_distrib.kl_divergence();
    let stat_distance = data_distrib.statistical_distance();

    let time = execution_start.elapsed();
    println!("{:.2?}, {}, {}, {}, {}, {}: {}, {}, {}, {}",  time, n, params.0, params.1, 
        &"data_distrib", 0, entropy, min_entropy, kl_div, stat_distance);

    Ok(data_distrib.fwht())
}

fn _make_timing_vec_1o(params: (f32,f32), rounds: usize, n: u32, 
         execution_start: Instant) -> Result<Vec<Density<u16>>, io::Error> 
{
    let N = ( 2_u64.pow(n) - 1 ) as u16;
    let mut t_vec: Vec<Density<u16>> = vec![Density::new(N)?; rounds];
    let mut t_vec_fourier: Vec<Fourier<u16>> = Vec::new();
    let mut delta_distrib = Density::new(N)?;

    println!("Building Fourier ({:.2?})", execution_start.elapsed());
    let mut fourier_t = Fourier::new(N)?;
    println!("\tFourier Initialisation done ({:.2?})", execution_start.elapsed());

    let (planner, planner_inv) = rayon::join(|| 
            fourier_t.get_fft_planner().unwrap(), || 
            fourier_t.get_fft_planner_inverse().unwrap()
        );
    println!("\tFourier Planner done ({:.2?})", execution_start.elapsed());

    println!("Building Cauchy ({:.2?})", execution_start.elapsed());
    let small_cauchy = Cauchy::new(params.0, params.1, N)?;
        println!("\tCauchy Function done ({:.2?})", execution_start.elapsed());
    delta_distrib.initialize(&small_cauchy, false);
        println!("\tCauchy Distribution done ({:.2?})", execution_start.elapsed());
    let fourier_delta = delta_distrib.fft(&planner);
        println!("\tCauchy FFT done ({:.2?})", execution_start.elapsed());

    for i in 0..rounds{
        println!("Computing T{} ({:.2?})", i+1, execution_start.elapsed());
        fourier_t.convolve(&fourier_delta);
        println!("\tT{}: Convolve done ({:.2?})", i+1, execution_start.elapsed());
        t_vec_fourier.push(fourier_t.clone());
        println!("\tT{}: Copy done ({:.2?})", i+1, execution_start.elapsed());
    }

    println!("time, bit_size, delta_scale, delta_location, function, round, entropy, min_entropy");
    let t_vec_fourier_mutex = Mutex::new(&t_vec_fourier);
    t_vec.par_iter_mut().enumerate().for_each(|(i, p)| {
        let temp_vec = t_vec_fourier_mutex.lock().unwrap();
        let temp = temp_vec[i].clone();
        drop(temp_vec);
        
        // let scale = 1.0/(N as f32);
        let temp = temp.fft_inverse(&planner_inv).normalize(); //* scale;

        let entropy = temp.entropy();
        let min_entropy = temp.min_entropy();
        let kl_div = temp.kl_divergence();
        let stat_distance = temp.statistical_distance();

        *p = temp.fwht();

        let time = execution_start.elapsed(); 
        println!("{:.2?}, {}, {}, {}, {}, {}: {}, {}, {}, {}",  time, n, params.0, params.1, 
            &["T", &(i+1).to_string()].join(""), i+1, entropy, min_entropy, kl_div, stat_distance);
    });

    Ok(t_vec)
}

fn make_timing_vec_2o(params: (f32,f32), rounds: usize, n: u32, 
         execution_start: Instant, d0: u16) -> Result<Vec<Density<u16>>, io::Error> 
{
    let N = ( 2_u64.pow(n) - 1 ) as u16;
    // let mut t_vec: Vec<Density<u16>> = vec![Density::new(N)?; rounds];
    let mut t_vec: Vec<Density<u16>> = Vec::new();
    let mut t_vec_fourier: Vec<Fourier<u16>> = Vec::new();

    let mut sigma_distrib = Density::new(N)?;
    let mut delta_distrib = Density::new(N)?;

    println!("Building Fourier ({:.2?})", execution_start.elapsed());
    let mut fourier_t = Fourier::new(N)?;
    println!("\tFourier Initialisation done ({:.2?})", execution_start.elapsed());

    let (planner, planner_inv) = rayon::join(|| 
            fourier_t.get_fft_planner().unwrap(), || 
            fourier_t.get_fft_planner_inverse().unwrap()
        );
    println!("\tFourier Planner done ({:.2?})", execution_start.elapsed());

    println!("Building Cauchy ({:.2?})", execution_start.elapsed());
    let small_cauchy = Cauchy::new(params.0, params.1, N)?;
    println!("\tCauchy Function done ({:.2?})", execution_start.elapsed());
    sigma_distrib.initialize(&small_cauchy, true);
    println!("\tCauchy Distribution done ({:.2?})", execution_start.elapsed());
    let fourier_sigma = sigma_distrib.fft(&planner);
    println!("\tCauchy FFT done ({:.2?})", execution_start.elapsed());

    println!("Building Delta ({:.2?})", execution_start.elapsed());
    delta_distrib.set_value(0, 0.0);
    delta_distrib.set_value(d0 as usize, 1.0);
    println!("\tDelta Distribution done ({:.2?})", execution_start.elapsed());
    let mut fourier_delta = delta_distrib.fft(&planner);
    println!("\tDelta FFT done ({:.2?})", execution_start.elapsed());

    for i in 0..rounds{
        println!("Computing T{} ({:.2?})", i+1, execution_start.elapsed());

        fourier_delta.convolve(&fourier_sigma);
        fourier_t.convolve(&fourier_delta);
        
        println!("\tT{}: Convolve done ({:.2?})", i+1, execution_start.elapsed());
        t_vec_fourier.push(fourier_t.clone());
        println!("\tT{}: Copy done ({:.2?})", i+1, execution_start.elapsed());
        t_vec.push(Density::new(N)?);
    }

    println!("time, bit_size, delta_scale, delta_location, function, round, entropy, min_entropy");
    let t_vec_fourier_mutex = Mutex::new(&t_vec_fourier);
    t_vec.par_iter_mut().enumerate().for_each(|(i, p)| {
        let temp_vec = t_vec_fourier_mutex.lock().unwrap();
        let temp = temp_vec[i].clone();
        drop(temp_vec);

        let temp = temp.fft_inverse(&planner_inv).normalize();
        let entropy = temp.entropy();
        let min_entropy = temp.min_entropy();
        let kl_div = temp.kl_divergence();
        let stat_distance = temp.statistical_distance();

        *p = temp.fwht();

        let time = execution_start.elapsed(); 
        println!("{:.2?}, {}, {}, {}, {}, {}: {}, {}, {}, {}",  time, n, params.0, params.1, 
            &["T", &(i+1).to_string()].join(""), i+1, entropy, min_entropy, kl_div, stat_distance);
    });

    Ok(t_vec)
}

fn test_rotations(t_vec: Vec<Density<u16>>, params: (f32,f32), 
        n: usize, execution_start: Instant, data_fwht: Density<u16>) -> Result<(), io::Error>
{
    let N = t_vec[0].size();
    fn rot(u: usize, alpha: usize) -> usize { u16::rotate_left(u, alpha) }
    
    let rounds = t_vec.len();
    // let t_vec_mutex = Mutex::new(&t_vec);

    (0..n).into_par_iter().for_each(|alpha| {
        let rot_i = |x: usize| -> usize {rot(x, alpha as usize)};

        let mut pool = Density::new(N).unwrap();
        for round in 0..rounds {
            if round == 0 {
                pool = t_vec[round].clone().fwht().normalize();
            } else{
                pool = pool.apply(&rot_i).fwht_left_convolve(&t_vec[round].clone());
                pool = pool.fwht_left_convolve(&data_fwht.clone());
            }   

            let entropy = pool.entropy();
            let min_entropy = pool.min_entropy();
            let kl_div = pool.kl_divergence();
            let stat_distance = pool.statistical_distance();

            let time = execution_start.elapsed();
            println!("{:.2?}, {}, {}, {}, {}, {}: {}, {}, {}, {}",  time, n, params.0, params.1, 
                &["rot_", &(alpha+1).to_string()].join(""), round+1, entropy, min_entropy, kl_div, stat_distance);
        }
    });
    
    Ok(())
}
