#![allow(non_snake_case)]

#[macro_use]
extern crate lazy_static;
extern crate num;

use std::io::Error;
use std::time::Instant;
use coloured::Colorize;

use entropy_gathering::UnsignedVec;
use entropy_gathering::functions::u16;
use entropy_gathering::density::Density;
use entropy_gathering::distributions::{step::Step, gauss::Gauss};

lazy_static! {
    static ref VERBOSE: bool = std::env::args().collect::<Vec<String>>()
        .contains(&"verbose".to_string());
}

fn print_options() {
    print!("\n{}", format!("Options: ").italic());
    if *VERBOSE {
        println!{"{} ", format!("Verbose").italic()};
    } else {
        println!{"{}", format!("None").italic()};
    }
}

#[macro_export]
macro_rules! printif {
    ($( $args:expr ),*) => {
        if *VERBOSE {
            println!( $( $args ),* );
        }
    }
}

fn main() -> Result<(), Error> {
    println!("\n################\
              \n# TEST FOURIER #\
              \n################");
    print_options(); 
    let execution_start = Instant::now();

    // Check directories exists
    let paths = vec!("out/", "out/plots", "out/csv", "out/plots/fft_convolution", "out/csv/fft_convolution");
    for path in paths {
        if !std::path::Path::new(&path).exists() {std::fs::create_dir(path)?;}
    }

    // Parameters
    let n = 16;
    let N = ( 2_u64.pow(n) - 1 ) as u16;

    
    // Creating distributions
    // // step 
    let Na = ( 2_u64.pow(n-2) - 1 ) as u16;
    let Nb = ( 2_u64.pow(n-1) - 1 ) as u16;
    let step = Step::new(Na + 1, Nb, N)?;
    let mut step_density = Density::new(N)?;
    step_density.initialize(&step, false);

    // // small step 
    let Na = ( 2_u64.pow(3) - 1 ) as u16;
    let Nb = ( 2_u64.pow(4) - 1 ) as u16;
    let small_step = Step::new(Na + 1, Nb, N)?;
    let mut small_step_density = Density::new(N)?;
    small_step_density.initialize(&small_step, false);
    
    // // gauss 
    let fN = (N/16) as f32;
    let gauss = Gauss::new(fN, (N/2).into(), N)?;
    let mut gauss_density = Density::new(N)?;
    gauss_density.initialize(&gauss, false);

    // // small gauss 
    let small_gauss = Gauss::new((N/2048).into(), (N/16).into(), N)?;
    let mut small_gauss_density = Density::new(N)?;
    small_gauss_density.initialize(&small_gauss, false);


    // Apply functions
    fn rot_5(u: usize) -> usize { u16::rotate_left(u, 5) }


    // Testing Fourier transform
    test_fourier(&step_density, "step")?;

    // Testing successive Hadamard convolutions
    multiple_fourier(&small_step_density, rot_5, 10, &"step", &"rot_5")?;


    println!("\n -> Total execution time: {:.2?}\n", execution_start.elapsed());
    Ok(())
}

fn test_fourier(density: &Density<u16>, out: &str) -> Result<(), Error> {
    // Timing
    let fft_convolve_start = Instant::now();
    println!("\nFFT convolution:");
    
    // TODO: plot fourier

    // Convolution
    let planner = density.get_fft_planner()?;
    let planner_inv = density.get_fft_planner_inverse()?;
    let conv = density.fft_convolve(&density, &planner, &planner_inv);
    let subtitle = &["Entropy: H = ", &conv.entropy().to_string()].join("");
    conv.plot(&["test_fourier-", out,"-fft-convolve.png"].join(""), "FFT convolution result", 
        subtitle, false)?;

    println!(" -> Execution time: {:.2?}", fft_convolve_start.elapsed());
    Ok(())
}

fn multiple_fourier(density: &Density<u16>, function: fn(usize) -> usize, rounds: i8, 
        distrib_name: &str, function_name: &str) -> Result<(), Error>
{
    // Timing
    let multiple_fft_convolve_start = Instant::now();
    println!("\nSuccessive FFT convolutions:");

    printif!(" - Register entropy: {}", density.entropy());

    // Loop
    let N = density.size();
    let mut result = Density::new(N)?;
    let planner = result.get_fft_planner()?;
    let planner_inv = result.get_fft_planner_inverse()?;
    for i in 1..=rounds {
        result = result.fft_convolve(&density, &planner, &planner_inv);

        // Print before apply
        let path1 = &["fft_convolution/", distrib_name, "-", &i.to_string(), 
            "-before_", function_name].join("");
        let title1 = &["Convolution result ", &i.to_string(), " pre-apply"].join("");
        let subtitle1 = &["Entropy: H = ", &result.entropy().to_string()].join(""); 
        
        result.plot(&[path1, ".png"].join(""), title1, subtitle1, false)?;
        // result.export(&[path1, ".csv"].join(""))?;
        
        // Apply
        result = result.apply(&function);

        // Print after apply
        let path2 = &["fft_convolution/", distrib_name, "-", &i.to_string(), 
            "-post_", function_name].join("");
        let title2 = &["Convolution result ", &i.to_string(), ", post-apply"].join("");
        let subtitle2 = &["Entropy: H = ", &result.entropy().to_string()].join(""); 
        
        result.plot(&[path2, ".png"].join(""), title2, subtitle2, false)?;
        // result.export(&[path2, ".csv"].join(""))?;

        printif!(" - Step {}: {}", i, result.entropy());
    }

    println!(" -> Execution time: {:.2?}", multiple_fft_convolve_start.elapsed());
    Ok(())
}