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
use entropy_gathering::distributions::step::Step;

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
    println!("\n####################\
              \n# TEST CONVOLUTION #\
              \n####################");
    print_options(); 
    let execution_start = Instant::now();

    // Parameters
    let n = 16;
    let N = ( 2_u64.pow(n) - 1 ) as u16;
    // printif!("\nMax value N: {:b}", N);

    
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


    // Apply functions
    fn rot_5(u: usize) -> usize { u16::rotate_left(u, 5) }

    
    // Testing convolution
    test_convolution(&step_density, "step")?;
    test_xor_convolution(&step_density, "step")?;

    // Testing successive convolutions
    test_multiple_convolution(&small_step_density, rot_5, 10, &"step", &"rot_5")?;
    test_multiple_xor_convolution(&small_step_density, rot_5, 10, &"step", &"rot_5")?;


    println!("\n -> Total execution time: {:.2?}\n", execution_start.elapsed());
    Ok(())
}

fn test_convolution(density: &Density<u16>, out: &str) -> Result<(), Error> {
    let convolve_start = Instant::now();
    println!("\nConvolution test:");

    let conv = density.convolve(&density);
    let subtitle = &["Entropy: H = ", &conv.entropy().to_string()].join("");
    conv.plot(&["test_convo-", out,"-convolve.png"].join(""), 
        "Convolution result", subtitle, false)?;

    println!(" -> Execution time: {:.2?}", convolve_start.elapsed());
    Ok(())
}

fn test_multiple_convolution(density: &Density<u16>, function: fn(usize) -> usize, rounds: i8, 
    distrib_name: &str, function_name: &str) -> Result<(), Error> {
    let N = density.size();

    // Successive convolution
    let multiple_convolve_start = Instant::now();
    println!("\nSuccessive convolutions:");

    printif!(" - Register entropy: {}", density.entropy());

    // Check directory exists
    let (path1, path2) = ("out/plots/convolution", "out/csv/convolution");
    if !std::path::Path::new(&path1).exists() {std::fs::create_dir(path1)?;}
    if !std::path::Path::new(&path2).exists() {std::fs::create_dir(path2)?;}

    // // Loop
    let mut result = Density::new(N)?;
    for i in 1..=rounds {
        result = result.convolve(&density);
        
        // Print before apply
        let path1 = &["convolution/", distrib_name, "-convolve-", &i.to_string(), 
            "-before_", function_name].join("");
        let title1 = &["Convolution result ", &i.to_string(), " pre-apply"].join("");
        let subtitle1 = &["Entropy: H = ", &result.entropy().to_string()].join("");
        
        result.plot(&[path1, ".png"].join(""), title1, subtitle1, false)?;
        // result.export(&[path1, ".csv"].join(""))?;
        
        // Apply
        result = result.apply(&function);

        // Print after apply
        let path2 = &["convolution/", distrib_name, "-convolve-", &i.to_string(), 
            "-post_", function_name].join("");
        let title2 = &["Convolution result ", &i.to_string(), " post-apply"].join("");
        let subtitle2 = &["Entropy: H = ", &result.entropy().to_string()].join("");

        result.plot(&[path2, ".png"].join(""), title2, subtitle2, false)?;
        // result.export(&[path2, ".csv"].join(""))?;


        printif!(" - Step {}: {}", i, result.entropy());
    }

    println!(" -> Execution time: {:.2?}", multiple_convolve_start.elapsed());
    Ok(())
}

fn test_xor_convolution(density: &Density<u16>, out: &str) -> Result<(), Error> {
    let xor_convolve_start = Instant::now();
    println!("\nXor convolution test:");

    let conv = density.xor_convolve(&density);
    let subtitle = &["Entropy: H = ", &conv.entropy().to_string()].join("");
    conv.plot(&["test_xor-", out,"-xor-convolve.png"].join(""), 
        "Xor convolution result", subtitle, true)?;

    println!(" -> Execution time: {:.2?}", xor_convolve_start.elapsed());
    Ok(())
}

fn test_multiple_xor_convolution(density: &Density<u16>, function: fn(usize) -> usize, 
        rounds: i8, distrib_name: &str, function_name: &str) -> Result<(), Error> {
    let N = density.size();

    // Successive xor convolution
    let multiple_convolve_start = Instant::now();
    println!("\nSuccessive xor convolutions:");

    printif!(" - Register entropy: {}", density.entropy());

    // Check directory exists
    let (path1, path2) = ("out/plots/xor_convolution", "out/csv/xor_convolution");
    if !std::path::Path::new(&path1).exists() {std::fs::create_dir(path1)?;}
    if !std::path::Path::new(&path2).exists() {std::fs::create_dir(path2)?;}

    // Loop
    let mut result = Density::new(N)?;
    for i in 1..=rounds {
        result = result.xor_convolve(&density);

        // Print before apply
        let path1 = &["xor_convolution/", distrib_name, "-xor_convolve-", &i.to_string(), 
            "-before_", function_name].join("");
        let title1 = &["Xor convolution result ", &i.to_string(), " pre-apply"].join("");
        let subtitle1 = &["Entropy: H = ", &result.entropy().to_string()].join("");

        result.plot(&[path1, ".png"].join(""), title1, subtitle1, true)?;
        // result.export(&[path1, ".csv"].join(""))?;
        
        // Apply
        result = result.apply(&function);

        // Print after apply
        let path2 = &["xor_convolution/", distrib_name, "-xor_convolve-", &i.to_string(), 
            "-post_", function_name].join("");
        let title2 = &["Xor convolution result ", &i.to_string(), " post-apply"].join("");
        let subtitle2 = &["Entropy: H = ", &result.entropy().to_string()].join("");
        
        result.plot(&[path2, ".png"].join(""), title2, subtitle2, true)?;
        // result.export(&[path2, ".csv"].join(""))?;

        printif!(" - Step {}: {}", i, result.entropy());
    }

    println!(" -> Execution time: {:.2?}", multiple_convolve_start.elapsed());
    Ok(())
}