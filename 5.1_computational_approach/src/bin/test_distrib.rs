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
use entropy_gathering::distributions::{step::Step, gauss::Gauss, cauchy::Cauchy};

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

fn main() -> Result<(), Error>
{
    println!("\n################\
              \n# TEST DISTRIB #\
              \n################");
    print_options(); 
    let execution_start = Instant::now();
    
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

    // // cauchy 
    let cauchy = Cauchy::new(fN, (N/2).into(), N)?;
    let mut cauchy_density = Density::new(N)?;
    cauchy_density.initialize(&cauchy, false);

    // // small cauchy 
    let small_cauchy = Cauchy::new((N/2048).into(), (N/16).into(), N)?;
    let mut small_cauchy_density = Density::new(N)?;
    small_cauchy_density.initialize(&small_cauchy, false);

    // Printing 
    let subtitle = &["Entropy: H = ", &step_density.entropy().to_string()].join("");
    step_density.plot("test_distrib-step.png", "Step distribution", subtitle, false)?;

    let subtitle = &["Entropy: H = ", &small_step_density.entropy().to_string()].join("");
    small_step_density.plot("test_distrib-small_step.png", "Small Step distribution", subtitle, false)?;
    
    let subtitle = &["Entropy: H = ", &gauss_density.entropy().to_string()].join("");
    gauss_density.plot("test_distrib-gauss.png", "Gauss distribution", subtitle, false)?;

    let subtitle = &["Entropy: H = ", &small_gauss_density.entropy().to_string()].join("");
    small_gauss_density.plot("test_distrib-small_gauss.png", "Small Gauss distribution", subtitle, false)?;

    let subtitle = &["Entropy: H = ", &cauchy_density.entropy().to_string()].join("");
    cauchy_density.plot("test_distrib-cauchy.png", "Cauchy distribution", subtitle, false)?;

    let subtitle = &["Entropy: H = ", &small_cauchy_density.entropy().to_string()].join("");
    small_cauchy_density.plot("test_distrib-small_cauchy.png", "Small Cauchy distribution", subtitle, false)?;

    println!("\n -> Total execution time: {:.2?}\n", execution_start.elapsed());
    Ok(())
}