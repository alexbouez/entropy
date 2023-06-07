#![warn(missing_docs)]
#![allow(non_snake_case)]
#![feature(rustdoc_missing_doc_code_examples)]

//! Entropy Gathering - Main function
//!
//! The sandbox area of the main function allows for direct use of the crate's tools. 
//! See examples in bin.  

#[macro_use]
extern crate lazy_static;
extern crate num;

use std::io::Error;
use std::time::Instant;
use coloured::Colorize;

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
/// Prints args if VERBOSE  
macro_rules! printif {
    ($( $args:expr ),*) => {
        if *VERBOSE {
            println!( $( $args ),* );
        }
    }
}

/// Main function of the crate. 
/// Contains sandbox area, allows access to all statistical tools.
fn main() -> Result<(), Error>{
    println!("\n#####################\n# ENTROPY GATHERING #\n#####################");
    print_options(); 
    let execution_start = Instant::now();
    
    // Sandox area
    // ...
    // End of Sanbox area

    println!("\n -> Total execution time: {:.2?}\n", execution_start.elapsed());
    Ok(())
}
