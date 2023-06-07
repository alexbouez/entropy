# Probability sandbox

This crate allows for efficient data computation surrounding this objective.
The main program is a sandbox area which gives access to all the statistical tools of the crate.
Examples on how to use the crate, as well as benchmarking code for rotation functions, 
are available in the bin/ folder. These can be built and run using the provided Makefile. 

### Dendencies

The main project uses Python3 and Rust. See the following pages for help with installation:
 * **The Rust Book:** https://doc.rust-lang.org/cargo/getting-started/installation.html

The Rust dependencies are installed automatically when using cargo. They are listed in
'Cargo.toml'.

It requires the installation of a number of dependencies to be installed for the 
dependencies to function properly: 
> cmake, gcc, g++, pkg-config, libfontconfig, libfontconfig1-dev, libfreetype6-dev, m4

### Instructions

The project is developped in Rust, the easiest option for launching is to use cargo:
 * cargo build      : Builds the project without running it. Installs missing dependencies.
 * cargo run        : Builds and runs the project. Installs missing dependencies.
 * cargo check      : Checks the program for errors/warnings without building the project.
 * cargo test       : Builds and runs every submodule test section.
 * cargo clean      : Deletes all compilation products.
 * cargo doc --open : Builds and opens the documentation in a browser.

Use "cargo build --release" to build all the alternative main functions from bin/. 
Executable files are located in target/release/. 
See the cargo man page for more information.

A Makefile is available with built-in quick access commands:
 * make             : Runs the main function (src/main.rs). 
 * make minimal     : Runs the main 16-bit rotation comparison function (src/bin/best_rot_xor_u16.rs). 
 * make minimal32   : Runs the main 32-bit rotation comparison function (src/bin/best_rot_xor_u32.rs). 
 * make distrib     : Runs the distribution test function (src/bin/test_distrib.rs).
 * make convolution : Runs the convolution test functions (src/bin/test_convolution.rs).
 * make fourier     : Runs the FFT and FFT convolution test functions (src/bin/test_fourier.rs).
 * make hadamard    : Runs the FHT and FHT convolution test functions (src/bin/test_hadamard.rs).

Some additional options are available when using the makefile: 
 * verbose          : Outputs all indicative and intermediary messages.
     - ex: 'cargo run verbose'     (not to be confused with 'cargo run -v' or 'cargo -v run')

Some useful commands are also available through the Makefile:
 * make clean       : Delete all compilation products.
 * make distclean   : Delete all compilation products and results (including plots).

## Interactive visualization

### Dependencies

The interactive visualization tool uses Python3 with Numpy and Bokeh. 
Bokeh can be installed with 'conda install bokeh' using Miniconda/Anaconda. 
    
### Instructions

The tool can be launched in the [interactive](interactive/) directory, using 'make plots'.
