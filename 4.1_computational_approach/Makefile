OUT = out/plots/ out/csv/ 
OPTIONS = distrib convolution fourier hadamard 

.PHONY: out build run clean distclean 

target: run

build: 
	@ cargo build

release:
	@ cargo build --release --target=x86_64-unknown-linux-gnu

run: 
	@ cargo run


clean: 
	@ cargo clean

distclean: clean
	@ rm -rf out/ __pycache__/


out: 
	@ mkdir -p $(OUT)


.PHONY: minimal minimal32 distrib convolution fourier hadamard

minimal: out
	@ cargo run --bin best_rot_xor_u16

minimal32: out
	@ cargo run --bin best_rot_xor_u32

distrib: out
	@ cargo run --bin test_distrib verbose

convolution: out
	@ cargo run --bin test_convolution verbose

fourier: out
	@ cargo run --bin test_fourier verbose

hadamard: out
	@ cargo run --bin test_hadamard verbose
	@ ./plot.py
