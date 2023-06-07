#!/usr/bin/python3

import time
import sys
import os
path = os.path.abspath(os.getcwd())

import matplotlib.pyplot as plt
import random as rd
import pandas as pd
import numpy as np
import csv

from nistrng import check_eligibility_all_battery, run_all_battery, SP800_22R1A_BATTERY, pack_sequence
from nistrng.sp800_22r1a import MonobitTest, FrequencyWithinBlockTest, RunsTest, LongestRunOnesInABlockTest, \
    DiscreteFourierTransformTest, NonOverlappingTemplateMatchingTest, SerialTest, ApproximateEntropyTest, \
    CumulativeSumsTest, RandomExcursionTest, RandomExcursionVariantTest

# numpy version: 1.23.5
# pandas version: 1.5.3
# matplotlib version: 3.7.0
# scipy version: 1.10.0
# nistrng version: 1.2.3

# Setup

## 1) Parameters

start_time = time.time()
bitsize: np.int64 = 32
mask: np.int64 = ((1<<(bitsize-1))-1)*2+1

if len(sys.argv) > 1:  # Check if an argument is passed
    argument = sys.argv[1]
    output_filename = str(argument) + '.csv'
else:
    output_filename = 'output.csv'

## 2) Utility fucntions

def printif(verbose, string):
    if verbose:
        print(string)

def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
def append_to_csv(filename, data):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

# I/ Operational functions

## 1) Binary operations

def xor(a, b): 
    if not(isinstance(a, int)):
        a = int(a)
    if not(isinstance(b, int)):
        b = int(b)
    return (a^b) & mask

def add(a, b): 
    return (a + b) & mask

def bitrol(n, r):
    return ((n << r) & mask) | ((n & mask) >> (bitsize - r))

# def pack_sequence(seq):
#     bits = []
#     for number in seq:
#         # Convert numpy int32 to standard Python int before calling to_bytes
#         number_bytes = int(number).to_bytes(4, 'big')  # Change 8 to 4 to get 32 bits instead of 64
#         bits.extend(np.unpackbits(np.frombuffer(number_bytes, dtype=np.uint8)))
#     return np.array(bits, dtype=np.int8)

## 2) Entropy gathering functions

def hsiphash_perm(pool):
    assert(bitsize==32)
    rol32 = bitrol
    
    a, b, c, d = pool[0], pool[1], pool[2], pool[3]
    
    a = add(a, b)
    b = rol32(b, 5)
    b = xor(b, a)
    a = rol32(a, 16)
    
    c = add(c, d)
    d = rol32(d, 8)
    d = xor(d, c)
    
    a = add(a, d)
    d = rol32(d, 7)
    d = xor(d, a)
    
    c = add(c, b)
    b = rol32(b, 13)
    b = xor(b, c)
    c = rol32(c, 16)
    
    return [a, b, c, d]

fastmix_perm = hsiphash_perm

def fast_mix(pool, input1, input2):
    pool[3] = xor(pool[3], input1)
    pool = fastmix_perm(pool)
    pool[0] = xor(pool[0], input1)
    
    pool[3] = xor(pool[3], input2)
    pool = fastmix_perm(pool)
    pool[0] = xor(pool[0], input2)
    
    return pool

def rot_mix(pool, input1, input2, r): 
    pool = xor(pool, input1)
    pool = xor(pool, input2)
    pool = bitrol(pool, r)
    
    return pool

## 3) Distributions

### A. Timestamp value

def generate_cauchy(gamma, x0, size):
    cauchy_array = np.random.standard_cauchy(size) * gamma + x0
    return cauchy_array.astype(int)

### B. Data value

data_value_distribution = [
    [4112549988, 0.729635], [3995109476, 0.177747], [3709896804, 0.072718],
    [4005702796, 0.010259], [3995352108, 0.004920], [4005727176, 0.000448],
    [4246767716, 0.000396], [4123167688, 0.000297], [4112865224, 0.000261],
    [3995247160, 0.000223], [3995185056, 0.000115], [4112960576, 0.000110],
    [4112786712, 0.000072], [4113351664, 0.000070], [4123167300, 0.000070],
    [3720514504, 0.000058], [3710212040, 0.000053], [3712703804, 0.000050],
    [3995424712, 0.000050], [4113352520, 0.000048]
]

data_elements = [element for element, probability in data_value_distribution]
data_probabilities = [probability for element, probability in data_value_distribution]

def draw_data_value(N):
    drawn_element = rd.choices(data_elements, data_probabilities, k=N)
    return drawn_element

# II/ Simulation

chosen_tests: dict = {
    'monobit': MonobitTest(),
    'frequency_within_block': FrequencyWithinBlockTest(),
    'runs': RunsTest(),
    'longest_run_ones_in_a_block': LongestRunOnesInABlockTest(),
    'dft': DiscreteFourierTransformTest(),
    # 'non_overlapping_template_matching': NonOverlappingTemplateMatchingTest(),
    # 'serial': SerialTest(),
    # 'approximate_entropy': ApproximateEntropyTest(),
    'cumulative sums': CumulativeSumsTest(),
    # 'random_excursion': RandomExcursionTest(),
    'random_excursion_variant': RandomExcursionVariantTest()
}

test_lengths: dict = {
    'monobit': 100,
    'frequency_within_block': 100,
    'runs': 100,
    'longest_run_ones_in_a_block': 128,
    'dft': 1000,
    'non_overlapping_template_matching': 1000000,
    'serial': 100,
    'approximate_entropy': 1000,
    'cumulative sums': 100,
    'random_excursion': 1000000,
    'random_excursion_variant': 1000000
}

## A. Siphash

def make_siphash_sim(gamma, x0, delay, rounds, N, tests, verbose): 
    pools = [ [0,0,0,0] for _ in range(N)]
    output = []
    
    delta_values = np.array([delay for _ in range(N)]).astype(int)
    timestamp_values = np.array([0 for _ in range(N)]).astype(int)
    
    for r in range(rounds): 
        printif(verbose, "Round" + str(r))
        
        # Compensating for the fact that rotations accumulate in a single register
        for _ in range(4):
            delta_values_prev, timestamp_values_prev = delta_values, timestamp_values
            
            theta_values = np.array(generate_cauchy(gamma, x0, N))
            delta_values = delta_values_prev + theta_values
            timestamp_values = timestamp_values_prev + delta_values
                
            data_values = draw_data_value(N)
            
            for i in range(N): 
                pools[i] = fast_mix(pools[i], timestamp_values[i], data_values[i])
        
        flat_pool = np.concatenate(pools)
        flat_pool_int = [int(x) for x in flat_pool]
        binary_sequence = pack_sequence(flat_pool_int)
        
        if tests == None: 
            eligible_battery: dict = check_eligibility_all_battery(
                binary_sequence, SP800_22R1A_BATTERY)
            tests = eligible_battery
        
        for test in tests:
            # print('\t\t'+str(r)+': '+str(test))

            # Limit the sequence to X times the recommended length, if available
            # test_len = min(len(binary_sequence), 20000 * test_lengths[test])
            # test_sequence = binary_sequence[:test_len]
            test_sequence = binary_sequence
            
            # Make sure to pass a dictionary containing only the current test to run_all_battery
            current_test = {test: tests[test]}
            results = run_all_battery(test_sequence, current_test, False)
            
            # Since we're running only one test at a time, we can directly get the result
            result, elapsed_time = results[0]
            
            output.append(["siphash", gamma, x0, delay, N, r, result.name, result.score, result.passed])
            
    return output

## B. Rotations

def make_rotation_sim(rot, gamma, x0, delay, rounds, N, tests, verbose): 
    pools = [ 0 for _ in range(N)]
    output = []
    
    delta_values = np.array([delay for _ in range(N)]).astype(int)
    timestamp_values = np.array([0 for _ in range(N)]).astype(int)
    
    for r in range(rounds): 
        printif(verbose, "Round" + str(r))
        
        delta_values_prev, timestamp_values_prev = delta_values, timestamp_values
        
        theta_values = np.array(generate_cauchy(gamma, x0, N))
        delta_values = delta_values_prev + theta_values
        timestamp_values = timestamp_values_prev + delta_values
            
        data_values = draw_data_value(N)
        
        for i in range(N): 
            pools[i] = rot_mix(pools[i], timestamp_values[i], data_values[i], rot)
        
        binary_sequence = pack_sequence(pools)
        
        if tests == None: 
            eligible_battery: dict = check_eligibility_all_battery(
                binary_sequence, SP800_22R1A_BATTERY)
            tests = eligible_battery
               
        for test in tests:
            # Limit the sequence to X times the recommended length, if available
            # test_len = min(len(binary_sequence), 20000 * test_lengths[test])
            # test_sequence = binary_sequence[:test_len]
            test_sequence = binary_sequence
            
            # Make sure to pass a dictionary containing only the current test to run_all_battery
            current_test = {test: tests[test]}
            results = run_all_battery(test_sequence, current_test, False)
            
            # Since we're running only one test at a time, we can directly get the result
            result, elapsed_time = results[0]
            
            output.append(["rot"+str(rot), gamma, x0, delay, N, r, result.name, result.score, result.passed])

    return output

## C. Averaged result

def init_csv():
    write_to_csv(output_filename, [['function','gamma','x0','delay','pools','round','test','score','passed']])

def run_simulation(gamma, x0, delay, rounds, N, repeats):
    for r in range(repeats):
        print("Repeat {}: {}, {}, {}".format(r, gamma, x0, time.time()-start_time))
        output = make_siphash_sim(gamma, x0, delay, rounds, N, chosen_tests, False)
        append_to_csv(output_filename, output)
        print('\tSiphash done ({}).'.format( time.time()-start_time))

        for rot in range(bitsize - 1):
            output = make_rotation_sim(rot+1, gamma, x0, delay, rounds, 4*N, chosen_tests, False) 
            append_to_csv(output_filename, output)
            print('\tRot ' + str(rot+1) + " done ({}).".format( time.time()-start_time))

# MAIN
init_csv()
run_simulation(444592, -2986, 572059897, 8, 100000, 1)
run_simulation(48031, 3818, 38229868, 8, 100000, 1)
run_simulation(73323, 5263, 38228910, 8, 100000, 1)
run_simulation(444592, 7476, 572059897, 8, 100000, 1)
run_simulation(47339, 7476, 38228910, 8, 100000, 1)
run_simulation(48125, 3804, 38229868, 8, 100000, 1)