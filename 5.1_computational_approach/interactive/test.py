#!/usr/bin/env python
import math
import numpy as np
from scipy import stats 

n = 16

def circular_right_shift16(x, a):
    return (x >> a) | (x << (n-a))  & 0xFFFF

def circular_left_shift16(x, a):
    return (x << a) & 0xFFFF | (x >> (n-a))

while True:
    num = int(input("Input binary value  : "), 2)
    a = int(input("Input shift value   : "), 10)
    print("Result : ", bin(circular_right_shift16(num,a)))