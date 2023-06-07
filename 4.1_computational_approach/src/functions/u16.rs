#![allow(non_snake_case)]

//! Entropy Gathering - Functions - u16
//!
//! Implement bitwise operations for 16-bit registers:
//! not, shift_left, shift_right, rotate_left, rotate_right

/// Returns u after applying `not` operation.
pub fn not(u: usize) -> usize{
    let new = !(u as u16) as usize;
    new
}

/// Returns u shifted a bits to the left.
pub fn shift_left(u: usize, a: usize) -> usize {
    ( (u as u16) << a ) as usize
}

/// Returns u shifted a bits to the right.
pub fn shift_right(u: usize, a: usize) -> usize {
    ( (u as u16) >> a ) as usize
}

/// Returns u rotated a bits to the left.
pub fn rotate_left(u: usize, a: usize) -> usize {
    ( (u as u16).rotate_left(a.try_into().unwrap()) ) as usize
}

/// Returns u rotated a bits to the right.
pub fn rotate_right(u: usize, a: usize) -> usize {
    ( (u as u16).rotate_right(a.try_into().unwrap()) ) as usize
}