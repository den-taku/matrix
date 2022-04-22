#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use matrix::matrix::*;

fn main() {
    let left = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
    let right = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
    let sum = left + right;
    println!("{sum}");

    let left = Matrix::<2, 3, u32>([3, 7, 2, 2, 4, 3]);
    let right = Matrix::<3, 3, u32>([2, 1, 4, 9, 2, 7, 8, 3, 2]);
    let prd = left * right;
    println!("{prd}");

    let left = Matrix::<2, 3, i32>([1, 2, 3, 4, 5, 6]);
    let right = Matrix::<2, 3, i32>([1, 2, 3, 4, 5, 6]);
    let diff = left - right;
    println!("{diff}");

    let a = Matrix::<4, 4, f64>([
        2.0, 3.0, -4.0, 5.0, 1.0, 1.0, 1.0, 1.0, -1.0, 2.0, -3.0, 1.0, 1.0, 2.0, 3.0, -4.0,
    ]);
    let b = Matrix::<4, 1, f64>([16.0, 10.0, -2.0, -2.0]);
    let x = solve_eqn_gauss(a, b);
    println!("{x}");
}
