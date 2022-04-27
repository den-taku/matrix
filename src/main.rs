#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use matrix::matrix::*;
use matrix::multi_dim::*;

fn main() {
    let left = Matrix::<2, 3, u32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    let right = Matrix::<2, 3, u32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    let sum = left + right;
    println!("{sum}");

    let left = Matrix::<2, 3, u32>::new([
        3, 7, 2, //
        2, 4, 3,
    ]);
    let right = Matrix::<3, 3, u32>::new([
        2, 1, 4, //
        9, 2, 7, //
        8, 3, 2,
    ]);
    let prd = left * right;
    println!("{prd}");

    let left = Matrix::<3, 2, f32>::new([
        4.5, 3.4, //
        2.5, 2.1, //
        1.3, 1.2,
    ]);
    let right = 3.0;
    let prd = left * right;
    println!("{prd}");

    let left = Matrix::<2, 3, i32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    let right = Matrix::<2, 3, i32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    let diff = left - right;
    println!("{diff}");

    let a = Matrix::<4, 4, f64>::new([
        2.0, 3.0, -4.0, 5.0, //
        1.0, 1.0, 1.0, 1.0, //
        -1.0, 2.0, -3.0, 1.0, //
        1.0, 2.0, 3.0, -4.0,
    ]);
    let b = Matrix::<4, 1, f64>::new([16.0, 10.0, -2.0, -2.0]);
    let x = solve_eqn_gauss(a, b);
    println!("{x}");

    let matrix = Matrix::<4, 4, f64>::new([
        2.0, 3.0, -4.0, 5.0, //
        1.0, 1.0, 1.0, 1.0, //
        -1.0, 2.0, -3.0, 1.0, //
        1.0, 2.0, 3.0, -4.0,
    ]);
    let (l, u) = matrix.lu_decomposition();
    println!("{l}");
    println!("{u}");
    println!("{}", l * u);

    let a = Matrix::<4, 4, f64>::new([
        2.0, 3.0, -4.0, 5.0, //
        1.0, 1.0, 1.0, 1.0, //
        -1.0, 2.0, -3.0, 1.0, //
        1.0, 2.0, 3.0, -4.0,
    ]);
    let b = Matrix::<4, 1, f64>::new([
        16.0, //
        10.0, //
        -2.0, //
        -2.0,
    ]);
    let x = solve_eqn(a, b);
    println!("{x}");
    assert!((1.0 - x.0[0]).abs() < 1e-10);
    assert!((2.0 - x.0[1]).abs() < 1e-10);
    assert!((3.0 - x.0[2]).abs() < 1e-10);
    assert!((4.0 - x.0[3]).abs() < 1e-10);

    let matrix = Matrix::<4, 3, u32>::new([
        1, 2, 3, //
        4, 5, 6, //
        7, 8, 9, //
        10, 11, 12,
    ]);
    println!("{matrix}");
    println!("↓");
    let trans = matrix.transpose();
    println!("{trans}");

    let multi_matrix = MultiMatrix::<2, 4, 3, u32>([
        3, 4, 2, 1, //
        4, 5, 4, 2, //
        //
        2, 3, 4, 2, //
        7, 5, 7, 7, //
        //
        2, 2, 4, 6, //
        7, 4, 6, 7, //
    ]);
    println!("{multi_matrix}");

    let matrix = Matrix::<10, 10, f64>::new([
        3.4, 5.3, 2.4, 4.7, 7.89, 3.2, 3.5, 2.1324, 3.0, 3.4, //
        1.4, 5.4, 2.4, 4.7, 7.89, 3.2, 4.5, 2.1324, 3.0, 3.4, //
        2.4, 5.5, 2.4, 4.7, 7.89, 3.2, 2.5, 2.1324, 3.0, 3.4, //
        3.4, 5.6, 2.4, 4.7, 7.89, 3.2, 4.5, 2.1324, 3.0, 3.4, //
        3.4, 5.9, 2.4, 4.7, 7.89, 3.2, 5.5, 2.1324, 3.0, 3.4, //
        5.4, 4.3, 2.4, 4.7, 7.89, 3.2, 4.5, 2.1324, 3.0, 3.4, //
        6.4, 3.3, 2.4, 4.7, 7.89, 3.2, 7.5, 2.1324, 3.0, 3.4, //
        7.4, 1.3, 2.4, 4.7, 7.89, 3.2, 9.5, 2.1324, 3.0, 3.4, //
        8.4, 2.3, 2.4, 4.7, 7.89, 3.2, 4.5, 2.1324, 3.0, 3.4, //
        9.4, 3.3, 2.4, 4.7, 7.89, 3.2, 1.5, 2.1324, 3.0, 3.4, //
    ]);
    let (l, u) = matrix.lu_decomposition();
    let diff = matrix - l * u;
    diff.map(|e| assert!(e.abs() < 1e-10));
}
