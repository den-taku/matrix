# matrix

https://github.com/den-taku/matrix/actions/workflows/on_merge.yml/badge.svg

Statically sized matrix using a definition with const generics (only for nightly)

## Getting Started

1. `docker compose up --detach`
2. `docker container exec --interactive --tty nightly bash`

## Test

use `cargo nextest run` instead of `cargo test`

## Examples

```rust
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use matrix::heap::*;
use matrix::matrix::*;
// use matrix::multi_dim::*;

fn main() {
    // A + B in stack
    let left = Matrix::<2, 3, _, i32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    let right = Matrix::<2, 3, _, i32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    assert_eq!(
        left + right,
        Matrix::<2, 3, _, i32>::new([
            2, 4, 6, //
            8, 10, 12
        ])
    );

    // A + B in heap
    let left = Matrix::<2, 3, _, _>::new(Heaped::<2, 3, i32>::new(Box::new([
        1, 2, 3, //
        4, 5, 6,
    ])));
    let right = Matrix::<2, 3, _, _>::new(Heaped::<2, 3, i32>::new(Box::new([
        1, 2, 3, //
        4, 5, 6,
    ])));
    assert_eq!(
        left + right,
        Matrix::<2, 3, _, i32>::new([
            2, 4, 6, //
            8, 10, 12
        ])
    );

    // A in heap + B in stack
    let left = Matrix::<2, 3, _, _>::new(Heaped::<2, 3, i32>::new(Box::new([
        1, 2, 3, //
        4, 5, 6,
    ])));
    let right = Matrix::<2, 3, _, i32>::new([
        1, 2, 3, //
        4, 5, 6,
    ]);
    assert_eq!(
        left + right,
        Matrix::<2, 3, _, i32>::new([
            2, 4, 6, //
            8, 10, 12
        ])
    );

    // A * B
    let left = Matrix::<2, 3, _, u32>::new([
        3, 7, 2, //
        2, 4, 3,
    ]);
    let right = Matrix::<3, 3, _, u32>::new([
        2, 1, 4, //
        9, 2, 7, //
        8, 3, 2,
    ]);
    assert_eq!(
        left * right,
        Matrix::<2, 3, _, u32>::new([
            85, 23, 65, //
            64, 19, 42
        ])
    );

    // LU decomposition
    let matrix = Matrix::<10, 10, _, f64>::new([
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
    diff.map::<_, _, [(); 10 * 10]>(|e| assert!(e.abs() < 1e-10));

    // Solve Ax = b with LU Decomposition
    //
    // 2a + 2b - 4c + 5d = 16
    //  a +  b +  c +  d = 10
    // -a + 2b - 3c -  d = -2
    //  a + 2b + 3c - 4d = -2
    //
    // (a, b, c, d) = (1, 2, 3, 4)
    let a = Matrix::<4, 4, _, f64>::new([
        2.0, 3.0, -4.0, 5.0, //
        1.0, 1.0, 1.0, 1.0, //
        -1.0, 2.0, -3.0, 1.0, //
        1.0, 2.0, 3.0, -4.0,
    ]);
    let b = Matrix::<4, 1, _, f64>::new([
        16.0, //
        10.0, //
        -2.0, //
        -2.0,
    ]);
    let x = solve_eqn(a, b);
    assert!((1.0 - x.0[0]).abs() < 1e-10);
    assert!((2.0 - x.0[1]).abs() < 1e-10);
    assert!((3.0 - x.0[2]).abs() < 1e-10);
    assert!((4.0 - x.0[3]).abs() < 1e-10);
}
```