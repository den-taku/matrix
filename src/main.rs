#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

struct Matrix<const N: usize, const M: usize, T>
where
    [(); N * M]:,
{
    array: [T; N * M],
}

fn main() {
    println!("Hello, world!");
}
