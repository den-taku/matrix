use std::convert::Into;
use std::ops::Index;

/// Matrix M[N x M]
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize, Array, T>(
    pub Array,
    pub std::marker::PhantomData<T>,
)
where
    [T; N * M]:,
    Array: Into<[T; N * M]> + Index<usize>;

impl<const N: usize, const M: usize, Array, T> Matrix<N, M, Array, T>
where
    [T; N * M]:,
    Array: Into<[T; N * M]> + Index<usize>,
{
    pub fn new(array: Array) -> Self {
        Self(array, std::marker::PhantomData::<T>)
    }
}

impl<const N: usize, const M: usize, Array, T> std::fmt::Display for Matrix<N, M, Array, T>
where
    T: std::fmt::Display + Copy + num::traits::Zero + PartialOrd,
    [T; N * M]:,
    Array: Into<[T; N * M]> + Index<usize, Output = T>,
{
    fn fmt(&self, dest: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut string = "[ ".to_string();
        for i in 0..N {
            if i != 0 {
                string.push_str("  ");
            }
            for j in 0..M {
                let pad = if self.0[i * M + j] >= T::zero() {
                    " ".to_string()
                } else {
                    "".to_string()
                };
                string = format!("{}{}{} ", string, pad, self.0[i * M + j].clone());
            }
            if i != N - 1 {
                string.push('\n');
            }
        }
        write!(dest, "{}]", string)
    }
}
