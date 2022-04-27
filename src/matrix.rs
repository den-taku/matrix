use crate::heap::Heaped;
use std::convert::Into;
use std::ops::{Index, IndexMut};

/// Matrix M[N x M]
#[derive(Clone, Debug)]
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
    /// generate Matrix
    pub fn new(array: Array) -> Self {
        Self(array, std::marker::PhantomData::<T>)
    }
}

impl<const N: usize, const M: usize, Lhs, Rhs, T> std::ops::Add<Matrix<N, M, Rhs, T>>
    for Matrix<N, M, Lhs, T>
where
    [T; N * M]:,
    Lhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize>,
    Rhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize>,
    T: std::ops::AddAssign,
{
    type Output = Self;
    /// A + B
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![allow(incomplete_features)]
    /// #![feature(generic_const_exprs)]
    /// use matrix::matrix::Matrix;
    /// use matrix::heap::Heaped;
    ///
    /// let left = Matrix::<2, 3, _, i32>::new([
    ///     1, 2, 3, //
    ///     4, 5, 6,
    /// ]);
    /// let right = Matrix::<2, 3, _, i32>::new([
    ///     1, 2, 3, //
    ///     4, 5, 6,
    /// ]);
    /// assert_eq!(
    ///     left + right,
    ///     Matrix::<2, 3, _, i32>::new([
    ///         2, 4, 6, //
    ///         8, 10, 12
    ///     ])
    /// );
    ///
    /// let left = Matrix::<2, 3, _, _>::new(Heaped::<2, 3, i32>::new(Box::new([
    ///     1, 2, 3, //
    ///     4, 5, 6,
    /// ])));
    /// let right = Matrix::<2, 3, _, _>::new(Heaped::<2, 3, i32>::new(Box::new([
    ///     1, 2, 3, //
    ///     4, 5, 6,
    /// ])));
    /// assert_eq!(
    ///     left + right,
    ///     Matrix::<2, 3, _, i32>::new([
    ///         2, 4, 6, //
    ///         8, 10, 12
    ///     ])
    /// );
    ///
    /// let left = Matrix::<2, 3, _, _>::new(Heaped::<2, 3, i32>::new(Box::new([
    ///     1, 2, 3, //
    ///     4, 5, 6,
    /// ])));
    /// let right = Matrix::<2, 3, _, i32>::new([
    ///     1, 2, 3, //
    ///     4, 5, 6,
    /// ]);
    /// assert_eq!(
    ///     left + right,
    ///     Matrix::<2, 3, _, i32>::new([
    ///         2, 4, 6, //
    ///         8, 10, 12
    ///     ])
    /// );
    /// ```
    fn add(self, other: Matrix<N, M, Rhs, T>) -> Self::Output {
        let mut new_matrix = self.0.into();
        for (l, r) in new_matrix.iter_mut().zip(other.0.into().into_iter()) {
            *l += r;
        }
        Self::new(Lhs::from(new_matrix))
    }
}

impl<const N: usize, const M: usize, Lhs, T> std::ops::Add<T> for Matrix<N, M, Lhs, T>
where
    [T; N * M]:,
    Lhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize>,
    T: std::ops::AddAssign + Copy,
{
    type Output = Self;
    /// A + k
    fn add(self, other: T) -> Self::Output {
        let mut new_matrix = self.0.into();
        for l in new_matrix.iter_mut() {
            *l += other;
        }
        Self::new(Lhs::from(new_matrix))
    }
}

impl<const N: usize, const M: usize, const L: usize, Rhs, T> std::ops::Mul<Matrix<M, L, Rhs, T>>
    for Matrix<N, M, [T; N * M], T>
where
    [T; N * M]: Into<[T; N * M]> + From<[T; N * M]> + Index<usize, Output = T>,
    [T; M * L]:,
    [T; N * L]:,
    Rhs: Into<[T; M * L]> + From<[T; M * L]> + Index<usize, Output = T>,
    T: std::ops::AddAssign + std::ops::Mul<Output = T> + Copy + num::traits::Zero,
{
    type Output = Matrix<N, L, [T; N * L], T>;
    /// A * B
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![allow(incomplete_features)]
    /// #![feature(generic_const_exprs)]
    /// use matrix::matrix::Matrix;
    /// use matrix::heap::Heaped;
    ///
    /// let left = Matrix::<2, 3, _, i32>::new([
    ///     3, 7, 2, //
    ///     2, 4, 3,
    /// ]);
    /// let right = Matrix::<3, 3, _, i32>::new([
    ///     2, 1, 4, //
    ///     9, 2, 7, //
    ///     8, 3, 2
    /// ]);
    /// assert_eq!(
    ///     left * right,
    ///     Matrix::<2, 3, _, i32>::new([
    ///         85, 23, 65, //
    ///         64, 19, 42
    ///     ])
    /// );
    /// ```
    fn mul(self, other: Matrix<M, L, Rhs, T>) -> Self::Output {
        let mut new_matrix = [T::zero(); N * L];
        for i in 0..N {
            for j in 0..L {
                for k in 0..M {
                    new_matrix[i * L + j] += self.0[i * M + k] * other.0[k * L + j];
                }
            }
        }
        Matrix::new(new_matrix)
    }
}

impl<const N: usize, const M: usize, Lhs, T> std::ops::Mul<T> for Matrix<N, M, Lhs, T>
where
    [T; N * M]:,
    Lhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize>,
    T: std::ops::MulAssign + Copy,
{
    type Output = Self;
    /// A * k
    fn mul(self, other: T) -> Self::Output {
        let mut new_matrix = self.0.into();
        for l in new_matrix.iter_mut() {
            *l *= other;
        }
        Self::new(Lhs::from(new_matrix))
    }
}

impl<const N: usize, const M: usize, const L: usize, Rhs, T> std::ops::Mul<Matrix<M, L, Rhs, T>>
    for Matrix<N, M, Heaped<N, M, T>, T>
where
    [T; N * M]:,
    [T; M * L]:,
    [T; N * L]:,
    Heaped<N, M, T>: Into<[T; N * M]> + From<[T; N * M]> + Index<usize, Output = T>,
    Rhs: Into<[T; M * L]> + From<[T; M * L]> + Index<usize, Output = T>,
    T: std::ops::AddAssign + std::ops::Mul<Output = T> + Copy + num::traits::Zero,
{
    type Output = Matrix<N, L, Heaped<N, L, T>, T>;
    /// A * B
    fn mul(self, other: Matrix<M, L, Rhs, T>) -> Self::Output {
        let mut new_matrix = [T::zero(); N * L];
        for i in 0..N {
            for j in 0..L {
                for k in 0..M {
                    new_matrix[i * L + j] += self.0[i * M + k] * other.0[k * L + j];
                }
            }
        }
        Matrix::new(Heaped::new(Box::new(new_matrix)))
    }
}

impl<const N: usize, const M: usize, Slice, T> std::ops::Neg for Matrix<N, M, Slice, T>
where
    [T; N * M]:,
    Slice: Into<[T; N * M]>
        + From<[T; N * M]>
        + Index<usize, Output = T>
        + IndexMut<usize, Output = T>,
    T: std::ops::Neg<Output = T> + Copy,
{
    type Output = Self;
    /// - A
    fn neg(self) -> Self::Output {
        let mut new_matrix = self.0;
        for i in 0..N {
            for j in 0..M {
                new_matrix[i * M + j] = -new_matrix[i * M + j]
            }
        }
        Self::new(new_matrix)
    }
}

impl<const N: usize, const M: usize, Lhs, Rhs, T> std::ops::Sub<Matrix<N, M, Rhs, T>>
    for Matrix<N, M, Lhs, T>
where
    [T; N * M]:,
    Lhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize>,
    Rhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize>,
    T: std::ops::SubAssign,
{
    type Output = Self;
    /// A - B
    fn sub(self, other: Matrix<N, M, Rhs, T>) -> Self::Output {
        let mut new_matrix = self.0.into();
        for (l, r) in new_matrix.iter_mut().zip(other.0.into().into_iter()) {
            *l -= r;
        }
        Self::new(Lhs::from(new_matrix))
    }
}

impl<const N: usize, const M: usize, Lhs, T> std::ops::Sub<T> for Matrix<N, M, Lhs, T>
where
    [T; N * M]:,
    Lhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize>,
    T: std::ops::SubAssign + Copy,
{
    type Output = Self;
    /// A - k
    fn sub(self, other: T) -> Self::Output {
        let mut new_matrix = self.0.into();
        for l in new_matrix.iter_mut() {
            *l -= other;
        }
        Self::new(Lhs::from(new_matrix))
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

impl<const N: usize, const M: usize, Lhs, Rhs, T> std::cmp::PartialEq<Matrix<N, M, Rhs, T>>
    for Matrix<N, M, Lhs, T>
where
    [T; N * M]:,
    Lhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize, Output = T>,
    Rhs: Into<[T; N * M]> + From<[T; N * M]> + Index<usize, Output = T>,
    T: std::cmp::PartialEq,
{
    fn eq(&self, other: &Matrix<N, M, Rhs, T>) -> bool {
        for i in 0..N {
            for j in 0..M {
                if self[i * M + j] != other[i * M + j] {
                    return false;
                }
            }
        }
        true
    }
}

impl<const N: usize, const M: usize, Array, T> std::ops::Index<usize> for Matrix<N, M, Array, T>
where
    [T; N * M]:,
    Array: Into<[T; N * M]> + Index<usize, Output = T>,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap::*;

    #[test]
    fn for_add() {
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
    }

    #[test]
    fn for_add_scala() {
        let left = Matrix::<2, 3, _, i32>::new([
            1, 2, 3, //
            4, 5, 6,
        ]);
        let right = 7;
        assert_eq!(
            left + right,
            Matrix::<2, 3, _, i32>::new([
                8, 9, 10, //
                11, 12, 13
            ])
        );
    }

    #[test]
    fn for_mul() {
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

        let left = Matrix::<2, 3, _, u32>::new(Heaped::<2, 3, u32>::new(Box::new([
            3, 7, 2, //
            2, 4, 3,
        ])));
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
    }

    #[test]
    fn for_mul_scala() {
        let left = Matrix::<2, 3, _, i32>::new([
            1, 2, 3, //
            4, 5, 6,
        ]);
        let right = 7;
        assert_eq!(
            left * right,
            Matrix::<2, 3, _, i32>::new([
                7, 14, 21, //
                28, 35, 42
            ])
        );
    }

    #[test]
    fn for_neg() {
        let matrix = Matrix::<2, 3, _, i32>::new([
            1, 2, 3, //
            4, 5, 6,
        ]);
        assert_eq!(
            -matrix,
            Matrix::<2, 3, _, i32>::new([
                -1, -2, -3, //
                -4, -5, -6
            ])
        );
    }

    #[test]
    fn for_sub() {
        let left = Matrix::<2, 3, _, i32>::new([
            1, 2, 3, //
            4, 5, 6,
        ]);
        let right = Matrix::<2, 3, _, i32>::new([
            6, 5, 4, //
            3, 2, 1,
        ]);
        assert_eq!(
            left - right,
            Matrix::<2, 3, _, i32>::new([
                -5, -3, -1, //
                1, 3, 5
            ])
        );
    }

    #[test]
    fn for_sub_scala() {
        let left = Matrix::<2, 3, _, i32>::new([
            1, 2, 3, //
            4, 5, 6,
        ]);
        let right = 5;
        assert_eq!(
            left - right,
            Matrix::<2, 3, _, i32>::new([
                -4, -3, -2, //
                -1, 0, 1
            ])
        )
    }
}
