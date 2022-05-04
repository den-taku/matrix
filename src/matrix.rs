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
    Array: Into<[T; N * M]> + Index<usize>;

impl<const N: usize, const M: usize, Array, T> Matrix<N, M, Array, T>
where
    Array: Into<[T; N * M]> + Index<usize>,
{
    /// generate Matrix
    pub fn new(array: Array) -> Self {
        Self(array, std::marker::PhantomData::<T>)
    }
}

impl<const N: usize, const M: usize, Slice, T> Matrix<N, M, Slice, T>
where
    Slice: Into<[T; N * M]> + Index<usize, Output = T>,
    T: Clone,
{
    /// Apply f to all value
    ///
    /// # Examples
    /// ```
    /// #![allow(incomplete_features)]
    /// #![feature(generic_const_exprs)]
    /// use dntk_matrix::matrix::Matrix;
    ///
    /// let matrix = Matrix::<4, 3, _, u32>::new([
    ///    1, 2, 3, //
    ///    4, 5, 6, //
    ///    7, 8, 9, //
    ///    10, 11, 12,
    ///]);
    ///assert_eq!(
    ///    matrix.clone().map::<_, _, [u32; 4 * 3]>(|e| 2 * e),
    ///    matrix * 2
    ///)
    /// ```
    pub fn map<U, F, Array>(self, f: F) -> Matrix<N, M, Array, U>
    where
        F: Fn(T) -> U,
        U: Copy,
        Array: Into<[U; N * M]> + Index<usize> + From<[U; N * M]>,
    {
        let mut new_matrix = [f(self.0[0].clone()); N * M];
        for (a, e) in new_matrix.iter_mut().zip(self.0.into().into_iter()).skip(1) {
            *a = f(e)
        }
        Matrix::new(Array::from(new_matrix))
    }
}

impl<const N: usize, const M: usize, T: Copy> Matrix<N, M, [T; N * M], T>
where
    [T; N * M]: Into<[T; N * M]> + Index<usize, Output = T>,
{
    /// A^t
    /// # Examples
    /// ```
    /// #![allow(incomplete_features)]
    /// #![feature(generic_const_exprs)]
    /// use dntk_matrix::matrix::Matrix;
    ///
    /// let matrix = Matrix::<4, 3, _, u32>::new([
    ///     1, 2, 3, //
    ///     4, 5, 6, //
    ///     7, 8, 9, //
    ///     10, 11, 12
    /// ]);
    /// assert_eq!(
    ///     matrix.transpose(),
    ///     Matrix::<3, 4, _, u32>::new([
    ///         1, 4, 7, 10, //
    ///         2, 5, 8, 11, //
    ///         3, 6, 9, 12
    ///     ])
    /// );
    /// ```
    pub fn transpose(self) -> Matrix<M, N, [T; M * N], T> {
        let mut matrix = [self.0[0]; M * N];
        for (i, e) in self.0.into_iter().enumerate() {
            matrix[(i % M) * N + i / M] = e;
        }
        Matrix::new(matrix)
    }
}

impl<const N: usize, Slice, F: num::traits::Float> Matrix<N, N, Slice, F>
where
    Slice: Into<[F; N * N]>
        + From<[F; N * N]>
        + Index<usize, Output = F>
        + IndexMut<usize, Output = F>
        + Clone,
{
    /// LU decomposition
    /// # Examples
    /// ```
    /// #![allow(incomplete_features)]
    /// #![feature(generic_const_exprs)]
    /// use dntk_matrix::matrix::Matrix;
    ///
    /// let matrix = Matrix::<4, 4, _, f64>::new([
    ///     2.0, 3.0, -4.0, 5.0, //
    ///     1.0, 1.0, 1.0, 1.0, //
    ///     -1.0, 2.0, -3.0, 1.0, //
    ///     1.0, 2.0, 3.0, -4.0,
    /// ]);
    /// let (l, u) = matrix.lu_decomposition();
    ///
    /// assert_eq!(l * u, matrix);
    /// ```
    pub fn lu_decomposition(&self) -> (Self, Self) {
        let mut l = Slice::from([F::zero(); N * N]);
        let mut u = Slice::from([F::zero(); N * N]);

        for i in 0..N {
            l[i * N + i] = F::one();
        }

        let mut dec = self.0.clone();
        for j in 0..N - 1 {
            let w = F::one() / dec[j * N + j];
            for i in j + 1..N {
                dec[i * N + j] = w * dec[i * N + j];
                for k in j + 1..N {
                    dec[i * N + k] = dec[i * N + k] - dec[i * N + j] * dec[j * N + k];
                }
            }
        }

        for j in 0..N {
            for i in 0..j + 1 {
                u[i * N + j] = dec[i * N + j];
            }
            for i in j + 1..N {
                l[i * N + j] = dec[i * N + j];
            }
        }

        (Self::new(l), Self::new(u))
    }
}

/// Solve Ax = b
/// A: N x N
/// b: N
/// # Examples
///
/// ```
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
/// use dntk_matrix::matrix::{Matrix, solve_eqn};
///
/// // Solve Ax = b with LU Decomposition
/// //
/// // 2a + 2b - 4c + 5d = 16
/// //  a +  b +  c +  d = 10
/// // -a + 2b - 3c -  d = -2
/// //  a + 2b + 3c - 4d = -2
/// //
/// // (a, b, c, d) = (1, 2, 3, 4)
///
/// let a = Matrix::<4, 4, _, f64>::new([
///     2.0, 3.0, -4.0, 5.0, //
///     1.0, 1.0, 1.0, 1.0, //
///     -1.0, 2.0, -3.0, 1.0, //
///     1.0, 2.0, 3.0, -4.0,
/// ]);
/// let b = Matrix::<4, 1, _, f64>::new([
///     16.0, //
///     10.0, //
///     -2.0, //
///     -2.0
/// ]);
/// let x = solve_eqn(a, b);
///
/// assert!((1.0 - x.0[0]).abs() < 1e-10);
/// assert!((2.0 - x.0[1]).abs() < 1e-10);
/// assert!((3.0 - x.0[2]).abs() < 1e-10);
/// assert!((4.0 - x.0[3]).abs() < 1e-10);
/// ```
#[allow(clippy::identity_op)] // compiler cannot inference N + 1 = N
pub fn solve_eqn<const N: usize, Slice, Vector, F>(
    a: Matrix<N, N, Slice, F>,
    b: Matrix<N, 1, Vector, F>,
) -> Matrix<N, 1, Vector, F>
where
    F: num::traits::Float,
    Slice: Into<[F; N * N]>
        + From<[F; N * N]>
        + Index<usize, Output = F>
        + IndexMut<usize, Output = F>
        + Clone,
    Vector: Into<[F; N * 1]> + Index<usize, Output = F> + IndexMut<usize, Output = F>,
{
    let mut b = b.0;
    let (l, u) = a.lu_decomposition();
    for i in 0..N - 1 {
        for j in i + 1..N {
            b[j] = b[j] - l.0[j * N + i] * b[i]
        }
    }
    for i in (0..N).rev() {
        b[i] = b[i] / u.0[i * N + i];
        for k in (0..i).rev() {
            b[k] = b[k] - u.0[k * N + i] * b[i];
        }
    }
    Matrix::new(b)
}

impl<const N: usize, const M: usize, Lhs, Rhs, T> std::ops::Add<Matrix<N, M, Rhs, T>>
    for Matrix<N, M, Lhs, T>
where
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
    /// use dntk_matrix::matrix::Matrix;
    /// use dntk_matrix::heap::Heaped;
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
    /// use dntk_matrix::matrix::Matrix;
    /// use dntk_matrix::heap::Heaped;
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
                string = format!("{}{}{} ", string, pad, self.0[i * M + j]);
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

    #[test]
    fn for_transpose() {
        let matrix = Matrix::<4, 2, _, u32>::new([
            3, 4, //
            2, 34, //
            5, 2, //
            3, 54,
        ]);
        assert_eq!(
            matrix.transpose(),
            Matrix::<2, 4, _, u32>::new([
                3, 2, 5, 3, //
                4, 34, 2, 54
            ])
        );

        let matrix = Matrix::<4, 3, _, u32>::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(
            matrix.transpose(),
            Matrix::<3, 4, _, u32>::new([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12])
        );
    }

    #[test]
    fn for_map() {
        let matrix = Matrix::<4, 3, _, u32>::new([
            1, 2, 3, //
            4, 5, 6, //
            7, 8, 9, //
            10, 11, 12,
        ]);
        assert_eq!(
            matrix.clone().map::<_, _, [u32; 4 * 3]>(|e| 2 * e),
            matrix * 2
        )
    }

    #[test]
    fn for_lu_decomposition() {
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
        for i in 0..10 {
            for j in 0..10 {
                if i < j {
                    assert!(l[i * 10 + j] == 0.0)
                } else if i == j {
                    assert!(l[i * 10 + j] == 1.0)
                } else {
                    assert!(u[i * 10 + j] == 0.0)
                }
            }
        }
        let diff = matrix - l * u;
        diff.map::<_, _, [(); 10 * 10]>(|e| assert!(e.abs() < 1e-10));
    }

    #[test]
    fn for_solve_eqn() {
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
}
