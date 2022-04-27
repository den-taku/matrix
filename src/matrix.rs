/// Matrix M[N x M]
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize, T>(pub [T; N * M])
where
    [T; N * M]:;

impl<const N: usize, const M: usize, T> Matrix<N, M, T>
where
    [T; N * M]:,
    T: Clone,
{
    pub fn map<U, F>(self, f: F) -> Matrix<N, M, U>
    where
        F: Fn(T) -> U,
        [U; N * M]:,
        U: Copy,
    {
        let mut new_matrix = [f(self.0[0].clone()); N * M];
        for (a, e) in new_matrix.iter_mut().zip(self.0.into_iter()).skip(1) {
            *a = f(e)
        }
        Matrix(new_matrix)
    }
}

impl<const N: usize, const M: usize, T> Matrix<N, M, T>
where
    [T; N * M]:,
{
    pub fn new(array: [T; N * M]) -> Self {
        Self(array)
    }
}

impl<const N: usize, const M: usize, T: Copy> Matrix<N, M, T>
where
    [T; N * M]:,
    [T; M * N]:,
{
    /// A^t
    /// # Examples
    /// ```
    /// use matrix::matrix::Matrix;
    ///
    /// let matrix = Matrix::<4, 3, u32>([
    ///     1, 2, 3, //
    ///     4, 5, 6, //
    ///     7, 8, 9, //
    ///     10, 11, 12
    /// ]);
    /// assert_eq!(
    ///     matrix.transpose(),
    ///     Matrix::<3, 4, u32>([
    ///         1, 4, 7, 10, //
    ///         2, 5, 8, 11, //
    ///         3, 6, 9, 12
    ///     ])
    /// );
    /// ```
    pub fn transpose(self) -> Matrix<M, N, T> {
        let mut matrix = [self.0[0]; M * N];
        for (i, e) in self.0.into_iter().enumerate() {
            matrix[(i % M) * N + i / M] = e;
        }
        Matrix(matrix)
    }
}

impl<const N: usize, F: num::traits::Float> Matrix<N, N, F>
where
    [F; N * N]:,
{
    /// LU decomposition
    /// # Examples
    /// ```
    /// use matrix::matrix::Matrix;
    ///
    /// let matrix = Matrix::<4, 4, f64>([
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
        let mut l = [F::zero(); N * N];
        let mut u = [F::zero(); N * N];

        for i in 0..N {
            l[i * N + i] = F::one();
        }

        let mut dec = self.0;
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

        (Self(l), Self(u))
    }
}

/// Solve Ax = b
/// A: N x N
/// b: N
/// # Examples
///
/// ```
/// use matrix::matrix::{Matrix, solve_eqn};
///
/// let a = Matrix::<4, 4, f64>([
///     2.0, 3.0, -4.0, 5.0, //
///     1.0, 1.0, 1.0, 1.0, //
///     -1.0, 2.0, -3.0, 1.0, //
///     1.0, 2.0, 3.0, -4.0,
/// ]);
/// let b = Matrix::<4, 1, f64>([
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
pub fn solve_eqn<const N: usize, F>(a: Matrix<N, N, F>, b: Matrix<N, 1, F>) -> Matrix<N, 1, F>
where
    [F; N * N]:,
    [F; N * 1]:,
    F: num::traits::Float,
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
    Matrix(b)
}

/// Solve Ax = b
/// A: N x N
/// b: N
///
/// # Examples
///
/// ```
/// use matrix::matrix::{Matrix, solve_eqn_gauss};
///
/// let a = Matrix::<4, 4, f64>([
///     2.0, 3.0, -4.0, 5.0, //
///     1.0, 1.0, 1.0, 1.0, //
///     -1.0, 2.0, -3.0, 1.0, //
///     1.0, 2.0, 3.0, -4.0,
/// ]);
/// let b = Matrix::<4, 1, f64>([
///     16.0, //
///     10.0, //
///     -2.0, //
///     -2.0
/// ]);
/// let x = solve_eqn_gauss(a, b);
///
/// assert!((1.0 - x.0[0]).abs() < 1e-10);
/// assert!((2.0 - x.0[1]).abs() < 1e-10);
/// assert!((3.0 - x.0[2]).abs() < 1e-10);
/// assert!((4.0 - x.0[3]).abs() < 1e-10);
/// ```
#[allow(clippy::identity_op)] // compiler cannot inference N + 1 = N
pub fn solve_eqn_gauss<const N: usize, F>(a: Matrix<N, N, F>, b: Matrix<N, 1, F>) -> Matrix<N, 1, F>
where
    [F; N * N]:,
    [F; N * 1]:,
    F: num::traits::Float,
{
    backward_substitute(forward_erase(a, b))
}

#[allow(clippy::identity_op)] // compiler cannot inference N + 1 = N
pub fn forward_erase<const N: usize, F>(
    a: Matrix<N, N, F>,
    b: Matrix<N, 1, F>,
) -> (Matrix<N, N, F>, Matrix<N, 1, F>)
where
    [F; N * N]:,
    [F; N * 1]:,
    F: num::traits::Float,
{
    let a = a.0;
    let b = b.0;
    let mut v_a = vec![vec![F::zero(); N + 1]; N];
    for (i, v) in v_a.iter_mut().enumerate() {
        for j in 0..N {
            v[j] = a[i * N + j];
        }
    }
    for (v, e) in v_a.iter_mut().zip(b.iter()) {
        v[N] = *e;
    }
    for i in 0..N {
        let index = {
            let mut v_tmp = Vec::new();
            for (j, v) in v_a.iter().enumerate().take(N).skip(i) {
                v_tmp.push((v[i], j));
            }
            v_tmp.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v_tmp.pop().unwrap().1
        };
        v_a.swap(i, index);
        let a0 = v_a[i][i];
        for j in i..N + 1 {
            v_a[i][j] = v_a[i][j] / a0;
        }
        for k in i + 1..N {
            let c = v_a[k][i];
            for l in i..N + 1 {
                v_a[k][l] = v_a[k][l] - c * v_a[i][l];
            }
        }
    }
    let mut new_a = a;
    let mut new_b = b;
    for i in 0..N {
        for j in 0..N {
            new_a[i * N + j] = v_a[i][j];
        }
        new_b[i] = v_a[i][N];
    }
    (Matrix(new_a), Matrix(new_b))
}

#[allow(clippy::identity_op)] // compiler cannot inference N + 1 = N
pub fn backward_substitute<const N: usize, F>(
    (a, mut b): (Matrix<N, N, F>, Matrix<N, 1, F>),
) -> Matrix<N, 1, F>
where
    [F; N * N]:,
    [F; N * 1]:,
    F: num::traits::Float,
{
    for i in (0..N).rev() {
        for j in 0..i {
            b.0[j] = b.0[j] - a.0[j * N + i] * b.0[i];
        }
    }
    b
}

impl<const N: usize, const M: usize, T> std::ops::Add for Matrix<N, M, T>
where
    T: std::ops::AddAssign + Copy,
    [T; N * M]:,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        let mut new_matrix = self.0;
        for (l, r) in new_matrix.iter_mut().zip(other.0.into_iter()) {
            *l += r
        }
        Self(new_matrix)
    }
}

impl<const N: usize, const M: usize, const L: usize, T> std::ops::Mul<Matrix<M, L, T>>
    for Matrix<N, M, T>
where
    T: std::ops::AddAssign + std::ops::Mul<Output = T> + Copy + num::traits::Zero,
    [T; N * M]:,
    [T; M * L]:,
    [T; N * L]:,
{
    type Output = Matrix<N, L, T>;
    fn mul(self, other: Matrix<M, L, T>) -> Self::Output {
        let mut new_matrix = [T::zero(); N * L];
        for i in 0..N {
            for j in 0..L {
                for k in 0..M {
                    new_matrix[i * L + j] += self.0[i * M + k] * other.0[k * L + j];
                }
            }
        }
        Matrix(new_matrix)
    }
}

impl<const N: usize, const M: usize, T> std::ops::Mul<T> for Matrix<N, M, T>
where
    T: std::ops::MulAssign + Copy,
    [T; N * M]:,
{
    type Output = Matrix<N, M, T>;
    fn mul(self, other: T) -> Self::Output {
        let mut new_matrix = self.0;
        for e in new_matrix.iter_mut() {
            *e *= other;
        }
        Matrix(new_matrix)
    }
}

impl<const N: usize, const M: usize, T> std::ops::Neg for Matrix<N, M, T>
where
    T: std::ops::Neg<Output = T> + Copy,
    [T; N * M]:,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut new_matrix = self.0;
        for e in new_matrix.iter_mut() {
            *e = -(*e);
        }
        Self(new_matrix)
    }
}

impl<const N: usize, const M: usize, T> std::ops::Sub for Matrix<N, M, T>
where
    T: std::ops::SubAssign + Copy,
    [T; N * M]:,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        let mut new_matrix = self.0;
        for (left, right) in new_matrix.iter_mut().zip(other.0.iter()) {
            *left -= *right
        }
        Self(new_matrix)
    }
}

impl<const N: usize, const M: usize, T> std::fmt::Display for Matrix<N, M, T>
where
    T: std::fmt::Display + Copy + num::traits::Zero + PartialOrd,
    [T; N * M]:,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_add() {
        let left = Matrix::<2, 3, u32>([
            1, 2, 3, //
            4, 5, 6,
        ]);
        let right = Matrix::<2, 3, u32>([
            1, 2, 3, //
            4, 5, 6,
        ]);
        let sum = left + right;
        assert_eq!(
            sum,
            Matrix::<2, 3, u32>([
                2, 4, 6, //
                8, 10, 12
            ])
        );
    }

    #[test]
    fn for_mul() {
        let left = Matrix::<2, 3, u32>([
            3, 7, 2, //
            2, 4, 3,
        ]);
        let right = Matrix::<3, 3, u32>([
            2, 1, 4, //
            9, 2, 7, //
            8, 3, 2,
        ]);
        assert_eq!(
            left * right,
            Matrix::<2, 3, u32>([
                85, 23, 65, //
                64, 19, 42
            ])
        );
    }

    #[test]
    fn for_mul2() {
        let left = Matrix::<3, 2, i32>([
            45, 34, //
            25, 21, //
            13, 12,
        ]);
        let right = 3;
        assert_eq!(
            left * right,
            Matrix::<3, 2, i32>([
                135, 102, //
                75, 63, //
                39, 36
            ])
        );
    }

    #[test]
    fn for_neg() {
        let matrix = Matrix::<4, 2, i32>([
            3, -2, //
            -3, 0, //
            0, 34, //
            12, -1,
        ]);
        assert_eq!(
            -matrix,
            Matrix::<4, 2, i32>([
                -3, 2, //
                3, 0, //
                0, -34, //
                -12, 1
            ])
        );
    }

    #[test]
    fn for_sub() {
        let left = Matrix::<2, 3, i32>([
            1, 2, 3, //
            4, 5, 6,
        ]);
        let right = Matrix::<2, 3, i32>([
            6, 5, 4, //
            3, 2, 1,
        ]);
        assert_eq!(
            left - right,
            Matrix::<2, 3, i32>([
                -5, -3, -1, //
                1, 3, 5
            ])
        );
    }

    #[test]
    fn for_gauss() {
        // 2a + 2b - 4c + 5d = 16
        //  a +  b +  c +  d = 10
        // -a + 2b - 3c -  d = -2
        //  a + 2b + 3c - 4d = -2
        //
        // (a, b, c, d) = (1, 2, 3, 4)

        let a = Matrix::<4, 4, f64>([
            2.0, 3.0, -4.0, 5.0, //
            1.0, 1.0, 1.0, 1.0, //
            -1.0, 2.0, -3.0, 1.0, //
            1.0, 2.0, 3.0, -4.0,
        ]);
        let b = Matrix::<4, 1, f64>([
            16.0, //
            10.0, //
            -2.0, //
            -2.0,
        ]);
        let x = solve_eqn_gauss(a, b);
        assert!((1.0 - x.0[0]).abs() < 1e-10);
        assert!((2.0 - x.0[1]).abs() < 1e-10);
        assert!((3.0 - x.0[2]).abs() < 1e-10);
        assert!((4.0 - x.0[3]).abs() < 1e-10);
    }

    #[test]
    fn for_lu_decomposition() {
        let matrix = Matrix::<4, 4, f64>([
            2.0, 3.0, -4.0, 5.0, //
            1.0, 1.0, 1.0, 1.0, //
            -1.0, 2.0, -3.0, 1.0, //
            1.0, 2.0, 3.0, -4.0,
        ]);
        let (l, u) = matrix.lu_decomposition();
        assert_eq!(l * u, matrix);
    }

    #[test]
    fn for_solve_eqn() {
        let a = Matrix::<4, 4, f64>([
            2.0, 3.0, -4.0, 5.0, //
            1.0, 1.0, 1.0, 1.0, //
            -1.0, 2.0, -3.0, 1.0, //
            1.0, 2.0, 3.0, -4.0,
        ]);
        let b = Matrix::<4, 1, f64>([
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

    #[test]
    fn for_transpose() {
        let matrix = Matrix::<4, 2, u32>([
            3, 4, //
            2, 34, //
            5, 2, //
            3, 54,
        ]);
        assert_eq!(
            matrix.transpose(),
            Matrix::<2, 4, u32>([
                3, 2, 5, 3, //
                4, 34, 2, 54
            ])
        );

        let matrix = Matrix::<4, 3, u32>([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(
            matrix.transpose(),
            Matrix::<3, 4, u32>([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12])
        );
    }

    #[test]
    fn for_lu_and_map() {
        let matrix = Matrix::<10, 10, f64>([
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

        let matrix = Matrix::<3, 4, i32>([
            1, 2, 3, 4, //
            5, 6, 7, 8, //
            9, 10, 11, 12, //
        ]);
        assert_eq!(matrix.clone().map(|e| e * 2), matrix * 2);
    }
}
