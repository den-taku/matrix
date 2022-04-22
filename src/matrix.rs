#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize, T>(pub [T; N * M])
where
    [T; N * M]:;

impl<const N: usize, const M: usize, T> Matrix<N, M, T> where [(); N * M]: {}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_add() {
        let left = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
        let right = Matrix::<2, 3, u32>([1, 2, 3, 4, 5, 6]);
        assert_eq!(left + right, Matrix::<2, 3, u32>([2, 4, 6, 8, 10, 12]));
    }

    #[test]
    fn for_mul() {
        let left = Matrix::<2, 3, u32>([3, 7, 2, 2, 4, 3]);
        let right = Matrix::<3, 3, u32>([2, 1, 4, 9, 2, 7, 8, 3, 2]);
        assert_eq!(left * right, Matrix::<2, 3, u32>([85, 23, 65, 64, 19, 42]));
    }
}
