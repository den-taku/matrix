/// Matrix [[N x M] x L]
#[derive(Clone, Debug, PartialEq)]
pub struct MultiMatrix<const N: usize, const M: usize, const L: usize, T>(pub [T; N * M * L])
where
    [T; N * M * L]:;

impl<const N: usize, const M: usize, const L: usize, T> std::ops::Add for MultiMatrix<N, M, L, T>
where
    T: std::ops::AddAssign + Copy,
    [T; N * M * L]:,
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

impl<const N: usize, const M: usize, const L: usize, T> std::fmt::Display
    for MultiMatrix<N, M, L, T>
where
    T: std::fmt::Display + Copy + num::traits::Zero + PartialOrd,
    [T; N * M * L]:,
{
    fn fmt(&self, dest: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut string = "[[ ".to_string();
        for k in 0..L {
            if k != 0 {
                string = format!("{} [ ", string);
            }
            for i in 0..N {
                if i != 0 {
                    string = format!("{}   ", string);
                }
                for j in 0..M {
                    let pad = if self.0[i * M + j + k * N * M] >= T::zero() {
                        " ".to_string()
                    } else {
                        "".to_string()
                    };
                    string = format!(
                        "{}{}{} ",
                        string,
                        pad,
                        self.0[i * M + j + k * N * M].clone()
                    );
                }
                if i != N - 1 {
                    string = format!("{}\n", string);
                } else if k != L - 1 {
                    string = format!("{}]\n", string);
                } else {
                    string = format!("{}]]", string);
                }
            }
        }
        write!(dest, "{}", string)
    }
}
