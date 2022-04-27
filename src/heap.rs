#[derive(Clone, Debug)]
pub struct Heaped<const N: usize, const M: usize, T>(pub Box<[T; N * M]>)
where
    [T; N * M]:;

impl<const N: usize, const M: usize, T> std::ops::Index<usize> for Heaped<N, M, T>
where
    [T; N * M]:,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[allow(clippy::from_over_into)]
impl<const N: usize, const M: usize, T> Into<[T; N * M]> for Heaped<N, M, T>
where
    [T; N * M]:,
{
    fn into(self) -> [T; N * M] {
        *self.0
    }
}

impl<const N: usize, const M: usize, T> From<[T; N * M]> for Heaped<N, M, T>
where
    [T; N * M]:,
{
    fn from(slice: [T; N * M]) -> Self {
        Self::new(Box::new(slice))
    }
}

impl<const N: usize, const M: usize, T> Heaped<N, M, T>
where
    [T; N * M]:,
{
    pub fn new(boxed_array: Box<[T; N * M]>) -> Self {
        Self(boxed_array)
    }
}
