use rand::Rng;
pub use std::slice::{Iter, IterMut};
pub use std::ops::{Deref, DerefMut, Add, Mul, Sub, Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct fVector<const N: usize>
{
   pub data:[f64; N] 
}

impl<const N: usize> fVector<N>
{
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut data = [0.0; N];
        let scale = 1.0 / (N as f64).sqrt();
        for i in 0..N {
            data[i] = rng.gen_range(-scale..scale);
        }
        Self { data }
    }

    pub fn from_array(array:[f64; N])->Self
    {
        Self{data:array}
    }

    pub fn iter(&self)->Iter<f64>
    {
        self.data.iter()
    }

    pub fn iter_mut(&mut self)->IterMut<f64>
    {
        self.data.iter_mut()
    }

    pub fn dot(&self, otherVec: &Self)->f64
    {
        self.data.iter().zip(otherVec.data.iter()).map(|(a,b)| a*b).sum()
    }

    pub fn log(&mut self)->()
    {
        self.data.iter_mut().for_each(|x| *x = x.ln());
    }

    pub fn outer_product<const M:usize>(&self, vec_b: &fVector::<M>) -> Vec::<fVector::<M>> {
        let mut result = vec![fVector::<M>::from_array([0.0; M]); N]; // Initialize a matrix with zeros

        for i in 0..N {
            for j in 0..M {
                result[i][j] = self[i] * vec_b[j];
            }
        }

        result
    }
}


impl<const N: usize> Add for fVector<N> 
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = self.data[i] + other.data[i];
        }
        Self { data: result }
    }
}

impl<const N: usize> Sub for fVector<N> {
     type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = self.data[i] - other.data[i];
        }
        Self { data: result }
    }
}

impl<const N: usize> Mul<f64> for fVector<N> {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = self.data[i] * scalar;
        }
        Self { data: result }
    }
}

impl<const N: usize> Index<usize> for fVector<N> 
{
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> IndexMut<usize> for fVector<N> 
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const N: usize> Deref for fVector<N> 
{
    type Target = [f64; N];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<const N: usize> DerefMut for fVector<N> 
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
