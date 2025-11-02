//! Fixed-size vector implementation with f64 elements.
//!
//! This module provides a generic fixed-size vector type that supports
//! common mathematical operations like dot product, outer product, and
//! standard arithmetic operations (addition, subtraction, scalar multiplication).

use rand::Rng;
pub use std::slice::{Iter, IterMut};
pub use std::ops::{Deref, DerefMut, Add, Mul, Sub, Index, IndexMut};

/// A fixed-size vector containing `N` elements of type `f64`.
///
/// This struct provides efficient vector operations using Rust's const generics
/// to ensure compile-time size checking. It supports standard mathematical
/// operations and can be used as a building block for machine learning algorithms.
///
/// # Type Parameters
///
/// * `N` - The number of elements in the vector (fixed at compile time)
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::f_Vector::fVector;
///
/// // Create a new vector with random initialization
/// let vec1 = fVector::<3>::new();
///
/// // Create a vector from an array
/// let vec2 = fVector::<3>::from_array([1.0, 2.0, 3.0]);
///
/// // Perform operations
/// let sum = vec1 + vec2;
/// let scaled = vec2 * 2.0;
/// let dot_product = vec1.dot(&vec2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct fVector<const N: usize>
{
   /// The underlying array storing the vector elements
   pub data:[f64; N] 
}

impl<const N: usize> fVector<N>
{
    /// Creates a new vector with random initialization using Xavier/He initialization.
    ///
    /// The elements are initialized with random values in the range
    /// `[-scale, scale]` where `scale = 1.0 / sqrt(N)`. This initialization
    /// is commonly used in neural networks to prevent gradient vanishing/explosion.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// let vec = fVector::<5>::new();
    /// assert_eq!(vec.len(), 5);
    /// ```
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut data = [0.0; N];
        let scale = 1.0 / (N as f64).sqrt();
        for i in 0..N {
            data[i] = rng.gen_range(-scale..scale);
        }
        Self { data }
    }

    /// Creates a vector from an existing array of `f64` values.
    ///
    /// # Arguments
    ///
    /// * `array` - An array of `f64` values with length `N`
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// let vec = fVector::<3>::from_array([1.0, 2.0, 3.0]);
    /// assert_eq!(vec[0], 1.0);
    /// ```
    pub fn from_array(array:[f64; N])->Self
    {
        Self{data:array}
    }

    /// Returns an iterator over the vector elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// let vec = fVector::<3>::from_array([1.0, 2.0, 3.0]);
    /// let sum: f64 = vec.iter().sum();
    /// assert_eq!(sum, 6.0);
    /// ```
    pub fn iter(&self)->Iter<f64>
    {
        self.data.iter()
    }

    /// Returns a mutable iterator over the vector elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// let mut vec = fVector::<3>::from_array([1.0, 2.0, 3.0]);
    /// vec.iter_mut().for_each(|x| *x *= 2.0);
    /// assert_eq!(vec[0], 2.0);
    /// ```
    pub fn iter_mut(&mut self)->IterMut<f64>
    {
        self.data.iter_mut()
    }

    /// Computes the dot product of this vector with another vector.
    ///
    /// The dot product is the sum of element-wise products: Σ(a\[i\] * b\[i\])
    ///
    /// # Arguments
    ///
    /// * `otherVec` - A reference to another vector of the same size
    ///
    /// # Returns
    ///
    /// The dot product as an `f64` value
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// let vec1 = fVector::<3>::from_array([1.0, 2.0, 3.0]);
    /// let vec2 = fVector::<3>::from_array([4.0, 5.0, 6.0]);
    /// let result = vec1.dot(&vec2);
    /// assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    pub fn dot(&self, otherVec: &Self)->f64
    {
        self.data.iter().zip(otherVec.data.iter()).map(|(a,b)| a*b).sum()
    }

    /// Applies the natural logarithm to each element in-place.
    ///
    /// This method modifies the vector by replacing each element with its
    /// natural logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// let mut vec = fVector::<3>::from_array([1.0, 2.718281828, 7.389056099]);
    /// vec.log();
    /// // Elements are now approximately [0.0, 1.0, 2.0]
    /// ```
    pub fn log(&mut self)->()
    {
        self.data.iter_mut().for_each(|x| *x = x.ln());
    }

    /// Computes the outer product of this vector with another vector.
    ///
    /// The outer product of vectors `a` (size N) and `b` (size M) produces
    /// an N×M matrix where element (i,j) = a\[i\] * b\[j\].
    ///
    /// # Type Parameters
    ///
    /// * `M` - The size of the other vector
    ///
    /// # Arguments
    ///
    /// * `vec_b` - A reference to the other vector
    ///
    /// # Returns
    ///
    /// A vector of vectors representing the N×M matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// let vec1 = fVector::<2>::from_array([1.0, 2.0]);
    /// let vec2 = fVector::<3>::from_array([3.0, 4.0, 5.0]);
    /// let outer = vec1.outer_product::<3>(&vec2);
    /// // Results in a 2x3 matrix:
    /// // [[3.0, 4.0, 5.0],
    /// //  [6.0, 8.0, 10.0]]
    /// ```
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


/// Implementation of the `Add` trait for element-wise vector addition.
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::f_Vector::fVector;
///
/// let vec1 = fVector::<3>::from_array([1.0, 2.0, 3.0]);
/// let vec2 = fVector::<3>::from_array([4.0, 5.0, 6.0]);
/// let result = vec1 + vec2;
/// assert_eq!(result[0], 5.0);
/// ```
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

/// Implementation of the `Sub` trait for element-wise vector subtraction.
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::f_Vector::fVector;
///
/// let vec1 = fVector::<3>::from_array([5.0, 7.0, 9.0]);
/// let vec2 = fVector::<3>::from_array([2.0, 3.0, 4.0]);
/// let result = vec1 - vec2;
/// assert_eq!(result[0], 3.0);
/// ```
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

/// Implementation of the `Mul` trait for scalar multiplication.
///
/// Multiplies each element of the vector by the given scalar value.
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::f_Vector::fVector;
///
/// let vec = fVector::<3>::from_array([1.0, 2.0, 3.0]);
/// let result = vec * 2.0;
/// assert_eq!(result[0], 2.0);
/// assert_eq!(result[1], 4.0);
/// ```
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

/// Implementation of the `Index` trait for read-only element access.
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::f_Vector::fVector;
///
/// let vec = fVector::<3>::from_array([1.0, 2.0, 3.0]);
/// assert_eq!(vec[1], 2.0);
/// ```
impl<const N: usize> Index<usize> for fVector<N> 
{
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

/// Implementation of the `IndexMut` trait for mutable element access.
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::f_Vector::fVector;
///
/// let mut vec = fVector::<3>::from_array([1.0, 2.0, 3.0]);
/// vec[1] = 5.0;
/// assert_eq!(vec[1], 5.0);
/// ```
impl<const N: usize> IndexMut<usize> for fVector<N> 
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Implementation of the `Deref` trait for transparent array access.
///
/// Allows treating the vector as a slice of `f64` values.
impl<const N: usize> Deref for fVector<N> 
{
    type Target = [f64; N];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Implementation of the `DerefMut` trait for mutable transparent array access.
///
/// Allows treating the vector as a mutable slice of `f64` values.
impl<const N: usize> DerefMut for fVector<N> 
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
