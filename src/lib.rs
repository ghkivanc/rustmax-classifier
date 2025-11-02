//! # Rustmax Classifier
//!
//! A lightweight softmax classifier implementation in Rust using compile-time
//! sized vectors for efficient machine learning operations.
//!
//! This library provides a complete implementation of a multi-class softmax
//! classifier with support for training via gradient descent and inference.
//! It also includes utilities for loading CSV data and converting it to the
//! appropriate format.
//!
//! ## Features
//!
//! - **Type-safe vectors**: Fixed-size vectors with compile-time size checking
//! - **Softmax classification**: Multi-class classification using softmax activation
//! - **Gradient descent**: Training via batch gradient descent
//! - **CSV support**: Load and prepare data from CSV files
//! - **One-hot encoding**: Automatic conversion of labels to one-hot vectors
//!
//! ## Example Usage
//!
//! ```no_run
//! use my_softmax_classifier::softmax::SoftmaxClassifier;
//! use my_softmax_classifier::load_real_data;
//!
//! // Define dimensions
//! const NUM_FEATURES: usize = 4;
//! const NUM_CLASSES: usize = 3;
//!
//! // Load data from CSV
//! let (dataset, labels) = load_real_data::<NUM_FEATURES, NUM_CLASSES>("path/to/data.csv");
//!
//! // Create and train classifier
//! let mut classifier = SoftmaxClassifier::<NUM_CLASSES, NUM_FEATURES>::new();
//! classifier.fit(&dataset, &labels, 1000, 0.01);
//!
//! // Make predictions
//! let predictions = classifier.infer(&dataset);
//! ```

pub mod f_Vector;
use crate::f_Vector::*;
use csv::ReaderBuilder;

/// Softmax classification module.
///
/// This module contains the implementation of a multi-class softmax classifier
/// that uses gradient descent for training.
pub mod softmax{
    use super::*;
    
    /// A multi-class softmax classifier using gradient descent.
    ///
    /// This classifier implements the softmax regression algorithm, which is
    /// a generalization of logistic regression to multiple classes. It uses
    /// a linear model followed by a softmax activation to produce class
    /// probabilities.
    ///
    /// # Type Parameters
    ///
    /// * `NUM_CLASSES` - The number of classes for classification
    /// * `NUM_FEATURES` - The number of features in each input sample
    ///
    /// # Fields
    ///
    /// * `weights` - An array of weight vectors, one per class. Each weight vector
    ///   has `NUM_FEATURES` elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_softmax_classifier::softmax::SoftmaxClassifier;
    /// use my_softmax_classifier::f_Vector::fVector;
    ///
    /// // Create a classifier for 3 classes with 4 features each
    /// let mut classifier = SoftmaxClassifier::<3, 4>::new();
    ///
    /// // Create some dummy data
    /// let data = vec![fVector::<4>::new(); 10];
    /// let labels = vec![fVector::<3>::from_array([1.0, 0.0, 0.0]); 10];
    ///
    /// // Train the classifier
    /// classifier.fit(&data, &labels, 100, 0.01);
    /// ```
    pub struct SoftmaxClassifier<const NUM_CLASSES: usize, const NUM_FEATURES: usize>
    {
        pub weights:[fVector::<NUM_FEATURES>; NUM_CLASSES]
    }

    impl <const NUM_CLASSES:usize, const NUM_FEATURES: usize> SoftmaxClassifier<NUM_CLASSES, NUM_FEATURES>
    {
        /// Creates a new softmax classifier with randomly initialized weights.
        ///
        /// The weights are initialized using Xavier/He initialization, which
        /// helps prevent gradient vanishing/explosion during training.
        ///
        /// # Returns
        ///
        /// A new `SoftmaxClassifier` instance with randomly initialized weights
        ///
        /// # Examples
        ///
        /// ```
        /// use my_softmax_classifier::softmax::SoftmaxClassifier;
        ///
        /// let classifier = SoftmaxClassifier::<3, 4>::new();
        /// ```
        pub fn new()->Self
        {
            let weights:Vec<fVector::<NUM_FEATURES>> = (0..NUM_CLASSES).map(|_| fVector::<NUM_FEATURES>::new()).collect();
            SoftmaxClassifier{weights:weights.try_into().expect("not the same size for vec to array conversion")} 
        }

        /// Performs inference on a dataset, returning class probability distributions.
        ///
        /// For each input sample, this method computes the softmax probabilities
        /// for all classes. The softmax function is numerically stable, using
        /// the max logit trick to prevent overflow.
        ///
        /// # Arguments
        ///
        /// * `dataset` - A reference to a vector of input samples
        ///
        /// # Returns
        ///
        /// A vector of probability distributions, one per input sample.
        /// Each distribution is a vector of length `NUM_CLASSES` where element
        /// `i` represents the probability that the sample belongs to class `i`.
        ///
        /// # Examples
        ///
        /// ```
        /// use my_softmax_classifier::softmax::SoftmaxClassifier;
        /// use my_softmax_classifier::f_Vector::fVector;
        ///
        /// let classifier = SoftmaxClassifier::<3, 4>::new();
        /// let data = vec![fVector::<4>::new(); 5];
        /// let predictions = classifier.infer(&data);
        /// assert_eq!(predictions.len(), 5);
        /// ```
        pub fn infer(&self, dataset: &Vec<fVector<NUM_FEATURES>>) -> Vec<fVector<NUM_CLASSES>> {
        let mut result: Vec<fVector<NUM_CLASSES>> = Vec::new();

                for instance in dataset.iter() {
                let mut dist = [0.0; NUM_CLASSES];
                let max_logit = self.weights.iter()
                    .map(|class_vec| class_vec.dot(instance))
                    .fold(f64::NEG_INFINITY, f64::max); // Find max logit
                let mut sum = 0.0;

                for (idx, class_vec) in self.weights.iter().enumerate() {
                    let temp = (class_vec.dot(instance) - max_logit).exp(); // Stabilize exp
                    sum += temp;
                    dist[idx] = temp;
                }

                result.push(fVector::<NUM_CLASSES>::from_array(dist) * (1.0 / sum));
            }
            result
        }

        /// Computes the gradient of the loss function with respect to the weights.
        ///
        /// This method calculates the gradient of the cross-entropy loss using
        /// the current predictions and the true labels. The gradient is used
        /// in the `fit` method to update the weights via gradient descent.
        ///
        /// # Arguments
        ///
        /// * `dataset` - A reference to a vector of training samples
        /// * `labels` - A reference to a vector of one-hot encoded labels
        ///
        /// # Returns
        ///
        /// An array of gradient vectors, one per class. Each gradient vector
        /// has `NUM_FEATURES` elements representing the partial derivatives
        /// of the loss with respect to that class's weights.
        ///
        /// # Examples
        ///
        /// ```
        /// use my_softmax_classifier::softmax::SoftmaxClassifier;
        /// use my_softmax_classifier::f_Vector::fVector;
        ///
        /// let classifier = SoftmaxClassifier::<3, 4>::new();
        /// let data = vec![fVector::<4>::new(); 5];
        /// let labels = vec![fVector::<3>::from_array([1.0, 0.0, 0.0]); 5];
        /// let gradient = classifier.gradient(&data, &labels);
        /// ```
        pub fn gradient(&self, dataset:&Vec<fVector::<NUM_FEATURES>>, labels:&Vec<fVector::<NUM_CLASSES>>)-> [fVector::<NUM_FEATURES>; NUM_CLASSES]
        {
            let preds = self.infer(dataset);
            let mut gradient_wrt_z: Vec<fVector::<NUM_CLASSES>> = Vec::new();  

            for (label, pred) in labels.iter().zip(preds.iter())
            {
               gradient_wrt_z.push(*pred - *label); //isntxC 
            }

            let mut gradient_wrt_w = [fVector::<NUM_CLASSES>::from_array([0.0; NUM_CLASSES]); NUM_FEATURES];
           
            for _ in 0..NUM_CLASSES//featuresXclasses
            {
               for j in 0..dataset.len()
               {
                    let outer = dataset[j].outer_product::<NUM_CLASSES>(&gradient_wrt_z[j]);
                    for k in 0..gradient_wrt_w.len()
                    {
                        gradient_wrt_w[k] = gradient_wrt_w[k] + outer[k];
                    }
               }
            }

            for i in 0..gradient_wrt_w.len()
            {
                gradient_wrt_w[i] = gradient_wrt_w[i] * (1.0/dataset.len() as f64);
            }
           
            transpose_matrix::<NUM_CLASSES, NUM_FEATURES>(gradient_wrt_w)
        }

        /// Trains the classifier using batch gradient descent.
        ///
        /// This method iteratively updates the classifier's weights to minimize
        /// the cross-entropy loss between predictions and true labels. It performs
        /// `epochs` iterations of gradient descent with learning rate `alpha`.
        ///
        /// # Arguments
        ///
        /// * `dataset` - A reference to a vector of training samples
        /// * `labels` - A reference to a vector of one-hot encoded labels
        /// * `epochs` - The number of training iterations to perform
        /// * `alpha` - The learning rate (step size) for gradient descent
        ///
        /// # Examples
        ///
        /// ```
        /// use my_softmax_classifier::softmax::SoftmaxClassifier;
        /// use my_softmax_classifier::f_Vector::fVector;
        ///
        /// let mut classifier = SoftmaxClassifier::<3, 4>::new();
        /// let data = vec![fVector::<4>::new(); 10];
        /// let labels = vec![fVector::<3>::from_array([1.0, 0.0, 0.0]); 10];
        ///
        /// // Train for 1000 epochs with learning rate 0.01
        /// classifier.fit(&data, &labels, 1000, 0.01);
        /// ```
        pub fn fit(&mut self, dataset:&Vec<fVector::<NUM_FEATURES>>, labels:&Vec<fVector::<NUM_CLASSES>>, epochs:usize, alpha:f64)
        {
            for _ in 0..epochs
            {
                let  gradient = self.gradient(dataset, labels);
                
                for j in 0..self.weights.len()
                {
                    self.weights[j] = self.weights[j] - gradient[j]*alpha;
                }
            }
        }
    }
}
/// Loads data from a CSV file without headers.
///
/// Reads a CSV file and converts each row into a vector of `f64` values.
/// The CSV file should not have headers.
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file
///
/// # Returns
///
/// A vector of vectors, where each inner vector represents one row from the CSV
///
/// # Panics
///
/// Panics if:
/// - The file cannot be opened
/// - A record cannot be read
/// - A value cannot be parsed as `f64`
///
/// # Examples
///
/// ```no_run
/// use my_softmax_classifier::load_csv;
///
/// let data = load_csv("data.csv");
/// println!("Loaded {} rows", data.len());
/// ```
pub fn load_csv(file_path: &str) -> Vec<Vec<f64>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false) // Set to false if your CSV has no headers
        .from_path(file_path)
        .expect("Failed to open file");

    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let row: Vec<f64> = record
            .iter()
            .map(|value| value.parse::<f64>().expect("Failed to parse value"))
            .collect();
        data.push(row);
    }

    data
}

/// Converts a vector of vectors into a vector of fixed-size vectors.
///
/// This function takes data in the form of `Vec<Vec<f64>>` and converts
/// it to `Vec<fVector<N>>` for use with the classifier.
///
/// # Type Parameters
///
/// * `N` - The size of each vector (number of elements per row)
///
/// # Arguments
///
/// * `data` - The input data as a vector of vectors
///
/// # Returns
///
/// A vector of `fVector<N>` instances
///
/// # Panics
///
/// Panics if any row's length does not match `N`
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::{convert_to_fvector, f_Vector::fVector};
///
/// let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let vectors: Vec<fVector<3>> = convert_to_fvector(data);
/// assert_eq!(vectors.len(), 2);
/// ```
pub fn convert_to_fvector<const N: usize>(data: Vec<Vec<f64>>) -> Vec<fVector<N>> {
    data.into_iter()
        .map(|row| {
            let array: [f64; N] = row.try_into().expect("Row length does not match fVector size");
            fVector::<N>::from_array(array)
        })
        .collect()
}


/// Converts class labels to one-hot encoded vectors.
///
/// Takes a vector of class indices and converts them to one-hot encoded
/// vectors suitable for training a classifier.
///
/// # Type Parameters
///
/// * `C` - The number of classes
///
/// # Arguments
///
/// * `labels` - A vector of class indices (0 to C-1)
///
/// # Returns
///
/// A vector of one-hot encoded label vectors
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::{convert_labels_to_one_hot, f_Vector::fVector};
///
/// let labels = vec![0, 1, 2, 0];
/// let one_hot: Vec<fVector<3>> = convert_labels_to_one_hot(labels);
/// assert_eq!(one_hot[0][0], 1.0);
/// assert_eq!(one_hot[1][1], 1.0);
/// ```
pub fn convert_labels_to_one_hot<const C: usize>(labels: Vec<usize>) -> Vec<fVector<C>> {
    labels
        .into_iter()
        .map(|label| {
            let mut one_hot = [0.0; C];
            one_hot[label] = 1.0; // Set the corresponding index to 1
            fVector::<C>::from_array(one_hot)
        })
        .collect()
}

/// Splits data into features and labels.
///
/// Assumes the last column of each row is the label and the first `N`
/// columns are the features.
///
/// # Type Parameters
///
/// * `N` - The number of feature columns
///
/// # Arguments
///
/// * `data` - The input data where each row contains features followed by a label
///
/// # Returns
///
/// A tuple of (features, labels) where:
/// - `features` is a vector of feature vectors
/// - `labels` is a vector of class indices
///
/// # Panics
///
/// Panics if any row has fewer than `N+1` columns
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::{split_features_and_labels, f_Vector::fVector};
///
/// let data = vec![
///     vec![1.0, 2.0, 3.0, 0.0],  // 3 features, label 0
///     vec![4.0, 5.0, 6.0, 1.0],  // 3 features, label 1
/// ];
/// let (features, labels): (Vec<fVector<3>>, Vec<usize>) = 
///     split_features_and_labels(data);
/// assert_eq!(features.len(), 2);
/// assert_eq!(labels, vec![0, 1]);
/// ```
pub fn split_features_and_labels<const N: usize>(
    data: Vec<Vec<f64>>,
) -> (Vec<fVector<N>>, Vec<usize>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for row in data {
        let (feature_row, label) = row.split_at(N); // Split features and label
        features.push(
            fVector::<N>::from_array(feature_row.try_into().expect("Feature size mismatch")),
        );
        labels.push(label[0] as usize); // Assuming the label is the last column
    }

    (features, labels)
}


/// Loads and prepares data from a CSV file for training.
///
/// This is a convenience function that combines loading, splitting, and
/// one-hot encoding in a single call. It loads data from a CSV file,
/// splits it into features and labels, and converts labels to one-hot encoding.
///
/// # Type Parameters
///
/// * `N` - The number of features per sample
/// * `C` - The number of classes
///
/// # Arguments
///
/// * `file_path` - Path to the CSV file
///
/// # Returns
///
/// A tuple of (features, labels) where:
/// - `features` is a vector of feature vectors
/// - `labels` is a vector of one-hot encoded label vectors
///
/// # Examples
///
/// ```no_run
/// use my_softmax_classifier::{load_real_data, f_Vector::fVector};
///
/// const NUM_FEATURES: usize = 4;
/// const NUM_CLASSES: usize = 3;
///
/// let (features, labels) = load_real_data::<NUM_FEATURES, NUM_CLASSES>("iris.csv");
/// println!("Loaded {} samples", features.len());
/// ```
pub fn load_real_data<const N: usize, const C: usize>(file_path: &str) -> (Vec<fVector<N>>, Vec<fVector<C>>) {
    // Step 1: Load data from CSV
    let raw_data = load_csv(file_path);

    // Step 2: Split features and labels
    let (features, raw_labels) = split_features_and_labels::<N>(raw_data);

    // Step 3: Convert labels to one-hot encoding
    let labels = convert_labels_to_one_hot::<C>(raw_labels);

    (features, labels)
}



/// Finds the index of the maximum element in a vector.
///
/// Returns the index of the maximum value in the vector, ignoring NaN values.
/// This is useful for determining the predicted class from a probability distribution.
///
/// # Type Parameters
///
/// * `N` - The size of the vector
///
/// # Arguments
///
/// * `vector` - The input vector
///
/// # Returns
///
/// `Some(index)` where `index` is the position of the maximum value,
/// or `None` if the vector is empty or contains only NaN values
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::{argmax, f_Vector::fVector};
///
/// let vec = fVector::<3>::from_array([0.2, 0.7, 0.1]);
/// let max_idx = argmax(vec);
/// assert_eq!(max_idx, Some(1));
/// ```
pub fn argmax<const N: usize>(vector: fVector<N>) -> Option<usize> {
    //println!("{:?}", vector);
    vector
        .iter()
        .enumerate()
        .filter(|(_, &value)| !value.is_nan()) // Exclude NaN values
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index) // Return the index of the maximum value
}

/// Transposes a matrix represented as an array of vectors.
///
/// Converts an M×N matrix to an N×M matrix by swapping rows and columns.
///
/// # Type Parameters
///
/// * `N` - The number of columns in the input matrix (rows in output)
/// * `M` - The number of rows in the input matrix (columns in output)
///
/// # Arguments
///
/// * `matrix` - The input matrix as an array of M vectors, each of size N
///
/// # Returns
///
/// The transposed matrix as an array of N vectors, each of size M
///
/// # Examples
///
/// ```
/// use my_softmax_classifier::{transpose_matrix, f_Vector::fVector};
///
/// let matrix = [
///     fVector::<3>::from_array([1.0, 2.0, 3.0]),
///     fVector::<3>::from_array([4.0, 5.0, 6.0]),
/// ];
/// let transposed = transpose_matrix::<3, 2>(matrix);
/// assert_eq!(transposed.len(), 3);
/// assert_eq!(transposed[0].len(), 2);
/// ```
pub fn transpose_matrix<const N: usize, const M: usize>(matrix: [fVector::<N>; M]) -> [fVector::<M>; N]{
    let mut transposed: [fVector::<M>; N] = [fVector::<M>::from_array([0.0; M]); N];

    for i in 0..M
    {
        for j in 0..N 
        {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

#[cfg(test)]
mod tests 
{
    use super::*;
    use crate::softmax::SoftmaxClassifier;

    #[test]
    fn from_array() {
        let test_vector = fVector::<5>::from_array([1.0; 5]);

        let sum = test_vector.iter().fold(0.0, |a, b| a + b);

        assert!(sum > 0.0)
    }   

    #[test]
    fn add_test()
    {
        let vec1 = fVector::<3>::new();
        let vec2 = fVector::<3>::new();

        let addVec = vec1.add(vec2);
        let mut result = [0.0; 3];
        result[0] = vec1[0] + vec2[0]; 
        result[1] = vec1[1] + vec2[1]; 
        result[2] = vec1[2] + vec2[2]; 
        let compVec = fVector::<3>{data:result};
        assert_eq!(addVec, compVec)
    }

    #[test]
    fn outer_test()
    {
        let vec1 = fVector::<2>::from_array([1.0, 2.3]);
        let vec2 = fVector::<3>::from_array([2.1, 5.3, 1.1]);

        let outer = vec1.outer_product::<3>(&vec2);
        assert!(outer.len() == 2 && outer[0].data.len() == 3);
    }

    #[test]
    fn mul_test()
    {
        let vec1 = fVector::<3>::new();
        let scaler = 5.0;

        let addVec = vec1.mul(5.0);
        let mut result = [0.0; 3];
        result[0] = vec1[0]*5.0; 
        result[1] = vec1[1]*5.0; 
        result[2] = vec1[2]*5.0; 
        let compVec = fVector::<3>{data:result};
        assert_eq!(addVec, compVec)
    }

    #[test]
    fn new_softmax_test()
    {
        let softmax:SoftmaxClassifier<5,4> = SoftmaxClassifier::new();
        assert!(softmax.weights.len() == 5 && softmax.weights[1].len() == 4);
    }

    #[test]
    fn infer_test()
    {
        let dataset = (0..10).map(|_| fVector::<2>::new()).collect();
        let softmax:SoftmaxClassifier<2,2> = SoftmaxClassifier::new();
        let result = softmax.infer(&dataset);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn gradient_test()
    {
        let dataset = (0..10).map(|_| fVector::<2>::new()).collect();
        let softmax:SoftmaxClassifier<2,2> = SoftmaxClassifier::new();
        let labels = (0..10).map(|_| fVector::<2>::from_array([0.0,1.0])).collect();
        let result = softmax.gradient(&dataset, &labels);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn fit_test()
    {
        const NUM_CLASSES:usize = 3;
        const NUM_FEATURES:usize = 4; 
        let (dataset, labels) = load_real_data::<NUM_FEATURES, NUM_CLASSES>("/home/kivanc/projects/debug_data/iris/iris.data"); 
        let mut softmax:SoftmaxClassifier<NUM_CLASSES,NUM_FEATURES> = SoftmaxClassifier::new();

        softmax.fit(&dataset, &labels, 500, 0.01);

        let result = softmax.infer(&dataset);
        
        let mut correct = 0.0;

        for i in 0..result.len() {
            correct += labels[i][argmax::<NUM_CLASSES>(result[i]).expect("oops")];
        }       
        let accuracy = correct as f64/result.len() as f64;
        println!("{:?}", accuracy);
        assert!(accuracy > 0.8);
    }

}
