# Rustmax Classifier

A lightweight, type-safe softmax classifier implementation in Rust, featuring compile-time sized vectors and efficient gradient descent training.

## Overview

Rustmax Classifier is a machine learning library that implements multi-class softmax regression (also known as multinomial logistic regression) in pure Rust. It leverages Rust's const generics for compile-time size checking and memory safety, making it both efficient and easy to use.

This project was created as a learning exercise to explore machine learning concepts in a systems programming language, combining type safety with performance.

## Features

- ✨ **Type-Safe Vectors**: Fixed-size vectors with compile-time dimension checking using const generics
- 🧠 **Softmax Classification**: Multi-class classification with numerically stable softmax activation
- 📈 **Gradient Descent**: Efficient batch gradient descent training
- 📊 **CSV Support**: Built-in utilities for loading and preprocessing CSV data
- 🔢 **One-Hot Encoding**: Automatic conversion of class labels to one-hot vectors
- 🎯 **No External ML Dependencies**: Pure Rust implementation (only uses `rand` and `csv` crates)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
softmax = { path = "path/to/rustmax-classifier" }
```

Note: The package name is `softmax`, but the library name is `my_softmax_classifier`, so you'll import it as:

```rust
use my_softmax_classifier::softmax::SoftmaxClassifier;
```

## Quick Start

Here's a simple example that demonstrates the complete workflow:

```rust
use my_softmax_classifier::softmax::SoftmaxClassifier;
use my_softmax_classifier::{load_real_data, argmax};

fn main() {
    // Define problem dimensions
    const NUM_FEATURES: usize = 4;  // e.g., 4 features per sample
    const NUM_CLASSES: usize = 3;   // e.g., 3 possible classes

    // Load and prepare data from CSV
    // CSV format: feature1,feature2,...,featureN,label
    let (dataset, labels) = load_real_data::<NUM_FEATURES, NUM_CLASSES>("data.csv");

    // Create a new classifier
    let mut classifier = SoftmaxClassifier::<NUM_CLASSES, NUM_FEATURES>::new();

    // Train the classifier
    // Parameters: dataset, labels, epochs, learning_rate
    classifier.fit(&dataset, &labels, 1000, 0.01);

    // Make predictions
    let predictions = classifier.infer(&dataset);

    // Evaluate accuracy
    let mut correct = 0.0;
    for i in 0..predictions.len() {
        let predicted_class = argmax(predictions[i]).unwrap();
        correct += labels[i][predicted_class];
    }
    let accuracy = correct / predictions.len() as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}
```

## Usage Guide

### 1. Creating a Classifier

The classifier requires two const generic parameters:
- `NUM_CLASSES`: The number of target classes
- `NUM_FEATURES`: The number of input features per sample

```rust
use my_softmax_classifier::softmax::SoftmaxClassifier;

// Create a classifier for 3 classes with 4 features each
let mut classifier = SoftmaxClassifier::<3, 4>::new();
```

The weights are randomly initialized using Xavier/He initialization for better training convergence.

### 2. Preparing Data

#### Loading from CSV

The library provides a convenient function to load data from CSV files:

```rust
use my_softmax_classifier::load_real_data;

const NUM_FEATURES: usize = 4;
const NUM_CLASSES: usize = 3;

// Assumes CSV format: feature1,feature2,...,featureN,label
// Label should be an integer from 0 to NUM_CLASSES-1
let (features, labels) = load_real_data::<NUM_FEATURES, NUM_CLASSES>("data.csv");
```

#### Manual Data Preparation

You can also prepare data manually:

```rust
use my_softmax_classifier::f_Vector::fVector;
use my_softmax_classifier::convert_labels_to_one_hot;

// Create feature vectors
let features = vec![
    fVector::<4>::from_array([5.1, 3.5, 1.4, 0.2]),
    fVector::<4>::from_array([4.9, 3.0, 1.4, 0.2]),
    fVector::<4>::from_array([7.0, 3.2, 4.7, 1.4]),
];

// Create labels (must be one-hot encoded)
let raw_labels = vec![0, 0, 1];
let labels = convert_labels_to_one_hot::<3>(raw_labels);
```

### 3. Training

Train the classifier using gradient descent:

```rust
// Parameters:
// - dataset: reference to feature vectors
// - labels: reference to one-hot encoded labels
// - epochs: number of training iterations
// - alpha: learning rate (typically 0.001 to 0.1)
classifier.fit(&dataset, &labels, 1000, 0.01);
```

**Tips for choosing hyperparameters:**
- Start with a learning rate of 0.01
- Use more epochs (1000-5000) for better convergence
- If loss doesn't decrease, try a smaller learning rate
- If training is too slow, try a larger learning rate

### 4. Making Predictions

Get probability distributions for each sample:

```rust
let predictions = classifier.infer(&test_data);

// predictions[i] is an fVector containing probabilities for each class
// predictions[i][j] = probability that sample i belongs to class j
```

To get the predicted class (the one with highest probability):

```rust
use my_softmax_classifier::argmax;

for prediction in predictions {
    let predicted_class = argmax(prediction).unwrap();
    println!("Predicted class: {}", predicted_class);
}
```

### 5. Working with Vectors

The library provides a custom vector type `fVector` with useful operations:

```rust
use my_softmax_classifier::f_Vector::fVector;

// Create vectors
let v1 = fVector::<3>::new();                      // Random initialization
let v2 = fVector::<3>::from_array([1.0, 2.0, 3.0]); // From array

// Vector operations
let sum = v1 + v2;                    // Element-wise addition
let diff = v1 - v2;                   // Element-wise subtraction
let scaled = v2 * 2.0;                // Scalar multiplication
let dot = v1.dot(&v2);                // Dot product

// Element access
println!("First element: {}", v2[0]);
let mut v3 = v2;
v3[1] = 5.0;  // Mutable access
```

## API Reference

### Core Types

#### `SoftmaxClassifier<NUM_CLASSES, NUM_FEATURES>`

The main classifier struct.

**Methods:**
- `new()` - Creates a new classifier with random weights
- `fit(&mut self, dataset, labels, epochs, alpha)` - Trains the classifier
- `infer(&self, dataset)` - Makes predictions
- `gradient(&self, dataset, labels)` - Computes loss gradient (used internally)

#### `fVector<N>`

A fixed-size vector of `f64` values.

**Methods:**
- `new()` - Creates a vector with random initialization
- `from_array(array)` - Creates a vector from an array
- `dot(&self, other)` - Computes dot product
- `outer_product(&self, other)` - Computes outer product
- `iter()` / `iter_mut()` - Iterators over elements

### Utility Functions

- `load_csv(file_path)` - Loads CSV data without headers
- `load_real_data<N, C>(file_path)` - Loads and prepares data for training
- `convert_labels_to_one_hot<C>(labels)` - Converts class indices to one-hot vectors
- `split_features_and_labels<N>(data)` - Splits data into features and labels
- `argmax<N>(vector)` - Finds index of maximum element
- `transpose_matrix<N, M>(matrix)` - Transposes a matrix

## Examples

### Iris Dataset Classification

```rust
use my_softmax_classifier::softmax::SoftmaxClassifier;
use my_softmax_classifier::{load_real_data, argmax};

const NUM_FEATURES: usize = 4;  // sepal length, sepal width, petal length, petal width
const NUM_CLASSES: usize = 3;   // setosa, versicolor, virginica

fn main() {
    // Load iris dataset
    let (dataset, labels) = load_real_data::<NUM_FEATURES, NUM_CLASSES>("iris.csv");

    // Create and train classifier
    let mut classifier = SoftmaxClassifier::<NUM_CLASSES, NUM_FEATURES>::new();
    classifier.fit(&dataset, &labels, 500, 0.01);

    // Evaluate
    let predictions = classifier.infer(&dataset);
    let mut correct = 0.0;
    for i in 0..predictions.len() {
        correct += labels[i][argmax(predictions[i]).unwrap()];
    }
    println!("Accuracy: {:.2}%", 100.0 * correct / predictions.len() as f64);
}
```

### Manual Training Loop

For more control over training, you can implement your own loop:

```rust
use my_softmax_classifier::softmax::SoftmaxClassifier;

let mut classifier = SoftmaxClassifier::<3, 4>::new();

for epoch in 0..1000 {
    let gradient = classifier.gradient(&dataset, &labels);
    
    // Update weights manually
    for i in 0..classifier.weights.len() {
        classifier.weights[i] = classifier.weights[i] - gradient[i] * 0.01;
    }
    
    if epoch % 100 == 0 {
        // Compute loss or accuracy here
        println!("Epoch {}", epoch);
    }
}
```

## Mathematical Background

### Softmax Function

For a sample with features **x** and class weights **w_k**, the probability of class *k* is:

```
P(y = k | x) = exp(w_k^T * x) / Σ_j exp(w_j^T * x)
```

where the sum is over all classes j.

### Cross-Entropy Loss

The loss function being minimized is the cross-entropy:

```
L = -(1/m) * Σ_i Σ_k y_{i,k} * log(p_{i,k})
```

where:
- m is the number of samples
- y_{i,k} is 1 if sample i belongs to class k (0 otherwise)
- p_{i,k} is the predicted probability

### Gradient Descent

Weights are updated using:

```
w_k := w_k - α * ∇_{w_k} L
```

where α is the learning rate and ∇ denotes the gradient operator.

## Performance Considerations

- **Compile-time checking**: Vector dimensions are checked at compile time, eliminating runtime overhead
- **No heap allocations in hot paths**: Fixed-size arrays avoid dynamic allocations during training
- **Numerical stability**: Uses the max-logit trick to prevent overflow in softmax computation
- **Batch operations**: All training is done in batch mode for efficiency

## Limitations

- Only supports batch gradient descent (no stochastic or mini-batch variants)
- No built-in regularization (L1/L2)
- No support for sparse data
- Training data must fit in memory
- No GPU acceleration

## Contributing

This is a pet project for learning purposes. Feel free to fork and experiment!

## License

This project is available for educational and research purposes.

## Acknowledgments

This project was created as an exercise in implementing machine learning algorithms in Rust. It demonstrates:
- Type-safe numerical computing with const generics
- Gradient-based optimization
- Practical application of ownership and borrowing principles
- Integration with the Rust ecosystem (csv, rand crates)

## Further Reading

- [Softmax Regression](https://en.wikipedia.org/wiki/Softmax_function)
- [Cross-Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Rust Const Generics](https://doc.rust-lang.org/reference/items/generics.html#const-generics)
