pub mod f_Vector;
use crate::f_Vector::*;
use csv::ReaderBuilder;

pub mod softmax{
    use super::*;
    pub struct SoftmaxClassifier<const NUM_CLASSES: usize, const NUM_FEATURES: usize>
    {
        pub weights:[fVector::<NUM_FEATURES>; NUM_CLASSES]
    }

    impl <const NUM_CLASSES:usize, const NUM_FEATURES: usize> SoftmaxClassifier<NUM_CLASSES, NUM_FEATURES>
    {
        pub fn new()->Self
        {
            let weights:Vec<fVector::<NUM_FEATURES>> = (0..NUM_CLASSES).map(|_| fVector::<NUM_FEATURES>::new()).collect();
            SoftmaxClassifier{weights:weights.try_into().expect("not the same size for vec to array conversion")} 
        }

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

pub fn convert_to_fvector<const N: usize>(data: Vec<Vec<f64>>) -> Vec<fVector<N>> {
    data.into_iter()
        .map(|row| {
            let array: [f64; N] = row.try_into().expect("Row length does not match fVector size");
            fVector::<N>::from_array(array)
        })
        .collect()
}


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


pub fn load_real_data<const N: usize, const C: usize>(file_path: &str) -> (Vec<fVector<N>>, Vec<fVector<C>>) {
    // Step 1: Load data from CSV
    let raw_data = load_csv(file_path);

    // Step 2: Split features and labels
    let (features, raw_labels) = split_features_and_labels::<N>(raw_data);

    // Step 3: Convert labels to one-hot encoding
    let labels = convert_labels_to_one_hot::<C>(raw_labels);

    (features, labels)
}



pub fn argmax<const N: usize>(vector: fVector<N>) -> Option<usize> {
    //println!("{:?}", vector);
    vector
        .iter()
        .enumerate()
        .filter(|(_, &value)| !value.is_nan()) // Exclude NaN values
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index) // Return the index of the maximum value
}

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
