use csv::ReaderBuilder;
use ndarray::{Array2, Axis};
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use std::collections::HashMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let data_path = "adult/adult.data";

    // Load and preprocess dataset
    let (features, labels) = load_and_preprocess_dataset(data_path)?;
    println!("Loaded {} samples from {}", labels.len(), data_path);

    // Split dataset into training and testing
    let test_size = 0.2;
    let (train_features, train_labels, test_features, test_labels) =
        split_data(features, labels, test_size);
    println!(
        "Training samples: {}, Testing samples: {}",
        train_labels.len(),
        test_labels.len()
    );

    // Train the model
    let model = train_model(train_features.clone(), train_labels.clone())?;
    println!("Model trained successfully!");

    // Evaluate the model
    evaluate_model(&model, test_features, test_labels);

    Ok(())
}

// Function to load and preprocess dataset (Basic Encoding + Scaling)
fn load_and_preprocess_dataset(file_path: &str) -> Result<(Array2<f64>, Vec<i64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut features = vec![];
    let mut labels = vec![];

    // Dictionary for encoding categorical features
    let mut categorical_encoders: HashMap<usize, HashMap<String, usize>> = HashMap::new();

    for result in rdr.records() {
        let record = result?;
        let mut feature_row = vec![];

        for (idx, value) in record.iter().enumerate() {
            if idx == record.len() - 1 {
                // Last column is the label
                let label: i64 = match value.trim() {
                    ">50K" => 1,
                    "<=50K" => 0,
                    _ => continue, // Skip rows with invalid labels
                };
                labels.push(label);
            } else if is_categorical(value) {
                // Encode categorical values
                let encoder = categorical_encoders
                    .entry(idx)
                    .or_insert_with(HashMap::new);
                let next_index = encoder.len();
                let encoded_value = *encoder
                    .entry(value.to_string())
                    .or_insert(next_index);
                feature_row.push(encoded_value as f64);
            } else {
                // Parse numeric features
                feature_row.push(value.parse::<f64>().unwrap_or(0.0));
            }
        }

        features.push(feature_row);
    }

    let num_rows = features.len();
    let num_cols = features[0].len();
    let features_array = Array2::from_shape_vec((num_rows, num_cols), features.into_iter().flatten().collect())?;

    // Scale numeric features
    let scaled_features = scale_features(features_array);

    Ok((scaled_features, labels))
}

// Helper function to identify categorical values
fn is_categorical(value: &str) -> bool {
    !value.trim().parse::<f64>().is_ok()
}

// Function to scale features
fn scale_features(mut features: Array2<f64>) -> Array2<f64> {
    for mut col in features.axis_iter_mut(Axis(1)) {
        let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        col.map_inplace(|x| *x = (*x - min) / (max - min));
    }
    features
}

// Function to split dataset into training and testing sets
fn split_data(
    features: Array2<f64>,
    labels: Vec<i64>,
    test_size: f64,
) -> (Array2<f64>, Vec<i64>, Array2<f64>, Vec<i64>) {
    let split_idx = (features.nrows() as f64 * (1.0 - test_size)) as usize;

    let (train_features, test_features) = features.view().split_at(Axis(0), split_idx);
    let (train_labels, test_labels) = labels.split_at(split_idx);

    (
        train_features.to_owned(),
        train_labels.to_vec(),
        test_features.to_owned(),
        test_labels.to_vec(),
    )
}

// Function to train a logistic regression model
fn train_model(
    features: Array2<f64>,
    labels: Vec<i64>,
) -> Result<LogisticRegression<f64, i64, DenseMatrix<f64>, Vec<i64>>, Box<dyn Error>> {
    let matrix = DenseMatrix::from_2d_vec(
        &features.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
    );
    let params = LogisticRegressionParameters::default();
    Ok(LogisticRegression::fit(&matrix, &labels, params)?)
}

// Function to evaluate the model's accuracy
fn evaluate_model(
    model: &LogisticRegression<f64, i64, DenseMatrix<f64>, Vec<i64>>,
    test_features: Array2<f64>,
    test_labels: Vec<i64>,
) {
    let matrix = DenseMatrix::from_2d_vec(
        &test_features.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
    );
    let predictions = model.predict(&matrix).unwrap();
    let acc = accuracy(&predictions, &test_labels);
    println!("Model Accuracy: {:.2}%", acc * 100.0);
}

