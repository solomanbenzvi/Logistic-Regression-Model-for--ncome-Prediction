use csv::ReaderBuilder;
use ndarray::Array2;
use rand::Rng;
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
    let split_idx = (features.nrows() as f64 * (1.0 - test_size)) as usize;

    let (train_features, test_features) = features.view().split_at(ndarray::Axis(0), split_idx);
    let (train_labels, test_labels) = labels.split_at(split_idx);

    // Train the model
    let model = train_model(
        train_features.to_owned(),
        train_labels.to_vec(),
    )?;
    println!("Model trained successfully!");

    // Evaluate the model
    evaluate_model(
        &model,
        test_features.to_owned(),
        test_labels.to_vec(),
    );

    Ok(())
}

// Load and preprocess dataset
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
                let label = match value.trim() {
                    ">50K" => 1,
                    "<=50K" => 0,
                    _ => continue,
                };
                labels.push(label);
            } else if is_categorical(value) {
                // Simplified encoding: Map most categories to a single group
                let encoder = categorical_encoders
                    .entry(idx)
                    .or_insert_with(HashMap::new);
                let encoded_value = *encoder.entry(value.to_string()).or_insert(0); // Map all to 0
                feature_row.push(encoded_value as f64);
            } else {
                let parsed_value = value.parse::<f64>().unwrap_or(0.0);
                let mut rng = rand::thread_rng();
                let noisy_value = parsed_value + (parsed_value * 0.15 * rng.gen_range(-0.5..0.5)); 
                feature_row.push(noisy_value);
            }
        }

        features.push(feature_row);
    }

    let num_rows = features.len();
    let num_cols = features[0].len();
    let features_array = Array2::from_shape_vec(
        (num_rows, num_cols),
        features.into_iter().flatten().collect(),
    )?;

    Ok((features_array, labels))
}

// Identify categorical values
fn is_categorical(value: &str) -> bool {
    !value.trim().parse::<f64>().is_ok()
}

// Train logistic regression
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

// Evaluate model accuracy
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

