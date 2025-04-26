use csv::ReaderBuilder;
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::error::Error;
use crate::utils::{is_categorical, scale_features};

/// Loads and preprocesses the dataset from a CSV file.
///
/// # Arguments
/// * `file_path` - Path to the CSV file.
///
/// # Returns
/// A tuple containing the feature matrix (`Array2<f64>`) and labels (`Vec<i64>`).
pub fn load_and_preprocess_dataset(file_path: &str) -> Result<(Array2<f64>, Vec<i64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut features = vec![];
    let mut labels = vec![];

    // Dictionary for one-hot encoding categorical features
    let mut categorical_encoders: HashMap<usize, HashMap<String, usize>> = HashMap::new();

    for result in rdr.records() {
        let record = result?;
        let mut feature_row = vec![];

        // Process each column in the record
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
                // One-hot encode categorical values
                let encoder = categorical_encoders
                    .entry(idx)
                    .or_insert_with(HashMap::new);
                let next_index = encoder.len();
                let encoded_value = *encoder.entry(value.to_string()).or_insert(next_index);
                feature_row.push(encoded_value as f64);
            } else {
                // Numeric features
                feature_row.push(value.parse::<f64>().unwrap_or(0.0));
            }
        }

        features.push(feature_row);
    }

    // Convert features to ndarray
    let num_rows = features.len();
    let num_cols = features.get(0).map_or(0, |row| row.len());
    if num_rows == 0 || num_cols == 0 {
        return Err("Dataset is empty or malformed".into());
    }
    let features_array = Array2::from_shape_vec(
        (num_rows, num_cols),
        features.into_iter().flatten().collect(),
    )?;

    // Scale the features
    let scaled_features = scale_features(features_array);

    Ok((scaled_features, labels))
}

/// Splits the dataset into training and testing sets.
///
/// # Arguments
/// * `features` - Feature matrix.
/// * `labels` - Labels vector.
/// * `test_size` - Fraction of the dataset to use as the test set (e.g., 0.2 for 20%).
///
/// # Returns
/// A tuple of (train_features, train_labels, test_features, test_labels).
pub fn split_data(
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_split_data() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let labels = vec![0, 1, 0, 1, 0];
        let test_size = 0.4; // 40% test, 60% train

        let (train_features, train_labels, test_features, test_labels) =
            split_data(features, labels, test_size);

        // Expected: 60% of 5 rows = 3 rows for training, 2 rows for testing
        assert_eq!(train_features.shape(), &[3, 2]);
        assert_eq!(train_labels.len(), 3);
        assert_eq!(test_features.shape(), &[2, 2]);
        assert_eq!(test_labels.len(), 2);

        // Check the content
        assert_eq!(train_features, array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        assert_eq!(train_labels, vec![0, 1, 0]);
        assert_eq!(test_features, array![[7.0, 8.0], [9.0, 10.0]]);
        assert_eq!(test_labels, vec![1, 0]);
    }

    // Note: Testing `load_and_preprocess_dataset` would require a mock CSV file,
    // which is more complex. You can add such a test by creating a temporary file.
}
