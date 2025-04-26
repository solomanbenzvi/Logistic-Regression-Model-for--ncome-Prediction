use data::{load_and_preprocess_dataset, split_data};
use model::{train_and_tune_model, ModelResult};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Define the path to the dataset
    let data_path = "adult/adult.data";

    // Load and preprocess the dataset
    let (features, labels) = load_and_preprocess_dataset(data_path)?;

    // Split the dataset into training and testing sets
    let test_size = 0.2;
    let (train_features, train_labels, test_features, test_labels) =
        split_data(features, labels, test_size);

    // Train and tune the model
    let (_best_model, best_accuracy) = train_and_tune_model(train_features, train_labels)?;

    // Print the results
    println!("Model trained successfully!");
    println!("Best Training Accuracy: {:.2}%", best_accuracy * 100.0);

    // Optionally, evaluate on test set (uncomment to use)
    // let test_accuracy = evaluate_model(&best_model, test_features, test_labels);
    // println!("Test Accuracy: {:.2}%", test_accuracy * 100.0);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_execution() {
        // This test ensures the main function runs without errors
        let result = main();
        assert!(result.is_ok(), "Main function failed: {:?}", result.err());
    }
}
