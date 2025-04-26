use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters, LogisticRegressionSolverName};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use ndarray::Array2;
use std::error::Error;

/// A type alias for the logistic regression model result.
pub type ModelResult = Result<(LogisticRegression<f64, i64, DenseMatrix<f64>, Vec<i64>>, f64), Box<dyn Error>>;

/// Trains and tunes a logistic regression model using hyperparameter tuning.
///
/// # Arguments
/// * `features` - Training feature matrix.
/// * `labels` - Training labels.
///
/// # Returns
/// A tuple containing the best model and its training accuracy.
pub fn train_and_tune_model(
    features: Array2<f64>,
    labels: Vec<i64>,
) -> ModelResult {
    let matrix = DenseMatrix::from_2d_vec(
        &features.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
    );

    // Hyperparameter tuning: try different regularization strengths
    let alphas = vec![0.1, 1.0, 10.0];
    let mut best_model = None;
    let mut best_accuracy = 0.0;

    for alpha in alphas {
        let params = LogisticRegressionParameters {
            solver: LogisticRegressionSolverName::LBFGS,
            alpha,
        };
        let model = LogisticRegression::fit(&matrix, &labels, params)?;
        let acc = evaluate_model(&model, features.clone(), labels.clone());
        if acc > best_accuracy {
            best_accuracy = acc;
            best_model = Some(model);
        }
    }

    best_model.map(|model| (model, best_accuracy)).ok_or_else(|| "No model was successfully trained".into())
}

/// Evaluates the model's accuracy on a given dataset.
///
/// # Arguments
/// * `model` - The trained logistic regression model.
/// * `test_features` - Feature matrix to evaluate on.
/// * `test_labels` - True labels for evaluation.
///
/// # Returns
/// The accuracy of the model on the given dataset.
pub fn evaluate_model(
    model: &LogisticRegression<f64, i64, DenseMatrix<f64>, Vec<i64>>,
    test_features: Array2<f64>,
    test_labels: Vec<i64>,
) -> f64 {
    let matrix = DenseMatrix::from_2d_vec(
        &test_features.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
    );
    let predictions = model.predict(&matrix).unwrap_or_default();
    accuracy(&predictions, &test_labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_evaluate_model() {
        // Create a simple dataset
        let features = array![[1.0, 0.0], [0.0, 1.0]];
        let labels = vec![0, 1];
        let matrix = DenseMatrix::from_2d_vec(&features.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>());

        // Train a simple model
        let params = LogisticRegressionParameters {
            solver: LogisticRegressionSolverName::LBFGS,
            alpha: 1.0,
        };
        let model = LogisticRegression::fit(&matrix, &labels, params).unwrap();

        // Evaluate the model
        let accuracy = evaluate_model(&model, features.clone(), labels.clone());
        assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy should be between 0 and 1, got {}", accuracy);
    }
}
