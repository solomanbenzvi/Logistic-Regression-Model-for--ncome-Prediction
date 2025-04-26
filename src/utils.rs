use ndarray::{Array2, Axis};

/// Determines if a value is categorical (i.e., cannot be parsed as a float).
pub fn is_categorical(value: &str) -> bool {
    !value.trim().parse::<f64>().is_ok()
}

/// Scales the features to the range [0, 1] using min-max scaling.
pub fn scale_features(mut features: Array2<f64>) -> Array2<f64> {
    for mut col in features.axis_iter_mut(Axis(1)) {
        let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if max == min {
            col.fill(0.0); // Avoid division by zero
        } else {
            col.map_inplace(|x| *x = (*x - min) / (max - min));
        }
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_is_categorical() {
        assert!(is_categorical("category"));
        assert!(is_categorical("123abc"));
        assert!(!is_categorical("123.45"));
        assert!(!is_categorical("0"));
    }

    #[test]
    fn test_scale_features() {
        let mut features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let expected = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
        let scaled = scale_features(features.clone());
        assert_eq!(scaled, expected);

        // Test with identical values (avoid division by zero)
        features = array![[1.0, 2.0], [1.0, 2.0]];
        let expected = array![[0.0, 0.0], [0.0, 0.0]];
        let scaled = scale_features(features);
        assert_eq!(scaled, expected);
    }
}
