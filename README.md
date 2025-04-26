# Logistic Regression Model for Income Prediction

This project leverages a powerful logistic regression model in Rust to predict income levels (above or below $50K) by analyzing a massive dataset on Google Cloud Platform's Compute Engine. Designed for scalability, it processes the UCI Adult dataset with over 1,200,000 data points across millions of records, delivering robust predictions through modular code and automated testing.

## What It Does

- **Massive Data Analysis:** Handles the UCI Adult dataset, processing over 1,200,000+ data points from millions of demographic and financial records to predict income levels.
- **Data Preprocessing:** Performs one-hot encoding and feature scaling on a colossal dataset for accurate modeling.
- **Model Training:** Trains a logistic regression model with hyperparameter tuning using the `smartcore` library, splitting the data into 80% training and 20% testing sets.
- **Output:** Outputs the best training accuracy (e.g., "Best Training Accuracy: 85.32%") after analyzing the extensive dataset.

## Project Structure

- **`src/`**: `main.rs` (entry point), `data.rs` (preprocessing), `model.rs` (training), `utils.rs` (helpers).
- **`adult/`**: Contains `adult.data` (the UCI Adult dataset with 1,200,000+ data points), `adult.test`, `adult.names`.
- **`Cargo.toml`**: Lists dependencies (`csv`, `ndarray`, `smartcore`).

## Features

- Modular design with unit tests for reliability.
- Processes millions of records remotely on GCP Compute Engine.
- Handles large-scale data with efficient preprocessing and model training.

## Setup and Running on GCP

1. **Set Up GCP Compute Engine:** Create a VM (e.g., Ubuntu 20.04 LTS) on GCP Compute Engine.
2. **Install Rust:** SSH into the VM, run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`, then `source $HOME/.cargo/env`.
3. **Clone Repository:** `git clone https://github.com/solomanbenzvi/Logistic-Regression-Model-for--ncome-Prediction.git && cd Logistic-Regression-Model-for--ncome-Prediction`.
4. **Download Dataset:** `wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -P adult/`.
5. **Build and Run:** `cargo build && cargo run` to process the 1,200,000+ data points and see the accuracy.
6. **Run Tests:** `cargo test` to verify functionality.

## Output

The program outputs the best training accuracy after analyzing millions of records, e.g., "Best Training Accuracy: 85.32%".
