import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gradient_descent import gradient_descent


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = "dataset_clean.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42

ALPHA = 1.0
NUM_ITERS = 1000


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("price", axis=1).to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)
    return X, y, df


# ============================================================
# 3. RANDOMIZED TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Return a reproducible shuffled holdout split.

    Use a chronological split instead when row order represents time.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    sample_count = len(X)
    if sample_count < 2:
        raise ValueError("At least two samples are required.")

    permutation = np.random.default_rng(random_state).permutation(sample_count)
    test_count = math.ceil(sample_count * test_size)
    test_indices = permutation[:test_count]
    train_indices = permutation[test_count:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ============================================================
# 4. FEATURE NORMALIZATION
# ============================================================

def normalize_features(X_train, X_test):
    """Normalize using training statistics only."""
    mean = X_train.mean(axis=0)
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    scale = max_val - min_val
    scale[scale == 0] = 1

    X_train_norm = (X_train - mean) / scale
    X_test_norm = (X_test - mean) / scale
    return X_train_norm, X_test_norm


# ============================================================
# 5. LINEAR REGRESSION MODEL
# ============================================================

def model(w, b, X):
    return np.dot(X, w) + b


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        raise ValueError("R² is undefined when the target is constant.")
    return 1 - (ss_res / ss_tot)


def train_gradient_descent(X_train, y_train):
    print("\n========================================")
    print("Gradient Descent Training")
    print("========================================")
    return gradient_descent(
        X_train, y_train, alpha=ALPHA, num_iters=NUM_ITERS
    )


def train_least_squares(X_train, y_train):
    """Closed-form least-squares solution with an intercept."""
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    theta = np.linalg.lstsq(X_b, y_train, rcond=None)[0]
    return theta[1:], theta[0]


def plot_cost(cost_history):
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost J(w, b)")
    plt.title("Gradient Descent — Cost Function Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect Prediction",
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_price(df, X_train, y_train, y_train_pred, feature_name):
    feature_index = df.drop("price", axis=1).columns.get_loc(feature_name)
    feature_values = X_train[:, feature_index]
    sort_indices = np.argsort(feature_values)

    plt.figure(figsize=(9, 6))
    plt.scatter(feature_values, y_train, alpha=0.3, label="Training Data")
    plt.plot(
        feature_values[sort_indices],
        y_train_pred[sort_indices],
        linewidth=2,
        label="Model Predictions",
    )
    plt.xlabel(feature_name)
    plt.ylabel("Price")
    plt.title(f"{feature_name} vs Price — Model Predictions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    X, y, df = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)

    w_gd, b_gd, cost_history = train_gradient_descent(X_train_norm, y_train)
    y_train_pred_gd = model(w_gd, b_gd, X_train_norm)
    y_test_pred_gd = model(w_gd, b_gd, X_test_norm)

    r2_train_gd = r2_score_np(y_train, y_train_pred_gd)
    r2_test_gd = r2_score_np(y_test, y_test_pred_gd)
    print("\nGradient Descent results:")
    print(f"R² Train: {r2_train_gd:.4f}")
    print(f"R² Test:  {r2_test_gd:.4f}")

    w_ls, b_ls = train_least_squares(X_train_norm, y_train)
    y_train_pred_ls = model(w_ls, b_ls, X_train_norm)
    y_test_pred_ls = model(w_ls, b_ls, X_test_norm)

    r2_train_ls = r2_score_np(y_train, y_train_pred_ls)
    r2_test_ls = r2_score_np(y_test, y_test_pred_ls)
    print("\nLeast Squares results:")
    print(f"R² Train: {r2_train_ls:.4f}")
    print(f"R² Test:  {r2_test_ls:.4f}")

    print("\nMODEL COMPARISON")
    print(
        f"Gradient Descent | Train R² = {r2_train_gd:.4f} "
        f"| Test R² = {r2_test_gd:.4f}"
    )
    print(
        f"Least Squares    | Train R² = {r2_train_ls:.4f} "
        f"| Test R² = {r2_test_ls:.4f}"
    )

    plot_cost(cost_history)
    plot_actual_vs_predicted(
        y_test, y_test_pred_gd, "Gradient Descent — Actual vs Predicted"
    )

    feature_name = "sqft_living"
    if feature_name in df.columns:
        plot_feature_vs_price(
            df, X_train, y_train, y_train_pred_gd, feature_name
        )


if __name__ == "__main__":
    main()
