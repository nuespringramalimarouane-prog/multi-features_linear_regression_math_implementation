import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import gradient_descent


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = "dataset_clean.csv"

TRAIN_SIZE = 17000
TEST_START = 17000
TEST_END = 20000

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
# 3. TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):
    X_train = X[:TRAIN_SIZE]
    y_train = y[:TRAIN_SIZE]

    X_test = X[TEST_START:TEST_END]
    y_test = y[TEST_START:TEST_END]

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. FEATURE NORMALIZATION
# ============================================================

def normalize_features(X_train, X_test):
    """
    Normalize features using statistics calculated
    only from the training data.

    Formula:

        X_norm = (X - mean) / (max - min)
    """

    mean = X_train.mean(axis=0)
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)

    scale = max_val - min_val

    # Avoid division by zero for constant features
    scale[scale == 0] = 1

    X_train_norm = (X_train - mean) / scale
    X_test_norm = (X_test - mean) / scale

    return X_train_norm, X_test_norm


# ============================================================
# 5. LINEAR REGRESSION MODEL
# ============================================================

def model(w, b, X):
    """
    Multi-feature Linear Regression:

        f(X) = Xw + b
    """

    return np.dot(X, w) + b


# ============================================================
# 6. R² SCORE
# ============================================================

def r2_score_np(y_true, y_pred):
    """
    Calculate R²:

             SS_res
        R² = 1 - -------
             SS_tot
    """

    ss_res = np.sum((y_true - y_pred) ** 2)

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    return 1 - (ss_res / ss_tot)


# ============================================================
# 7. TRAIN USING GRADIENT DESCENT
# ============================================================

def train_gradient_descent(X_train, y_train):

    print("\n========================================")
    print("Gradient Descent Training")
    print("========================================")

    w, b, cost_history = gradient_descent(
        X_train,
        y_train,
        alpha=ALPHA,
        num_iters=NUM_ITERS
    )

    return w, b, cost_history


# ============================================================
# 8. TRAIN USING LEAST SQUARES
# ============================================================

def train_least_squares(X_train, y_train):
    """
    Closed-form Least Squares solution.

    Adds a column of ones to represent the bias:

        X_b = [1, x1, x2, ..., xn]

    Then solves:

        theta = argmin ||X_b theta - y||²
    """

    X_b = np.c_[
        np.ones((X_train.shape[0], 1)),
        X_train
    ]

    theta = np.linalg.lstsq(
        X_b,
        y_train,
        rcond=None
    )[0]

    b = theta[0]
    w = theta[1:]

    return w, b


# ============================================================
# 9. PLOT COST CONVERGENCE
# ============================================================

def plot_cost(cost_history):

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(len(cost_history)),
        cost_history
    )

    plt.xlabel("Iterations")
    plt.ylabel("Cost J(w, b)")
    plt.title("Gradient Descent — Cost Function Convergence")

    plt.grid(True)
    plt.tight_layout()

    plt.show()


# ============================================================
# 10. PLOT ACTUAL VS PREDICTED
# ============================================================

def plot_actual_vs_predicted(y_true, y_pred, title):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )

    # Perfect prediction line
    min_value = min(
        y_true.min(),
        y_pred.min()
    )

    max_value = max(
        y_true.max(),
        y_pred.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect Prediction"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. PLOT ONE FEATURE VS PRICE
# ============================================================

def plot_feature_vs_price(
    df,
    X_train,
    y_train,
    y_train_pred,
    feature_name
):

    feature_index = df.drop(
        "price",
        axis=1
    ).columns.get_loc(feature_name)

    feature_values = X_train[:, feature_index]

    plt.figure(figsize=(9, 6))

    plt.scatter(
        feature_values,
        y_train,
        alpha=0.3,
        label="Training Data"
    )

    # Sort values so the prediction line is visually clean
    sort_indices = np.argsort(feature_values)

    plt.plot(
        feature_values[sort_indices],
        y_train_pred[sort_indices],
        linewidth=2,
        label="Model Predictions"
    )

    plt.xlabel(feature_name)
    plt.ylabel("Price")

    plt.title(
        f"{feature_name} vs Price — Model Predictions"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 12. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y, df = load_data(DATASET_PATH)

    print("Dataset shape:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")


    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    X_train_norm, X_test_norm = normalize_features(
        X_train,
        X_test
    )


    # ========================================================
    # GRADIENT DESCENT
    # ========================================================

    w_gd, b_gd, cost_history = train_gradient_descent(
        X_train_norm,
        y_train
    )

    print("\nGradient Descent parameters:")
    print(f"b = {b_gd}")
    print(f"w = {w_gd}")


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_train_pred_gd = model(
        w_gd,
        b_gd,
        X_train_norm
    )

    y_test_pred_gd = model(
        w_gd,
        b_gd,
        X_test_norm
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    r2_train_gd = r2_score_np(
        y_train,
        y_train_pred_gd
    )

    r2_test_gd = r2_score_np(
        y_test,
        y_test_pred_gd
    )

    print("\nGradient Descent results:")
    print(f"R² Train: {r2_train_gd:.4f}")
    print(f"R² Test:  {r2_test_gd:.4f}")


    # ========================================================
    # LEAST SQUARES
    # ========================================================

    w_ls, b_ls = train_least_squares(
        X_train_norm,
        y_train
    )

    y_train_pred_ls = model(
        w_ls,
        b_ls,
        X_train_norm
    )

    y_test_pred_ls = model(
        w_ls,
        b_ls,
        X_test_norm
    )

    r2_train_ls = r2_score_np(
        y_train,
        y_train_pred_ls
    )

    r2_test_ls = r2_score_np(
        y_test,
        y_test_pred_ls
    )

    print("\nLeast Squares results:")
    print(f"R² Train: {r2_train_ls:.4f}")
    print(f"R² Test:  {r2_test_ls:.4f}")


    # ========================================================
    # COMPARE MODELS
    # ========================================================

    print("\n========================================")
    print("MODEL COMPARISON")
    print("========================================")

    print(
        f"Gradient Descent | "
        f"Train R² = {r2_train_gd:.4f} | "
        f"Test R² = {r2_test_gd:.4f}"
    )

    print(
        f"Least Squares   | "
        f"Train R² = {r2_train_ls:.4f} | "
        f"Test R² = {r2_test_ls:.4f}"
    )


    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    # 1. Cost convergence
    plot_cost(cost_history)


    # 2. Actual vs predicted
    plot_actual_vs_predicted(
        y_test,
        y_test_pred_gd,
        "Gradient Descent — Actual vs Predicted"
    )


    # 3. One feature vs price
    #
    # Change this to another feature if desired.
    #
    feature_name = "sqft_living"

    if feature_name in df.columns:

        plot_feature_vs_price(
            df,
            X_train,
            y_train,
            y_train_pred_gd,
            feature_name
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()