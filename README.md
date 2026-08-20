# Multiple Linear Regression with Gradient Descent

This project trains a multiple linear regression model from scratch with NumPy. It uses batch gradient descent to predict house prices from several property features, plots the training cost as it changes, and reports predictions and an R-squared score for a small held-out slice of the dataset.

## Project Structure

| File | Purpose |
| --- | --- |
| `main.py` | Loads the cleaned data, prepares training and evaluation arrays, trains the model, plots cost convergence, prints predictions, and calculates R-squared. |
| `gradient_descent.py` | Contains the batch gradient-descent loop, gradient calculations, and mean-squared-error cost function. |
| `dataset_clean.csv` | Dataset used by the program. It contains the target and the selected numeric features. |
| `dataset.csv` | Original dataset. It is included for reference but is not loaded by the current program. |

## Requirements

- Python 3.8 or newer
- NumPy
- pandas
- Matplotlib

The project does not currently include a `requirements.txt` file. Install the dependencies with:

```bash
python -m pip install numpy pandas matplotlib
```

On Windows, a virtual environment can be created and activated with:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib
```

## Running the Project

Run the program from the project directory so that the relative CSV path resolves correctly:

```bash
python main.py
```

The program will:

1. Read `dataset_clean.csv` into a pandas DataFrame.
2. Use the first 15,000 rows for training.
3. Use rows 19,990 through 19,999 for evaluation.
4. Normalize every feature using the training data's mean, minimum, and maximum.
5. Train a linear model for 1,000 iterations with a learning rate of `0.1`.
6. Display a cost-versus-iteration plot.
7. Print each evaluation prediction alongside its actual price.
8. Print the evaluation R-squared score.

## Dataset

`dataset_clean.csv` has 21,613 data rows and these columns:

| Column | Description |
| --- | --- |
| `price` | Target house price. |
| `bedrooms` | Number of bedrooms. |
| `bathrooms` | Number of bathrooms. |
| `sqft_living` | Interior living area in square feet. |
| `sqft_lot` | Lot area in square feet. |
| `floors` | Number of floors. |
| `waterfront` | Whether the property has a waterfront view. |
| `view` | View-quality rating. |
| `condition` | Overall condition rating. |
| `grade` | Construction and design grade. |
| `sqft_above` | Above-ground area in square feet. |
| `sqft_basement` | Basement area in square feet. |
| `yr_built` | Original construction year. |
| `lat` | Latitude. |
| `long` | Longitude. |

The `price` column is removed from the feature matrix. The remaining 14 columns are used as model inputs. `dataset.csv` contains the original, wider schema, including fields such as `id`, `date`, `yr_renovated`, `zipcode`, `sqft_living15`, and `sqft_lot15`, but those fields are not used by `main.py`.

## Model

For each property, the model multiplies every input feature by its corresponding weight, adds the results together, and then adds the bias. The result is the predicted price.

### Prediction

The prediction for one property is calculated as:

Prediction = (feature 1 x weight 1) + (feature 2 x weight 2) + ... + (feature n x weight n) + bias

Here, n is the number of features. In this project, n is 14. The weights describe how strongly each feature affects the predicted price, and the bias is the starting value before the feature contributions are added.

The training cost is the half mean squared error. This is the average of the squared differences between the predicted and actual prices, divided by two. A lower cost means that the predictions are closer to the training prices.

### Cost

The cost for all training rows is calculated as:

Cost = (1 / (2 x number of rows)) x sum of (prediction - actual price)^2

The square makes every error positive and gives larger errors more influence. Dividing by the number of rows produces an average, and dividing by two makes the gradient calculation simpler.

During every iteration, the model calculates one gradient for the weights and another gradient for the bias. A gradient measures how much a parameter contributed to the current prediction error across all training rows.

For each weight, the gradient is:

Weight gradient = (1 / number of rows) x sum of (feature value x prediction error)

The bias gradient is:

Bias gradient = (1 / number of rows) x sum of prediction errors

The prediction error for one row is:

Prediction error = predicted price - actual price

The model updates the weights and bias by moving them in the opposite direction of their gradients. The learning rate controls the size of each update.

The update rules are:

New weight = old weight - (learning rate x weight gradient)

New bias = old bias - (learning rate x bias gradient)

`gradient_descent()` starts all weights and the bias at zero. It returns the learned weights, bias, and a cost-history list used by the plot in `main.py`.

## Feature Normalization

Training features are normalized by subtracting the training mean from each value and dividing by the training range. The training range is the training maximum minus the training minimum.

Normalized value = (original value - training mean) / (training maximum - training minimum)

The same training statistics are applied to the evaluation rows. This keeps feature scales comparable and prevents information from the evaluation slice from influencing the normalization.

## Evaluation

The script calculates the coefficient of determination, also called R-squared. This metric compares the model's squared prediction errors with the variation of the actual prices around their average.

R-squared = 1 - (sum of squared prediction errors / sum of squared differences from the actual average)

An R-squared value of `1` indicates perfect predictions, `0` means the model performs like predicting the evaluation average, and a negative value means it performs worse than that baseline. The current evaluation uses only 10 rows, so its score can vary substantially and should not be treated as a robust estimate of generalization performance.

## Implementation Notes and Limitations

- The training and evaluation ranges are hard-coded in `main.py`; changing the dataset size or using the script for general-purpose prediction requires updating those slices.
- There is no randomized train/test split or cross-validation.
- The evaluation rows are separated from the training rows, but the script does not report metrics on a larger test set.
- The learning rate and iteration count are fixed in `main.py` when `gradient_descent()` is called.
- `gradient_descent.py` prints progress every 100 iterations and `main.py` opens an interactive Matplotlib window.
- Normalization assumes every feature has a non-zero range. A constant feature would require special handling to avoid division by zero.
- The model is linear. It does not capture nonlinear relationships or feature interactions unless those features are added explicitly.

## Extending the Project

Useful next improvements would be to:

1. Move the dataset path, training size, evaluation size, learning rate, and iteration count into command-line arguments or a configuration file.
2. Replace the fixed evaluation slice with a reproducible randomized split.
3. Fit normalization statistics on the training split and reuse them for validation and test data.
4. Add metrics such as mean absolute error and root mean squared error.
5. Add tests for the prediction, gradient, cost, normalization, and R-squared functions.
6. Handle missing values and zero-range features explicitly.
