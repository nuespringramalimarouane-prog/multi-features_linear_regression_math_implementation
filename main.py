import pandas as pd
import numpy as np
from gradient_descent import gradient_descent
from matplotlib import pyplot as plt

def main():
    # Load the CSV file into a DataFrame
    df = pd.read_csv("dataset_clean.csv")

    # Load the features from the CSV file into a DataFrame
    data_x = np.array(df.drop("price", axis=1)[:15000])
    y_train = np.array(df.get("price")[:15000])
    y_test = np.array(df.get("price")[19990:20000])

    # Normalize the features data and the test data using mean normalization
    mean = data_x.mean(axis=0)
    min_val = data_x.min(axis=0)
    max_val = data_x.max(axis=0)

    x_train = (data_x - mean) / (max_val - min_val)
    x_test  = (np.array(df.drop("price", axis=1)[19990:20000]) - mean) / (max_val - min_val)

    # Calculating weights vector and the biais also ploting the cost function 
    w, b, cost_history = gradient_descent(x_train, y_train, alpha=0.1, num_iters=1000)

    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.show()


    for i in range(y_test.shape[0]):
        # predict the price of the test data 
        y_pred = model(w,b,x_test[i])

        print(f"prediction indx({i}): {y_pred} , real y : {y_test[i]}")

    y_preds = [model(w, b, x) for x in x_test]
    # evaluate the model effeciency using the R² score 
    # this model is a linear model so the best score you could get is 0.46 
    # so the model almost gets correct predictions by 46% 
    # maybe using a non-linear model will fit the data correctly
    print("R²:", r2_score_np(y_test, y_preds))

 

def model(w,b,x):
    return np.dot(w,x) + b

def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)   # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
    return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    main()