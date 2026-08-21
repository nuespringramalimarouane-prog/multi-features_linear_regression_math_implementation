import pandas as pd
import numpy as np
from gradient_descent import gradient_descent
from matplotlib import pyplot as plt

def main():
    # Load the CSV file into a DataFrame
    df = pd.read_csv("dataset_clean.csv")

    # Load the features from the CSV file into a DataFrame
    data_x = np.array(df.drop("price", axis=1)[:17000])
    y_train = np.array(df.get("price")[:17000])
    y_test = np.array(df.get("price")[17000:20000])

    # Normalize the features data and the test data using mean normalization
    mean = data_x.mean(axis=0)
    min_val = data_x.min(axis=0)
    max_val = data_x.max(axis=0)

    scale = max_val - min_val
    scale[scale == 0] = 1

    x_train = (data_x - mean) / scale
    x_test  = (np.array(df.drop("price", axis=1)[17000:20000]) - mean) / scale

    # Calculating weights vector and the biais also ploting the cost function 
    w, b, cost_history = gradient_descent(x_train, y_train, alpha=1, num_iters=1000)

    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.show()


    for i in range(y_test.shape[0]):
        # predict the price of the test data 
        y_pred = model(w,b,x_test[i])

        print(f"prediction indx({i}): {y_pred} , real y : {y_test[i]}")

    own_data = [4,2,1903,7550,2,0,0,4,8,1800,200,2005,47.73847,-122.342]
    own_data[own_data == 0] = 1

    y_pr = model(w,b,((own_data - mean) / scale))


    print(f"my own data test : {y_pr}")

    y_preds = [model(w, b, x) for x in x_test]
    # evaluate the model effeciency using the R² score 
    print("R² Gradient Descent:", r2_score_np(y_test, y_preds))

    #this here is using least squares algorithm to get w,b no alpha , no iterations needed 
    X_b = np.c_[np.ones((x_train.shape[0], 1)), x_train]

    theta = np.linalg.lstsq(X_b, y_train, rcond=None)[0]

    b_normal = theta[0]
    w_normal = theta[1:]

    y_preds_normal = np.dot(x_test, w_normal) + b_normal

  
    print("R² Least Squares:", r2_score_np(y_test, y_preds_normal))

 

def model(w,b,x):
    return np.dot(w,x) + b

def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)   # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
    return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    main()