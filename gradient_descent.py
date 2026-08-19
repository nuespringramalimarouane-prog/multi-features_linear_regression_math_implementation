import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x_train, y_train, alpha=0.0001, num_iters=1000):
    n_features = x_train.shape[1]
    m_samples = x_train.shape[0]

    w = np.zeros(n_features)
    b = 0.0
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = dj_wb(x_train, y_train, w, b, m_samples)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Track cost
        cost = cost_function(x_train, y_train, w, b, m_samples)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iter {i}: Cost={cost:.4f}, w={w}, b={b}")

    return w, b, cost_history

def dj_wb(x_train, y_train, w, b, m_samples):
    error = np.dot(x_train, w) + b - y_train
    dj_dw = (np.dot(x_train.T, error)) / m_samples
    dj_db = np.sum(error) / m_samples
    return dj_dw, dj_db

def cost_function(x_train, y_train, w, b, m_samples):
    y_pred = np.dot(x_train, w) + b
    cost = (1/(2*m_samples)) * np.sum((y_pred - y_train)**2)
    return cost

