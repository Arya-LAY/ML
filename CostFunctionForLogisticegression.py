import numpy as np
import matplotlib.pyplot as plt


def compute_cost_logistic(X,y,w,b):
    m = X.shape[0]
    loss = 0.0
    for i in range(m):

        z = np.dot(w, X[i]) + b
        f_wb = 1 / (1 + np.exp(-z))
        loss = loss - y[i] * np.log(f_wb) - (1-y[i])*np.log(1-f_wb)

    loss = loss / m
    return loss

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

