import copy

import numpy as np

def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    print(m, n)
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        err = 1 / (np.exp(-(np.dot(w,X[i]) + b)) + 1) - y[i]

        dj_db = dj_db + err

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]

    dj_db = dj_db / m
    dj_dw = dj_dw / m

    return dj_dw, dj_db

def gradient_descent(X,y,w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in


    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b







X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2., 3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )


