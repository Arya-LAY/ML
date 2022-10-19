import copy
import numpy as np

def cost(x,w,b):
    #计算一行，
    f_wb = np.dot(x, w) + b
    return f_wb

def gradient(X,y,w,b):
    dj_db = 0.
    m, n = X.shape
    dj_dw = np.zeros(n) #初始化数组时，一定要初始化其个数

    for i in range(m):
        err = cost(X[i], w, b)-y[i]
        dj_db = dj_db + err
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(X,y,w,b,alpha,iterations):

    w_tmp = copy.deepcopy(w)
    b_tmp = b
    J_history = []
    m = X.shape[0]
    cost_now=0
    for i in range(iterations):

        dj_dw, dj_db = gradient(X,y,w_tmp,b_tmp)

        w_tmp = w_tmp - alpha * dj_dw
        b_tmp = b_tmp - alpha * dj_db

        for j in range(m):
            cost_now = cost_now + ((cost(X[j], w_tmp, b_tmp)-y[j])**2)

        cost_now = cost_now /(2*m)
        J_history.append(cost_now)

    return J_history, w_tmp, b_tmp



