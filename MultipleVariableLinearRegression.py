import numpy as np
import copy, math
import matplotlib.pyplot as plt

# problem statement
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
print(X_train.shape[0]) #矩阵行数


# parameter vector w,b
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def prediect_single_loop(x,w,b):

    f_wb = 0
    n = x.shape[0]

    for i in range(n):
        f_wb = f_wb + x[i]*w[i]
    f_wb = f_wb + b
    return f_wb

x_vec = X_train[0,:]

print(f"prediect_loop：{prediect_single_loop(x_vec, w_init, b_init)}")

def prediect(x,w,b):

    f_wb = np.dot(x, w)  #向量化计算
    f_wb = f_wb + b
    return f_wb

print(f"prediect:{prediect(x_vec, w_init, b_init)}")

# compute cost with multiple variables
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        cost = cost + ((prediect(x[i],w,b)) - y[i])**2
    cost = cost /(2*m)
    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"cost:{cost}")

# 计算梯度/导数
def compute_gradient(X, y, w, b):
    m = X.shape[0] #行数
    n = X[0].shape[0] #列数

    dj_dw = np.array([1.0, 0, 0, 0]) #w是一个矩阵

    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w)+b)- y[i] #重复出现的式子可以先算出来
        dj_db = dj_db + err
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]

    dj_db = dj_db / m
    dj_dw = dj_dw/m
    return dj_dw, dj_db

tmp_dj_dw,tmp_dj_db = compute_gradient(X_train, y_train,w_init,b_init)
print(f"开始处的导数：{tmp_dj_dw},{tmp_dj_db}")

# 实现梯度下降

def compute_gradient_descent(x,y,w,b,item,alpha):

    b_tmp = b
    w_tmp = copy.deepcopy(w)
   #dj_dw, dj_db = compute_gradient(x, y, w_tmp, b_tmp) 写到迭代里面！！!!
    n = w_tmp.shape[0]
    J_history = []
    for i in range(item):
        dj_dw, dj_db = compute_gradient(x, y, w_tmp, b_tmp)
        b_tmp = b_tmp - alpha * dj_db
        w_tmp = w_tmp - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(x, y, w_tmp, b_tmp))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(item / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    J = compute_cost(x, y,w_tmp,b_tmp)
    return w_tmp, b_tmp,J_history

iterations = 1000
alpha = 5.0e-7
w_init = np.zeros_like(w_init)
b_init = 0.

w_final,b_final,J_final= compute_gradient_descent(X_train, y_train, w_init, b_init, iterations, alpha)
print(f"final：{w_final},{b_final:0.2f},{J_final}")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_final)
ax2.plot(100 + np.arange(len(J_final[100:])), J_final[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()