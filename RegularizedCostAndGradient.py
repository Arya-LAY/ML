import numpy as np

def compute_cost_linear_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost = 0
    reg_cost = 0
    for i in range(m):
        cost = cost + (np.dot(w, X[i])+b - y[i])**2
    cost = cost / (2*m)

    for j in range(n):
        reg_cost = reg_cost + w[j]**2

    reg_cost = lambda_ * reg_cost / (2 * m)

    total_cost = cost + reg_cost

    return total_cost

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

def compute_cost_logistic_reg(X, y, w, b, lambda_1=1):
    m, n = X.shape
    cost = 0
    cost_reg = 0

    for i in range(m):
        f_wb = 1/(1+np.exp(- (np.dot(w, X[i])+b)))
        cost = cost - y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
    cost = cost / m

    for j in range(n):
        cost_reg = cost_reg + (w[j])**2
    cost_reg = cost_reg * lambda_1 / (2*m)
    total_cost = cost + cost_reg

    return total_cost

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

def compute_gradient_linear_reg(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = np.dot(w, X[i])+b-y[i]
        dj_db = dj_db + err
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
    dj_db = dj_db / m
    dj_dw = dj_dw / m

    for j in range(n):
        dj_dw[j] = dj_dw[j]+lambda_/m * w[j]

    return dj_db, dj_dw

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )


def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.0  # scalar

    for i in range(m):
        f_wb_i = 1 / (1+np.exp(-(np.dot(X[i], w) + b))) # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_db, dj_dw
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )