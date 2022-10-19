import numpy as np
import matplotlib.pyplot as plt

from JustGradientDescent import gradient_descent

# problem statement
data = np.loadtxt("housedata.txt", delimiter=',')

X_train = data[:, :4] #取前四列
y_train = data[:, -1] #取最后一行
# print(y_train)
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# show data
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("price (1000's")
plt.show()

m, n = X_train.shape
w = np.zeros(n)

b = 0
itertions = 1000
iter = np.arange(1000)
alpha = 1e-1
J_hist, w_final, _, = gradient_descent(X_train, y_train, w, b, alpha, itertions)

plt.plot(iter, J_hist)
plt.show()

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0) #平均值，结果是一维数组
    sigma = np.std(X, axis=0) # 方差，
    X_norm = (X-mu) / sigma
    return X_norm, mu, sigma

X_norm,mu,sigma = zscore_normalize_features(X_train)
X_mean = X_train -mu

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:, 0], X_train[:, 3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

print(f"X_mu:{mu}")
print(f"X_sigms:{sigma}")

J_hist, w_final, _, = gradient_descent(X_norm, y_train, w, b, alpha, itertions)
plt.plot(iter, J_hist)
plt.show()