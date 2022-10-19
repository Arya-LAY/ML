import numpy as np
import time


# vector creation
a = np.zeros(4)
print(f"np.zeros(4):a={a},a.shape={a.shape},a data type={a.dtype}")
b = np.zeros((4,))
print(f"np.zeros(4,):b={b},b.shape={b.shape},b data type={b.dtype}")
c = np.random.random_sample(4)
print(f"np.random.random_sample(4):c={c},c shape={c.shape},c data type={c.dtype}")

e = np.arange(4.)
print(f"np.arange(4.): e={e},e.shape={e.shape},e data type={e.dtype}")
f = np.random.rand(4)
print(f"np.random.rand(4): f={f},f.shape={f.shape},f data type={f.dtype}")

m = np.array([5, 4, 3, 2])
print(f"np.array([5,4,3,2]):m={m},m.shape={m.shape},m data type={m.dtype}")
n = np.array([5., 4, 3, 2])
print(f"np.array([5.0,4,3,2]):n={n},n.shape={n.shape},n data type={n.dtype}")



# Indexing on Vectors
a = np.arange(10)
print(a)
# access an element
print(f"a[2].shape:{a[2].shape},a[2]={a[2]}")
# access the last element
print(f"a[-1]={a[-1]}")
# indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print(f"the error message you will see:")
    print(e)


# Slicing on Vectors
a = np.arange(10)
print(a)
# access 5 consecutive element (strat:stop:step)
c = a[2:7:1]
print(c)

c = a[2:7:2]
print(c)

c = a[3:]
print(c)

c = a[:3]
print(c)
# access all element
c = a[:]
print(c)


# single vectors operations
a = np.array([1, 2, 3, 4])
# negate elements of a
b = -a
print(b)
# sum all elements of a,get a scalar
b = np.sum(a)
print(b)
print(b.dtype)
b = np.mean(a)
print(b)
b = a**2
print(b)


# 矢量性操作,维数一致才行
a = np.array([1, 2, 3, 4])
b = np.array([1, 3, 4, 5])
c = a + b
d = 5 * a
print(c)
print(d)


# vector vector dot product
def my_dot(a,b):
    x = 0
    m = a.shape[0]
    for i in range(m):
        x = x + a[i]*b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([4, 5, 6, 7])
print(f"a,b的点积：{my_dot(a,b)}")
c = np.dot(a, b)  #向量化计算，和上面用循环计算结果一样，但是代码更简单，速度更快
print(f"a,b的np.dot:{c}，c.shape:{c.shape}")

# show common Course 1 example
X = np.array([[1],[2],[3],[4]]) #数组的元素还是数组
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")
print(X[1])
print(c)

# matrix creation
a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

mm = np.array([[0,2],
              [1,3]])
print(mm)


# indexing on matrices
# 6个元素，分成两列，-1算是补参数位
a = np.arange(6).reshape(-1, 2)
print(f"a:{a},a.shape:{a.shape}")
# access an element
print(a[2, 0])
# access a row
b = a[2]
print(f"b:{b}")
print(b.shape)


# slicing on matrices
a = np.arange(20).reshape(-1, 10)
print(a)
print(a[0, 2:7:1])  # 在第一行取
print(a[:, 2:7:1])  # 在每一行取
b = a  # 取全部
print(b)
print(a[:, :])
# 取一行
print(a[1])
print(a[1, :])

