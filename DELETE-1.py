import numpy as np

a = np.ones((2, 2))
a2 = np.ones((2, 2)) * 3

a_vec = np.ones((2, 1))
b = np.zeros((2, 2))
c = np.vstack((a, b))
d = np.hstack((a, b))
print(d)

A = np.eye(2) * 2
print(np.power(a2, 2))
print(a2 * a2)
print(np.dot(a2, a2))

# print(a2.dot(a_vec))
# print('hello')
# print(a2 * 3)
