# # Import packages.
# import cvxpy as cp
# import numpy as np
#
# a = np.ones((5, 1))*3
# b = np.ones((5, 1))*2
# print(a)
# print(b)
# print(type(a))
# print(a.T.dot(b))
# print(a*b)


import cvxpy as cp

x = cp.Variable()

# An infeasible problem.
prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)

# An unbounded problem.
prob = cp.Problem(cp.Minimize(x))
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)


import numpy
m = 1
n = 5
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
print(type(A));print(A.shape)
print(type(A*x));print((A*x).shape)
#objective = cp.Minimize(cp.sum_squares(A*x - b))
objective = cp.Minimize(A*x)
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

print("Optimal value", prob.solve())
print("Optimal var")
print(x.value) # A numpy ndarray.
