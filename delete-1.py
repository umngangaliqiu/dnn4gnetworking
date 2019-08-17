import numpy as np
from numpy.linalg import inv

# mean = np.ones(5)
# C = np.ones((5, 5))
#
# pos = np.random.multivariate_normal(mean, C, size=[3, 2])


# print(numpy.shape(pos))
# def trunc_norm(x, mean, bounds, cov):
#     if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
#         return -np.inf
#     else:
#         return -0.5*(x-mean).dot(inv(cov)).dot(x-mean)
#
#
# S = emcee.EnsembleSampler(Nwalkers, Ndim, lnprob_trunc_norm, args = (mean, bounds, C))
#
# rtmvn_gibbs(n, p, Mean, Sigma_chol, R, a, b, z)



from scipy.stats import truncnorm
# a = 0
# b = 1
# r = truncnorm.rvs(a, b, size=1000)

a, b = (-1 - .5) / 1, (1 - .5) / 1
qg_draw = truncnorm.rvs(a, b, loc=.5, scale=1, size=10)

x = np.linspace(truncnorm.ppf(0.01, a, b), truncnorm.ppf(0.99, a, b), 3)
y = truncnorm.pdf(x, .9*x, 1.1*x, loc=x, scale=np.abs(x))

print(y)

x