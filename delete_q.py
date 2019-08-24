from utils import *
import numpy as np
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import optimizers
from matplotlib import pyplot as plt
import scipy.io as sio
from networkmpc import mpc
from scipy.stats import truncnorm
import tensorflow as tf
import copy
# SEED=1234
# np.random.seed(SEED)


nm = 41
no_pv = 5
pf = 0.8
alpha = 0.05
beta = 0.05
no_trajectories = 1
learning_rate = 0.001

train_size = 8000

bus, branch = mpc(pf, beta)
from_to = branch[:, 0:2]
# pv_bus = np.array([bus[0, 11], bus[13, 11], bus[14, 11], bus[16, 11], bus[17, 11]])
pv_set = np.array([1, 14, 15, 17, 18])
cap_set = np.array([2, 31, 41]) # this numbering referred to the bus mps wich has 42 rows
qg_min, qg_max = np.float32(bus[pv_set, 12]), np.float32(bus[pv_set, 11])
v_max = bus[1:, 3]
v_min = bus[1:, 4]
r_vector = np.zeros(nm)
x_vector = np.zeros(nm)

A_tilde = np.zeros((nm, nm+1))

for i in range(nm):
    A_tilde[i, i+1] = -1
    for k in range(nm):
        if branch[k, 1] == i + 2:
            A_tilde[i, int(from_to[k, 0])-1] = 1
            r_vector[i] = branch[k, 2]
            x_vector[i] = branch[k, 3]

a0 = A_tilde[:, 0]
a_matrix = A_tilde[:, 1:]
a_inv = np.linalg.inv(a_matrix)
r_matrix = np.diagflat(r_vector)
x_matrix = np.diagflat(x_vector)
v0 = np.ones(1)

# cvx_ac(p, q, r, x, nm, v_max, v_min)

# load data
n_load = sio.loadmat("bus_47_load_data.mat")
n_solar = sio.loadmat("bus_47_solar_data.mat")
load_data = n_load['bus47loaddata']
solar_data = n_solar['bus47solardata']

permutation_indices = np.random.permutation(10000)

load_data_train = load_data[permutation_indices[:train_size], :]
solar_data_train = solar_data[permutation_indices[:train_size], :]


load_data_test = load_data[permutation_indices[train_size:], :]
solar_data_test = solar_data[permutation_indices[train_size:], :]


pc_train, pg_train, qc_train = pre_process_data(load_data_train, solar_data_train, bus, alpha)  # pc_train pg_train qc_train all 41*10000
p_train = pg_train - pc_train  # 41*10000
data_set_temp_train = np.vstack((p_train, qc_train))
data_set_train = data_set_temp_train.T  # 10000*82
# aq, av, bv, an, bn, cn, dn = pre_process_cvx_ac_matrix(p_train, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, nm)
pc_test, pg_test, qc_test = pre_process_data(load_data_test, solar_data_test, bus, alpha)  # pc_train pg_train qc_train all 41*10000
p_test = pg_test - pc_test  # 41*10000
data_set_temp_test = np.vstack((p_test, qc_test))
data_set_test = data_set_temp_test.T  # 10000*82


data_test_sample = np.squeeze(data_set_test[1, :])



cvx_wqg, qg = cvx_ac_qg(data_test_sample[:nm], data_test_sample[nm:], r_vector, x_vector, a_inv, a_matrix, r_matrix, x_matrix, v_min, v_max, nm, a0, v0, branch, bus)

# qg = np.zeros(nm)
# qg[cap_set - 1] = bus[cap_set - 1, 11]
print(qg[0])
qg[0] = -.001

cvx_withoutqg = cvx_ac(data_test_sample[:nm], qg - data_test_sample[nm:], r_vector, x_vector, a_inv, a_matrix, r_matrix, x_matrix, v_min, v_max, nm, a0, v0, branch, bus)

print("qg is variable:= %f" %cvx_wqg)
print("\nqg is given := %f" %cvx_withoutqg )