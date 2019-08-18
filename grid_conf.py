import numpy as np
import scipy.io as sio
from utils import preprocess_data
from networkmpc import mpc


class Grid:

    nm = 41
    no_pv = 5
    total_iteration = 100
    # load mpc
    pf = 0.8
    alpha = 0.8
    beta = 0.2
    bus, branch = mpc(pf, beta)
    from_to = branch[:, 0:2]
    pv_bus = np.array([bus[1, 11], bus[14, 11], bus[15, 11], bus[17, 11], bus[18, 11]])
    pv_set = np.array([1, 14, 15, 17, 18])
    qg_min, qg_max = np.float32(bus[pv_set, 12]), np.float32(bus[pv_set, 11])

    r = np.zeros((nm, 1))
    x = np.zeros((nm, 1))
    A_tilde = np.zeros((nm, nm+1))

    for i in range(nm):
        A_tilde[i, i+1] = -1
        for k in range(nm):
            if branch[k, 1] == i + 1:
                A_tilde[i, int(from_to[k, 0])] = 1
                r[i] = branch[k, 2]
                x[i] = branch[k, 3]

    a0 = A_tilde[:, 0]
    A = A_tilde[:, 1:]
    A_inv = np.linalg.inv(A)
    R = np.diagflat(r)
    X = np.diagflat(x)
    v0 = np.ones(1)

    # load data
    n_load = sio.loadmat("bus_47_load_data.mat")
    n_solar = sio.loadmat("bus_47_solar_data.mat")
    load_data = n_load['bus47loaddata']
    solar_data = n_solar['bus47solardata']

    pc, pg, qc = preprocess_data(load_data, solar_data, bus, alpha)
    p = pg - pc
    data_set_temp = np.vstack((p, qc))
    data_set = data_set_temp.T
