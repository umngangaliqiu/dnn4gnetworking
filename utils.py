import numpy as np
import cvxpy as cp


def preprocess_data(loaddata, solardata, bus, alpha):
    pc_max = bus[1:, 5]
    load_data_pc = np.dot(loaddata, np.diag(pc_max))
    qc_max = bus[1:, 7]
    load_data_qc = np.dot(loaddata, np.diag(qc_max))
    pg_max = bus[1:, 9]
    solardata = np.dot(solardata, np.diag(pg_max))

    pc = load_data_pc.transpose()
    qc = load_data_qc.transpose()
    pg = alpha*solardata
    pg = pg.transpose()

    return pc, pg, qc


def cvx_fun(p, q, r, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, bus, nm):
    v_max = 1.5*bus[1:, 3]
    v_min = .6*bus[1:, 4]

    v = cp.Variable(nm)
    p_flow = cp.Variable(nm)
    q_flow = cp.Variable(nm)
    # obj = cp.Minimize(r.T * (np.power(p_flow, 2) + np.power(q_flow, 2)))
    print(np.shape(np.power(p_flow, 2)))

    obj = cp.Minimize(r.T * (np.power(p_flow, 2) + np.power(q_flow, 2)))

    constraints = [a_matrix.T * p_flow == p,
                   a_matrix.T * q_flow == q,
                   v == 2 * a_inv * x_matrix * q_flow + 2 * a_inv * r_matrix * p_flow - a_inv.dot(a0) * v0,
                   # v >= np.power(v_min, 2),
                   # v <= np.power(v_max, 2),
                   # q_flow[0] <= bus[0, 5],
                   # p_flow[0] >= bus[0, 6],
                   # q_flow[0] <= bus[0, 7],
                   # q_flow[0] >= bus[0, 8]
                   ]
    prob = cp.Problem(obj, constraints)
    result = prob.solve()

    print(prob.value, v.value)

    return


def construct_feed_dict(features):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update(features)
    return feed_dict
