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


def cvx_dc(p, q, r, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, bus, nm):
    v_max = 1.5*bus[1:, 3]
    v_min = .6*bus[1:, 4]

    v_node = cp.Variable((nm, 1))
    p_flow = cp.Variable((nm, 1))
    q_flow = cp.Variable((nm, 1))
    print(np.shape(r.T*(np.power(p_flow, 2) + np.power(q_flow, 2))))
    obj_dc = cp.Minimize(r.T*(np.power(p_flow, 2) + np.power(q_flow, 2)))
    constraints = [a_matrix.T * p_flow == p,
                   a_matrix.T * q_flow == q,
                   v_node == 2 * a_inv * x_matrix * q_flow + 2 * a_inv * r_matrix * p_flow - a_inv * a0 * v0,
                   # v >= np.power(v_min, 2),
                   # v <= np.power(v_max, 2),
                   # q_flow[0] <= bus[0, 5],
                   # p_flow[0] >= bus[0, 6],
                   # q_flow[0] <= bus[0, 7],
                   # q_flow[0] >= bus[0, 8]
                   ]

    prob = cp.Problem(obj_dc, constraints)
    result_dc = prob.solve()

    return result_dc


def cvx_ac(p, q, r, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, bus, nm):
    v_max = 1.5*bus[1:, 3]
    v_min = .6*bus[1:, 4]
    ap = np.hstack((np.array((nm, nm)), -a_inv.T * r_matrix))
    bp = a_inv * p

    av1 = a_inv * x_matrix
    av2 = -a_inv * (2 * r_matrix * a_inv.T * r_matrix + r_matrix * r_matrix + x_matrix * x_matrix)
    av = np.hstack((2*av1, av2))
    bv = a_inv * (2 * r_matrix * a_inv.T * p - a0 * v0)
    aq = np.hstack((a_matrix.T, x_matrix))

    an = []
    bn = []
    cn = []
    dn = []

    # define socp constraints w.s.p matrix-vector
    e_matrix = np.eye(nm)

    for k in range(nm):

        if k == 0:
            an.append(np.array([[2 * e_matrix[:, k].T*ap],
                                [np.hstack(2 * e_matrix[:, k].T, np.zeros(1,nm))],
                                [np.hstack([np.zeros(1, nm), -e_matrix[:, k].T])]]
                               ))

            bn.append(np.array([[2 * e_matrix[:, k].T*bp],
                                [0],
                                [1]]
                               ))

            cn.append(np.array([[np.hstack(np.zeros(1, nm), e_matrix[:, k].T)]]).T)
            dn.append(np.array([[1]]))
        else:
            for j in range(nm):
                if a_matrix[k, j] == 1:
                    pik = j
                break

            an.append(np.array([[2 * e_matrix[:, k].T*ap],
                                [np.hstack(2 * e_matrix[:, k].T, np.zeros(1,nm))],
                                [e_matrix[:, pik].T * av - np.hstack(np.zeros(1,nm), e_matrix[:,k].T)]]
                               ))

            bn.append(np.array([[2 * e_matrix[:, k].T*bp],
                                [0],
                                [e_matrix[:, pik].T*bv]]
                               ))

            cn.append(np.array([[e_matrix[:, pik].T*av + np.hstack(np.zeros(1,nm), e_matrix[:,k].T)]]).T)
            dn.append(np.array([[e_matrix[:, pik].T*bv]]))

    # cvx begin
    q_flow = cp.Variable((nm, 1))
    l_flow = cp.Variable((nm, 1))

    obj_ac = cp.Minimize(np.hstack(np.zeros((nm, 1)), r.T)*np.hstack(q_flow.T, l_flow.T).T)
    soc_constraints = [cp.SOC(cn[i]*np.hstack(q_flow.T, l_flow.T).T + dn[i], an[i]*np.hstack(q_flow.T, l_flow.T).T + bn[i]) for i in range(nm)]
    prob = cp.Problem(cp.Minimize(obj_ac), soc_constraints + [aq * np.hstack(q_flow.T, l_flow.T).T == q,
                                                              av * np.hstack(q_flow.T, l_flow.T).T + bv >= np.power(v_min, 2),
                                                              av * np.hstack(q_flow.T, l_flow.T).T + bv <= np.power(v_max, 2)
                                                              ])

    result_ac = prob.solve()
    return result_ac





